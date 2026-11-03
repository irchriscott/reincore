//! Heuristic session naming.
//!
//! A session name is derived from the latest user input, without spending an
//! LLM call. Client scaffolding (context tags, attachments, fenced code) is
//! stripped, then opening lines are scored for how much they describe an actual
//! task. Greetings, smoke tests, and vague openers score low, so they only ever
//! produce a *provisional* name that a later, more specific turn is allowed to
//! replace via [`improved_task_summary`].

use crate::models::{ChatCompletionRequest, ChatMessage};

/// Legacy character budget used by external-tool labels. User titles use words.
pub const MAX_TASK_SUMMARY_CHARS: usize = 60;

/// Name used when no user turn carries any usable signal.
pub const FALLBACK_TASK_SUMMARY: &str = "New session awaiting user request";

/// Score at or above which a candidate is treated as a durable session name.
pub(crate) const SPECIFIC_SCORE: u32 = 8;

/// Extra score a replacement must carry before it overwrites a stored name.
const REPLACEMENT_MARGIN: u32 = 2;

/// Only the opening lines of a turn are considered; later lines are usually
/// pasted context rather than the request itself.
const MAX_CANDIDATE_LINES: usize = 3;

/// Shortest prefix that may be treated as a complete sentence.
const MIN_CLAUSE_CHARS: usize = 8;

/// Politeness and framing that adds nothing to a session name.
const FILLER_PREFIXES: &[&str] = &[
    "i would appreciate it if you could",
    "i'd appreciate it if you could",
    "i was wondering if you could",
    "i'm wondering if you could",
    "what would be the best way to",
    "what should i do to",
    "what should we do to",
    "what is the best way to",
    "what's the best way to",
    "what is a good way to",
    "what's a good way to",
    "would it be possible to",
    "is it possible to",
    "when you have a chance",
    "could you help me",
    "can you help me",
    "do me a favor and",
    "how should i",
    "how should we",
    "how can i",
    "how can we",
    "how would i",
    "how would we",
    "how do i",
    "how do we",
    "we would like you to",
    "we'd like you to",
    "i would like you to",
    "i'd like you to",
    "we want you to",
    "we need you to",
    "i want you to",
    "i need you to",
    "could we please",
    "can we please",
    "would we please",
    "could you please",
    "can you please",
    "would you please",
    "could we",
    "can we",
    "would we",
    "could you",
    "can you",
    "would you",
    "please help me",
    "help me",
    "our goal is to",
    "the goal is to",
    "my goal is to",
    "i am trying to",
    "i'm trying to",
    "we would like to",
    "we'd like to",
    "i would like to",
    "i'd like to",
    "we want to",
    "we need to",
    "i want to",
    "i need to",
    "let us",
    "let's",
    "lets",
    "for this request",
    "for this task",
    "in this repository",
    "in this codebase",
    "in this workspace",
    "in this application",
    "in this system",
    "so basically",
    "basically",
    "in this project",
    "in this repo",
    "in this app",
    "for my project",
    "in my project",
    "as a next step",
    "to start with",
    "first of all",
    "at the moment",
    "at present",
    "right now",
    "currently",
    "quick question",
    "one question",
    "by the way",
    "hey there",
    "hi there",
    "hello",
    "hey",
    "hi",
    "please",
    "kindly",
    "just",
    "actually",
    "alright",
    "okay",
    "ok",
    "also",
    "next",
    "then",
    "now",
    "for now",
    "if possible",
    "if you can",
    "if you could",
];

/// Openers that make a clause an explicit request rather than background
/// context. When a prompt first describes a problem and then asks for an
/// outcome, the request clause makes the better title.
const REQUEST_CLAUSE_PREFIXES: &[&str] = &[
    "i would appreciate it if you could",
    "i'd appreciate it if you could",
    "i was wondering if you could",
    "i'm wondering if you could",
    "what would be the best way to",
    "what should i do to",
    "what should we do to",
    "what is the best way to",
    "what's the best way to",
    "what is a good way to",
    "what's a good way to",
    "would it be possible to",
    "is it possible to",
    "could you help me",
    "can you help me",
    "do me a favor and",
    "how should i",
    "how should we",
    "how can i",
    "how can we",
    "how would i",
    "how would we",
    "how do i",
    "how do we",
    "we would like you to",
    "we'd like you to",
    "i would like you to",
    "i'd like you to",
    "we want you to",
    "we need you to",
    "i want you to",
    "i need you to",
    "could we please",
    "can we please",
    "would we please",
    "could you please",
    "can you please",
    "would you please",
    "could we",
    "can we",
    "would we",
    "could you",
    "can you",
    "would you",
    "please help me",
    "help me",
    "our goal is to",
    "the goal is to",
    "my goal is to",
    "i am trying to",
    "i'm trying to",
    "we would like to",
    "we'd like to",
    "i would like to",
    "i'd like to",
    "we want to",
    "we need to",
    "i want to",
    "i need to",
    "let us",
    "let's",
    "lets",
    "please",
    "kindly",
];

/// Turns that carry no task at all.
const GENERIC_PHRASES: &[&str] = &[
    "hi",
    "hello",
    "hello there",
    "hey",
    "greetings",
    "howdy",
    "yo",
    "sup",
    "hi there",
    "good day",
    "good morning",
    "good afternoon",
    "good evening",
    "test",
    "testing",
    "checking",
    "check",
    "test message",
    "smoke test",
    "hello world",
    "ping",
    "pong",
    "are you there",
    "anyone there",
    "are you working",
    "is this working",
    "do you work",
    "who are you",
    "what can you do",
    "what do you do",
    "help",
    "thanks",
    "thanks again",
    "thank you",
    "much appreciated",
    "no thanks",
    "ok",
    "okay",
    "ok thanks",
    "okay thanks",
    "yes",
    "no",
    "got it",
    "understood",
    "sounds good",
    "looks good",
    "all good",
    "perfect",
    "great",
    "fine",
    "cool",
    "nice",
    "awesome",
    "done",
    "continue",
    "continue please",
    "go on",
    "keep going",
    "keep working",
    "carry on",
    "proceed",
    "resume",
    "retry",
    "again",
    "next",
    "more",
    "status",
    "update",
    "updates",
    "any update",
    "any updates",
    "progress",
    "what next",
    "what's next",
    "what now",
];

struct Candidate {
    text: String,
    score: u32,
}

/// Name a new session from the request that created it.
pub fn derive_task_summary(request: &ChatCompletionRequest) -> String {
    latest_title_input(request)
        .map(|input| summarize_title_input(&input))
        .unwrap_or_else(|| FALLBACK_TASK_SUMMARY.to_string())
}

/// Extract only the latest genuine user input, never the surrounding history.
pub fn latest_title_input(request: &ChatCompletionRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && message.tool_call_id.is_none())
        .and_then(|message| {
            let text = message.full_text();
            crate::user_content::session_user_request(&text)
                .map(clean_message)
                .filter(|text| !text.trim().is_empty())
        })
}

/// Offline, title-only summarization with a word budget instead of a character cut.
pub fn summarize_title_input(input: &str) -> String {
    let message = ChatMessage {
        provider_state: None,
        role: "user".into(),
        content: Some(clean_message(input).into()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let title = best_candidate(&[message])
        .map(|candidate| candidate.text)
        .unwrap_or_else(|| FALLBACK_TASK_SUMMARY.into());
    fit_title_words(&title)
}

fn fit_title_words(title: &str) -> String {
    let words: Vec<_> = title.split_whitespace().take(10).collect();
    match words.len() {
        0 => FALLBACK_TASK_SUMMARY.into(),
        1 => format!("Review request about {}", words[0]),
        2 => format!("Review request: {}", words.join(" ")),
        3 => format!("Task: {}", words.join(" ")),
        _ => words.join(" "),
    }
}

/// Reject malformed LLM output instead of publishing a clipped sentence.
pub fn validate_generated_title(text: &str) -> Option<String> {
    let text = text.trim().trim_matches(['"', '\'']);
    let count = text.split_whitespace().count();
    (text.lines().count() == 1
        && (4..=10).contains(&count)
        && text.chars().count() <= 200
        && !text.contains(['<', '>', '`']))
    .then(|| text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Propose a better name for a session that was opened on a weak first turn.
/// Returns `None` once the stored name is durable enough to keep, so a session
/// is renamed at most a couple of times and never churns.
pub fn improved_task_summary(current: &str, request: &ChatCompletionRequest) -> Option<String> {
    let current_score = specificity_score(current);
    let current_durable = durable_name(current);
    // A bare score above the threshold is not enough to freeze a name. Long
    // problem descriptions and meta/context openers ("I am sending an image
    // but getting this error...", "I was working on this feature...") reach the
    // threshold purely from length, yet still describe no concrete task. Only
    // a name with a topical noun anchor or a concrete reference is durable.
    if current_score >= SPECIFIC_SCORE && current_durable {
        return None;
    }
    // Recency-aware full scan: the replacement name should come from the most
    // recent durable task turn, not the (already-stored) opener.
    let input = latest_title_input(request)?;
    let text = summarize_title_input(&input);
    let candidate = Candidate {
        score: specificity_score(&text),
        text,
    };
    if candidate.score >= SPECIFIC_SCORE && candidate.text != current {
        // When the stored name is non-durable (a vague fragment that scored
        // high from length alone), any durable, specific candidate replaces it
        // outright — there is no quality worth protecting. Only when both are
        // in the same durability class do we demand a margin, so two good
        // names don't churn back and forth.
        let same_class = current_durable == durable_name(&candidate.text);
        if !same_class && durable_name(&candidate.text) {
            return Some(candidate.text);
        }
        if candidate.score > current_score + REPLACEMENT_MARGIN {
            return Some(candidate.text);
        }
    }
    None
}

/// Whether a stored name is stable enough that later turns should NOT rewrite
/// it. The score threshold alone is too permissive: it rewards sheer length,
/// so vague sentence-fragments score high. A durable task name is phrased as a
/// request — an imperative or a direct question — and names a concrete topic.
/// First/second-person *narrative* ("I am sending an image but getting an
/// error...", "I was working on this feature...") describes a situation rather
/// than requesting an outcome, so it always stays renamable.
fn durable_name(text: &str) -> bool {
    if text.split_whitespace().any(is_specific_reference) {
        return true;
    }
    if is_narrative(text) {
        return false;
    }
    has_action_with_topic(text)
}

/// First/second-person framing marks a sentence as describing a situation,
/// not requesting work. Genuine task names are imperatives or questions.
fn is_narrative(text: &str) -> bool {
    const NARRATIVE_OPENERS: &[&str] = &[
        "i", "i'm", "i am", "i've", "i have", "i was", "i'll", "i will", "i would", "i'd", "we",
        "we're", "we are", "we've", "we have", "we were", "we'll", "we will", "my", "me",
    ];
    let lower = text.trim().to_ascii_lowercase();
    NARRATIVE_OPENERS.iter().any(|opener| {
        lower
            .strip_prefix(opener)
            .is_some_and(|rest| rest.starts_with(|c: char| !c.is_alphanumeric()))
    })
}

/// An action verb paired with a topical noun is the minimal signal that a
/// phrase names a concrete task rather than describing a situation. The noun
/// must be a different token from the verb itself.
fn has_action_with_topic(text: &str) -> bool {
    if !crate::lexical::contains_action_verb(text) {
        return false;
    }
    let topics = crate::lexical::relevance_terms(text);
    // Require a topical noun that is not itself an action-form token, so
    // "Review and fix this" (verbs only) and "Continue with the task" (the
    // only surviving token is the continuation verb) do not qualify.
    let verb_tokens: std::collections::HashSet<String> = crate::lexical::words(text)
        .into_iter()
        .filter(|word| crate::lexical::is_action_token(word))
        .collect();
    topics
        .iter()
        .filter(|term| term.chars().count() >= 3)
        .any(|term| !verb_tokens.contains(term))
}

/// Highest-scoring line from the earliest user turn that describes a task.
fn best_candidate(messages: &[ChatMessage]) -> Option<Candidate> {
    best_candidate_impl(messages, false)
}

/// Scan user turns for the strongest task name.
///
/// Selection is recency-aware: a later, durable task description outranks an
/// earlier vague one even when the vague one scored higher from sheer length.
/// That is what lets a session opened on a problem statement ("I am sending
/// an image but getting an error...") be retitled by the follow-up that names
/// the actual fix. Among equally-recency-preferred lines we keep the highest
/// score. When `for_rename` is set the whole transcript is scanned; otherwise
/// the first specific turn short-circuits, which keeps new-session naming
/// anchored to the opening request.
fn best_candidate_impl(messages: &[ChatMessage], for_rename: bool) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for message in messages.iter().filter(|message| message.role == "user") {
        let Some(content) = message.content_str() else {
            continue;
        };
        // A durable candidate from a *later* turn replaces an earlier vague
        // winner. A durable candidate never loses to a later non-durable one.
        let mut turn_best: Option<Candidate> = None;
        for line in candidate_lines(&clean_message(content)) {
            let Some(text) = title_from_line(&line) else {
                continue;
            };
            let score = specificity_score(&text);
            if score == 0 {
                continue;
            }
            if turn_best
                .as_ref()
                .map_or(true, |current| score > current.score)
            {
                turn_best = Some(Candidate { text, score });
            }
        }
        if let Some(candidate) = turn_best {
            let candidate_durable = durable_name(&candidate.text);
            let best_durable = best.as_ref().map_or(false, |b| durable_name(&b.text));
            let replace = match (&best, candidate_durable, best_durable) {
                (None, _, _) => true,
                // Later durable beats earlier non-durable (the core fix).
                (Some(_), true, false) => true,
                // Same durability class: keep the higher score.
                (Some(current), c, b) if c == b => candidate.score > current.score,
                // Earlier durable, later non-durable: keep the durable one.
                _ => false,
            };
            if replace {
                best = Some(candidate);
            }
        }
        if !for_rename
            && best.as_ref().is_some_and(|candidate| {
                candidate.score >= SPECIFIC_SCORE && durable_name(&candidate.text)
            })
        {
            break;
        }
    }
    best
}

/// Phrases that editors and agents prepend before mentioning a workspace or
/// file path. These parsing-only preambles carry no task signal and, if left
/// in place, can dominate the session name because a path is a concrete
/// reference. They do not participate in domain classification.
const PATH_PREAMBLES: [&str; 12] = [
    "the filepath is",
    "the file path is",
    "the path is",
    "in the workspace at",
    "in the directory at",
    "in the folder at",
    "in the project at",
    "the workspace is",
    "the project is located at",
    "the project is at",
    "working directory is",
    "current directory is",
];

/// Connector words that can follow a stripped preamble phrase.
const CONNECTORS: [&str; 4] = ["and", "investigate", "this is", "this is its"];

/// Remove editor/agent path-context preambles and the filepaths themselves,
/// leaving only the user's actual request.  This is the core fix for session
/// names like *"The filepath is C:\Users\...\package.json and this is its..."* or
/// *"In the workspace at C:\..., investigate the Rust engine code to answer:..."*.
fn strip_path_context(input: &str) -> String {
    let mut work = strip_filepaths(input);

    // Iteratively strip known path preambles until no more match.
    loop {
        let lower = work.to_ascii_lowercase();
        let Some(rest) = PATH_PREAMBLES.iter().find_map(|preamble| {
            lower
                .strip_prefix(preamble)
                .map(|rest| (preamble.len(), rest))
        }) else {
            break;
        };

        let after = &work[rest.0..];
        // Consume any leftover whitespace.
        let after = after.trim_start();
        work = after.to_string();

        // Strip a connector phrase if present ("and ...", "investigate ...", etc.).
        let lower = work.to_ascii_lowercase();
        if let Some((len, _)) = CONNECTORS.iter().find_map(|conn| {
            lower
                .starts_with(conn)
                .then(|| (conn.len(), &work[conn.len()..]))
        }) {
            work = work[len..].trim_start().to_string();
        }
    }

    // Clean up leftover artifacts: leading punctuation, empty clauses.
    work = work
        .lines()
        .map(|line| {
            let trimmed = line
                .trim_start_matches([':', ',', ';', '.', '-', ' '])
                .trim_end();
            trimmed.to_string()
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    // Collapse any double spaces left behind by path removal.
    while work.contains("  ") {
        work = work.replace("  ", " ");
    }

    work.trim().to_string()
}

/// Count the number of path segments in `text` until a whitespace terminator.
/// `/a/b/c` → 3, `/v1/messages` → 2, `~/projects/reincore` → 3.
fn count_path_segments(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0;
    let mut i = 0;
    let mut expect_segment = true; // After a separator, we expect a segment.

    while i < text.len() {
        let ch = bytes[i];

        if ch == b'/' || ch == b'\\' {
            if !expect_segment {
                // Two separators in a row → not a real path.
                return 0;
            }
            count += 1;
            expect_segment = false;
            i += 1;
            continue;
        }

        if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
            break;
        }

        // Sentence terminators end path scanning.
        if matches!(ch, b',' | b':' | b';')
            && bytes.get(i + 1).map_or(true, |b| b.is_ascii_whitespace())
        {
            break;
        }

        if ch.is_ascii_alphanumeric() || matches!(ch, b'_' | b'.' | b'-') {
            expect_segment = true;
            i += 1;
            continue;
        }

        // Unknown character → not a path.
        break;
    }

    // If we ended expecting a segment (trailing separator with nothing after),
    // the last separator didn't open a real segment.
    if !expect_segment && count > 0 {
        count -= 1;
    }

    count
}

/// Replace Windows (`C:\Users\...`) and Unix (`/home/...`) filepaths with a
/// placeholder, so the surrounding sentence text survives for session naming.
///
/// We cannot use the `regex` crate in `engine-core`, so detection is manual.
/// A path is identified by a drive letter or a leading `/`, then extends to
/// include `\`, `/`, alphanumerics, spaces, dots, dashes, and underscores until
/// the sequence hits a sentence terminator (`,`, `:`, `;`) followed by a space
/// or a newline.
fn strip_filepaths(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < text.len() {
        // Try to match a filepath starting at position `i`.
        if let Some(end) = detect_filepath(text, bytes, i) {
            // Preserve the last path segment (filename) so the session
            // name retains useful context, e.g. "app.rs" instead of "...".
            let path = &text[i..end];
            let last_segment = path.rsplit(|c| c == '\\' || c == '/').next().unwrap_or("");
            if !last_segment.is_empty() {
                result.push_str(last_segment);
            } else {
                result.push_str("...");
            }
            i = end;
        } else {
            // Copy one UTF-8 character.
            let ch = text[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }
    }

    result
}

/// If a filepath starts at byte offset `start`, return the byte offset just past
/// the end of the path.  Otherwise return `None`.
fn detect_filepath(text: &str, bytes: &[u8], start: usize) -> Option<usize> {
    // Windows drive-letter path: `C:\...` or `c:\...`
    // Two-byte lookahead: letter + ':' + backslash.
    if bytes.len() >= start + 3
        && bytes[start].is_ascii_alphabetic()
        && bytes[start + 1] == b':'
        && (bytes[start + 2] == b'\\' || bytes[start + 2] == b'/')
    {
        return Some(consume_path(text, bytes, start));
    }

    // UNC path: `\\server\share\...`
    if bytes.len() >= start + 3 && bytes[start] == b'\\' && bytes[start + 1] == b'\\' {
        return Some(consume_path(text, bytes, start));
    }

    // Unix absolute path: `/home/user/...`
    // Require at least 3 real segments (e.g. `/home/user/projects`) so that
    // short API paths like `/v1/messages` are not mistaken for file paths.
    if bytes.len() >= start + 2 && bytes[start] == b'/' && bytes[start + 1].is_ascii_alphanumeric()
    {
        if count_path_segments(&text[start..]) >= 3 {
            return Some(consume_path(text, bytes, start));
        }
    }

    // Tilde path: `~/projects/...`
    if bytes.len() >= start + 2 && bytes[start] == b'~' && bytes[start + 1] == b'/' {
        if count_path_segments(&text[start + 1..]) >= 3 {
            return Some(consume_path(text, bytes, start));
        }
    }

    None
}

/// Consume a path starting at `start` and return the offset just past its end.
///
/// Paths may contain spaces (e.g. `C:\Users\Christian Scott\...`), so we keep
/// going as long as the next *non-space* character is a valid path character
/// (`\`, `/`, `_`, `.`, `-`, alphanumeric).  The path ends when:
///   - we hit a sentence terminator (`,`, `:`, `;`) followed by whitespace, or
///   - we hit a newline, or
///   - the next non-space character is not a valid path continuation.
fn consume_path(text: &str, bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < text.len() {
        let ch = bytes[i];

        // Newline always terminates a path.
        if ch == b'\n' || ch == b'\r' {
            break;
        }

        // Sentence terminators end the path when followed by whitespace.
        // A colon *not* followed by whitespace is path-internal (e.g. the
        // drive separator in `c:\Users\...`) and must be consumed, not
        // treated as a terminator.
        if matches!(ch, b',' | b';') {
            let next = bytes.get(i + 1);
            if next.map_or(true, |b| b.is_ascii_whitespace()) {
                break;
            }
        }
        if matches!(ch, b':') {
            let next = bytes.get(i + 1);
            if next.map_or(true, |b| b.is_ascii_whitespace()) {
                break;
            }
            // Non-terminating colon - consume as path-internal.
            i += 1;
            continue;
        }

        // Path-internal characters: always consumed.
        if matches!(ch, b'\\' | b'/' | b'_' | b'.' | b'-') || ch.is_ascii_alphanumeric() {
            i += 1;
            continue;
        }

        // Spaces: consumed only if a path separator (`\` or `/`) appears
        // in the next word (before another space).  This lets us consume
        // spaces inside directory names (e.g. "Christian Scott" in a
        // Windows path) while stopping at the end of the path.
        if ch == b' ' || ch == b'\t' {
            let lookahead = &bytes[i..];
            let mut j = 0;
            // Skip spaces.
            while j < lookahead.len() && (lookahead[j] == b' ' || lookahead[j] == b'\t') {
                j += 1;
            }
            // Scan the next word for a path separator.
            let mut found_sep = false;
            while j < lookahead.len() && lookahead[j] != b' ' && lookahead[j] != b'\t' {
                if lookahead[j] == b'\\' || lookahead[j] == b'/' {
                    found_sep = true;
                    break;
                }
                j += 1;
            }
            if found_sep {
                i += 1;
                continue;
            } else {
                break;
            }
        }

        // Anything else terminates the path.
        break;
    }

    // Trim trailing spaces/punctuation that might have been consumed.
    while i > start && text.as_bytes()[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    i
}

/// Strip client scaffolding and normalize whitespace, keeping line structure.
pub(crate) fn clean_message(raw: &str) -> String {
    let unwrapped = crate::user_content::effective_user_request(raw);
    let without_noise = crate::user_content::strip_user_scaffolding(unwrapped);
    let without_code = strip_code_fences(&without_noise);
    let without_framing = strip_agent_framing(&without_code);
    let without_paths = strip_path_context(&without_framing);
    without_paths
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Plain-text labels injected by agentic clients (editors, AI assistants)
/// that frame the user's request but are not part of it.  These are not XML
/// scaffolding tags (handled by [`strip_user_scaffolding`]) but free-form
/// prefixes like `ACTIVE TASK**:` or `User said:`.
///
/// Each entry is matched case-insensitively at the start of a line.  When the
/// remaining text after stripping is wrapped in quotes, the quotes are also
/// removed so the session name is the actual request, not the framing.
const AGENT_FRAMING_PREFIXES: &[&str] = &[
    // Copilot-style active-task framing
    "active task**",
    "active task**:",
    "active task:",
    "active task",
    // Generic agent labels
    "user said",
    "user said:",
    "user request:",
    "user request",
    "user prompt:",
    "user prompt",
    "user message:",
    "user message",
    "task:",
    "task",
    "request:",
    "request",
    "prompt:",
    "prompt",
    "instruction:",
    "instruction",
    "query:",
    "query",
    "instruction to model:",
    "instruction to model",
];

/// Remove agent/editor plain-text framing prefixes from each line.  When a
/// line consists entirely of `Prefix: "quoted content"`, the quotes are
/// unwrapped so only the user's words remain.  Multiple nested prefixes
/// (e.g. `ACTIVE TASK**: User said: "..."`) are stripped iteratively.
fn strip_agent_framing(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let mut work = line.trim().to_string();

            // Iteratively strip framing prefixes, matching the longest one at
            // each pass.  This handles nested labels like
            // `ACTIVE TASK**: User said: "Add multi-part content"`.
            for _ in 0..AGENT_FRAMING_PREFIXES.len() {
                let lower = work.to_ascii_lowercase();

                // Find the longest matching prefix at the current position.
                let mut best: Option<(&str, &str)> = None;
                for &prefix in AGENT_FRAMING_PREFIXES {
                    if lower.starts_with(prefix) {
                        let rest_after = &work[prefix.len()..];
                        // The prefix must be followed by a word boundary so we
                        // don't strip "task" from "tasklist".
                        let boundary = rest_after
                            .chars()
                            .next()
                            .map_or(true, |c| !c.is_alphanumeric());
                        if boundary && best.map_or(true, |(prev, _)| prefix.len() > prev.len()) {
                            best = Some((prefix, rest_after));
                        }
                    }
                }

                let Some((prefix, _)) = best else { break };

                work = work[prefix.len()..]
                    .trim_start_matches(['*', ':', ' ', '\t', '-', '>'])
                    .trim()
                    .to_string();

                // If nothing remains after stripping, stop.
                if work.is_empty() {
                    break;
                }
            }

            // Unwrap surrounding quotes if the entire remainder is quoted.
            unwrap_quotes(&work).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove matching surrounding quote characters from `text` when the entire
/// string is wrapped.  Supports `"..."`, `'...'`, and `""...""`.
fn unwrap_quotes(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        let first = bytes[0];
        let last = bytes[trimmed.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return trimmed[1..trimmed.len() - 1].trim();
        }
    }
    trimmed
}

fn candidate_lines(cleaned: &str) -> Vec<String> {
    cleaned
        .lines()
        .map(strip_list_marker)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            !line.split_whitespace().next().is_some_and(|word| {
                word.contains(':') && word.chars().all(|ch| ch.is_ascii_digit() || ch == ':')
            })
        })
        .take(MAX_CANDIDATE_LINES)
        .map(str::to_string)
        .collect()
}

fn strip_list_marker(line: &str) -> &str {
    let trimmed = line.trim_start_matches(['#', '>', '-', '*', '•']).trim();
    match trimmed.split_once('.') {
        // Ordered-list markers ("1." / "12.") only.
        Some((head, rest)) if !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()) => {
            rest.trim()
        }
        _ => trimmed,
    }
}

fn strip_code_fences(input: &str) -> String {
    input
        .split("```")
        .step_by(2)
        .collect::<Vec<_>>()
        .join("\n")
        .replace('`', "")
}

/// Articles that make a name read like a sentence rather than a title.
const LEADING_ARTICLES: [&str; 3] = ["the", "a", "an"];

fn title_from_line(line: &str) -> Option<String> {
    let line = line.split(" - ").next().unwrap_or(line);
    let clause = explicit_request_clause(line).unwrap_or_else(|| first_clause(line));
    if is_generic_phrase(clause) {
        return None;
    }
    let preceding_context = line
        .find(clause)
        .map(|start| &line[..start])
        .unwrap_or_default();
    let stripped = strip_fillers(clause);
    // Remove markdown emphasis markers (**bold**, *italic*, __bold__, _italic_)
    // so they don't leak into session names.
    let de_marked = strip_markdown_emphasis(stripped);
    let trimmed = de_marked.trim_matches(|c: char| {
        c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '-' | '"' | '\'')
    });
    // Strip a leading article ("the", "a", "an") so the name reads like
    // a concise title rather than a sentence.
    let without_article = strip_leading_article(trimmed);
    if without_article.is_empty() {
        return None;
    }
    let resolved = resolve_anaphoric_target(without_article, preceding_context);
    if resolved.split_whitespace().count() <= 10 {
        return Some(capitalize(&resolved));
    }
    // Long prose used to be cut mid-clause. Extract its topic across the whole
    // line so the symptom after an introductory clause can reach the title.
    let mut seen = std::collections::HashSet::new();
    let action = resolved
        .split_whitespace()
        .next()
        .filter(|word| crate::lexical::is_action_token(&word.to_lowercase()));
    let source = if action.is_some() {
        resolved.as_str()
    } else {
        line
    };
    let terms = crate::lexical::relevance_terms(source);
    let mut depth = 0usize;
    let without_asides: String = source
        .chars()
        .filter(|ch| {
            if *ch == '(' {
                depth += 1;
                return false;
            }
            if *ch == ')' {
                depth = depth.saturating_sub(1);
                return false;
            }
            depth == 0
        })
        .collect();
    let topic: Vec<_> = without_asides
        .split_whitespace()
        .map(|word| word.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'))
        .filter(|word| {
            let lower = word.to_lowercase();
            (terms.contains(&lower)
                || lower.contains('/')
                || (word.chars().next().is_some_and(char::is_uppercase)
                    && crate::lexical::is_action_token(&lower))
                || (action.is_some() && lower == "app"))
                && !matches!(
                    lower.as_str(),
                    "issue"
                        | "trying"
                        | "got"
                        | "staring"
                        | "previous"
                        | "tries"
                        | "instead"
                        | "launching/trying"
                )
                && !action.is_some_and(|action| action.eq_ignore_ascii_case(word))
                && (action.is_some() || lower != "app")
                && seen.insert(lower)
        })
        .take(9)
        .collect();
    if topic.is_empty() {
        return Some(fit_title_words(&capitalize(&resolved)));
    }
    Some(capitalize(&format!(
        "{} {}",
        action.unwrap_or("Review"),
        topic.join(" ")
    )))
}

/// Prefer a later clause that explicitly asks for an outcome over an opening
/// clause that merely describes the current state or its symptoms.
fn explicit_request_clause(line: &str) -> Option<&str> {
    sentence_clauses(line).into_iter().find(|clause| {
        let lower = clause.to_ascii_lowercase();
        REQUEST_CLAUSE_PREFIXES
            .iter()
            .any(|prefix| starts_with_phrase(&lower, prefix))
            && crate::lexical::contains_action_verb(clause)
    })
}

fn starts_with_phrase(text: &str, phrase: &str) -> bool {
    text.strip_prefix(phrase).is_some_and(|rest| {
        rest.is_empty() || rest.starts_with(|character: char| !character.is_alphanumeric())
    })
}

/// Split prose into sentence-like clauses without breaking identifiers such
/// as `db.rs` or decimal values such as `1.5`.
fn sentence_clauses(line: &str) -> Vec<&str> {
    let mut clauses = Vec::new();
    let mut start = 0;
    let bytes = line.as_bytes();
    for (index, character) in line.char_indices() {
        if index.saturating_sub(start) < MIN_CLAUSE_CHARS
            || !matches!(character, '.' | '?' | '!' | ';')
        {
            continue;
        }
        let next = bytes.get(index + character.len_utf8());
        if next.is_some_and(|byte| !byte.is_ascii_whitespace()) {
            continue;
        }
        let clause = line[start..index].trim();
        if !clause.is_empty() {
            clauses.push(clause);
        }
        start = index + character.len_utf8();
    }
    let tail = line[start..].trim();
    if !tail.is_empty() {
        clauses.push(tail);
    }
    clauses
}

/// Replace a demonstrative object in an explicit request with the nearest
/// concrete identifier from the descriptive context. For example,
/// `optimize that page` becomes `optimize SessionTimeline page`.
fn resolve_anaphoric_target(title: &str, context: &str) -> String {
    let Some(reference) = context
        .split_whitespace()
        .filter(|token| is_specific_reference(token))
        .map(clean_reference_token)
        .filter(|token| !token.is_empty())
        .last()
    else {
        return title.to_string();
    };
    let Some((action, object)) = title.split_once(' ') else {
        return title.to_string();
    };
    if !crate::lexical::contains_action_verb(action) {
        return title.to_string();
    }
    let object_lower = object.to_ascii_lowercase();
    for demonstrative in ["that", "this"] {
        if let Some(rest) = object_lower.strip_prefix(demonstrative).filter(|rest| {
            rest.is_empty() || rest.starts_with(|character: char| !character.is_alphanumeric())
        }) {
            return format!(
                "{action} {reference}{}",
                &object[object.len() - rest.len()..]
            );
        }
    }
    if matches!(object_lower.as_str(), "it" | "itself") {
        return format!("{action} {reference}");
    }
    title.to_string()
}

fn clean_reference_token(token: &str) -> &str {
    token.trim_matches(|character: char| {
        !character.is_alphanumeric()
            && !matches!(character, '_' | '.' | '/' | ':' | '-' | '@' | '#')
    })
}

/// Remove markdown emphasis markers from a line while preserving the text
/// they wrap.  Handles `**bold**`, `*italic*`, `__bold__`, and `_italic_`.
fn strip_markdown_emphasis(text: &str) -> String {
    // Repeatedly strip pairs of emphasis markers.  We do simple passes rather
    // than using regex (unavailable in engine-core).
    let mut result = text.to_string();
    for marker in ["**", "__", "*", "_"] {
        while let Some(start) = result.find(marker) {
            let after_start = start + marker.len();
            if let Some(end) = result[after_start..].find(marker) {
                let end_abs = after_start + end;
                // Keep text before marker + text inside markers + text after closing marker.
                result = format!(
                    "{}{}{}",
                    &result[..start],
                    &result[after_start..end_abs],
                    &result[end_abs + marker.len()..]
                );
            } else {
                // Unpaired marker - strip it.
                result = format!("{}{}", &result[..start], &result[after_start..]);
            }
        }
    }
    result
}

/// Remove a single leading article (case-insensitive) and return the rest.
fn strip_leading_article(text: &str) -> &str {
    let lower = text.to_ascii_lowercase();
    for article in LEADING_ARTICLES {
        if let Some(rest) = lower.strip_prefix(article) {
            // Only strip when the next char is a word boundary (space), so
            // "theory" or "android" are left intact.
            if rest.starts_with(' ') {
                return text[article.len()..].trim_start();
            }
        }
    }
    text
}

fn first_clause(line: &str) -> &str {
    let line = line.trim();
    let bytes = line.as_bytes();
    for (idx, ch) in line.char_indices() {
        if idx < MIN_CLAUSE_CHARS || !matches!(ch, '.' | '?' | '!' | ';' | '-') {
            continue;
        }
        // A terminator only ends the clause when a space or the line follows it,
        // so "db.rs" and "1.5" stay intact.
        let next = bytes.get(idx + ch.len_utf8());
        if next.map_or(true, |byte| byte.is_ascii_whitespace()) {
            return line[..idx].trim();
        }
    }
    line
}

fn strip_fillers(text: &str) -> &str {
    let mut current = text.trim();
    for _ in 0..FILLER_PREFIXES.len() {
        let lower = current.to_ascii_lowercase();
        let Some(rest) = FILLER_PREFIXES.iter().find_map(|prefix| {
            let rest = lower
                .starts_with(prefix)
                .then(|| &current[prefix.len()..])
                .filter(|rest| !rest.starts_with(|c: char| c.is_alphanumeric()))?;
            (!rest.trim().is_empty()).then_some(rest)
        }) else {
            break;
        };
        current = rest.trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ':'));
    }
    current
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// How much a candidate looks like a real task description.
pub(crate) fn specificity_score(text: &str) -> u32 {
    let lower = text.trim().to_ascii_lowercase();
    if is_generic_phrase(&lower) {
        return 0;
    }

    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut score = words.len().min(10) as u32;
    if crate::lexical::contains_action_verb(&lower) {
        score += 5;
    }
    if crate::lexical::is_question(&lower) {
        score += 2;
    }
    if text.split_whitespace().any(is_specific_reference) {
        score += 4;
    }
    score
}

fn is_generic_phrase(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    let normalized = lower.trim_matches(|character: char| !character.is_alphanumeric());
    normalized.is_empty() || GENERIC_PHRASES.contains(&normalized)
}

/// Concrete references are useful naming signal in every domain: filenames
/// and identifiers, but also account/campaign ids, email addresses, hashtags,
/// dates, and other mixed alpha-numeric labels.
fn is_specific_reference(word: &str) -> bool {
    let token = word.trim_matches(|c: char| {
        !c.is_alphanumeric() && !matches!(c, '_' | '.' | '/' | ':' | '-' | '@' | '#')
    });
    if token.len() < 3 {
        return false;
    }
    if token.contains('_') || token.contains("::") || token.contains('/') || token.contains('@') {
        return true;
    }
    if token.starts_with('#') && token.chars().count() > 3 {
        return true;
    }
    if looks_like_version(token) {
        return false;
    }
    if looks_like_date(token) {
        return true;
    }
    let has_letter = token.chars().any(|c| c.is_alphabetic());
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    if has_letter && has_digit {
        return true;
    }
    if looks_like_filename(token) {
        return true;
    }
    // camelCase / PascalCase identifiers.
    token.chars().skip(1).any(|c| c.is_ascii_uppercase()) && token.chars().any(|c| c.is_lowercase())
}

fn looks_like_version(token: &str) -> bool {
    (token.starts_with('v') || token.starts_with('V'))
        && token[1..].contains('.')
        && token[1..].chars().all(|c| c.is_ascii_digit() || c == '.')
}

fn looks_like_date(token: &str) -> bool {
    let parts: Vec<_> = token.split('-').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

/// `name.ext` in any language, without an extension allow-list to maintain.
/// The extension must contain a letter so version numbers stay out.
fn looks_like_filename(token: &str) -> bool {
    let Some((stem, extension)) = token.rsplit_once('.') else {
        return false;
    };
    stem.len() >= 2
        && stem.chars().any(|c| c.is_alphanumeric())
        && (1..=6).contains(&extension.chars().count())
        && extension.chars().all(|c| c.is_ascii_alphanumeric())
        && extension.chars().any(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageContent;

    fn user(content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn request(contents: &[&str]) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages: contents.iter().map(|content| user(content)).collect(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn reported_problem_titles_keep_the_topic_and_word_budget() {
        for (input, topics) in [
            ("In the app, when navigating to Launch or Settings page, the page is not staring from top. It scrolls from the scroll position of the previous page", vec!["launch", "settings", "scroll"]),
            ("Help me fix this issue in the app:\nWhen launching/trying to launch an app with ollama (and do all the setup for it), the app tries to start the engine instead of checking if it running and just do the setup/configuration and launch the app. I tried to launch claude code with ollama and this is what I got:\n15:54:51 → Setting up Ollama in local mode ...", vec!["ollama", "engine"]),
        ] {
            let title = summarize_title_input(input);
            assert!((4..=10).contains(&title.split_whitespace().count()), "{title}");
            for topic in topics { assert!(title.to_lowercase().contains(topic), "{title} missing {topic}"); }
            assert!(!title.contains("15:54"));
            println!("{title}");
        }
    }

    #[test]
    fn titles_never_use_earlier_turns_or_wrapped_context() {
        let req = request(&["Implement secret payments integration", "<context>old billing conversation</context><userRequest>Fix navigation scroll position</userRequest>"]);
        assert_eq!(derive_task_summary(&req), "Fix navigation scroll position");
        assert_eq!(
            derive_task_summary(&request(&["Implement secret payments integration", "hi"])),
            FALLBACK_TASK_SUMMARY
        );
        for input in ["", "Help", "Fix auth", "Fix auth errors", "修复错误", "🦀"] {
            assert!((4..=10).contains(&summarize_title_input(input).split_whitespace().count()));
        }
    }

    #[test]
    fn llm_title_validation_enforces_word_bounds() {
        assert_eq!(
            validate_generated_title("\"Fix navigation scroll position\""),
            Some("Fix navigation scroll position".into())
        );
        for bad in [
            "",
            "Fix scrolling",
            "one two three four five six seven eight nine ten eleven",
            "First title has words\nSecond title has words",
            "<tag>Ignore this title please</tag>",
        ] {
            assert!(validate_generated_title(bad).is_none());
        }
    }

    #[test]
    fn drops_politeness_and_keeps_the_task() {
        let summary = derive_task_summary(&request(&[
            "Hey, can you please add rate limiting to the /v1/messages endpoint?",
        ]));
        // "the" in "the /v1/messages endpoint" is internal, not leading.
        assert_eq!(summary, "Add rate limiting to the /v1/messages endpoint");
    }

    #[test]
    fn greetings_and_smoke_tests_do_not_name_a_session() {
        assert_eq!(
            derive_task_summary(&request(&["hi"])),
            FALLBACK_TASK_SUMMARY
        );
        assert_eq!(
            derive_task_summary(&request(&["testing"])),
            FALLBACK_TASK_SUMMARY
        );
        for prompt in [
            "greetings",
            "hello world",
            "is this working",
            "got it",
            "sounds good",
            "much appreciated",
            "continue please",
            "any updates",
            "what's next",
        ] {
            assert_eq!(
                derive_task_summary(&request(&[prompt])),
                FALLBACK_TASK_SUMMARY,
                "generic phrase should not become a title: {prompt}"
            );
        }
    }

    #[test]
    fn expanded_filler_prefixes_leave_only_the_requested_outcome() {
        for (prompt, expected) in [
            (
                "In this repository, at the moment, could you please configure retry handling in worker.rs?",
                "Configure retry handling in worker.rs",
            ),
            (
                "I'd appreciate it if you could archive expired customer records",
                "Archive expired customer records",
            ),
            (
                "Our goal is to export the monthly invoice report",
                "Export the monthly invoice report",
            ),
            (
                "Do me a favor and sync the interview calendar",
                "Sync the interview calendar",
            ),
        ] {
            assert_eq!(derive_task_summary(&request(&[prompt])), expected, "{prompt}");
        }
    }

    #[test]
    fn scaffolding_and_code_blocks_are_ignored() {
        let content = concat!(
            "<environment_info>OS: Windows</environment_info>\n",
            "<userRequest>Fix the session rollover bug in http.rs</userRequest>\n",
            "```rust\nfn main() {}\n```"
        );
        assert_eq!(
            derive_task_summary(&request(&[content])),
            "Fix the session rollover bug in http.rs"
        );
    }

    #[test]
    fn later_specific_turn_beats_a_generic_opener() {
        let summary = derive_task_summary(&request(&[
            "hello",
            "what does this project do",
            "Now refactor the budget tier state machine",
        ]));
        // "the" in "the budget tier state machine" is internal, not leading.
        assert_eq!(summary, "Refactor the budget tier state machine");
    }

    #[test]
    fn exploratory_opener_is_only_provisional() {
        let provisional = derive_task_summary(&request(&["what does this project do"]));
        assert_eq!(provisional, "What does this project do");

        let improved = improved_task_summary(
            &provisional,
            &request(&[
                "what does this project do",
                "Add a retry policy to the Ollama provider",
            ]),
        );
        // "a" and "the" are internal articles, not leading ones.
        assert_eq!(
            improved.as_deref(),
            Some("Add a retry policy to the Ollama provider")
        );
    }

    #[test]
    fn informational_work_is_a_first_class_task() {
        for (prompt, expected_name) in [
            (
                "Explain Uganda VAT filing deadlines",
                "Explain Uganda VAT filing deadlines",
            ),
            (
                "Summarize Q3 sales performance",
                "Summarize Q3 sales performance",
            ),
            (
                "Describe onboarding options for contractors",
                "Describe onboarding options for contractors",
            ),
            (
                "What are Uganda VAT filing deadlines?",
                "What are Uganda VAT filing deadlines",
            ),
        ] {
            assert!(
                specificity_score(prompt) >= SPECIFIC_SCORE,
                "{prompt} should be considered a specific task"
            );
            assert_eq!(derive_task_summary(&request(&[prompt])), expected_name);
        }
    }

    #[test]
    fn a_specific_name_is_never_rewritten() {
        let current = "Add a retry policy to the Ollama provider";
        assert!(improved_task_summary(
            current,
            &request(&["Also rename the checkpoint token helper"])
        )
        .is_none());
    }

    #[test]
    fn durable_names_require_a_lexical_anchor() {
        // Names with a topical noun or concrete reference are durable.
        for name in [
            "Add a retry policy to the Ollama provider",
            "Task: Optimize SessionTimeline page",
            "What are Uganda VAT filing deadlines",
            "Fix the session-naming logic in summary.rs",
            "Reconcile the July ledger for account ACCT-2048",
        ] {
            assert!(durable_name(name), "{name} should be durable");
        }
        // Long but vague procedural/referential fragments are not.
        for name in [
            "Implement the feature",
            "Implement this feature",
            "Review and fix this",
            "Continue with the task",
        ] {
            assert!(!durable_name(name), "{name} should stay renamable");
        }
    }

    #[test]
    fn a_vague_problem_opener_stays_renamable() {
        // Opens describing a problem (long, weak verb, no topic anchor). This
        // is the shape that previously locked the session to a bad name.
        let opener = "I am sending an image in the request but I am getting this error from ollama";
        // The opener may score high from length alone, but it is not durable,
        // so a later clean request fixes the name.
        let improved = improved_task_summary(
            opener,
            &request(&[
                opener,
                "Please validate the Ollama vision capabilities in ollama.rs",
            ]),
        );
        assert_eq!(
            improved.as_deref(),
            Some("Validate the Ollama vision capabilities in ollama.rs")
        );
    }

    #[test]
    fn a_meta_opening_turn_does_not_block_a_specific_rename() {
        let opener = "I was working on this feature and my chat was cut short before it finished";
        let improved = improved_task_summary(
            opener,
            &request(&[opener, "Fix the budget tier state machine in engine-core"]),
        );
        assert!(
            improved.is_some(),
            "a non-anchored opener must not lock the session name"
        );
    }

    #[test]
    fn long_requests_are_truncated_on_a_word_boundary() {
        let long = format!("Implement {}", "session naming heuristics ".repeat(12));
        let summary = derive_task_summary(&request(&[long.as_str()]));
        assert!((4..=10).contains(&summary.split_whitespace().count()));
        assert!(
            !summary.ends_with("..."),
            "Titles should not end with ellipsis: {summary}"
        );
        assert!(summary.starts_with("Implement session naming"));
    }

    #[test]
    fn explicit_outcome_names_the_session_instead_of_problem_preamble() {
        let prompt = "In my project, currently the SessionTimeline are taking quite a long time to load and it is a very long list as it lists all requests and their events. What is the best way to optimize that page?";

        assert_eq!(
            derive_task_summary(&request(&[prompt])),
            "Task: Optimize SessionTimeline page"
        );
    }

    #[test]
    fn explicit_request_clause_does_not_break_identifiers_or_informational_questions() {
        assert_eq!(
            derive_task_summary(&request(&[
                "The response from db.rs is slow. Can you optimize it?",
            ])),
            "Review request: Optimize db.rs"
        );
        assert_eq!(
            derive_task_summary(&request(&["What are Uganda VAT filing deadlines?"])),
            "What are Uganda VAT filing deadlines"
        );
        assert_eq!(
            derive_task_summary(&request(&[
                "The Q3Report.csv export is incomplete. Would it be possible to archive it?",
            ])),
            "Review request: Archive Q3Report.csv"
        );
    }

    #[test]
    fn file_references_count_as_signal_in_any_language() {
        for token in [
            "Main.java",
            "app.vue",
            "schema.prisma",
            "main.tf",
            "user.rb",
        ] {
            assert!(
                is_specific_reference(token),
                "{token} should read as a file"
            );
        }
        for token in ["e.g", "etc.", "3.14", "v1.2.3", "sentence."] {
            assert!(
                !is_specific_reference(token),
                "{token} should not read as a file"
            );
        }
    }

    #[test]
    fn non_coding_actions_are_specific_task_names() {
        for prompt in [
            "Reconcile the July ledger for account ACCT-2048",
            "Schedule interviews for regional sales candidates",
            "Translate the procurement policy into French",
            "Research suppliers for the Kampala office",
        ] {
            assert!(
                specificity_score(prompt) >= SPECIFIC_SCORE,
                "{prompt} should be considered a specific task"
            );
        }
    }

    #[test]
    fn business_identifiers_count_as_specific_references() {
        for token in [
            "ACCT-2048",
            "Q3-CAMPAIGN",
            "owner@example.com",
            "#renewal",
            "2026-08-24",
        ] {
            assert!(
                is_specific_reference(token),
                "{token} should be a concrete reference"
            );
        }
    }

    // Path stripping tests

    #[test]
    fn strips_windows_path_preamble() {
        let content = "The filepath is c:\\Users\\Christian Scott\\Desktop\\Projects\\ReinCore\\reincore-extension\\package.json and this is its package.json file";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("c:\\"),
            "Windows path should not appear in summary: {summary}"
        );
        assert!(
            !summary.to_lowercase().contains("the filepath is"),
            "Preamble should be stripped: {summary}"
        );
        assert!(
            summary.contains("package.json"),
            "Useful filename should survive: {summary}"
        );
    }

    #[test]
    fn strips_workspace_path_preamble() {
        let content = "In the workspace at c:\\Users\\Christian Scott\\Desktop\\Projects\\ReinCore, investigate the Rust engine code to answer: how does session rollover work?";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("c:\\"),
            "Workspace path should not appear: {summary}"
        );
        assert!(
            !summary.to_lowercase().contains("in the workspace at"),
            "Workspace preamble should be stripped: {summary}"
        );
        assert!(
            summary.contains("session rollover") || summary.contains("Rust engine"),
            "Task content should survive: {summary}"
        );
    }

    #[test]
    fn strips_unix_path_preamble() {
        let content = "In the workspace at /home/dev/projects/reincore, fix the token counting bug";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("/home/dev"),
            "Unix path should not appear: {summary}"
        );
        assert!(
            summary.contains("token counting bug"),
            "Task should survive: {summary}"
        );
    }

    #[test]
    fn strips_bare_filepath_in_text() {
        let content = "Fix the bug in c:\\Users\\dev\\code\\app.rs related to auth";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("c:\\Users"),
            "Bare filepath should not appear: {summary}"
        );
        assert!(
            summary.contains("auth"),
            "Task word should survive: {summary}"
        );
    }

    #[test]
    fn clean_message_strips_path_context() {
        let raw = "The filepath is c:\\Users\\Christian Scott\\Desktop\\Projects\\ReinCore\\reincore-extension\\package.json and this is its contents";
        let cleaned = clean_message(raw);
        assert!(
            !cleaned.contains("c:\\"),
            "Path should be stripped from cleaned message: {cleaned}"
        );
        assert!(
            !cleaned.to_lowercase().contains("the filepath is"),
            "Preamble should be stripped: {cleaned}"
        );
        // The useful tail should survive.
        assert!(
            cleaned.contains("package.json"),
            "Filename should survive: {cleaned}"
        );
    }

    #[test]
    fn strip_filepaths_handles_spaces_in_path() {
        let text = "The path is C:\\Program Files\\My App\\config.toml and check it";
        let stripped = strip_filepaths(text);
        assert!(
            !stripped.contains("C:\\Program Files"),
            "Path with spaces should be stripped: {stripped}"
        );
        assert!(
            stripped.contains("config.toml") || stripped.contains("check it"),
            "Remaining text should survive: {stripped}"
        );
    }
    // Agent framing tests

    #[test]
    fn strips_active_task_framing() {
        let content = "ACTIVE TASK**: User said: \"Please add multi-part content\"";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.to_lowercase().contains("active task"),
            "Agent framing should not appear in summary: {summary}"
        );
        assert!(
            !summary.to_lowercase().contains("user said"),
            "Agent framing should not appear in summary: {summary}"
        );
        assert_eq!(summary, "Task: Add multi-part content");
    }

    #[test]
    fn strips_task_label_framing() {
        let content = "Task: Add multi-part content support to the extension";
        let summary = derive_task_summary(&request(&[content]));
        assert_eq!(summary, "Add multi-part content support to the extension");
    }

    #[test]
    fn strips_user_request_label_framing() {
        let content = "User request: \"Please add multi-part content support\"";
        let summary = derive_task_summary(&request(&[content]));
        assert_eq!(summary, "Add multi-part content support");
    }

    #[test]
    fn strips_prompt_label_framing() {
        let content = "Prompt: Fix the session rollover bug in http.rs";
        let summary = derive_task_summary(&request(&[content]));
        assert_eq!(summary, "Fix the session rollover bug in http.rs");
    }

    #[test]
    fn strip_agent_framing_unwraps_quotes() {
        let input = "User said: \"Add multi-part content\"";
        let result = strip_agent_framing(input);
        assert_eq!(result, "Add multi-part content");
    }

    #[test]
    fn strip_agent_framing_handles_multiple_lines() {
        let input =
            "ACTIVE TASK**: User said: \"Add multi-part content\"\nNext, fix the rollover bug";
        let result = strip_agent_framing(input);
        assert!(
            !result.to_lowercase().contains("active task"),
            "Framing should be stripped: {result}"
        );
        assert!(result.contains("Add multi-part content"));
        assert!(result.contains("fix the rollover bug"));
    }

    #[test]
    fn strip_agent_framing_preserves_non_framed_lines() {
        let input = "Add multi-part content support";
        let result = strip_agent_framing(input);
        assert_eq!(result, "Add multi-part content support");
    }

    #[test]
    fn strip_agent_framing_does_not_partial_match_words() {
        // "task" should not be stripped from "tasklist"
        let input = "tasklist update the session naming";
        let result = strip_agent_framing(input);
        assert_eq!(result, "tasklist update the session naming");
    }

    #[test]
    fn unwrap_quotes_handles_single_and_double() {
        assert_eq!(unwrap_quotes("\"hello\""), "hello");
        assert_eq!(unwrap_quotes("'hello'"), "hello");
        assert_eq!(unwrap_quotes("hello"), "hello");
        assert_eq!(unwrap_quotes("\"hello world\""), "hello world");
    }

    #[test]
    fn strips_markdown_bold_from_session_name() {
        let content =
            "Help me implement this feature: **Desktop app budget/provider endpoints** - noted as placeholders in memory.";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains('*'),
            "Markdown bold markers should not appear in summary: {summary}"
        );
        assert!(
            !summary.contains("noted as"),
            "Text after em dash should not appear in summary: {summary}"
        );
        assert!(
            summary.contains("Desktop app budget/provider"),
            "Summary should contain the task text: {summary}"
        );
    }

    #[test]
    fn strips_markdown_italic_from_session_name() {
        let content = "Add *rate limiting* to the API";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains('*'),
            "Markdown italic markers should not appear in summary: {summary}"
        );
        assert_eq!(summary, "Add rate limiting to the API");
    }

    #[test]
    fn strips_markdown_underscore_bold_from_session_name() {
        let content = "Fix the __session rollover__ bug";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("__"),
            "Markdown underscore bold should not appear in summary: {summary}"
        );
        assert_eq!(summary, "Fix the session rollover bug");
    }

    #[test]
    fn em_dash_splits_clause_in_session_name() {
        let content = "Desktop app budget/provider endpoints - noted as placeholders in memory";
        let summary = derive_task_summary(&request(&[content]));
        assert!(
            !summary.contains("noted as"),
            "Text after em dash should not appear in summary: {summary}"
        );
        assert!(
            summary.contains("Desktop app budget/provider endpoints"),
            "Summary should contain text before em dash: {summary}"
        );
    }
}
