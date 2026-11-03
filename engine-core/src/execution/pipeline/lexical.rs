//! Domain-neutral lexical helpers shared by task intelligence heuristics.
//!
//! ReinCore can front any kind of agent. Keyword signals therefore describe
//! general work (research, communication, operations, analysis, creative
//! production, administration, and software) rather than assuming a coding
//! assistant. None of these lists is authoritative: word-count, structure,
//! and lexical-overlap fallbacks continue to work for unlisted domains.

use std::collections::HashSet;

/// Verbs that commonly introduce a concrete unit of agent work.
pub(crate) const ACTION_VERBS: &[&str] = &[
    "accept",
    "add",
    "adapt",
    "administer",
    "analyze",
    "annotate",
    "answer",
    "apply",
    "approve",
    "archive",
    "arrange",
    "assess",
    "assign",
    "attach",
    "audit",
    "automate",
    "backfill",
    "benchmark",
    "book",
    "brainstorm",
    "build",
    "cache",
    "call",
    "calculate",
    "cancel",
    "categorize",
    "change",
    "chart",
    "check",
    "classify",
    "clean",
    "close",
    "coach",
    "collapse",
    "collect",
    "combine",
    "compare",
    "compose",
    "configure",
    "confirm",
    "connect",
    "consolidate",
    "contact",
    "conduct",
    "convert",
    "copy",
    "coordinate",
    "correct",
    "create",
    "curate",
    "debug",
    "decide",
    "delete",
    "deliver",
    "deploy",
    "design",
    "describe",
    "detect",
    "diagnose",
    "disable",
    "dispatch",
    "display",
    "distribute",
    "document",
    "download",
    "draft",
    "edit",
    "email",
    "enable",
    "enforce",
    "enrich",
    "escalate",
    "estimate",
    "evaluate",
    "execute",
    "expand",
    "explain",
    "export",
    "expose",
    "extract",
    "file",
    "filter",
    "finalize",
    "find",
    "fix",
    "forecast",
    "format",
    "fulfill",
    "generate",
    "give",
    "grade",
    "group",
    "handle",
    "hide",
    "hire",
    "identify",
    "illustrate",
    "import",
    "implement",
    "improve",
    "index",
    "inspect",
    "install",
    "integrate",
    "introduce",
    "investigate",
    "interview",
    "invoice",
    "label",
    "launch",
    "list",
    "load",
    "localize",
    "map",
    "measure",
    "merge",
    "migrate",
    "model",
    "moderate",
    "monitor",
    "move",
    "negotiate",
    "notify",
    "onboard",
    "open",
    "optimize",
    "order",
    "organize",
    "outline",
    "paginate",
    "parse",
    "patch",
    "pay",
    "perform",
    "persist",
    "pin",
    "plan",
    "post",
    "prepare",
    "prioritize",
    "process",
    "procure",
    "produce",
    "profile",
    "promote",
    "proofread",
    "publish",
    "purchase",
    "quote",
    "read",
    "reconcile",
    "recommend",
    "reduce",
    "refactor",
    "refresh",
    "register",
    "release",
    "remind",
    "remove",
    "rename",
    "render",
    "reorganize",
    "replace",
    "reply",
    "report",
    "reproduce",
    "reserve",
    "research",
    "resolve",
    "respond",
    "restore",
    "retrieve",
    "review",
    "rewrite",
    "route",
    "run",
    "scan",
    "schedule",
    "screen",
    "search",
    "select",
    "send",
    "set",
    "shorten",
    "show",
    "simplify",
    "submit",
    "solve",
    "sort",
    "source",
    "start",
    "stop",
    "suggest",
    "summarize",
    "summarise",
    "support",
    "switch",
    "sync",
    "teach",
    "tell",
    "test",
    "track",
    "translate",
    "transcribe",
    "triage",
    "troubleshoot",
    "tune",
    "uninstall",
    "unpublish",
    "update",
    "upgrade",
    "upload",
    "validate",
    "verify",
    "view",
    "virtualize",
    "visualize",
    "visualise",
    "wire",
    "write",
];

/// Words and phrases whose meaning depends on prior conversational context.
/// These are domain-independent discourse signals rather than task keywords.
const PRIOR_CONTEXT_WORDS: &[&str] = &[
    "additional",
    "above",
    "else",
    "further",
    "it",
    "its",
    "otherwise",
    "this",
    "that",
    "those",
    "these",
    "them",
    "they",
    "same",
    "again",
    "also",
    "instead",
    "previous",
    "previously",
    "earlier",
    "continue",
    "resume",
    "proceed",
    "another",
    "remaining",
    "more",
    "still",
    "your",
    "yours",
    "revert",
    "undo",
    "retry",
    "next",
    "now",
    "then",
];

const PRIOR_CONTEXT_PHRASES: &[&str] = &[
    "as i said",
    "like i said",
    "as before",
    "as discussed",
    "as mentioned",
    "mentioned before",
    "mentioned earlier",
    "as a follow up",
    "following up from",
    "follow up to",
    "from before",
    "from earlier",
    "from there",
    "with that in mind",
    "having said that",
    "that said",
    "on second thought",
    "you just",
    "last one",
    "previous one",
    "prior one",
    "earlier one",
    "other one",
    "previous answer",
    "earlier answer",
    "last answer",
    "previous response",
    "earlier response",
    "last response",
    "previous request",
    "earlier request",
    "last request",
    "previous version",
    "earlier version",
    "last version",
    "the rest",
    "remaining work",
    "what remains",
    "finish the rest",
    "complete the rest",
    "keep the rest",
    "leave the rest",
    "everything else",
    "keep going",
    "keep it going",
    "carry on",
    "continue with",
    "continue from",
    "continue where",
    "continue working",
    "continue the previous",
    "please continue",
    "can you continue",
    "could you continue",
    "would you continue",
    "let us continue",
    "let's continue",
    "resume with",
    "resume from",
    "resume the previous",
    "please resume",
    "can you resume",
    "could you resume",
    "pick up where",
    "pick up from",
    "where we left off",
    "where you left off",
    "pick this back up",
    "pick that back up",
    "go on with",
    "go on from",
    "please go on",
    "move on to the next",
    "back to the",
    "go back to",
    "go back and",
    "come back to",
    "return to the",
    "return to that",
    "back to that",
    "follow up on",
    "following up on",
    "build on that",
    "building on that",
    "based on that",
    "add to that",
    "do the same",
    "same as before",
    "same as last time",
    "same approach",
    "same format",
    "same style",
    "same layout",
    "same structure",
    "try that again",
    "try it again",
    "do that again",
    "try the other",
    "once more",
    "one more change",
    "one more adjustment",
    "one more thing",
    "what about",
    "how about",
    "why did",
    "same thing",
    "instead of",
    "other than that",
    "apart from that",
    "besides that",
    "except for that",
];

/// Generic definite objects that usually need an antecedent when no topical
/// qualifier is present: "implement the feature", "apply the recommendation",
/// and similar requests are incomplete without the preceding conversation.
const ANAPHORIC_DEFINITE_NOUNS: &[&str] = &[
    "approach",
    "change",
    "design",
    "feature",
    "fix",
    "idea",
    "implementation",
    "option",
    "plan",
    "proposal",
    "recommendation",
    "request",
    "solution",
    "suggestion",
    "task",
    "work",
];

const ANAPHORIC_DEFINITE_PLURAL_NOUNS: &[&str] = &[
    "approaches",
    "changes",
    "designs",
    "features",
    "fixes",
    "ideas",
    "implementations",
    "options",
    "plans",
    "proposals",
    "recommendations",
    "requests",
    "solutions",
    "suggestions",
    "tasks",
    "works",
];

const ANAPHORIC_DEFINITE_MODIFIERS: &[&str] = &[
    "chosen",
    "preferred",
    "proposed",
    "recommended",
    "selected",
    "suggested",
];

fn is_anaphoric_definite_noun(word: &str) -> bool {
    let candidates = ANAPHORIC_DEFINITE_NOUNS
        .iter()
        .chain(ANAPHORIC_DEFINITE_PLURAL_NOUNS);
    candidates.clone().any(|candidate| word == *candidate)
        || (word.is_ascii()
            && candidates
                .filter(|candidate| candidate.len() >= 4)
                .any(|candidate| within_one_typo(word.as_bytes(), candidate.as_bytes())))
}

/// Match a single missing, extra, substituted, or transposed ASCII byte.
/// Callers keep this deliberately scoped to the small reference vocabulary;
/// it is not a general fuzzy matcher over user text.
fn within_one_typo(word: &[u8], candidate: &[u8]) -> bool {
    if word.len().abs_diff(candidate.len()) > 1 {
        return false;
    }

    if word.len() == candidate.len() {
        let mut first_mismatch = None;
        let mut second_mismatch = None;
        for index in 0..word.len() {
            if word[index] == candidate[index] {
                continue;
            }
            if first_mismatch.is_none() {
                first_mismatch = Some(index);
            } else if second_mismatch.is_none() {
                second_mismatch = Some(index);
            } else {
                return false;
            }
        }
        return match (first_mismatch, second_mismatch) {
            (None, None) | (Some(_), None) => true,
            (Some(first), Some(second)) => {
                second == first + 1
                    && word[first] == candidate[second]
                    && word[second] == candidate[first]
            }
            (None, Some(_)) => false,
        };
    }

    let (longer, shorter) = if word.len() > candidate.len() {
        (word, candidate)
    } else {
        (candidate, word)
    };
    let (mut long_index, mut short_index, mut skipped) = (0, 0, false);
    while long_index < longer.len() && short_index < shorter.len() {
        if longer[long_index] == shorter[short_index] {
            long_index += 1;
            short_index += 1;
        } else if skipped {
            return false;
        } else {
            skipped = true;
            long_index += 1;
        }
    }
    true
}

/// Function words and request framing that carry little topical identity.
/// Action verbs are also excluded by [`relevance_terms`] so two unrelated
/// tasks do not match merely because both say "create" or "review".
const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "within",
    "without",
    "through",
    "across",
    "around",
    "over",
    "under",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "been",
    "being",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "must",
    "please",
    "want",
    "need",
    "help",
    "make",
    "task",
    "feature",
    "request",
    "work",
    "working",
    "use",
    "using",
    "about",
    "after",
    "before",
    "between",
    "because",
    "since",
    "until",
    "their",
    "there",
    "here",
    "what",
    "when",
    "where",
    "which",
    "while",
    "your",
    "ours",
    "they",
    "them",
    "then",
    "than",
    "all",
    "not",
    "also",
    "just",
    "same",
    "each",
    "every",
    "either",
    "neither",
    "some",
    "any",
    "many",
    "much",
    "most",
    "more",
    "few",
    "other",
    "another",
    "only",
    "very",
    "quite",
    "really",
    "rather",
    "currently",
    "current",
    "existing",
    "good",
    "better",
    "best",
    "thing",
    "things",
    "way",
    "like",
    "one",
    "project",
    "repository",
    "repo",
    "codebase",
    "application",
    "system",
    "issue",
    "problem",
];

/// Lowercase word tokens. Unicode alphanumeric text is retained, so overlap
/// matching is not restricted to English or ASCII vocabulary.
pub(crate) fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn contains_action_verb(text: &str) -> bool {
    words(text).iter().any(|word| is_action_word(word))
}

/// Whether a single lowercase token is (an inflection of) an action verb.
/// Exposed for heuristics that must exclude the verb itself when looking for
/// the *topic* a task acts on.
pub(crate) fn is_action_token(word: &str) -> bool {
    is_action_word(word)
}

fn is_action_word(word: &str) -> bool {
    if ACTION_VERBS.contains(&word) {
        return true;
    }
    for suffix in ["ing", "ed", "es", "s"] {
        let Some(stem) = word.strip_suffix(suffix).filter(|stem| stem.len() >= 3) else {
            continue;
        };
        if ACTION_VERBS.contains(&stem) {
            return true;
        }
        if matches!(suffix, "ing" | "ed") {
            let restored = format!("{stem}e");
            if ACTION_VERBS.contains(&restored.as_str()) {
                return true;
            }
        }
    }
    false
}

/// Whether the request is phrased as a direct question. This is structural
/// naming signal, not a domain vocabulary lookup.
pub(crate) fn is_question(text: &str) -> bool {
    const QUESTION_OPENERS: &[&str] = &[
        "are", "can", "could", "did", "do", "does", "how", "is", "may", "should", "what", "when",
        "where", "which", "who", "why", "will", "would",
    ];
    words(text)
        .first()
        .is_some_and(|word| QUESTION_OPENERS.contains(&word.as_str()))
}

/// Explicit work changes override discourse words such as "next" or "now".
/// Only a leading declaration with a concrete body establishes a boundary;
/// references to a new feature inside an existing request do not.
pub(crate) fn starts_new_task(text: &str) -> bool {
    // Only the opening declaration matters. Do not lowercase/tokenize an
    // entire feature specification (or attachment) for this prefix check.
    let words = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .take(7)
        .collect::<Vec<_>>();
    let words = words.as_slice();
    let mut words = if words
        .first()
        .is_some_and(|word| word.eq_ignore_ascii_case("please"))
    {
        &words[1..]
    } else {
        words
    };
    // One leading discourse token ("OK, next feature: ...") is not part of
    // the declaration itself.
    if words.first().is_some_and(|word| {
        ["ok", "okay", "thanks", "so", "well"]
            .iter()
            .any(|opener| word.eq_ignore_ascii_case(opener))
    }) {
        words = &words[1..];
    }
    [
        &["next", "feature"][..],
        &["next", "features"][..],
        &["new", "feature"][..],
        &["new", "features"][..],
        &["next", "task"][..],
        &["next", "tasks"][..],
        &["new", "task"][..],
        &["new", "tasks"][..],
        &["new", "request"][..],
        &["new", "topic"][..],
        &["another", "task"][..],
        &["another", "topic"][..],
        &["another", "question"][..],
        &["separate", "task"][..],
        &["unrelated", "task"][..],
        &["change", "topic"][..],
        &["change", "of", "topic"][..],
        &["moving", "on"][..],
        &["next", "item"][..],
        &["new", "question"][..],
        &["switch", "to", "a", "new", "task"][..],
        &["start", "a", "new", "task"][..],
    ]
    .iter()
    .any(|prefix| {
        words.len() > prefix.len()
            && words
                .iter()
                .zip(prefix.iter())
                .all(|(word, prefix)| word.eq_ignore_ascii_case(prefix))
    })
}

/// Whether a request linguistically depends on earlier conversation.
///
/// Single-word signals are checked only near the beginning. This preserves
/// genuine relative clauses such as "Draft a proposal that covers renewals"
/// while still recognizing "That proposal needs another section" as a
/// follow-up. Multi-word discourse phrases may occur anywhere.
pub(crate) fn refers_to_prior_context(text: &str) -> bool {
    refers_to_prior_context_inner(text, true)
}

fn refers_to_prior_context_inner(text: &str, include_weak: bool) -> bool {
    if starts_new_task(text) {
        return false;
    }
    if PRIOR_CONTEXT_PHRASES
        .iter()
        .any(|phrase| contains_signal(text, phrase))
    {
        return true;
    }
    let words = words(text);
    // A labeled answer is a reference to a choice offered earlier, even when
    // its label contains enough action words to look like a standalone task.
    // Keep this near the start so discussing numbered options is not itself
    // mistaken for an answer to a clarification.
    let selection_verb = |word: &str| {
        matches!(
            word,
            "choose" | "chose" | "select" | "pick" | "use" | "prefer"
        )
    };
    let label_index = match words.first().map(String::as_str) {
        Some("option" | "choice") => Some(0),
        Some("i" | "please") if words.get(1).is_some_and(|word| selection_verb(word)) => Some(2),
        Some(word) if selection_verb(word) => Some(1),
        _ => None,
    };
    if label_index.is_some_and(|index| {
        words
            .get(index)
            .is_some_and(|word| matches!(word.as_str(), "option" | "choice"))
            && words.get(index + 1).is_some_and(|word| {
                word.chars().all(|ch| ch.is_ascii_digit())
                    || (word.len() == 1 && word.chars().all(|ch| ch.is_ascii_alphabetic()))
            })
    }) {
        return true;
    }
    let has_definite_reference = words.iter().enumerate().any(|(index, word)| {
        if word != "the" {
            return false;
        }
        let next = words.get(index + 1).map(String::as_str);
        let after_modifier = words.get(index + 2).map(String::as_str);
        next.is_some_and(is_anaphoric_definite_noun)
            || (next.is_some_and(|word| ANAPHORIC_DEFINITE_MODIFIERS.contains(&word))
                && after_modifier.is_some_and(is_anaphoric_definite_noun))
    });
    let has_underspecified_definite_reference = has_definite_reference && {
        let terms = relevance_terms(text);
        terms.is_empty()
            || terms.iter().all(|term| {
                is_anaphoric_definite_noun(term)
                    || ANAPHORIC_DEFINITE_MODIFIERS.contains(&term.as_str())
            })
    };
    if has_underspecified_definite_reference {
        return true;
    }
    if words
        .first()
        .is_some_and(|word| matches!(word.as_str(), "and" | "but"))
    {
        return true;
    }
    include_weak && weak_prior_context_word(text)
}

/// Prior-context signals that unambiguously depend on earlier conversation:
/// discourse phrases, labeled answers, underspecified definite references,
/// and leading conjunctions. Excludes the weak single-word scan.
pub(crate) fn refers_to_prior_context_strong(text: &str) -> bool {
    refers_to_prior_context_inner(text, false)
}

fn weak_prior_context_word(text: &str) -> bool {
    words(text)
        .iter()
        .take(3)
        .any(|word| PRIOR_CONTEXT_WORDS.contains(&word.as_str()))
}

/// Whether the first word is a weak discourse marker (`now`, `next`,
/// `retry`, ...). Such openers are ambiguous: they can continue prior work
/// or introduce a self-contained new request.
pub(crate) fn starts_with_prior_context_word(text: &str) -> bool {
    words(text)
        .first()
        .is_some_and(|word| PRIOR_CONTEXT_WORDS.contains(&word.as_str()))
}

/// Light inflection stripping for classification scoring: `fixing`/`fixed`
/// score as `fix`, while unrelated words (`planetary`) stay intact.
pub(crate) fn stem_word(word: &str) -> String {
    for suffix in ["ing", "ed"] {
        if let Some(stem) = word.strip_suffix(suffix).filter(|stem| stem.len() >= 3) {
            return stem.to_string();
        }
    }
    if let Some(stem) = word.strip_suffix("es").filter(|stem| stem.len() >= 3) {
        return stem.to_string();
    }
    if let Some(stem) = word.strip_suffix('s').filter(|stem| stem.len() >= 3) {
        return stem.to_string();
    }
    word.to_string()
}

/// Whether a request is chatter rather than work: greetings, thanks,
/// acknowledgements, and small meta questions. Such turns should not create
/// tasks or be counted as follow-up work.
pub(crate) fn is_non_task_chatter(text: &str) -> bool {
    const CHATTER: &[&str] = &[
        "hi",
        "hello",
        "hey",
        "hello there",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thanks a lot",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "got it",
        "understood",
        "sounds good",
        "perfect",
        "great",
        "nice",
        "cool",
        "what can you do",
        "who are you",
        "what are you",
        "help",
    ];
    let normalized = words(text).join(" ");
    CHATTER.contains(&normalized.as_str())
}

/// Boundary-aware keyword/phrase lookup. This avoids substring accidents
/// such as matching `plan` inside `planet`, and works for multi-word signals.
pub(crate) fn contains_signal(text: &str, signal: &str) -> bool {
    let haystack = words(text);
    let needle = words(signal);
    !needle.is_empty()
        && needle.len() <= haystack.len()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle.as_slice())
}

/// Topical terms used for relevance matching, independent of agent domain.
pub(crate) fn relevance_terms(text: &str) -> HashSet<String> {
    words(text)
        .into_iter()
        .filter(|word| word.chars().count() > 2)
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .filter(|word| !is_action_word(word))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_vocabulary_covers_non_coding_agent_work() {
        for request in [
            "Draft a renewal proposal",
            "Reconcile the July ledger",
            "Schedule interviews",
            "Translate the policy",
            "Research regional suppliers",
            "Compose a campaign brief",
            "Scheduling candidate interviews",
            "Summarized the quarterly report",
            "Archive expired customer records",
            "Configure the payment provider",
            "Export the monthly invoices",
            "Hire an account manager",
            "Import the supplier contacts",
            "Route support tickets by region",
            "Sync the interview calendar",
            "Upload the signed contract",
        ] {
            assert!(contains_action_verb(request), "missing action in {request}");
        }
    }

    #[test]
    fn action_vocabulary_covers_common_software_work() {
        for request in [
            "Backfill the migration records",
            "Benchmark API latency",
            "Collapse the request panel",
            "Paginate the event stream",
            "Profile loading performance",
            "Reproduce the cache failure",
            "Troubleshoot the worker process",
            "Virtualized the session timeline",
        ] {
            assert!(contains_action_verb(request), "missing action in {request}");
        }
    }

    #[test]
    fn questions_are_recognized_without_domain_terms() {
        assert!(is_question("What are Uganda VAT filing deadlines?"));
        assert!(is_question("How should we prioritize these suppliers?"));
        assert!(!is_question("Prepare the quarterly forecast"));
    }

    #[test]
    fn signal_lookup_uses_word_boundaries() {
        assert!(contains_signal("Create a quarterly plan", "plan"));
        assert!(!contains_signal("Study planetary motion", "plan"));
        assert!(contains_signal(
            "Generate an executive report",
            "executive report"
        ));
    }

    #[test]
    fn relevance_terms_preserve_domain_vocabulary() {
        let terms = relevance_terms("Draft a renewal proposal for Acme procurement");
        assert!(terms.contains("renewal"));
        assert!(terms.contains("proposal"));
        assert!(terms.contains("procurement"));
        assert!(!terms.contains("draft"));
    }

    #[test]
    fn relevance_terms_remove_expanded_framing_and_inflected_actions() {
        let terms =
            relevance_terms("Currently configuring and testing the existing project retry policy");
        assert_eq!(
            terms,
            HashSet::from(["retry".to_string(), "policy".to_string()])
        );
    }

    #[test]
    fn prior_context_signals_do_not_capture_relative_clauses() {
        assert!(refers_to_prior_context(
            "That proposal needs another section"
        ));
        assert!(refers_to_prior_context("What about the Kampala office?"));
        assert!(!refers_to_prior_context(
            "Draft a proposal that covers the renewal terms"
        ));
        assert!(!refers_to_prior_context(
            "Prepare a report for your regional sales team"
        ));
        assert!(refers_to_prior_context("And the tests?"));
        assert!(refers_to_prior_context("But move the control inside"));
    }

    #[test]
    fn prior_context_signals_cover_natural_continuation_requests() {
        for request in [
            "The layout looks polished. But instead of keeping the control outside, move it inside",
            "Can you please continue from where you stopped?",
            "Please resume the work",
            "Pick up where we left off",
            "Following up on the earlier analysis, add the regional figures",
            "Do the same for the Kampala office",
            "Finish the rest",
            "Return to the budget review",
            "Build on that recommendation",
            "As discussed, prepare the meeting notes",
            "Once more, but in a warmer tone",
            "With that in mind, shorten the introduction",
            "On second thought, archive the draft",
            "Come back to the supplier analysis",
            "Use the same format for the regional report",
            "One more adjustment: align the totals",
            "Other than that, keep the layout unchanged",
            "Will this value be set automatically?",
            "Please implement this feature",
            "Please implement the feature",
            "Please implement the changes",
            "Please implement the chnges",
            "Please implement the changess",
            "Please implement the chanegs",
            "Apply the proposed fixes",
            "Apply the recommendation",
            "Proceed with the suggested option",
            "Option 3 - Change docker postgres port mapping",
            "I choose option 3: change the Docker Postgres port mapping",
            "Choice B - Use the local database for migrations",
        ] {
            assert!(
                refers_to_prior_context(request),
                "expected prior-context signal in: {request}"
            );
        }
    }

    #[test]
    fn continuation_vocabulary_does_not_capture_self_contained_requests() {
        for request in [
            "Explain how teams continue operating during an outage",
            "Write a guide to resume interrupted downloads",
            "Prepare a business continuity plan",
            "Research prior authorization requirements",
            "Draft a follow-up email for new customers",
            "Back up the production database",
            "Compare former and current employee policies",
            "Use Postgres rather than SQLite for analytics",
            "Research small change detection in economics",
            "Implement the feature flag system",
            "Fix the billing issue",
            "Draft the proposal for Acme renewal",
            "Explain option 3 in the database documentation",
            "Implement option 3 parsing for the CLI",
        ] {
            assert!(
                !refers_to_prior_context(request),
                "unexpected prior-context signal in: {request}"
            );
        }
    }

    #[test]
    fn typo_matching_is_bounded_to_one_edit() {
        for typo in ["chnges", "changess", "chanegs", "chamges"] {
            assert!(is_anaphoric_definite_noun(typo), "missed typo: {typo}");
        }
        for unrelated in ["chariots", "featureset", "six"] {
            assert!(
                !is_anaphoric_definite_noun(unrelated),
                "overmatched unrelated word: {unrelated}"
            );
        }
    }
}
