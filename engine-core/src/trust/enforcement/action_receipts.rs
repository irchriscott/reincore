//! Tool-host action declarations and bounded, correlated execution receipts.
//! These are evidence contracts, never permission grants or execution instructions.
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::OnceLock;

pub const RECEIPTS_KEY: &str = "reincore_action_receipts";
pub const MAX_RECEIPTS: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractFields {
    action: String,
    target_pointer: String,
    completion_template: String,
    #[serde(default)]
    confirmation_pointer: Option<String>,
}

/// Optional metadata on a tool definition. Providers receive only its ordinary tool schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ActionContract {
    pub action: String,
    pub target_pointer: String,
    pub completion_template: String,
    /// Where an ordinary result carries its confirmation reference, for hosts that return no
    /// receipt: "/id" for a Gmail-style `{"id": "msg-123"}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmation_pointer: Option<String>,
}

impl<'de> Deserialize<'de> for ActionContract {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = ContractFields::deserialize(deserializer)?;
        let contract = Self {
            action: fields.action,
            target_pointer: fields.target_pointer,
            completion_template: fields.completion_template,
            confirmation_pointer: fields.confirmation_pointer,
        };
        contract.validate().map_err(serde::de::Error::custom)?;
        Ok(contract)
    }
}

impl ActionContract {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.action.is_empty()
            || self.action.len() > 96
            || !self.action.bytes().all(|b| {
                b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
            })
        {
            return Err("action must be a bounded lowercase identifier");
        }
        if !valid_pointer(&self.target_pointer) {
            return Err("target_pointer must be a valid non-root JSON pointer");
        }
        if self
            .confirmation_pointer
            .as_deref()
            .is_some_and(|pointer| !valid_pointer(pointer))
        {
            return Err("confirmation_pointer must be a valid non-root JSON pointer");
        }
        let template = &self.completion_template;
        if template.len() > 256
            || template.matches("{target}").count() != 1
            || template.replace("{target}", "").contains(['{', '}'])
            || template.chars().any(char::is_control)
            || template.replace("{target}", "").trim().len() < 4
        {
            return Err("completion_template requires one target placeholder and bounded completion wording");
        }
        Ok(())
    }

    pub fn targets(&self, args: &Value) -> Option<Vec<String>> {
        self.validate().ok()?;
        let value = args.pointer(&self.target_pointer)?;
        let values: Vec<_> = match value {
            Value::String(_) => vec![value],
            Value::Array(values) if !values.is_empty() && values.len() <= MAX_RECEIPTS => {
                values.iter().collect()
            }
            _ => return None,
        };
        let mut seen = HashSet::new();
        values
            .into_iter()
            .map(|value| {
                let target = value.as_str()?;
                if !valid_text(target, 512) || !seen.insert(target) {
                    return None;
                }
                Some(target.to_string())
            })
            .collect()
    }

    /// Match declared wording, preserving target identity (including case).
    pub fn claim_target(&self, claim: &str) -> Option<String> {
        self.validate().ok()?;
        let template = literal(&self.completion_template);
        let (before, after) = template.split_once("{target}")?;
        let text = literal(claim);
        if !text.get(..before.len())?.eq_ignore_ascii_case(before)
            || !text
                .get(text.len().checked_sub(after.len())?..)?
                .eq_ignore_ascii_case(after)
        {
            return None;
        }
        let end = text.len().checked_sub(after.len())?;
        let target = text
            .get(before.len()..end)?
            .trim()
            .trim_matches(['`', '"', '\'']);
        valid_text(target, 512).then(|| target.to_string())
    }

    /// Recognize related completion wording without upgrading a paraphrase to proof.
    pub fn mentions_completion(&self, claim: &str) -> bool {
        let claim = words(claim);
        if claim.iter().any(|w| NOT_COMPLETED.contains(&w.as_str())) {
            return false;
        }
        let words = words(&self.completion_template.replace("{target}", ""));
        let anchors: Vec<_> = words
            .iter()
            .filter(|w| {
                ![
                    "a", "an", "the", "i", "we", "is", "are", "was", "were", "has", "have", "been",
                    "to", "at", "for", "of", "on", "in", "with",
                ]
                .contains(&w.as_str())
            })
            .collect();
        !anchors.is_empty() && anchors.iter().all(|w| claim.contains(w))
    }
}

fn literal(text: &str) -> String {
    text.trim().trim_matches(['*', '.']).trim().to_string()
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn normalize(text: &str) -> String {
    text.trim()
        .trim_matches(['*', '.'])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn valid_text(text: &str, limit: usize) -> bool {
    !text.is_empty()
        && text.len() <= limit
        && text.trim() == text
        && !text.chars().any(char::is_control)
}

fn valid_pointer(pointer: &str) -> bool {
    pointer.starts_with('/')
        && pointer.len() <= 256
        && !pointer.chars().any(char::is_control)
        && pointer
            .split('~')
            .skip(1)
            .all(|s| s.starts_with(['0', '1']))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionOutcome {
    Succeeded,
    Failed,
    Pending,
    Cancelled,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionReceipt {
    pub version: u8,
    pub tool_call_id: String,
    pub action: String,
    pub target: String,
    pub outcome: ActionOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmation: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionEvidence {
    pub action: String,
    pub target: String,
    pub outcome: ActionOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmation: Option<String>,
    /// Fixed engine explanation; never raw tool instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issue: Option<String>,
    /// Success came from the contract's declared `confirmation_pointer`, not a host receipt.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mapped: bool,
}

/// Missing/invalid receipts still create unknown observations for expected targets:
/// a subsequent pending or failed attempt must not borrow an earlier success.
pub fn collect(
    contract: &ActionContract,
    call_id: &str,
    args: &Value,
    result: &Value,
) -> Vec<ActionEvidence> {
    let Some(targets) = contract.targets(args) else {
        return Vec::new();
    };
    let mut evidence: Vec<_> = targets
        .iter()
        .map(|target| ActionEvidence {
            action: contract.action.clone(),
            target: target.clone(),
            outcome: ActionOutcome::Unknown,
            confirmation: None,
            issue: Some("No valid action receipt was returned for this target.".into()),
            mapped: false,
        })
        .collect();
    let direct = result.get(RECEIPTS_KEY);
    let structured = result
        .get("structuredContent")
        .and_then(|v| v.get(RECEIPTS_KEY));
    // Two competing envelopes are ambiguous, even if one looks successful.
    if direct.is_some() && structured.is_some() {
        return evidence;
    }
    let Some(envelope) = direct.or(structured) else {
        // With no envelope at all, a declared confirmation field in the ordinary result stands in.
        if let Some(pointer) = &contract.confirmation_pointer {
            map_result(&mut evidence, pointer, result);
        }
        return evidence;
    };
    let Value::Array(values) = envelope else {
        return evidence;
    };
    if values.is_empty() || values.len() > MAX_RECEIPTS {
        return evidence;
    }
    let mut seen = HashSet::new();
    let receipts: Option<Vec<ActionReceipt>> = values
        .iter()
        .map(|value| {
            let receipt: ActionReceipt = serde_json::from_value(value.clone()).ok()?;
            if receipt.version != 1
                || receipt.tool_call_id != call_id
                || receipt.action != contract.action
                || !targets.contains(&receipt.target)
                || !seen.insert(receipt.target.clone())
                || receipt
                    .confirmation
                    .as_ref()
                    .is_some_and(|id| !valid_text(id, 256))
                || receipt.outcome == ActionOutcome::Succeeded && receipt.confirmation.is_none()
            {
                return None;
            }
            Some(receipt)
        })
        .collect();
    let Some(receipts) = receipts else {
        return evidence;
    };
    for receipt in receipts {
        let entry = evidence
            .iter_mut()
            .find(|e| e.target == receipt.target)
            .expect("validated target");
        entry.outcome = receipt.outcome;
        entry.confirmation = receipt.confirmation;
        entry.issue = None;
    }
    evidence
}

/// Apply a declared confirmation field when the host returned no receipt. Claim grounding still
/// applies the outer result's error, pending, cached, and truncation signals afterwards.
fn map_result(evidence: &mut [ActionEvidence], pointer: &str, result: &Value) {
    // MCP servers often return JSON as the only text content item.
    let text = result
        .get("content")
        .and_then(Value::as_array)
        .filter(|content| content.len() == 1)
        .and_then(|content| content[0].get("text")?.as_str())
        .and_then(|text| serde_json::from_str::<Value>(text).ok());
    let sources: Vec<&Value> = [Some(result), result.get("structuredContent"), text.as_ref()]
        .into_iter()
        .flatten()
        .collect();
    let failed = sources.iter().any(|source| {
        ["error", "errors"]
            .iter()
            .any(|key| source.get(*key).is_some_and(reported))
    });
    let confirmation = sources.iter().find_map(|source| {
        let id = match source.pointer(pointer)? {
            Value::String(id) => id.clone(),
            Value::Number(id) => id.to_string(),
            _ => return None,
        };
        valid_text(&id, 256).then_some(id)
    });
    for entry in evidence {
        if failed {
            entry.outcome = ActionOutcome::Failed;
            entry.issue = Some("The tool result reports an error.".into());
        } else if let Some(id) = &confirmation {
            entry.outcome = ActionOutcome::Succeeded;
            entry.confirmation = Some(id.clone());
            entry.issue = None;
            entry.mapped = true;
        } else {
            entry.issue =
                Some("The declared confirmation field is missing from the tool result.".into());
        }
    }
}

/// Whether an `error` or `errors` field actually reports something.
fn reported(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
        Value::Number(_) => true,
    }
}

/// Shared vocabulary for recognition, not permission or proof. Arbitrary domains can
/// declare their own templates. Keep all three common-claim helpers on this catalog.
struct CommonAction {
    action: &'static str,
    noun: &'static str,
    verb: &'static str,
    active: &'static str,
    preposition: &'static str,
}

const COMMON_ACTIONS: &[CommonAction] = &[
    CommonAction {
        action: "email.send",
        noun: "email",
        verb: "sent",
        active: "sent an email",
        preposition: "to",
    },
    CommonAction {
        action: "message.send",
        noun: "message",
        verb: "sent",
        active: "sent a message",
        preposition: "to",
    },
    CommonAction {
        action: "booking.confirm",
        noun: "booking",
        verb: "confirmed",
        active: "confirmed booking",
        preposition: "for",
    },
    CommonAction {
        action: "booking.cancel",
        noun: "booking",
        verb: "cancelled",
        active: "cancelled booking",
        preposition: "for",
    },
    CommonAction {
        action: "refund.process",
        noun: "refund",
        verb: "processed",
        active: "processed refund",
        preposition: "for",
    },
    CommonAction {
        action: "payment.process",
        noun: "payment",
        verb: "processed",
        active: "processed payment",
        preposition: "for",
    },
    CommonAction {
        action: "meeting.schedule",
        noun: "meeting",
        verb: "scheduled",
        active: "scheduled meeting",
        preposition: "for",
    },
    CommonAction {
        action: "meeting.cancel",
        noun: "meeting",
        verb: "cancelled",
        active: "cancelled meeting",
        preposition: "for",
    },
    CommonAction {
        action: "ticket.create",
        noun: "ticket",
        verb: "created",
        active: "created ticket",
        preposition: "for",
    },
    CommonAction {
        action: "ticket.close",
        noun: "ticket",
        verb: "closed",
        active: "closed ticket",
        preposition: "for",
    },
    CommonAction {
        action: "record.create",
        noun: "record",
        verb: "created",
        active: "created record",
        preposition: "for",
    },
    CommonAction {
        action: "record.update",
        noun: "record",
        verb: "updated",
        active: "updated record",
        preposition: "for",
    },
    CommonAction {
        action: "record.delete",
        noun: "record",
        verb: "deleted",
        active: "deleted record",
        preposition: "for",
    },
    CommonAction {
        action: "report.publish",
        noun: "report",
        verb: "published",
        active: "published report",
        preposition: "to",
    },
    CommonAction {
        action: "document.create",
        noun: "document",
        verb: "created",
        active: "created document",
        preposition: "for",
    },
    CommonAction {
        action: "document.upload",
        noun: "document",
        verb: "uploaded",
        active: "uploaded document",
        preposition: "to",
    },
    CommonAction {
        action: "order.place",
        noun: "order",
        verb: "placed",
        active: "placed order",
        preposition: "for",
    },
    CommonAction {
        action: "order.cancel",
        noun: "order",
        verb: "cancelled",
        active: "cancelled order",
        preposition: "for",
    },
    CommonAction {
        action: "shipment.dispatch",
        noun: "shipment",
        verb: "dispatched",
        active: "dispatched shipment",
        preposition: "for",
    },
    CommonAction {
        action: "subscription.cancel",
        noun: "subscription",
        verb: "cancelled",
        active: "cancelled subscription",
        preposition: "for",
    },
    CommonAction {
        action: "contact.create",
        noun: "contact",
        verb: "created",
        active: "created contact",
        preposition: "for",
    },
    CommonAction {
        action: "task.create",
        noun: "task",
        verb: "created",
        active: "created task",
        preposition: "for",
    },
    CommonAction {
        action: "task.complete",
        noun: "task",
        verb: "completed",
        active: "completed task",
        preposition: "for",
    },
    CommonAction {
        action: "file.create",
        noun: "file",
        verb: "created",
        active: "created file",
        preposition: "at",
    },
    CommonAction {
        action: "file.update",
        noun: "file",
        verb: "updated",
        active: "updated file",
        preposition: "at",
    },
    CommonAction {
        action: "file.delete",
        noun: "file",
        verb: "deleted",
        active: "deleted file",
        preposition: "at",
    },
    CommonAction {
        action: "file.move",
        noun: "file",
        verb: "moved",
        active: "moved file",
        preposition: "to",
    },
    CommonAction {
        action: "test.run",
        noun: "tests",
        verb: "passed",
        active: "passed tests",
        preposition: "for",
    },
    CommonAction {
        action: "build.run",
        noun: "build",
        verb: "completed",
        active: "completed build",
        preposition: "for",
    },
    CommonAction {
        action: "lint.run",
        noun: "lint",
        verb: "passed",
        active: "passed lint",
        preposition: "for",
    },
    CommonAction {
        action: "deployment.deploy",
        noun: "deployment",
        verb: "succeeded",
        active: "deployed",
        preposition: "to",
    },
    CommonAction {
        action: "deployment.rollback",
        noun: "deployment",
        verb: "rolled back",
        active: "rolled back deployment",
        preposition: "for",
    },
    CommonAction {
        action: "commit.create",
        noun: "commit",
        verb: "created",
        active: "created commit",
        preposition: "for",
    },
    CommonAction {
        action: "branch.create",
        noun: "branch",
        verb: "created",
        active: "created branch",
        preposition: "for",
    },
    CommonAction {
        action: "branch.push",
        noun: "branch",
        verb: "pushed",
        active: "pushed branch",
        preposition: "to",
    },
    CommonAction {
        action: "pull_request.create",
        noun: "pull request",
        verb: "created",
        active: "created pull request",
        preposition: "for",
    },
    CommonAction {
        action: "pull_request.merge",
        noun: "pull request",
        verb: "merged",
        active: "merged pull request",
        preposition: "for",
    },
    CommonAction {
        action: "migration.apply",
        noun: "migration",
        verb: "applied",
        active: "applied migration",
        preposition: "for",
    },
    CommonAction {
        action: "package.publish",
        noun: "package",
        verb: "published",
        active: "published package",
        preposition: "to",
    },
    CommonAction {
        action: "dependency.install",
        noun: "dependency",
        verb: "installed",
        active: "installed dependency",
        preposition: "for",
    },
    CommonAction {
        action: "backup.create",
        noun: "backup",
        verb: "created",
        active: "created backup",
        preposition: "for",
    },
    CommonAction {
        action: "backup.restore",
        noun: "backup",
        verb: "restored",
        active: "restored backup",
        preposition: "for",
    },
];

/// Wording that asserts the same completed action as a catalog entry. An alias never states a
/// different or stronger outcome: "delivered" is not "sent", because a send receipt does not
/// prove delivery, and "resolved" is not "closed", because many trackers keep those states apart.
struct Alias {
    action: &'static str,
    /// Past participles usable wherever the catalog verb is, including inside longer sentences:
    /// "I've filed a ticket about the outage."
    verbs: &'static [&'static str],
    /// Adjectives that state completion only after the noun: "The task is done" is a claim,
    /// while "Complete the task" is a request.
    states: &'static [&'static str],
    /// Verbs whose object is the target: "Emailed alice@example.com."
    direct: &'static [&'static str],
    /// Nouns matched only in exact wording; "issue", "row", and "PR" are too common in code
    /// prose to detect inside other sentences.
    exact_nouns: &'static [&'static str],
    /// Verbs matched only in exact wording, for the same reason: "opened", "completed", "edited".
    exact_verbs: &'static [&'static str],
}

const NO_ALIAS: Alias = Alias {
    action: "",
    verbs: &[],
    states: &[],
    direct: &[],
    exact_nouns: &[],
    exact_verbs: &[],
};

/// Deliberately absent: "added" (added a field), "call" and "event" (function calls, event
/// handlers), "reserved" (reserved words), "paid" (paid plans), and "unsubscribed" (listeners).
const ALIASES: &[Alias] = &[
    Alias {
        action: "email.send",
        direct: &["emailed"],
        ..NO_ALIAS
    },
    Alias {
        action: "message.send",
        verbs: &["posted"],
        direct: &["messaged", "texted"],
        exact_nouns: &["dm", "text message"],
        ..NO_ALIAS
    },
    Alias {
        action: "booking.confirm",
        direct: &["booked"],
        exact_nouns: &["reservation"],
        ..NO_ALIAS
    },
    Alias {
        action: "booking.cancel",
        verbs: &["canceled"],
        exact_nouns: &["reservation"],
        ..NO_ALIAS
    },
    Alias {
        action: "refund.process",
        verbs: &["issued"],
        direct: &["refunded"],
        ..NO_ALIAS
    },
    Alias {
        action: "payment.process",
        exact_verbs: &["completed"],
        ..NO_ALIAS
    },
    Alias {
        action: "meeting.schedule",
        verbs: &["booked", "set up"],
        exact_nouns: &["appointment"],
        ..NO_ALIAS
    },
    Alias {
        action: "meeting.cancel",
        verbs: &["canceled"],
        exact_nouns: &["appointment"],
        ..NO_ALIAS
    },
    Alias {
        action: "ticket.create",
        verbs: &["filed"],
        exact_nouns: &["issue"],
        exact_verbs: &["opened"],
        ..NO_ALIAS
    },
    Alias {
        action: "ticket.close",
        exact_nouns: &["issue"],
        ..NO_ALIAS
    },
    Alias {
        action: "record.create",
        exact_nouns: &["entry", "row"],
        exact_verbs: &["inserted"],
        ..NO_ALIAS
    },
    Alias {
        action: "record.update",
        exact_nouns: &["entry", "row"],
        exact_verbs: &["modified", "edited"],
        ..NO_ALIAS
    },
    Alias {
        action: "record.delete",
        exact_nouns: &["entry", "row"],
        exact_verbs: &["removed"],
        ..NO_ALIAS
    },
    Alias {
        action: "report.publish",
        verbs: &["posted"],
        ..NO_ALIAS
    },
    Alias {
        action: "document.create",
        exact_nouns: &["doc"],
        ..NO_ALIAS
    },
    Alias {
        action: "document.upload",
        exact_nouns: &["doc"],
        ..NO_ALIAS
    },
    Alias {
        action: "order.place",
        verbs: &["submitted"],
        ..NO_ALIAS
    },
    Alias {
        action: "order.cancel",
        verbs: &["canceled"],
        ..NO_ALIAS
    },
    Alias {
        action: "shipment.dispatch",
        verbs: &["shipped"],
        ..NO_ALIAS
    },
    Alias {
        action: "subscription.cancel",
        verbs: &["canceled"],
        ..NO_ALIAS
    },
    Alias {
        action: "task.create",
        exact_nouns: &["todo"],
        ..NO_ALIAS
    },
    Alias {
        action: "task.complete",
        verbs: &["finished"],
        states: &["complete", "done"],
        exact_nouns: &["todo"],
        ..NO_ALIAS
    },
    Alias {
        action: "pull_request.create",
        exact_nouns: &["pr"],
        exact_verbs: &["opened"],
        ..NO_ALIAS
    },
    Alias {
        action: "pull_request.merge",
        exact_nouns: &["pr"],
        ..NO_ALIAS
    },
];

/// Negated or conditional wording is not a completion: "The email isn't sent yet",
/// "Once the task is done, ...". `words` splits contractions, so "isn't" yields "isn".
const NOT_COMPLETED: &[&str] = &[
    "not", "never", "cannot", "isn", "wasn", "aren", "weren", "hasn", "haven", "didn", "couldn",
    "won", "when", "once", "until", "unless", "whenever",
];

/// Words allowed between a noun and its completion state: "The task is now marked as done."
const LINKING: &[&str] = &[
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "been",
    "now",
    "all",
    "fully",
    "successfully",
    "marked",
    "as",
];

/// Lowercase wording for one catalog action.
struct Wording {
    action: &'static str,
    /// Whole targetless claims, such as "task is complete".
    claims: Vec<String>,
    /// Prefixes followed by a target, such as "email has been sent to ".
    prefixes: Vec<String>,
    /// Verbs followed directly by a target, such as "emailed ".
    direct: Vec<String>,
}

fn alias(action: &str) -> &'static Alias {
    ALIASES
        .iter()
        .find(|alias| alias.action == action)
        .unwrap_or(&NO_ALIAS)
}

fn wording() -> &'static [Wording] {
    static WORDING: OnceLock<Vec<Wording>> = OnceLock::new();
    WORDING.get_or_init(|| {
        COMMON_ACTIONS
            .iter()
            .map(|entry| {
                let alias = alias(entry.action);
                let nouns: Vec<&str> = std::iter::once(entry.noun)
                    .chain(alias.exact_nouns.iter().copied())
                    .collect();
                let verbs: Vec<&str> = alias
                    .verbs
                    .iter()
                    .chain(alias.exact_verbs)
                    .copied()
                    .collect();
                let mut passive = Vec::new();
                for noun in &nouns {
                    for word in std::iter::once(entry.verb)
                        .chain(verbs.iter().copied())
                        .chain(alias.states.iter().copied())
                    {
                        for link in [
                            "",
                            "is ",
                            "are ",
                            "was ",
                            "were ",
                            "has been ",
                            "have been ",
                        ] {
                            for adverb in ["", "successfully "] {
                                passive.push(format!("{noun} {link}{adverb}{word}"));
                            }
                        }
                    }
                }
                // Active forms take a target directly only without an article: "Merged PR #42",
                // but "updated the record type in src/models.rs" names no record.
                let mut bare = vec![entry.active.to_string()];
                let mut articled = Vec::new();
                // Noun-less forms such as "deployed" have nothing to vary.
                if let Some(phrase) = entry.active.strip_suffix(entry.noun) {
                    let phrase = phrase.trim_end();
                    let phrase = [" an", " a", " the"]
                        .iter()
                        .find_map(|article| phrase.strip_suffix(article))
                        .unwrap_or(phrase);
                    for verb in std::iter::once(phrase).chain(verbs.iter().copied()) {
                        for noun in &nouns {
                            bare.push(format!("{verb} {noun}"));
                            for article in ["a", "an", "the"] {
                                articled.push(format!("{verb} {article} {noun}"));
                            }
                        }
                    }
                }
                for forms in [&mut bare, &mut articled] {
                    let adverbs: Vec<_> =
                        forms.iter().map(|f| format!("successfully {f}")).collect();
                    forms.extend(adverbs);
                    forms.sort();
                    forms.dedup();
                }
                // Prepositional prefixes come first so "sent the email to Alice" keeps "Alice".
                let mut prefixes: Vec<_> = passive
                    .iter()
                    .chain(&bare)
                    .chain(&articled)
                    .map(|stem| format!("{stem} {} ", entry.preposition))
                    .collect();
                prefixes.extend(bare.iter().map(|stem| format!("{stem} ")));
                passive.extend(bare);
                passive.extend(articled);
                Wording {
                    action: entry.action,
                    claims: passive,
                    prefixes,
                    direct: alias.direct.iter().map(|verb| format!("{verb} ")).collect(),
                }
            })
            .collect()
    })
}

fn claim_body(claim: &str) -> String {
    let text = literal(claim);
    // Longer subjects first, so "I have sent" is not reduced to "have sent".
    for prefix in [
        "i have ", "we have ", "i've ", "we've ", "i’ve ", "we’ve ", "the ", "i ", "we ",
    ] {
        if text
            .get(..prefix.len())
            .is_some_and(|s| s.eq_ignore_ascii_case(prefix))
        {
            return text[prefix.len()..].to_string();
        }
    }
    text
}

pub fn common_action(claim: &str) -> Option<&'static str> {
    let text = normalize(&claim_body(claim)).to_lowercase();
    wording()
        .iter()
        .find(|wording| wording.claims.contains(&text))
        .map(|wording| wording.action)
}

pub fn common_claim(claim: &str) -> Option<(String, Option<String>)> {
    if let Some(action) = common_action(claim) {
        return Some((action.into(), None));
    }
    let text = claim_body(claim);
    let target = |prefix: &str| {
        text.get(..prefix.len())
            .is_some_and(|s| s.eq_ignore_ascii_case(prefix))
            .then(|| text[prefix.len()..].trim().trim_matches(['`', '"', '\'']))
            .filter(|target| valid_text(target, 512))
    };
    // Direct verbs are tried last, so "Booked a meeting for Monday" schedules a meeting.
    for direct in [false, true] {
        for wording in wording() {
            let prefixes = if direct {
                &wording.direct
            } else {
                &wording.prefixes
            };
            if let Some(found) = prefixes.iter().find_map(|prefix| target(prefix)) {
                return Some((wording.action.into(), Some(found.into())));
            }
        }
    }
    None
}

/// Loose detection inside longer sentences uses only catalog nouns and unambiguous aliases.
pub fn mentions_common_completion(claim: &str) -> bool {
    let claim_words = words(claim);
    if claim_words
        .iter()
        .any(|w| NOT_COMPLETED.contains(&w.as_str()))
    {
        return false;
    }
    COMMON_ACTIONS.iter().any(|entry| {
        let alias = alias(entry.action);
        let noun = words(entry.noun);
        let has_noun = noun.iter().all(|noun| {
            claim_words
                .iter()
                .any(|w| w == noun || w == &format!("{noun}s"))
        });
        let has_verb = std::iter::once(entry.verb)
            .chain(alias.verbs.iter().copied())
            .any(|verb| words(verb).iter().all(|verb| claim_words.contains(verb)));
        has_noun && (has_verb || stated(&claim_words, &noun, alias.states))
            || alias
                .direct
                .iter()
                .any(|verb| claim_words.iter().any(|w| w == verb))
    })
}

/// Qualifiers that make a completion partial: "Almost done."
const PARTIAL: &[&str] = &[
    "almost",
    "nearly",
    "mostly",
    "partially",
    "partly",
    "halfway",
];

/// Contractions and adverbs that may follow a subject: "It's all done", "We're completely finished."
const SUBJECT_LINKING: &[&str] = &[
    "s",
    "re",
    "m",
    "am",
    "ve",
    "completely",
    "entirely",
    "officially",
];

/// Conditions and futures report neither a completion nor what remains: "Once it's mostly done...".
const NOT_A_REPORT: &[&str] = &[
    "when", "once", "until", "unless", "whenever", "if", "will", "would", "should", "could",
    "might", "must", "won",
];

/// Negations that turn a completion into an admission that work remains: "It isn't done yet".
const NEGATIONS: &[&str] = &["not", "isn", "aren", "wasn", "haven", "hasn"];

/// Words a negation carries along: "not quite done", "not fully finished", "isn't done yet".
const NEGATED: &[&str] = &["quite", "fully", "entirely", "completely", "yet", "t"];

/// Phrases that name all of the work rather than an action.
const ALL_WORK: &[&str] = &[
    "everything you asked for",
    "everything requested",
    "all requested changes",
    "the requested changes",
    "all of the work",
    "all the work",
    "all changes",
    "the changes",
    "all of it",
    "the work",
    "the job",
    "everything",
    "it",
    "that",
    "this",
];

/// A completion claim that names no action: "Done.", "All set!", "Everything is finished.",
/// "I've completed everything you asked for." The completion must end its clause, so
/// "Done — the README covers setup" counts while "Finished products ship Monday" does not.
pub fn general_completion(claim: &str) -> bool {
    !words(claim)
        .iter()
        .any(|w| NOT_COMPLETED.contains(&w.as_str()))
        && !partial_completion(claim)
        && clauses(claim)
            .first()
            .is_some_and(|clause| finished(&strs(clause)))
}

/// An admission that the work is only partly finished: "Mostly done.", "Done, except for the
/// docs.", "Not quite finished.", "I haven't finished everything yet." Only the completion's own
/// clause and a directly following qualifier or exception count, so "Done; the changes are mostly
/// cosmetic" is a completion.
pub fn partial_completion(claim: &str) -> bool {
    if words(claim)
        .iter()
        .any(|w| NOT_A_REPORT.contains(&w.as_str()))
    {
        return false;
    }
    let clauses = clauses(claim);
    let Some(first) = clauses.first().map(|clause| strs(clause)) else {
        return false;
    };
    let without = |dropped: &[&str]| -> Vec<&str> {
        first
            .iter()
            .copied()
            .filter(|w| !dropped.contains(w))
            .collect()
    };
    // "Mostly done", "Everything is nearly complete"
    if first.iter().any(|w| PARTIAL.contains(w)) && finished(&without(PARTIAL)) {
        return true;
    }
    // "Not quite done", "It isn't finished yet", "I haven't finished everything"
    if first.iter().any(|w| NEGATIONS.contains(w))
        && finished(&without(&[NEGATIONS, NEGATED].concat()))
    {
        return true;
    }
    // "Done except for the docs", "Everything is complete apart from the changelog"
    if let Some(at) = exception(&first) {
        return at > 0 && finished(&first[..at]);
    }
    // "Done, mostly.", "All set, except the migration still needs review."
    finished(&first)
        && clauses.get(1).is_some_and(|next| {
            let next = strs(next);
            next.iter().all(|w| PARTIAL.contains(w)) || exception(&next) == Some(0)
        })
}

/// Where an exception to a completion begins: "except", "apart from", "aside from", "besides",
/// or "other than".
fn exception(words: &[&str]) -> Option<usize> {
    (0..words.len()).find(|&i| {
        ["except", "apart", "aside", "besides"].contains(&words[i])
            || words[i] == "other" && words.get(i + 1) == Some(&"than")
    })
}

/// A sentence's clauses as lowercase words, split at clause punctuation.
fn clauses(claim: &str) -> Vec<Vec<String>> {
    claim
        .replace(" - ", " — ")
        .split(['.', '!', ',', ';', ':', '—', '–'])
        .map(words)
        .filter(|clause| !clause.is_empty())
        .collect()
}

fn strs(words: &[String]) -> Vec<&str> {
    words.iter().map(String::as_str).collect()
}

/// Whether a clause says all of the work is finished, with nothing after the completion.
fn finished(clause: &[&str]) -> bool {
    let state = |rest: &[&str], after_all: bool| match rest {
        [word] => {
            ["done", "complete", "completed", "finished"].contains(word)
                || after_all && *word == "set"
        }
        _ => false,
    };
    // "Done", "All done", "All set"
    if state(clause, false) || clause.first() == Some(&"all") && state(&clause[1..], true) {
        return true;
    }
    let starts_with = |phrase: &str| {
        let phrase: Vec<&str> = phrase.split_whitespace().collect();
        (clause.len() >= phrase.len() && clause.iter().zip(&phrase).all(|(a, b)| a == b))
            .then_some(phrase.len())
    };
    // "Everything is finished", "It's all done", "I'm done", "We're all set"
    for subject in ALL_WORK.iter().chain(&["i", "we"]) {
        if let Some(start) = starts_with(subject) {
            let rest = &clause[start..];
            let links = rest
                .iter()
                .take_while(|w| LINKING.contains(*w) || SUBJECT_LINKING.contains(*w))
                .count();
            if state(&rest[links..], rest[..links].contains(&"all")) {
                return true;
            }
        }
    }
    // "I've completed everything you asked for", "Finished the work"
    let start = clause
        .iter()
        .take_while(|w| ["i", "we", "have", "ve"].contains(*w))
        .count();
    if let [verb, object @ ..] = &clause[start..] {
        return ["completed", "finished"].contains(verb)
            && ALL_WORK
                .iter()
                .any(|phrase| phrase.split_whitespace().eq(object.iter().copied()));
    }
    false
}

/// A completion state counts only after its noun, through linking words: "The task is now done."
fn stated(claim: &[String], noun: &[String], states: &[&str]) -> bool {
    (0..claim.len()).any(|start| {
        let end = start + noun.len();
        let Some(candidate) = claim.get(start..end) else {
            return false;
        };
        candidate
            .iter()
            .zip(noun)
            .all(|(word, noun)| word == noun || *word == format!("{noun}s"))
            && claim[end..]
                .iter()
                .find(|word| !LINKING.contains(&word.as_str()))
                .is_some_and(|word| states.contains(&word.as_str()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn action_receipt_catalog_covers_every_action_and_preserves_targets() {
        for entry in COMMON_ACTIONS {
            let short = format!("{} {}.", entry.noun, entry.verb);
            assert_eq!(common_action(&short), Some(entry.action), "{short}");
            assert!(mentions_common_completion(&short), "{short}");
            for claim in [
                format!(
                    "{} {} {} `Target  A`.",
                    entry.noun, entry.verb, entry.preposition
                ),
                format!("I {} {} Target  A.", entry.active, entry.preposition),
            ] {
                assert_eq!(
                    common_claim(&claim),
                    Some((entry.action.into(), Some("Target  A".into()))),
                    "{claim}"
                );
            }
            assert!(!mentions_common_completion(&format!(
                "{} was not {}",
                entry.noun, entry.verb
            )));
        }
        assert_eq!(
            common_claim("Pull request merged for PR-42."),
            Some(("pull_request.merge".into(), Some("PR-42".into())))
        );
        assert!(common_claim("A build completed with 3 warnings for app.").is_none());
    }

    fn contract() -> ActionContract {
        serde_json::from_value(json!({
            "action":"email.send",
            "target_pointer":"/to",
            "completion_template":"Email sent to {target}."
        }))
        .unwrap()
    }

    fn receipt(target: &str) -> Value {
        json!({
            "version": 1,
            "tool_call_id": "call-1",
            "action": "email.send",
            "target": target,
            "outcome": "succeeded",
            "confirmation": "message-123"
        })
    }

    #[test]
    fn action_receipt_declarations_validate_bounds_and_both_tool_formats() {
        let contract = serde_json::to_value(contract()).unwrap();
        let flat =
            json!({"name":"mail__send","parameters":{"type":"object"},"action_contract":contract});
        for value in [flat.clone(), json!({"type":"function","function":flat})] {
            let tool: crate::ToolDefinition = serde_json::from_value(value).unwrap();
            assert_eq!(tool.action_contract.unwrap().action, "email.send");
            assert!(!tool.safe_to_memoize);
        }
        for (key, value) in [
            ("action", json!("")),
            ("action", json!("Email Send")),
            ("target_pointer", json!("to")),
            ("target_pointer", json!("/bad~2escape")),
            ("completion_template", json!("Email sent")),
            ("completion_template", json!("Sent {target} {target}")),
            ("permissions", json!("all")),
        ] {
            let mut invalid = contract.clone();
            invalid[key] = value;
            assert!(serde_json::from_value::<ActionContract>(invalid).is_err());
        }
        let legacy: crate::ToolDefinition =
            serde_json::from_value(json!({"name":"legacy","action_contract":null})).unwrap();
        assert!(legacy.action_contract.is_none());
    }

    #[test]
    fn action_receipts_bind_exact_call_action_and_target_and_require_confirmation() {
        let contract = contract();
        let args = json!({"to":"alice@example.com"});
        assert_eq!(
            collect(
                &contract,
                "call-1",
                &args,
                &json!({RECEIPTS_KEY:[receipt("alice@example.com")]})
            )[0]
            .outcome,
            ActionOutcome::Succeeded
        );
        for (key, value) in [
            ("version", json!(2)),
            ("tool_call_id", json!("other")),
            ("action", json!("refund.process")),
            ("target", json!("bob@example.com")),
            ("confirmation", Value::Null),
            ("outcome", json!("queued")),
            ("permissions", json!("all")),
        ] {
            let mut invalid = receipt("alice@example.com");
            invalid[key] = value;
            let result = collect(&contract, "call-1", &args, &json!({RECEIPTS_KEY:[invalid]}));
            assert_eq!(result[0].outcome, ActionOutcome::Unknown, "{key}");
            assert!(result[0].confirmation.is_none());
        }
    }

    #[test]
    fn action_receipt_arrays_are_atomic_and_missing_targets_stay_unknown() {
        let contract = contract();
        let args = json!({"to":["alice","bob"]});
        let one = json!({RECEIPTS_KEY:[receipt("alice")]});
        let results = collect(&contract, "call-1", &args, &one);
        assert_eq!(results[0].outcome, ActionOutcome::Succeeded);
        assert_eq!(results[1].outcome, ActionOutcome::Unknown);
        for values in [
            vec![receipt("alice"), receipt("alice")],
            vec![receipt("alice"), receipt("mallory")],
            vec![receipt("alice"); MAX_RECEIPTS + 1],
        ] {
            assert!(
                collect(&contract, "call-1", &args, &json!({RECEIPTS_KEY:values}))
                    .iter()
                    .all(|r| r.outcome == ActionOutcome::Unknown)
            );
        }
        let structured = json!({"content":[],"structuredContent":one});
        assert_eq!(
            collect(&contract, "call-1", &args, &structured)[0].outcome,
            ActionOutcome::Succeeded
        );
        let competing = json!({RECEIPTS_KEY:[receipt("alice")], "structuredContent":one});
        assert!(collect(&contract, "call-1", &args, &competing)
            .iter()
            .all(|r| r.outcome == ActionOutcome::Unknown));
        assert!(contract
            .targets(&json!({"to":["alice", "alice"]}))
            .is_none());
    }

    #[test]
    fn action_receipt_wording_preserves_target_identity_and_custom_domains() {
        let contract = contract();
        assert_eq!(
            contract.claim_target("Email sent to `Alice  Smith`."),
            Some("Alice  Smith".into())
        );
        assert_eq!(
            contract.claim_target("EMAIL SENT TO Alice."),
            Some("Alice".into())
        );
        assert!(contract
            .claim_target("Email sent to Alice and then another message.")
            .is_some()); // unmatched target, never inferred as Alice
        assert!(contract.mentions_completion("The email was successfully sent to Alice."));
        assert!(!contract.mentions_completion("The email was not sent to Alice."));
        assert!(mentions_common_completion(
            "A refund of $500 was processed for order-1."
        ));
        assert!(common_claim("A refund of $500 was processed for order-1.").is_none());
        assert_eq!(
            common_claim("Email sent to Alice  Smith."),
            Some(("email.send".into(), Some("Alice  Smith".into())))
        );
    }

    #[test]
    fn action_receipt_wording_accepts_equivalent_forms_and_rejects_requests() {
        for (claim, action, target) in [
            (
                "The email has been sent to Alice  Smith.",
                "email.send",
                Some("Alice  Smith"),
            ),
            (
                "I've sent the email to alice@example.com.",
                "email.send",
                Some("alice@example.com"),
            ),
            (
                "The email was successfully sent to Alice.",
                "email.send",
                Some("Alice"),
            ),
            (
                "I emailed alice@example.com.",
                "email.send",
                Some("alice@example.com"),
            ),
            ("We’ve messaged #general.", "message.send", Some("#general")),
            ("Refunded order-1.", "refund.process", Some("order-1")),
            ("The task is complete.", "task.complete", None),
            ("Task done.", "task.complete", None),
            (
                "I finished the task for ENG-12.",
                "task.complete",
                Some("ENG-12"),
            ),
            (
                "The meeting is scheduled for 2026-09-17.",
                "meeting.schedule",
                Some("2026-09-17"),
            ),
            (
                "I updated the record for acme.com.",
                "record.update",
                Some("acme.com"),
            ),
        ] {
            assert_eq!(
                common_claim(claim),
                Some((action.into(), target.map(Into::into))),
                "{claim}"
            );
            assert!(mentions_common_completion(claim), "{claim}");
        }
        // A scope claim is recognized but never bound to a single target.
        assert!(mentions_common_completion(
            "All tasks are now marked as done."
        ));
        assert!(common_claim("All tasks are now marked as done.").is_none());
        for claim in [
            "Complete the task by Friday.",
            "Let me know when the task is done.",
            "Once the email is sent, the ticket can be closed.",
            "The email isn't sent yet.",
            "The task wasn't finished.",
            "Delivered the email to Alice.",
        ] {
            assert!(!mentions_common_completion(claim), "{claim}");
            assert!(common_claim(claim).is_none(), "{claim}");
        }
        assert!(!contract().mentions_completion("The email isn't sent to Alice yet."));
    }

    #[test]
    fn action_receipt_aliases_cover_common_synonyms_without_loose_code_matches() {
        for (claim, action, target) in [
            (
                "Reservation canceled for booking-7.",
                "booking.cancel",
                Some("booking-7"),
            ),
            (
                "Order canceled for order-9.",
                "order.cancel",
                Some("order-9"),
            ),
            (
                "The subscription was canceled for acct-3.",
                "subscription.cancel",
                Some("acct-3"),
            ),
            (
                "Appointment canceled for Alice.",
                "meeting.cancel",
                Some("Alice"),
            ),
            (
                "I set up the meeting for 2026-09-18.",
                "meeting.schedule",
                Some("2026-09-18"),
            ),
            (
                "Booked a meeting for Monday.",
                "meeting.schedule",
                Some("Monday"),
            ),
            ("Booked table-4.", "booking.confirm", Some("table-4")),
            ("Issue created for BUG-12.", "ticket.create", Some("BUG-12")),
            (
                "I've filed a ticket for BUG-12.",
                "ticket.create",
                Some("BUG-12"),
            ),
            ("Opened PR #42.", "pull_request.create", Some("#42")),
            ("Merged PR #42.", "pull_request.merge", Some("#42")),
            (
                "Refund issued for order-1.",
                "refund.process",
                Some("order-1"),
            ),
            (
                "Payment completed for invoice-42.",
                "payment.process",
                Some("invoice-42"),
            ),
            ("Order submitted for Alice.", "order.place", Some("Alice")),
            ("Texted +15551234567.", "message.send", Some("+15551234567")),
            ("DM sent to Bob.", "message.send", Some("Bob")),
            (
                "Posted the report to #finance.",
                "report.publish",
                Some("#finance"),
            ),
            (
                "Row updated for customer-42.",
                "record.update",
                Some("customer-42"),
            ),
            (
                "Doc uploaded to Shared Drive.",
                "document.upload",
                Some("Shared Drive"),
            ),
            ("Todo done.", "task.complete", None),
        ] {
            assert_eq!(
                common_claim(claim),
                Some((action.into(), target.map(Into::into))),
                "{claim}"
            );
        }
        for claim in [
            "The meeting was canceled.",
            "I've filed a ticket about the outage.",
            "Refund issued.",
            "Posted the report.",
        ] {
            assert!(mentions_common_completion(claim), "{claim}");
        }
        // Aliases never outrank another action's own wording.
        assert_eq!(
            common_claim("Posted the message to #general."),
            Some(("message.send".into(), Some("#general".into())))
        );
        // Ambiguous nouns and verbs count only in exact wording, never inside code prose.
        for claim in [
            "Fixed the issue where the ticket opened twice.",
            "Completed the payment integration in the checkout flow.",
            "Edited the record struct to add a field.",
            "Opened the pull request to review the diff.",
        ] {
            assert!(!mentions_common_completion(claim), "{claim}");
            assert!(common_claim(claim).is_none(), "{claim}");
        }
    }

    #[test]
    fn declared_confirmation_pointer_maps_ordinary_results_without_receipts() {
        let mut declared = serde_json::to_value(contract()).unwrap();
        assert!(declared.get("confirmation_pointer").is_none());
        declared["confirmation_pointer"] = json!("/id");
        let mapped: ActionContract = serde_json::from_value(declared.clone()).unwrap();
        let args = json!({"to":"alice@example.com"});
        for result in [
            json!({"id":"msg-123","threadId":"t-1"}),
            json!({"content":[],"structuredContent":{"id":"msg-123"}}),
            json!({"content":[{"type":"text","text":"{\"id\":\"msg-123\"}"}]}),
        ] {
            let evidence = collect(&mapped, "call-1", &args, &result);
            assert_eq!(evidence[0].outcome, ActionOutcome::Succeeded, "{result}");
            assert_eq!(evidence[0].confirmation.as_deref(), Some("msg-123"));
            assert!(evidence[0].mapped);
        }
        let error = json!({"id":"msg-123","error":{"message":"quota exceeded"}});
        assert_eq!(
            collect(&mapped, "call-1", &args, &error)[0].outcome,
            ActionOutcome::Failed
        );
        for result in [
            json!({}),
            json!({"id":""}),
            json!({"id":{"nested":true}}),
            json!("sent"),
            // Any receipt envelope, even an invalid one, is never replaced by the mapping.
            json!({RECEIPTS_KEY:[{"version":2}],"id":"msg-123"}),
        ] {
            let evidence = collect(&mapped, "call-1", &args, &result);
            assert_eq!(evidence[0].outcome, ActionOutcome::Unknown, "{result}");
            assert!(!evidence[0].mapped);
        }
        assert_eq!(
            collect(&contract(), "call-1", &args, &json!({"id":"msg-123"}))[0].outcome,
            ActionOutcome::Unknown
        );
        for pointer in [json!("id"), json!(""), json!("/bad~2escape"), json!(7)] {
            let mut invalid = declared.clone();
            invalid["confirmation_pointer"] = pointer;
            assert!(serde_json::from_value::<ActionContract>(invalid).is_err());
        }
    }

    #[test]
    fn general_completion_requires_a_finished_clause_without_hedges() {
        for claim in [
            "Done.",
            "All set!",
            "✅ Done",
            "Everything is finished.",
            "It's all done.",
            "I'm done.",
            "The requested changes are complete.",
            "I've completed everything you asked for.",
            "Done — the README now covers setup.",
        ] {
            assert!(general_completion(claim), "{claim}");
        }
        for claim in [
            "Almost done.",
            "Not done yet.",
            "Let me know when you're done.",
            "Finished products ship Monday.",
            "The work is complete except for the docs.",
            "This is done by the scheduler.",
            "Done reading the config.",
        ] {
            assert!(!general_completion(claim), "{claim}");
        }
    }

    #[test]
    fn partial_completion_reports_hedged_and_excepted_work() {
        for claim in [
            "Mostly done.",
            "Almost finished — just the docs left.",
            "Everything is nearly complete.",
            "Done, mostly.",
            "Done except for the docs.",
            "All set, except the migration still needs review.",
            "Everything is complete apart from the changelog.",
            "Not quite done.",
            "It isn't finished yet.",
            "I haven't finished everything yet.",
        ] {
            assert!(partial_completion(claim), "{claim}");
            assert!(!general_completion(claim), "{claim}");
        }
        for claim in [
            "Done.",
            "Done; the changes are mostly cosmetic.",
            "Once it's mostly done, I'll run the tests.",
            "The docs aren't done yet.",
            "Mostly cosmetic changes.",
            "Done, but note the API key rotates daily.",
        ] {
            assert!(!partial_completion(claim), "{claim}");
        }
        assert!(general_completion("Done; the changes are mostly cosmetic."));
    }
}
