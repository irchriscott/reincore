//! Domain-neutral task-type classification.
//!
//! A task gets one primary type from its first prompt. Signals are weighted and
//! scored together rather than checked in first-match order, so the intended
//! outcome can beat incidental vocabulary. For example, "research vendors and
//! draft a report" is content creation because the requested deliverable is a
//! report, while "research regional vendors" remains research.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// The primary outcome of a task, independent of the agent's industry.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Feature,
    BugFix,
    Research,
    Analysis,
    Planning,
    ContentCreation,
    Communication,
    Operations,
    Review,
    Learning,
    Support,
    #[default]
    General,
}

impl TaskType {
    pub const ALL: [Self; 12] = [
        Self::Feature,
        Self::BugFix,
        Self::Research,
        Self::Analysis,
        Self::Planning,
        Self::ContentCreation,
        Self::Communication,
        Self::Operations,
        Self::Review,
        Self::Learning,
        Self::Support,
        Self::General,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Feature => "feature",
            Self::BugFix => "bug_fix",
            Self::Research => "research",
            Self::Analysis => "analysis",
            Self::Planning => "planning",
            Self::ContentCreation => "content_creation",
            Self::Communication => "communication",
            Self::Operations => "operations",
            Self::Review => "review",
            Self::Learning => "learning",
            Self::Support => "support",
            Self::General => "general",
        }
    }
}

impl FromStr for TaskType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "feature" | "enhancement" => Ok(Self::Feature),
            "bug" | "bugfix" | "bug_fix" | "fix" => Ok(Self::BugFix),
            "research" => Ok(Self::Research),
            "analysis" | "analytics" => Ok(Self::Analysis),
            "plan" | "planning" => Ok(Self::Planning),
            "content" | "content_creation" | "writing" => Ok(Self::ContentCreation),
            "communication" | "communications" => Ok(Self::Communication),
            "operation" | "operations" => Ok(Self::Operations),
            "review" | "validation" => Ok(Self::Review),
            "learning" | "education" | "explanation" => Ok(Self::Learning),
            "support" | "assistance" => Ok(Self::Support),
            "general" | "other" => Ok(Self::General),
            _ => Err(()),
        }
    }
}

type WeightedSignal = (&'static str, u16);

const FEATURE: &[WeightedSignal] = &[
    ("feature", 7),
    ("new capability", 7),
    ("add support", 6),
    ("implementation", 6),
    ("implement", 5),
    ("integrate", 5),
    ("enhance", 4),
    ("extend", 4),
    ("develop", 4),
    ("build", 3),
    ("launch", 3),
];

const BUG_FIX: &[WeightedSignal] = &[
    ("bug fix", 9),
    ("bug", 7),
    ("defect", 7),
    ("regression", 7),
    ("not working", 6),
    ("resolve issue", 6),
    ("broken", 5),
    ("incorrect", 5),
    ("troubleshoot", 5),
    ("repair", 5),
    ("fix", 4),
    ("error", 4),
    ("failure", 4),
    ("failing", 4),
    ("crash", 5),
    ("exception", 4),
    ("fail", 4),
];

const RESEARCH: &[WeightedSignal] = &[
    ("research", 7),
    ("literature review", 7),
    ("market research", 7),
    ("investigate", 6),
    ("look up", 6),
    ("find out", 6),
    ("gather information", 6),
    ("discover", 4),
    ("sources", 3),
    ("evidence", 3),
];

const ANALYSIS: &[WeightedSignal] = &[
    ("root cause", 7),
    ("analyze", 6),
    ("analyse", 6),
    ("assess", 5),
    ("compare", 5),
    ("evaluate", 5),
    ("audit", 5),
    ("reconcile", 5),
    ("forecast", 4),
    ("calculate", 4),
    ("diagnose", 4),
    ("model", 3),
    ("explain why", 3),
];

const PLANNING: &[WeightedSignal] = &[
    ("roadmap", 7),
    ("plan", 6),
    ("strategy", 6),
    ("prioritize", 5),
    ("prioritise", 5),
    ("schedule", 4),
    ("organize", 4),
    ("organise", 4),
    ("coordinate", 4),
    ("agenda", 4),
    ("itinerary", 4),
];

const CONTENT_CREATION: &[WeightedSignal] = &[
    ("lesson plan", 7),
    ("marketing copy", 7),
    ("draft", 6),
    ("write", 6),
    ("compose", 6),
    ("proposal", 5),
    ("presentation", 5),
    ("report", 5),
    ("brief", 4),
    ("article", 4),
    ("document", 4),
    ("curriculum", 4),
    ("transcript", 4),
    ("illustration", 4),
    ("produce", 3),
    ("generate", 3),
    ("create", 2),
];

const COMMUNICATION: &[WeightedSignal] = &[
    ("email", 7),
    ("reply", 6),
    ("respond", 5),
    ("notify", 5),
    ("contact", 5),
    ("message", 5),
    ("announce", 5),
    ("translate", 4),
    ("send", 4),
    ("pitch", 4),
];

const OPERATIONS: &[WeightedSignal] = &[
    ("update record", 7),
    ("book", 5),
    ("reserve", 5),
    ("order", 5),
    ("purchase", 5),
    ("pay", 5),
    ("submit", 5),
    ("onboard", 5),
    ("dispatch", 5),
    ("execute", 4),
    ("monitor", 4),
    ("track", 4),
    ("assign", 4),
    ("file", 3),
];

const REVIEW: &[WeightedSignal] = &[
    ("quality assurance", 7),
    ("compliance review", 7),
    ("proofread", 7),
    ("review", 6),
    ("verify", 5),
    ("validate", 5),
    ("approve", 5),
    ("inspect", 4),
    ("check", 3),
];

const LEARNING: &[WeightedSignal] = &[
    ("walk me through", 7),
    ("explain", 6),
    ("teach", 6),
    ("tutorial", 5),
    ("learn", 5),
    ("understand", 4),
    ("how does", 4),
    ("what is", 4),
    ("what are", 4),
    ("why does", 4),
];

const SUPPORT: &[WeightedSignal] = &[
    ("customer support", 8),
    ("service request", 7),
    ("help customer", 7),
    ("resolve request", 6),
    ("support", 5),
    ("assist", 5),
    ("triage", 5),
    ("help", 3),
    ("customer", 2),
];

const CLASSIFIERS: &[(TaskType, &[WeightedSignal])] = &[
    (TaskType::BugFix, BUG_FIX),
    (TaskType::Feature, FEATURE),
    (TaskType::Research, RESEARCH),
    (TaskType::Analysis, ANALYSIS),
    (TaskType::Planning, PLANNING),
    (TaskType::ContentCreation, CONTENT_CREATION),
    (TaskType::Communication, COMMUNICATION),
    (TaskType::Operations, OPERATIONS),
    (TaskType::Review, REVIEW),
    (TaskType::Learning, LEARNING),
    (TaskType::Support, SUPPORT),
];

fn signal_score(prompt: &str, signals: &[WeightedSignal]) -> u16 {
    // Compare stemmed words so `fixing`, `crashed`, and `drafted` score
    // like their dictionary forms.
    let haystack: Vec<String> = crate::lexical::words(prompt)
        .iter()
        .map(|word| crate::lexical::stem_word(word))
        .collect();
    signals
        .iter()
        .filter(|(signal, _)| {
            let needle: Vec<String> = crate::lexical::words(signal)
                .iter()
                .map(|word| crate::lexical::stem_word(word))
                .collect();
            !needle.is_empty()
                && needle.len() <= haystack.len()
                && haystack
                    .windows(needle.len())
                    .any(|window| window == needle.as_slice())
        })
        .map(|(_, weight)| *weight)
        .sum()
}

/// Classify the primary outcome requested by a task's first prompt.
///
/// Ties use the stable order in [`CLASSIFIERS`]. `General` is returned when no
/// meaningful signal exists, avoiding a fabricated type for novel domains.
pub fn classify_task_type(prompt: &str) -> TaskType {
    const MINIMUM_SCORE: u16 = 4;
    let mut best = (TaskType::General, 0_u16);
    for (task_type, signals) in CLASSIFIERS {
        let mut score = signal_score(prompt, signals);
        // The question bonus only confirms learning intent already present;
        // it must not turn any question into a learning task by itself.
        if *task_type == TaskType::Learning && score > 0 && crate::lexical::is_question(prompt) {
            score = score.saturating_add(2);
        }
        if score >= MINIMUM_SCORE && score > best.1 {
            best = (*task_type, score);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_inflected_and_weak_signals() {
        assert_eq!(
            classify_task_type("Fixing the login crash"),
            TaskType::BugFix
        );
        assert_eq!(
            classify_task_type("Why does the export fail?"),
            TaskType::BugFix
        );
        assert_eq!(
            classify_task_type("Reviewing vendor contracts"),
            TaskType::Review
        );
        assert_eq!(
            classify_task_type("Drafted the memo"),
            TaskType::ContentCreation
        );
        // Lone weak signals no longer fabricate a type.
        assert_eq!(
            classify_task_type("Create a billing dashboard"),
            TaskType::General
        );
        assert_eq!(
            classify_task_type("Check the deployment"),
            TaskType::General
        );
    }

    #[test]
    fn classifies_primary_types_across_agent_domains() {
        let cases = [
            ("Implement recurring invoice reminders", TaskType::Feature),
            ("Fix the incorrect VAT total on invoices", TaskType::BugFix),
            ("Research regional suppliers in Kampala", TaskType::Research),
            ("Analyze quarterly customer churn", TaskType::Analysis),
            ("Plan the graduate recruitment campaign", TaskType::Planning),
            (
                "Draft a renewal proposal for Acme",
                TaskType::ContentCreation,
            ),
            ("Email the renewal summary to Acme", TaskType::Communication),
            (
                "Book interviews for shortlisted candidates",
                TaskType::Operations,
            ),
            (
                "Review the procurement policy for compliance",
                TaskType::Review,
            ),
            ("Explain Uganda VAT filing deadlines", TaskType::Learning),
            ("Triage this customer support request", TaskType::Support),
            ("Remember my preferred timezone", TaskType::General),
        ];
        for (prompt, expected) in cases {
            assert_eq!(classify_task_type(prompt), expected, "{prompt}");
        }
    }

    #[test]
    fn requested_deliverable_can_outscore_incidental_work() {
        assert_eq!(
            classify_task_type("Research suppliers and draft a comparison report"),
            TaskType::ContentCreation
        );
        assert_eq!(
            classify_task_type("Analyze the findings and prepare a remediation roadmap"),
            TaskType::Planning
        );
    }

    #[test]
    fn implementation_outscores_incidental_request_for_help() {
        assert_eq!(
            classify_task_type(
                "Help me complete the implementation of rust sdk. Use go as bluprint"
            ),
            TaskType::Feature
        );
    }

    #[test]
    fn matching_is_boundary_aware() {
        assert_eq!(
            classify_task_type("Study planetary motion"),
            TaskType::General
        );
    }

    #[test]
    fn parses_api_override_aliases() {
        assert_eq!("bug-fix".parse(), Ok(TaskType::BugFix));
        assert_eq!("writing".parse(), Ok(TaskType::ContentCreation));
        assert!("made_up".parse::<TaskType>().is_err());
    }
}
