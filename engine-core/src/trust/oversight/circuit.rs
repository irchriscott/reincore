//! Progress-aware circuit breaker (anti-loop detection).
//!
//! Detects when the agent is stuck in a repetition loop by hashing each
//! (input, output) step pair and keeping a bounded ring buffer of recent
//! hashes. When the same hash appears more than `repetition_threshold`
//! times within the window, an intervention event is emitted so the runtime
//! can break the cycle - for example by escalating to a stronger model or
//! injecting a corrective system message.
//!
//! Detection is the last resort, not the first. [`dedupe_repeated_tool_results`]
//! runs on every outgoing request and answers an already-satisfied read or
//! search with a pointer to the copy the transcript already carries, which both
//! saves the tokens a refusal would have protected and tells the model what it
//! already has. Only repetition that survives that correction is scored.
//!
//! Scoring weighs evidence rather than counting bytes. A search that changes its
//! query is a new hypothesis even when it misses again, so empty results are
//! held to a higher threshold than identical successful ones; and progress the
//! agent does make - a fresh instruction, a written answer, a run of genuinely
//! new findings - decays the accumulated redundancy. Without that decay the
//! counters only ratchet upward, and a long exploration phase on a large
//! codebase reaches the hard stop by arithmetic alone.

use crate::error::EngineResult;
use crate::event::{EventStore, LoopInterventionPayload, SessionEvent, SessionEventType};
use crate::models::{ChatCompletionRequest, MessageContent};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Consecutive non-redundant exploration results that earn a decay of the
/// accumulated redundancy evidence.
const CLEAN_RESULTS_BEFORE_DECAY: u32 = 3;

/// Lowest redundancy score worth recording. A score of 1 means the lookup was
/// not redundant at all, so reporting it would write a health event per turn.
const MIN_REPORTABLE_REDUNDANCY: u32 = 2;

/// Length at which a written answer counts as progress rather than another
/// blind lookup.
const SUBSTANTIVE_RESPONSE_CHARS: usize = 200;

/// Shortest tool result worth replacing with a pointer. Below this the pointer
/// costs about as much as the content it replaces.
const MIN_DEDUPE_CHARS: usize = 512;

/// Longest result still eligible to be classified as a negative lookup.
/// Anything larger is real content, whatever words it happens to contain.
const MAX_NEGATIVE_RESULT_CHARS: usize = 200;

/// Phrases tools use to report that a lookup found nothing.
const NEGATIVE_RESULT_MARKERS: [&str; 8] = [
    "no matches",
    "no results",
    "no files found",
    "not found",
    "no such file",
    "nothing found",
    "0 results",
    "0 matches",
];

/// Command fragments that identify verification performed through a generic
/// terminal tool. These are deliberately data, rather than branches in
/// `is_verification_call`, so deployments can replace the list when their
/// build system uses different commands.
const DEFAULT_VERIFICATION_COMMANDS: &[&str] = &[
    // Rust
    "cargo test",
    "cargo nextest",
    "cargo check",
    "cargo clippy",
    "cargo build",
    "cargo fmt --check",
    // JavaScript and TypeScript
    "npm test",
    "npm run test",
    "npm run build",
    "npm run lint",
    "npm run check",
    "npm run typecheck",
    "npm run type-check",
    "pnpm test",
    "pnpm build",
    "pnpm lint",
    "pnpm check",
    "pnpm typecheck",
    "yarn test",
    "yarn build",
    "yarn lint",
    "yarn check",
    "yarn typecheck",
    "bun test",
    "bun run test",
    "bun run build",
    "bun run lint",
    "deno test",
    "deno check",
    "deno lint",
    "node --test",
    "vitest",
    "jest",
    "playwright test",
    "cypress run",
    "npx tsc",
    "tsc",
    "eslint",
    "biome check",
    // Python
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "python -m unittest",
    "python3 -m unittest",
    "ruff check",
    "mypy",
    "pyright",
    "pylint",
    "flake8",
    "tox",
    "nox",
    "black --check",
    "isort --check",
    // Go
    "go test",
    "go vet",
    "go build",
    "golangci-lint",
    // .NET
    "dotnet test",
    "dotnet build",
    "msbuild",
    // JVM
    "mvn test",
    "mvn verify",
    "mvn package",
    "gradle test",
    "gradle check",
    "gradle build",
    "gradlew test",
    "gradlew check",
    "gradlew build",
    "sbt test",
    // C, C++, and general build runners
    "cmake --build",
    "ctest",
    "make test",
    "make check",
    "make lint",
    "ninja test",
    "meson test",
    "bazel test",
    "bazel build",
    "just test",
    "just check",
    "just lint",
    "pre-commit run",
    // Other common ecosystems and validators
    "bundle exec rspec",
    "rspec",
    "rake test",
    "rubocop",
    "phpunit",
    "composer test",
    "phpstan",
    "psalm",
    "mix test",
    "mix credo",
    "mix compile",
    "swift test",
    "swift build",
    "dart test",
    "dart analyze",
    "flutter test",
    "flutter analyze",
    "flutter build",
    "shellcheck",
    "terraform validate",
    "tofu validate",
    "ansible-lint",
    "nix flake check",
    "invoke-pester",
];

fn default_verification_commands() -> Vec<String> {
    DEFAULT_VERIFICATION_COMMANDS
        .iter()
        .map(|command| (*command).to_string())
        .collect()
}

/// Hash of an (input, output) pair used for loop detection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepHash(pub String);

impl StepHash {
    pub fn new(input: &str, output: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hasher.update(b"\0");
        hasher.update(output.as_bytes());
        Self(hex::encode(hasher.finalize()))
    }
}

/// Configuration for the progress-aware circuit breaker.
///
/// Serializable so a caller can be shown the *resolved* configuration - what
/// will actually apply after normalization - rather than only the overrides it
/// asked for.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CircuitBreakerConfig {
    /// Whether repetition may nudge or refuse a turn at all.
    ///
    /// The off switch. When false the breaker still scores and records what it
    /// sees - the telemetry keeps working - but never intervenes.
    pub enabled: bool,
    /// Whether repeated read/search results are replaced by a pointer to the
    /// copy already in the transcript. Deliberately independent of `enabled`:
    /// dedup only ever saves tokens, it never refuses a turn, so turning off
    /// interventions should not silently start costing money.
    pub dedupe_tool_results: bool,
    /// How many repetitions of the same (input, output) hash trigger intervention.
    pub repetition_threshold: u32,
    /// How many repetitions mean nudging has failed and the turn must be refused.
    pub hard_stop_threshold: u32,
    /// Nudge threshold for repeated read/search results.
    ///
    /// Deliberately larger than [`repetition_threshold`]: a wedged command
    /// repeating itself is a loop by the third try, but exploration legitimately
    /// revisits ground, and the same budget applied to reads stops a model that
    /// is still finding its way around an unfamiliar codebase.
    ///
    /// [`repetition_threshold`]: Self::repetition_threshold
    pub exploration_repetition_threshold: u32,
    /// Refusal threshold for repeated read/search results.
    pub exploration_hard_stop_threshold: u32,
    /// Size of the bounded recent window to inspect.
    pub window_size: usize,
    /// Multiplier applied to both exploration thresholds for empty / "no
    /// matches" results. Ruling a hypothesis out is progress, so a run of
    /// failed lookups needs to be longer than a run of identical successful
    /// ones to count as a loop.
    pub negative_result_threshold_multiplier: u32,
    /// Model turns allowed after an all-complete todo checkpoint or the most
    /// recent subsequent substantive progress before asking the agent to finalize.
    pub post_completion_threshold: u32,
    /// Model turns allowed after that checkpoint before refusing another turn.
    pub post_completion_hard_stop_threshold: u32,
    /// Consecutive per-request-group turns without substantive progress before
    /// a one-time finalization nudge.
    pub request_group_step_threshold: u32,
    /// Consecutive turns without substantive progress before refusing another
    /// turn.
    pub request_group_hard_stop_threshold: u32,
    /// Emergency ceiling on total model turns in one request group. This is a
    /// final fuse, independent of progress-aware stall scoring.
    pub request_group_absolute_hard_stop_threshold: u32,
    /// Case-insensitive command fragments that count a successful invocation
    /// of a generic terminal tool as verification progress.
    ///
    /// This classifies progress only; it neither allows nor blocks commands.
    /// An explicitly configured list replaces the built-in defaults.
    #[serde(default = "default_verification_commands")]
    pub verification_commands: Vec<String>,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dedupe_tool_results: true,
            repetition_threshold: 3,
            hard_stop_threshold: 6,
            exploration_repetition_threshold: 5,
            exploration_hard_stop_threshold: 12,
            window_size: 16,
            // Lower than it would need to be on its own: a *different* query is
            // already different evidence, so this now scales only the case of
            // one identical failing lookup repeating.
            negative_result_threshold_multiplier: 2,
            post_completion_threshold: 8,
            post_completion_hard_stop_threshold: 16,
            request_group_step_threshold: 64,
            request_group_hard_stop_threshold: 128,
            request_group_absolute_hard_stop_threshold: 512,
            verification_commands: default_verification_commands(),
        }
    }
}

/// Persisted loop-detection settings, layered over the defaults.
///
/// Every field is optional so a partial setting leaves the rest at its default
/// rather than silently pinning it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoopSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedupe_tool_results: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hard_stop_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_repetition_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_hard_stop_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_result_threshold_multiplier: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_completion_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_completion_hard_stop_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_group_step_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_group_hard_stop_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_group_absolute_hard_stop_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_commands: Option<Vec<String>>,
}

impl LoopSettings {
    /// Read the same knobs from the environment, for deployments that configure
    /// the engine by env rather than through the settings store.
    ///
    /// Cached: this sits on the per-request path and the process environment
    /// does not change under a running engine.
    pub fn from_env() -> Self {
        static CACHED: std::sync::OnceLock<LoopSettings> = std::sync::OnceLock::new();
        CACHED.get_or_init(Self::read_env).clone()
    }

    fn read_env() -> Self {
        fn flag(key: &str) -> Option<bool> {
            match std::env::var(key)
                .ok()?
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                // Blank or unrecognized leaves the layer unset rather than
                // guessing. A compose file forwarding `${VAR:-}` passes an
                // empty string, which must not read as "disabled".
                _ => None,
            }
        }

        fn number<T: std::str::FromStr>(key: &str) -> Option<T> {
            std::env::var(key).ok()?.trim().parse().ok()
        }

        fn string_list(key: &str) -> Option<Vec<String>> {
            let value = std::env::var(key).ok()?;
            let value = value.trim();
            if value.is_empty() {
                return None;
            }
            if value.starts_with('[') {
                return serde_json::from_str(value).ok();
            }
            Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(str::to_owned)
                    .collect(),
            )
        }

        Self {
            enabled: flag("REINCORE_LOOP_ENABLED"),
            dedupe_tool_results: flag("REINCORE_LOOP_DEDUPE"),
            repetition_threshold: number("REINCORE_LOOP_REPETITION_THRESHOLD"),
            hard_stop_threshold: number("REINCORE_LOOP_HARD_STOP_THRESHOLD"),
            exploration_repetition_threshold: number("REINCORE_LOOP_EXPLORATION_THRESHOLD"),
            exploration_hard_stop_threshold: number(
                "REINCORE_LOOP_EXPLORATION_HARD_STOP_THRESHOLD",
            ),
            window_size: number("REINCORE_LOOP_WINDOW_SIZE"),
            negative_result_threshold_multiplier: number("REINCORE_LOOP_NEGATIVE_MULTIPLIER"),
            post_completion_threshold: number("REINCORE_LOOP_POST_COMPLETION_THRESHOLD"),
            post_completion_hard_stop_threshold: number(
                "REINCORE_LOOP_POST_COMPLETION_HARD_STOP_THRESHOLD",
            ),
            request_group_step_threshold: number("REINCORE_LOOP_REQUEST_GROUP_STEP_THRESHOLD"),
            request_group_hard_stop_threshold: number(
                "REINCORE_LOOP_REQUEST_GROUP_HARD_STOP_THRESHOLD",
            ),
            request_group_absolute_hard_stop_threshold: number(
                "REINCORE_LOOP_REQUEST_GROUP_ABSOLUTE_HARD_STOP_THRESHOLD",
            ),
            verification_commands: string_list("REINCORE_LOOP_VERIFICATION_COMMANDS"),
        }
    }

    /// Per-request escape hatches. Unknown or malformed values are ignored so a
    /// bad override never breaks a request.
    pub fn from_extra(extra: &HashMap<String, serde_json::Value>) -> Self {
        let flag = |key: &str| extra.get(key).and_then(serde_json::Value::as_bool);
        let number = |key: &str| {
            extra
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
        };
        let string_list = |key: &str| {
            extra.get(key).and_then(|value| {
                value
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(serde_json::Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .or_else(|| {
                        value.as_str().map(|items| {
                            items
                                .split(',')
                                .map(str::trim)
                                .filter(|item| !item.is_empty())
                                .map(str::to_owned)
                                .collect()
                        })
                    })
            })
        };
        Self {
            enabled: flag("reincore_loop_enabled"),
            dedupe_tool_results: flag("reincore_loop_dedupe"),
            repetition_threshold: number("reincore_loop_repetition_threshold"),
            hard_stop_threshold: number("reincore_loop_hard_stop_threshold"),
            exploration_repetition_threshold: number("reincore_loop_exploration_threshold"),
            exploration_hard_stop_threshold: number(
                "reincore_loop_exploration_hard_stop_threshold",
            ),
            window_size: extra
                .get("reincore_loop_window_size")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok()),
            negative_result_threshold_multiplier: number("reincore_loop_negative_multiplier"),
            post_completion_threshold: number("reincore_loop_post_completion_threshold"),
            post_completion_hard_stop_threshold: number(
                "reincore_loop_post_completion_hard_stop_threshold",
            ),
            request_group_step_threshold: number("reincore_loop_request_group_step_threshold"),
            request_group_hard_stop_threshold: number(
                "reincore_loop_request_group_hard_stop_threshold",
            ),
            request_group_absolute_hard_stop_threshold: number(
                "reincore_loop_request_group_absolute_hard_stop_threshold",
            ),
            verification_commands: string_list("reincore_loop_verification_commands"),
        }
    }

    /// Overlay `other` on top of `self`, with `other` winning where it is set.
    pub fn overlay(self, other: Self) -> Self {
        Self {
            enabled: other.enabled.or(self.enabled),
            dedupe_tool_results: other.dedupe_tool_results.or(self.dedupe_tool_results),
            repetition_threshold: other.repetition_threshold.or(self.repetition_threshold),
            hard_stop_threshold: other.hard_stop_threshold.or(self.hard_stop_threshold),
            exploration_repetition_threshold: other
                .exploration_repetition_threshold
                .or(self.exploration_repetition_threshold),
            exploration_hard_stop_threshold: other
                .exploration_hard_stop_threshold
                .or(self.exploration_hard_stop_threshold),
            window_size: other.window_size.or(self.window_size),
            negative_result_threshold_multiplier: other
                .negative_result_threshold_multiplier
                .or(self.negative_result_threshold_multiplier),
            post_completion_threshold: other
                .post_completion_threshold
                .or(self.post_completion_threshold),
            post_completion_hard_stop_threshold: other
                .post_completion_hard_stop_threshold
                .or(self.post_completion_hard_stop_threshold),
            request_group_step_threshold: other
                .request_group_step_threshold
                .or(self.request_group_step_threshold),
            request_group_hard_stop_threshold: other
                .request_group_hard_stop_threshold
                .or(self.request_group_hard_stop_threshold),
            request_group_absolute_hard_stop_threshold: other
                .request_group_absolute_hard_stop_threshold
                .or(self.request_group_absolute_hard_stop_threshold),
            verification_commands: other.verification_commands.or(self.verification_commands),
        }
    }

    /// Resolve to a usable config. Normalization happens in
    /// [`CircuitBreaker::new`], so out-of-range settings cannot produce a
    /// breaker that stops before it nudges.
    pub fn resolve(&self) -> CircuitBreakerConfig {
        let base = CircuitBreakerConfig::default();
        CircuitBreakerConfig {
            enabled: self.enabled.unwrap_or(base.enabled),
            dedupe_tool_results: self.dedupe_tool_results.unwrap_or(base.dedupe_tool_results),
            repetition_threshold: self
                .repetition_threshold
                .unwrap_or(base.repetition_threshold),
            hard_stop_threshold: self.hard_stop_threshold.unwrap_or(base.hard_stop_threshold),
            exploration_repetition_threshold: self
                .exploration_repetition_threshold
                .unwrap_or(base.exploration_repetition_threshold),
            exploration_hard_stop_threshold: self
                .exploration_hard_stop_threshold
                .unwrap_or(base.exploration_hard_stop_threshold),
            window_size: self.window_size.unwrap_or(base.window_size),
            negative_result_threshold_multiplier: self
                .negative_result_threshold_multiplier
                .unwrap_or(base.negative_result_threshold_multiplier),
            post_completion_threshold: self
                .post_completion_threshold
                .unwrap_or(base.post_completion_threshold),
            post_completion_hard_stop_threshold: self
                .post_completion_hard_stop_threshold
                .unwrap_or(base.post_completion_hard_stop_threshold),
            request_group_step_threshold: self
                .request_group_step_threshold
                .unwrap_or(base.request_group_step_threshold),
            request_group_hard_stop_threshold: self
                .request_group_hard_stop_threshold
                .unwrap_or(base.request_group_hard_stop_threshold),
            request_group_absolute_hard_stop_threshold: self
                .request_group_absolute_hard_stop_threshold
                .unwrap_or(base.request_group_absolute_hard_stop_threshold),
            verification_commands: self
                .verification_commands
                .clone()
                .unwrap_or(base.verification_commands),
        }
    }
}

/// What the breaker saw on one turn, whether or not it acted on it.
#[derive(Clone, Debug)]
pub struct LoopObservation {
    hash: StepHash,
    /// The nudge or refusal text, used only when this intervenes.
    message: String,
    /// How redundant the latest step was scored.
    pub count: u32,
    /// The threshold `count` was compared against, after evidence-class scaling.
    pub threshold: u32,
    /// Whether nudging has already failed and the turn must be refused.
    pub hard_stop: bool,
    /// Whether clients may automatically retry a resulting hard stop.
    retryable: bool,
    /// Whether `count` reached `threshold`. When false the turn proceeds
    /// untouched and only the advisory health finding is written.
    pub intervenes: bool,
    /// The policy response selected independently from the detector score.
    pub action: crate::intervention::InterventionAction,
    /// Whether the repeated result was an empty / "no matches" lookup.
    pub negative: bool,
    /// Canonical tool name, absent for the identical-step detector.
    pub tool: Option<String>,
    /// The file path or search query being repeated.
    pub source: Option<String>,
}

impl LoopObservation {
    /// Project onto the per-turn score stamped into the `LlmCall` event.
    pub fn redundancy(&self) -> crate::event::ExplorationRedundancy {
        crate::event::ExplorationRedundancy {
            count: self.count,
            threshold: self.threshold,
            negative: self.negative,
            tool: self.tool.clone().unwrap_or_else(|| "step".to_string()),
            source: self.source.clone(),
        }
    }

    /// Advisory finding for repetition that did *not* stop the turn. Always
    /// `Warning` and never `should_stop`: this is a record that the guard-rails
    /// held, not a reason to refuse anything.
    ///
    /// `None` once the observation intervenes, because the `LoopIntervention`
    /// event then carries the same information.
    pub fn health_finding(&self) -> Option<crate::agent_health::AgentHealthFinding> {
        use crate::agent_health::{AgentHealthFinding, AgentHealthKind, AgentHealthSeverity};
        if self.intervenes {
            return None;
        }
        Some(AgentHealthFinding {
            kind: AgentHealthKind::RedundantLookup,
            severity: AgentHealthSeverity::Warning,
            message: format!(
                "redundant lookup scored {} of {} without intervening{}",
                self.count,
                self.threshold,
                match self.source.as_deref() {
                    Some(source) => format!(" ({source})"),
                    None => String::new(),
                }
            ),
            tool_name: self.tool.clone(),
        })
    }

    /// The advisory finding as a standalone report.
    pub fn health_report(&self) -> crate::agent_health::AgentHealthReport {
        let findings: Vec<_> = self.health_finding().into_iter().collect();
        let policy = crate::intervention::current_policy();
        let interventions = findings
            .iter()
            .map(|finding| {
                crate::intervention::InterventionRecord::from_finding(finding, &policy, self.action)
            })
            .collect();
        crate::agent_health::AgentHealthReport {
            findings,
            should_stop: false,
            claim_grounding: None,
            interventions,
        }
    }
}

/// A persisted intervention awaiting application on the next turn.
#[derive(Clone, Debug)]
pub struct PendingIntervention {
    pub message: String,
    /// When true the caller must refuse the turn rather than bill it.
    pub hard_stop: bool,
    /// False for terminal request-group budgets, where retrying the same
    /// request would simply restart the runaway continuation.
    pub retryable: bool,
}

/// Progress-aware circuit breaker (anti-loop detection).
#[derive(Clone, Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    recent: VecDeque<StepHash>,
}

impl CircuitBreaker {
    pub fn new(mut config: CircuitBreakerConfig) -> Self {
        // Public configuration may come from user settings. Normalize zero or
        // contradictory values so the ring buffer stays bounded and a hard
        // stop can never precede the first corrective intervention.
        config.window_size = config.window_size.max(1);
        config.repetition_threshold = config.repetition_threshold.max(1);
        config.hard_stop_threshold = config.hard_stop_threshold.max(config.repetition_threshold);
        config.exploration_repetition_threshold = config.exploration_repetition_threshold.max(1);
        config.exploration_hard_stop_threshold = config
            .exploration_hard_stop_threshold
            .max(config.exploration_repetition_threshold);
        config.negative_result_threshold_multiplier =
            config.negative_result_threshold_multiplier.max(1);
        config.post_completion_threshold = config.post_completion_threshold.max(1);
        config.post_completion_hard_stop_threshold = config
            .post_completion_hard_stop_threshold
            .max(config.post_completion_threshold);
        config.request_group_step_threshold = config.request_group_step_threshold.max(1);
        config.request_group_hard_stop_threshold = config
            .request_group_hard_stop_threshold
            .max(config.request_group_step_threshold);
        config.request_group_absolute_hard_stop_threshold = config
            .request_group_absolute_hard_stop_threshold
            .max(config.request_group_hard_stop_threshold);
        let mut seen_verification_commands = std::collections::HashSet::new();
        config.verification_commands = config
            .verification_commands
            .into_iter()
            .map(|command| command.trim().to_ascii_lowercase())
            .filter(|command| {
                !command.is_empty() && seen_verification_commands.insert(command.clone())
            })
            .collect();
        let window_size = config.window_size;
        Self {
            config,
            recent: VecDeque::with_capacity(window_size),
        }
    }

    pub fn default_config() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Record a step and return an intervention message if a loop is detected.
    ///
    /// The session is NOT stopped; instead a hidden system message is returned
    /// for injection into the next turn.
    pub fn record_step(&mut self, input: &str, output: &str) -> Option<String> {
        let hash = StepHash::new(input, output);

        // Count occurrences within the bounded window.
        let count = self.recent.iter().filter(|h| **h == hash).count() as u32 + 1;

        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(hash);

        if count >= self.config.repetition_threshold {
            Some(format!(
                "You are repeating yourself. This approach has failed {} times. Try a different tool or a fundamentally different approach.",
                count
            ))
        } else {
            None
        }
    }

    /// Log a loop intervention event to the session event log.
    ///
    /// `hard_stop` is supplied by the caller rather than recomputed here: the
    /// threshold that applies depends on the evidence class (see
    /// [`CircuitBreakerConfig::negative_result_threshold_multiplier`]), so
    /// deriving it from the raw count would disagree with the message text.
    pub async fn log_intervention<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
        repeated_hash: &StepHash,
        repetition_count: u32,
        injected_message: &str,
        hard_stop: bool,
    ) -> EngineResult<()> {
        let action = if hard_stop {
            match crate::intervention::current_policy().loop_detection {
                crate::intervention::HeuristicMode::Block => {
                    crate::intervention::InterventionAction::ActionBlocked
                }
                _ => crate::intervention::InterventionAction::Paused,
            }
        } else {
            crate::intervention::InterventionAction::RepairRequested
        };
        self.log_intervention_record(
            event_store,
            session_id,
            repeated_hash,
            repetition_count,
            injected_message,
            hard_stop,
            true,
            action,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn log_intervention_record<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
        repeated_hash: &StepHash,
        repetition_count: u32,
        injected_message: &str,
        hard_stop: bool,
        retryable: bool,
        action: crate::intervention::InterventionAction,
        tool_name: Option<String>,
    ) -> EngineResult<()> {
        let payload = LoopInterventionPayload {
            repeated_hash: repeated_hash.0.clone(),
            repetition_count,
            injected_message: injected_message.to_string(),
            hard_stop,
            retryable,
            intervention: Some(self.intervention_record(
                repetition_count,
                injected_message,
                action,
                tool_name,
            )),
        };
        event_store
            .append_event(
                session_id,
                SessionEventType::LoopIntervention,
                serde_json::to_value(payload)?,
                None,
            )
            .await?;
        Ok(())
    }

    /// Return a persisted intervention that should be injected into the next
    /// request. Because this reads the event log, it works across cloud
    /// replicas and process restarts instead of relying on in-memory state.
    pub async fn pending_intervention<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
    ) -> EngineResult<Option<PendingIntervention>> {
        let Some(latest) = event_store.latest_seq(session_id).await? else {
            return Ok(None);
        };
        let pending_window = self.config.window_size.max(64);
        let since = latest.saturating_sub(pending_window as i64 - 1).max(1);
        let events = event_store
            .read_events(session_id, Some(since), Some(pending_window))
            .await?;
        let Some(intervention_index) = events.iter().rposition(|event| {
            matches!(
                event.event_type,
                SessionEventType::LoopIntervention | SessionEventType::LlmCall
            )
        }) else {
            return Ok(None);
        };
        let event = &events[intervention_index];
        if event.event_type != SessionEventType::LoopIntervention {
            return Ok(None);
        }
        let Some(message) = event
            .payload
            .get("injected_message")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
        else {
            return Ok(None);
        };
        // An empty message is the tombstone written by `clear_intervention`.
        if message.is_empty() {
            return Ok(None);
        }
        let hard_stop = event
            .payload
            .get("hard_stop")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let retryable = event
            .payload
            .get("retryable")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        // The model call is logged before its external tool result arrives. A
        // hard stop armed at that boundary must not discard a later successful
        // mutation, verification, todo update, or novel lookup. Allow exactly
        // one finalization turn; that LLM event then supersedes this pending
        // intervention and can be scored normally if it asks for more tools.
        if hard_stop {
            let progress_ids =
                substantive_progress_result_ids(&events, &self.config.verification_commands);
            let progressed_after_intervention = events[intervention_index + 1..]
                .iter()
                .filter_map(successful_tool_result)
                .any(|(id, _)| progress_ids.contains(id));
            if progressed_after_intervention {
                return Ok(Some(PendingIntervention {
                    message: "The tool call that completed after the loop limit made substantive progress. Use its result and produce the final answer now; call another tool only to resolve a concrete failure.".to_string(),
                    hard_stop: false,
                    retryable: true,
                }));
            }
        }

        Ok(Some(PendingIntervention {
            message,
            hard_stop,
            retryable,
        }))
    }

    /// Consume the pending intervention so the caller may take one more turn.
    ///
    /// A hard stop must not brick the session: once the refusal has been
    /// delivered, an explicit retry is allowed. If the agent is still stuck the
    /// next inspection re-arms the stop, so each retry costs at most one call.
    pub async fn clear_intervention<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
    ) -> EngineResult<()> {
        let payload = LoopInterventionPayload {
            repeated_hash: String::new(),
            repetition_count: 0,
            injected_message: String::new(),
            hard_stop: false,
            retryable: true,
            intervention: None,
        };
        event_store
            .append_event(
                session_id,
                SessionEventType::LoopIntervention,
                serde_json::to_value(payload)?,
                None,
            )
            .await?;
        Ok(())
    }

    /// Inspect the recent persisted event window and log an intervention when
    /// the latest LLM input/output pair has repeated without a tool result
    /// (our state-progression signal).
    ///
    /// Equivalent to [`observe`](Self::observe) followed by
    /// [`log_observation`](Self::log_observation). Callers that also need to
    /// stamp the observation onto a payload should use the two halves directly,
    /// so the event window is read once rather than twice.
    pub async fn inspect_and_log<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
    ) -> EngineResult<bool> {
        let Some(observation) = self.observe(event_store, session_id).await? else {
            return Ok(false);
        };
        if !observation.intervenes {
            // Standalone path: there is no caller-owned health report to merge
            // the near-miss into, so record it on its own.
            crate::agent_health::log_agent_health(
                event_store,
                session_id,
                &observation.health_report(),
            )
            .await?;
            return Ok(false);
        }
        self.log_observation(event_store, session_id, &observation)
            .await
    }

    /// Score the recent event window without writing anything.
    ///
    /// Returns an observation whenever there is *any* repetition to report,
    /// including repetition below the intervention threshold - that near-miss is
    /// the signal that tells you whether dedup and the evidence split are
    /// working, and it is invisible in the event log otherwise.
    pub async fn observe<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
    ) -> EngineResult<Option<LoopObservation>> {
        self.observe_with_pending(event_store, session_id, None)
            .await
    }

    /// Score the window including a turn that has not been persisted yet.
    ///
    /// Callers that score before their accounting write must pass that turn in
    /// (see [`crate::event::LlmCallPayload::pending_audit`]); otherwise the
    /// newest LLM call is invisible and every LLM-derived signal - the
    /// identical-step detector, the progress decay, the reset on a mutating
    /// tool call - lags a turn behind.
    pub async fn observe_with_pending<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
        pending_llm_call: Option<serde_json::Value>,
    ) -> EngineResult<Option<LoopObservation>> {
        let Some(latest) = event_store.latest_seq(session_id).await? else {
            return Ok(None);
        };
        // Tool execution/accounting can insert many events between LLM calls.
        // Use a larger bounded window for result-based exploration detection.
        let group_window = (self.config.request_group_absolute_hard_stop_threshold as usize)
            .saturating_mul(8)
            .clamp(128, 4096);
        let window = self.config.window_size.max(128).max(group_window);
        let since = latest.saturating_sub(window as i64 - 1).max(1);
        let mut events = event_store
            .read_events(session_id, Some(since), Some(window))
            .await?;
        if let Some(payload) = pending_llm_call {
            events.push(SessionEvent {
                session_id,
                seq: latest.saturating_add(1),
                event_type: SessionEventType::LlmCall,
                payload,
                cost_cents: None,
                created_at: chrono::Utc::now(),
            });
        }
        // A response without tool calls is already terminal. The phase-level
        // request-group guard must not poison the next user request by calling
        // that successful completion an unbounded tool loop. Exact repetition
        // scoring below remains available for non-agentic/framework events.
        let latest_response_is_terminal = events
            .iter()
            .rev()
            .find(|event| event.event_type == SessionEventType::LlmCall)
            .and_then(llm_response_continues_with_tools)
            == Some(false);
        let recent = &events[events.len().saturating_sub(self.config.window_size)..];
        // Semantic/phase-level guard for loops whose individual lookups vary.
        // This catches a long re-inspection spiral after the client has marked
        // every todo complete, and supplies an absolute backstop for one user
        // request that never produces a terminal answer.
        // Non-intervening observations are kept as candidates, not returned:
        // an advisory group/exploration report must not mask the
        // identical-step hard-stop detector below.
        let mut pending_observation: Option<LoopObservation> = None;
        let keep_pending = |candidate: LoopObservation, pending: &mut Option<LoopObservation>| {
            let keep = pending
                .as_ref()
                .is_none_or(|existing| candidate.count > existing.count);
            if keep {
                *pending = Some(candidate);
            }
        };
        if let Some(found) = (!latest_response_is_terminal)
            .then(|| request_group_stall(&events, &self.config))
            .flatten()
        {
            let (intervenes, hard_stop, action) = self.response_for(true, found.hard_stop);
            let observation = LoopObservation {
                hash: StepHash::new(found.kind, &found.group_id),
                count: found.count,
                threshold: found.threshold,
                hard_stop,
                retryable: !found.hard_stop,
                intervenes,
                action,
                negative: false,
                tool: Some("request_group".to_string()),
                source: Some(found.group_id),
                message: if hard_stop {
                    format!("Stopped: {} Further tool calls would continue an unbounded request without producing a final answer.", found.reason)
                } else {
                    format!("{} Use the evidence already collected, report any remaining blocker, and produce the final answer now instead of starting another inspection pass.", found.reason)
                },
            };
            if intervenes {
                return Ok(Some(observation));
            }
            keep_pending(observation, &mut pending_observation);
        }
        // Scored unfiltered: repetition below the threshold is still reported,
        // it just does not intervene.
        if let Some(found) = repeated_exploration_results(&events)
            .filter(|found| found.count >= MIN_REPORTABLE_REDUNDANCY)
        {
            let threshold = self.exploration_threshold(found.negative);
            // `enabled` gates intervening, not observing: switching the breaker
            // off should stop it refusing turns without also going blind.
            let crossed = found.count >= threshold;
            let hard_threshold = found.count >= self.exploration_hard_stop(found.negative);
            let (intervenes, hard_stop, action) = self.response_for(crossed, hard_threshold);
            let message = format!(
                "{}{} Use the findings already collected to complete the task, or explain the blocker. Do not repeat these lookups.",
                if hard_stop { "Stopped: " } else { "You are repeating yourself. " },
                found.describe(),
            );
            let observation = LoopObservation {
                hash: found.hash,
                count: found.count,
                threshold,
                hard_stop,
                retryable: true,
                intervenes,
                action,
                negative: found.negative,
                tool: Some(found.tool),
                source: found.source,
                message,
            };
            if intervenes {
                return Ok(Some(observation));
            }
            keep_pending(observation, &mut pending_observation);
        }
        // The identical-step detector reports only at or above its threshold,
        // so anything it returns always intervenes.
        let Some((hash, count)) = self.detect_persisted_loop(recent) else {
            return Ok(pending_observation);
        };
        let (intervenes, hard_stop, action) =
            self.response_for(true, count >= self.config.hard_stop_threshold);
        let message = if hard_stop {
            format!(
                "Stopped: the same step has repeated {} times with no progress, and the corrective nudge did not change the outcome. Further turns would burn tokens without advancing the task.",
                count
            )
        } else {
            format!(
                "You are repeating yourself. This approach has failed {} times. Try a different tool or a fundamentally different approach.",
                count
            )
        };
        Ok(Some(LoopObservation {
            hash,
            count,
            threshold: self.config.repetition_threshold,
            hard_stop,
            retryable: true,
            intervenes,
            action,
            negative: false,
            tool: None,
            source: None,
            message,
        }))
    }

    /// Persist an intervention when the observation crosses its threshold, and
    /// report whether it did.
    ///
    /// Below the threshold this writes nothing: the near-miss belongs on the
    /// turn's own `AgentHealth` event via
    /// [`LoopObservation::health_finding`], which the caller merges with its
    /// other findings so a turn writes one health event rather than two.
    pub async fn log_observation<E: EventStore>(
        &self,
        event_store: &E,
        session_id: Uuid,
        observation: &LoopObservation,
    ) -> EngineResult<bool> {
        if !observation.intervenes {
            return Ok(false);
        }
        self.log_intervention_record(
            event_store,
            session_id,
            &observation.hash,
            observation.count,
            &observation.message,
            observation.hard_stop,
            observation.retryable,
            observation.action,
            observation.tool.clone(),
        )
        .await?;
        Ok(true)
    }

    fn response_for(
        &self,
        crossed_threshold: bool,
        crossed_hard_threshold: bool,
    ) -> (bool, bool, crate::intervention::InterventionAction) {
        use crate::intervention::{HeuristicMode, InterventionAction};
        if !self.config.enabled || !crossed_threshold {
            return (false, false, InterventionAction::Observed);
        }
        match crate::intervention::current_policy().loop_detection {
            HeuristicMode::Observe => (false, false, InterventionAction::Observed),
            HeuristicMode::Warn => (false, false, InterventionAction::Warned),
            HeuristicMode::RepairOnce if crossed_hard_threshold => {
                (true, true, InterventionAction::Paused)
            }
            HeuristicMode::RepairOnce => (true, false, InterventionAction::RepairRequested),
            HeuristicMode::Block => (true, true, InterventionAction::ActionBlocked),
            HeuristicMode::Pause => (true, true, InterventionAction::Paused),
        }
    }

    fn intervention_record(
        &self,
        repetition_count: u32,
        message: &str,
        action: crate::intervention::InterventionAction,
        tool_name: Option<String>,
    ) -> crate::intervention::InterventionRecord {
        use crate::agent_health::{AgentHealthFinding, AgentHealthKind, AgentHealthSeverity};
        use crate::intervention::{InterventionAction, InterventionRecord};
        let policy = crate::intervention::current_policy();
        let finding = AgentHealthFinding {
            kind: AgentHealthKind::ToolLoop,
            severity: if matches!(
                action,
                InterventionAction::ActionBlocked | InterventionAction::Paused
            ) {
                AgentHealthSeverity::Critical
            } else {
                AgentHealthSeverity::Warning
            },
            message: format!("{message} (repetition count: {repetition_count})"),
            tool_name,
        };
        InterventionRecord::from_finding(&finding, &policy, action)
    }

    /// The resolved configuration this breaker is running with.
    pub fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Replace read/search results already in the transcript with a pointer,
    /// honouring the `dedupe_tool_results` switch.
    pub fn dedupe_tool_results(&self, request: &mut ChatCompletionRequest) -> ToolResultDedup {
        if !self.config.dedupe_tool_results {
            return ToolResultDedup::default();
        }
        dedupe_repeated_tool_results(request)
    }

    /// Threshold for a first corrective nudge on repeated read/search results.
    fn exploration_threshold(&self, negative: bool) -> u32 {
        self.scale_for_evidence(self.config.exploration_repetition_threshold, negative)
    }

    /// Threshold at which repeated read/search results refuse the turn.
    fn exploration_hard_stop(&self, negative: bool) -> u32 {
        self.scale_for_evidence(self.config.exploration_hard_stop_threshold, negative)
    }

    /// Empty / "no matches" results get a proportionally larger budget. A miss
    /// rules a hypothesis out, which is progress even though the bytes returned
    /// are identical every time; only sustained repetition of the *same* failing
    /// lookup is evidence of a loop.
    fn scale_for_evidence(&self, threshold: u32, negative: bool) -> u32 {
        if negative {
            threshold.saturating_mul(self.config.negative_result_threshold_multiplier)
        } else {
            threshold
        }
    }

    fn detect_persisted_loop(&self, events: &[SessionEvent]) -> Option<(StepHash, u32)> {
        let after_progress = events
            .iter()
            .rposition(|event| event.event_type == SessionEventType::ToolResult)
            .map(|index| &events[index + 1..])
            .unwrap_or(events);
        let hashes: Vec<StepHash> = after_progress
            .iter()
            .filter(|event| event.event_type == SessionEventType::LlmCall)
            .filter_map(step_hash_from_event)
            .collect();
        let latest = hashes.last()?.clone();
        let count = hashes.iter().filter(|hash| **hash == latest).count() as u32;
        (count >= self.config.repetition_threshold).then_some((latest, count))
    }
}

#[derive(Clone, Debug)]
struct RequestGroupStall {
    group_id: String,
    kind: &'static str,
    count: u32,
    threshold: u32,
    hard_stop: bool,
    reason: String,
}

fn llm_response_continues_with_tools(event: &SessionEvent) -> Option<bool> {
    let response = event.payload.get("response_message")?;
    Some(
        response
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|calls| !calls.is_empty()),
    )
}

fn successful_tool_result(event: &SessionEvent) -> Option<(&str, &serde_json::Value)> {
    if event.event_type != SessionEventType::ToolResult {
        return None;
    }
    let result = event.payload.get("result")?;
    let id = result.get("tool_call_id")?.as_str()?;
    if result
        .get("is_error")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let text = result
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if text.starts_with("error:") || text.contains("input to the tool was invalid") {
        return None;
    }
    Some((id, result))
}

fn canonical_tool_name(name: &str) -> String {
    crate::tool_names::canonical_name(name).to_ascii_lowercase()
}

fn is_mutation_tool(name: &str, arguments: &serde_json::Value) -> bool {
    let action = crate::tool_names::classify(name, arguments, None);
    if action != crate::tool_names::Action::Unknown {
        return action.mutation();
    }
    let name = canonical_tool_name(name);
    name == "apply_patch"
        || name == "create_directory"
        || name == "make_directory"
        || (name.contains("file")
            && [
                "create", "write", "edit", "insert", "replace", "patch", "save", "delete", "remove",
            ]
            .iter()
            .any(|verb| name.contains(verb)))
}

fn is_todo_tool(name: &str) -> bool {
    matches!(
        canonical_tool_name(name).as_str(),
        "manage_todo_list" | "update_todo_list" | "write_todos"
    )
}

fn parsed_call_arguments(call: &serde_json::Value) -> serde_json::Value {
    let arguments = &call["function"]["arguments"];
    arguments
        .as_str()
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_else(|| arguments.clone())
}

#[cfg(test)]
fn is_verification_call(
    name: &str,
    arguments: &serde_json::Value,
    verification_commands: &[String],
) -> bool {
    is_verification_call_for_profile(name, arguments, verification_commands, None)
}

fn is_verification_call_for_profile(
    name: &str,
    arguments: &serde_json::Value,
    verification_commands: &[String],
    profile: Option<&str>,
) -> bool {
    let action = crate::tool_names::classify(name, arguments, profile);
    if matches!(
        action,
        crate::tool_names::Action::Orchestrate | crate::tool_names::Action::Output
    ) {
        return false;
    }
    let name = canonical_tool_name(name);
    if ["test", "check", "verify", "lint", "build"]
        .iter()
        .any(|signal| name.contains(signal))
    {
        return true;
    }
    if action != crate::tool_names::Action::Execute {
        return false;
    }
    let command = arguments.to_string().to_ascii_lowercase();
    verification_commands
        .iter()
        .any(|signal| command.contains(signal))
}

/// Successful result ids that demonstrate concrete progress. Mutations, todo
/// updates and verification are explicit signals. A read/search result counts
/// only the first time its source/content fingerprint appears in this window.
fn substantive_progress_result_ids(
    events: &[SessionEvent],
    verification_commands: &[String],
) -> std::collections::HashSet<String> {
    use std::collections::HashSet;

    let successful: HashMap<&str, &serde_json::Value> =
        events.iter().filter_map(successful_tool_result).collect();
    let profiles: HashMap<&str, &str> = events
        .iter()
        .filter(|event| event.event_type == SessionEventType::ToolCall)
        .filter_map(|event| {
            Some((
                event.payload["tool_call"]["id"].as_str()?,
                event.payload["delegation_profile"].as_str()?,
            ))
        })
        .collect();
    let mut progress = HashSet::new();
    let mut seen_calls = HashSet::new();
    let mut seen_evidence = HashSet::new();
    let mut seen_mutations = HashSet::new();
    let mut seen_todo_states = HashSet::new();

    let mut inspect_call = |call: &serde_json::Value, profile: Option<&str>| {
        let Some(id) = call.get("id").and_then(serde_json::Value::as_str) else {
            return;
        };
        if !seen_calls.insert(id.to_string()) {
            return;
        }
        let profile = profiles.get(id).copied().or(profile);
        let Some(_result) = successful.get(id) else {
            return;
        };
        let name = call
            .pointer("/function/name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let arguments = parsed_call_arguments(call);
        let mutation_progress = is_mutation_tool(name, &arguments)
            && seen_mutations.insert(format!("{}\0{}", canonical_tool_name(name), arguments));
        let todo_progress = is_todo_tool(name) && seen_todo_states.insert(arguments.to_string());
        let made_progress = mutation_progress
            || todo_progress
            || is_verification_call_for_profile(name, &arguments, verification_commands, profile)
            || (crate::tool_names::classify(name, &arguments, profile).exploration() && {
                let source = exploration_call_source(name, &arguments).unwrap_or_default();
                // Successful exploration of a new source is progress. Keying
                // by tool/source avoids re-hashing large result bodies here;
                // the dedicated repetition detector below still performs the
                // content-sensitive comparison.
                seen_evidence.insert(format!(
                    "{}\0{source}",
                    crate::tool_names::effective_name(name, &arguments, profile)
                ))
            });
        if made_progress {
            progress.insert(id.to_string());
        }
    };

    for event in events {
        if event.event_type == SessionEventType::LlmCall {
            for call in event
                .payload
                .pointer("/response_message/tool_calls")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
            {
                inspect_call(call, event.payload["delegation_profile"].as_str());
            }
        } else if event.event_type == SessionEventType::ToolCall {
            inspect_call(
                &event.payload["tool_call"],
                event.payload["delegation_profile"].as_str(),
            );
        }
    }
    progress
}

fn soft_group_intervention_seen(
    events: &[SessionEvent],
    kind: &str,
    group_id: &str,
    after_seq: i64,
) -> bool {
    let hash = StepHash::new(kind, group_id);
    events.iter().any(|event| {
        event.event_type == SessionEventType::LoopIntervention
            && event.seq > after_seq
            && event.payload["repeated_hash"].as_str() == Some(hash.0.as_str())
            && !event.payload["hard_stop"].as_bool().unwrap_or(false)
            && event.payload["injected_message"]
                .as_str()
                .is_some_and(|message| !message.is_empty())
    })
}

fn call_marks_all_todos_complete(call: &serde_json::Value) -> bool {
    let name = call
        .pointer("/function/name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .rsplit("__")
        .next()
        .unwrap_or_default()
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(
        name.as_str(),
        "manage_todo_list" | "update_todo_list" | "write_todos"
    ) {
        return false;
    }
    let Some(arguments) = call
        .pointer("/function/arguments")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
    else {
        return false;
    };
    let Some(todos) = arguments
        .get("todoList")
        .or_else(|| arguments.get("todos"))
        .and_then(serde_json::Value::as_array)
        .filter(|todos| !todos.is_empty())
    else {
        return false;
    };
    todos.iter().all(|todo| {
        todo.get("status")
            .or_else(|| todo.get("state"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| {
                matches!(
                    status.to_ascii_lowercase().as_str(),
                    "completed" | "complete" | "done"
                )
            })
    })
}

/// Detect phase-level stalls inside one request group. Exact-result matching
/// alone cannot catch an agent that rereads a different file on every turn.
fn request_group_stall(
    events: &[SessionEvent],
    config: &CircuitBreakerConfig,
) -> Option<RequestGroupStall> {
    let group_id = events
        .iter()
        .rev()
        .filter(|event| event.event_type == SessionEventType::LlmCall)
        .find_map(|event| event.payload.get("request_group_id")?.as_str())?
        .to_string();
    let successful_results: std::collections::HashSet<&str> = events
        .iter()
        .filter_map(successful_tool_result)
        .map(|(id, _)| id)
        .collect();
    let progress_results = substantive_progress_result_ids(events, &config.verification_commands);
    let calls: Vec<&SessionEvent> = events
        .iter()
        .filter(|event| {
            event.event_type == SessionEventType::LlmCall
                && event
                    .payload
                    .get("request_group_id")
                    .and_then(serde_json::Value::as_str)
                    == Some(group_id.as_str())
        })
        .collect();
    let total = u32::try_from(calls.len()).unwrap_or(u32::MAX);

    let mut completion_checkpoint = None;
    let mut last_progress = None;
    for (index, event) in calls.iter().enumerate() {
        let tool_calls = event
            .payload
            .pointer("/response_message/tool_calls")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for call in tool_calls {
            let id = call.get("id").and_then(serde_json::Value::as_str);
            if !id.is_some_and(|id| successful_results.contains(id)) {
                continue;
            }
            if id.is_some_and(|id| progress_results.contains(id)) {
                last_progress = Some((index, event.seq));
            }
            if id.is_some_and(|id| progress_results.contains(id))
                && call_marks_all_todos_complete(call)
            {
                completion_checkpoint = Some((index, event.seq));
            }
        }
    }

    if let Some(checkpoint) = completion_checkpoint {
        const KIND: &str = "request_group_post_completion";
        let anchor = last_progress
            .filter(|(index, _)| *index > checkpoint.0)
            .unwrap_or(checkpoint);
        let count = u32::try_from(calls.len().saturating_sub(anchor.0 + 1)).unwrap_or(u32::MAX);
        let hard_stop = count >= config.post_completion_hard_stop_threshold;
        if hard_stop
            || (count >= config.post_completion_threshold
                && !soft_group_intervention_seen(events, KIND, &group_id, anchor.1))
        {
            return Some(RequestGroupStall {
                group_id,
                kind: KIND,
                count,
                threshold: config.post_completion_threshold,
                hard_stop,
                reason: format!(
                    "The task had an all-complete todo checkpoint and then continued for {count} model turns without later substantive progress."
                ),
            });
        }
    }

    let stalled = last_progress
        .map(|(index, _)| calls.len().saturating_sub(index + 1))
        .unwrap_or(calls.len());
    let stalled = u32::try_from(stalled).unwrap_or(u32::MAX);
    let progress_seq = last_progress.map(|(_, seq)| seq).unwrap_or(0);
    const STALL_KIND: &str = "request_group_no_progress";
    let hard_stop = stalled >= config.request_group_hard_stop_threshold;
    if hard_stop
        || (stalled >= config.request_group_step_threshold
            && !soft_group_intervention_seen(events, STALL_KIND, &group_id, progress_seq))
    {
        return Some(RequestGroupStall {
            group_id,
            kind: STALL_KIND,
            count: stalled,
            threshold: config.request_group_step_threshold,
            hard_stop,
            reason: format!(
                "This user request has continued for {stalled} model turns without a successful mutation, verification, todo update, or new lookup result."
            ),
        });
    }

    (total >= config.request_group_absolute_hard_stop_threshold).then(|| RequestGroupStall {
        group_id,
        kind: "request_group_absolute_limit",
        count: total,
        threshold: config.request_group_absolute_hard_stop_threshold,
        hard_stop: true,
        reason: format!(
            "This single user request reached the emergency ceiling of {total} model turns."
        ),
    })
}

/// How a lookup repeated. Determines what the intervention message says, so a
/// user reading it can tell an agent re-reading overlapping slices of one file
/// from one firing the identical search over and over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RepetitionKind {
    /// The lookup found nothing, again.
    EmptyResult,
    /// A byte-identical result came back.
    IdenticalResult,
    /// A different range whose every line had already been read from this file.
    OverlappingRead,
}

impl RepetitionKind {
    /// Phrase describing `count` repetitions of this kind.
    fn evidence(self, count: u32) -> String {
        match self {
            Self::EmptyResult => format!("{count} empty results"),
            Self::IdenticalResult => format!("{count} identical results"),
            Self::OverlappingRead => format!("{count} overlapping reads"),
        }
    }
}

/// Repetition found among read/search results, carrying the evidence class the
/// caller needs to pick a threshold and to explain itself to the user.
#[derive(Clone, Debug)]
struct ExplorationRepetition {
    hash: StepHash,
    count: u32,
    /// Canonical name of the tool that produced the repeated result.
    tool: String,
    /// What the call was looking at - a file path, or a search query.
    source: Option<String>,
    /// Empty / "no matches" results. Weak evidence: a miss rules a hypothesis
    /// out, so these are held to a proportionally higher threshold.
    negative: bool,
    kind: RepetitionKind,
}

impl ExplorationRepetition {
    /// Name the actual evidence. "the same information N times" tells a user
    /// nothing they can act on; the tool, the path or query, and the kind of
    /// repetition tell them exactly what to narrow.
    fn describe(&self) -> String {
        let target = match self.source.as_deref() {
            Some(source) => format!("`{}` {source}", self.tool),
            None => format!("`{}`", self.tool),
        };
        let evidence = self.kind.evidence(self.count);
        match self.kind {
            RepetitionKind::EmptyResult => format!("{target} keeps finding nothing ({evidence})."),
            _ => format!("{target} returned content already in context ({evidence})."),
        }
    }
}

/// Halve accumulated redundancy rather than erasing it.
///
/// Progress does not prove the earlier repeats never happened, so a genuine
/// loop that resumes after a detour still reaches the threshold - but a
/// productive detour no longer carries the whole prior streak into a stop.
/// Without this the counters only ever ratchet upward, and a long exploration
/// phase on a large codebase reaches the hard stop by arithmetic alone.
fn decay_exploration_evidence(counts: &mut HashMap<String, u32>, redundant_streak: &mut u32) {
    *redundant_streak /= 2;
    counts.retain(|_, count| {
        *count /= 2;
        *count > 0
    });
}

/// Whether this LLM call begins a fresh user turn rather than continuing an
/// agentic tool loop. A new instruction is new work; it should not inherit the
/// redundancy accumulated while answering the previous one.
fn starts_new_user_turn(event: &SessionEvent) -> bool {
    // `request_messages` is deliberately compacted for durable audit and often
    // contains only the original user request even when the actual provider
    // request ends in a tool result. Prefer the separately persisted full role
    // shape or every continuation is misclassified as a fresh user turn and
    // repetition evidence is decayed on every step.
    if let Some(role) = event
        .payload
        .get("message_roles")
        .and_then(|value| value.as_array())
        .and_then(|roles| roles.last())
        .and_then(|role| role.as_str())
    {
        return role.eq_ignore_ascii_case("user");
    }
    event
        .payload
        .get("request_messages")
        .and_then(|value| value.as_array())
        .and_then(|messages| messages.last())
        .and_then(|message| message["role"].as_str())
        .is_some_and(|role| role.eq_ignore_ascii_case("user"))
}

/// Whether the model wrote a substantive answer rather than firing another
/// blind lookup. Handles both plain-text and multipart content.
fn wrote_substantive_text(event: &SessionEvent) -> bool {
    let Some(content) = event.payload.pointer("/response_message/content") else {
        return false;
    };
    let length = match content {
        serde_json::Value::String(text) => text.trim().len(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .map(|text| text.trim().len())
            .sum(),
        _ => 0,
    };
    length >= SUBSTANTIVE_RESPONSE_CHARS
}

/// Whether a tool result reports that the lookup found nothing.
fn is_negative_result(content: &serde_json::Value) -> bool {
    let Some(text) = content.as_str() else {
        // Structured results are treated as real findings; only text results
        // carry the "nothing found" phrasing this classifies.
        return false;
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.len() > MAX_NEGATIVE_RESULT_CHARS {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    // Markers must anchor the result: a substantive answer that merely
    // mentions "not found" (e.g. "404 Not Found: key x, using default") is
    // progress, not a failed lookup.
    let non_empty_lines: Vec<&str> = lowered.lines().filter(|l| !l.trim().is_empty()).collect();
    let first_line = non_empty_lines.first().copied().unwrap_or("");
    non_empty_lines.len() <= 2
        && NEGATIVE_RESULT_MARKERS.iter().any(|marker| {
            first_line.starts_with(marker)
                || (first_line.contains(marker) && first_line.len() <= marker.len() + 16)
        })
}

/// Canonical tool events include tools executed by external clients such as
/// Copilot. Compact LLM audit messages deliberately omit tool history.
fn repeated_exploration_results(events: &[SessionEvent]) -> Option<ExplorationRepetition> {
    use std::collections::HashSet;
    let mut counts = HashMap::<String, u32>::new();
    let mut seen = HashSet::<String>::new();
    let mut group = None;
    let mut latest = None;
    let mut sources = HashMap::<String, HashSet<u64>>::new();
    let mut call_sources = HashMap::<String, String>::new();
    let mut call_names = HashMap::<String, String>::new();
    let mut redundant_streak = 0u32;
    let mut clean_streak = 0u32;
    for event in events {
        if let Some(next) = event
            .payload
            .get("request_group_id")
            .and_then(|v| v.as_str())
        {
            if group.is_some_and(|previous| previous != next) {
                counts.clear();
                seen.clear();
                latest = None;
                sources.clear();
                call_sources.clear();
                call_names.clear();
                redundant_streak = 0;
                clean_streak = 0;
            }
            group = Some(next);
        }
        if event.event_type == SessionEventType::LlmCall {
            if let Some(calls) = event
                .payload
                .pointer("/response_message/tool_calls")
                .and_then(|v| v.as_array())
            {
                for call in calls {
                    remember_exploration_source(
                        call,
                        &mut call_sources,
                        &mut call_names,
                        event.payload["delegation_profile"].as_str(),
                    );
                }
                if calls.iter().any(|call| {
                    !crate::tool_names::classify(
                        call["function"]["name"].as_str().unwrap_or(""),
                        &parsed_call_arguments(call),
                        event.payload["delegation_profile"].as_str(),
                    )
                    .exploration()
                }) {
                    counts.clear();
                    latest = None;
                    sources.clear();
                    redundant_streak = 0;
                    clean_streak = 0;
                }
            }
            // Decay on progress the agent actually made, so redundancy earned
            // during one line of enquiry does not stop a later, different one.
            if starts_new_user_turn(event) || wrote_substantive_text(event) {
                decay_exploration_evidence(&mut counts, &mut redundant_streak);
                clean_streak = 0;
            }
        }
        if event.event_type == SessionEventType::ToolCall {
            remember_exploration_source(
                &event.payload["tool_call"],
                &mut call_sources,
                &mut call_names,
                event.payload["delegation_profile"].as_str(),
            );
        }
        if event.event_type != SessionEventType::ToolResult {
            continue;
        }
        let result = &event.payload["result"];
        let Some(id) = result["tool_call_id"].as_str() else {
            continue;
        };
        if !seen.insert(id.to_string()) {
            continue;
        }
        let Some(name) = result["name"].as_str() else {
            continue;
        };
        let name = call_names.get(id).map(String::as_str).unwrap_or(name);
        if !is_exploration_tool(name) {
            counts.clear();
            latest = None;
            sources.clear();
            redundant_streak = 0;
            clean_streak = 0;
            continue;
        }
        // Preserve result content: fuzzy matching could mistake new findings
        // for repetition. Query/argument changes reach the fingerprint only
        // through `source`, which is the file path for reads and the query for
        // searches - so a *different* search is different evidence.
        let content = &result["content"];
        if content.is_null() || content.as_str() == Some("") {
            continue;
        }
        let negative = is_negative_result(content);
        let source = call_sources.get(id).map(String::as_str).unwrap_or("");
        let hash = exploration_result_hash(
            exploration_tool_name(name),
            source,
            content,
            &result["is_error"],
        );
        let count = {
            let entry = counts.entry(hash.0.clone()).or_default();
            *entry += 1;
            *entry
        };
        // A changed line range is not progress when all meaningful three-line
        // sequences have already been read from this same file. Require exact
        // sequences, not semantic similarity, so even a one-line code change
        // or a different file is new evidence. Bound memory and scan work.
        let covered = call_sources
            .get(id)
            .and_then(|source| {
                let text = content.as_str()?;
                if text.len() > 256 * 1024 {
                    return None;
                }
                let lines: Vec<_> = text
                    .lines()
                    .map(exploration_line)
                    .filter(|line| !line.trim().is_empty())
                    .collect();
                if lines.len() < 3 || lines.len() > 16_384 {
                    return None;
                }
                let evidence = sources.entry(source.clone()).or_default();
                let mut new = false;
                for window in lines.windows(3) {
                    let fingerprint = line_window_fingerprint(window);
                    if !evidence.contains(&fingerprint) {
                        new = true;
                        if evidence.len() < 16_384 {
                            evidence.insert(fingerprint);
                        }
                    }
                }
                Some(!new)
            })
            .unwrap_or(false);
        if covered || count > 1 {
            redundant_streak += 1;
            clean_streak = 0;
        } else {
            redundant_streak = 0;
            clean_streak += 1;
            // A run of genuinely new findings is the clearest progress signal
            // there is: the agent is exploring, not spinning.
            if clean_streak >= CLEAN_RESULTS_BEFORE_DECAY {
                clean_streak = 0;
                decay_exploration_evidence(&mut counts, &mut redundant_streak);
            }
        }
        // Classified for the message, not the threshold: an empty result is
        // already its own evidence class, so it is reported as one first.
        let kind = if negative {
            RepetitionKind::EmptyResult
        } else if count > 1 {
            RepetitionKind::IdenticalResult
        } else {
            RepetitionKind::OverlappingRead
        };
        latest = Some((
            hash,
            exploration_tool_name(name).to_string(),
            Some(source.to_string()).filter(|source| !source.is_empty()),
            negative,
            kind,
        ));
    }
    // Score at the end rather than per-result, so a decay earned *after* the
    // last lookup (a written answer, a fresh instruction) still lowers what is
    // reported. Scoring inline would let the pre-decay count reach the caller.
    let (hash, tool, source, negative, kind) = latest?;
    let count = counts
        .get(&hash.0)
        .copied()
        .unwrap_or_default()
        .max(redundant_streak.saturating_add(1));
    Some(ExplorationRepetition {
        hash,
        count,
        tool,
        source,
        negative,
        kind,
    })
}

/// Fingerprint a tool result without materializing a copy of its content.
///
/// The obvious form - wrapping the content in a `json!` object and calling
/// `to_string()` - clones the content into a fresh `Value` and then serializes
/// it, for every result in the window on every completion. On a heavy
/// exploration window that is megabytes of copying to hash bytes we already
/// hold. Hashing the parts in order is equivalent and allocation-free for the
/// overwhelmingly common case of text content.
fn exploration_result_hash(
    tool: &str,
    source: &str,
    content: &serde_json::Value,
    is_error: &serde_json::Value,
) -> StepHash {
    let mut hasher = Sha256::new();
    hasher.update(tool.as_bytes());
    hasher.update(b"\0");
    hasher.update(source.as_bytes());
    hasher.update(b"\0");
    match content.as_str() {
        Some(text) => hasher.update(text.as_bytes()),
        // Structured results are rare; pay the serialization only for them.
        None => hasher.update(content.to_string().as_bytes()),
    }
    hasher.update(b"\0");
    hasher.update(if is_error.as_bool().unwrap_or(false) {
        b"1" as &[u8]
    } else {
        b"0"
    });
    StepHash(hex::encode(hasher.finalize()))
}

/// Cheap fingerprint for one three-line window.
///
/// This runs for every line of every read/search result in the window, on every
/// completion - one 5,000-line file is 5,000 of them. A SHA-256 plus a
/// 64-character hex `String` per line was most of the detector's cost and most
/// of its memory. These fingerprints are only ever compared against other
/// windows from the same session, so 64 bits and no allocation is the right
/// trade: a collision would at worst mark one window among thousands as
/// already-seen, which cannot by itself flip a read to "covered".
fn line_window_fingerprint(window: &[&str]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for line in window {
        line.hash(&mut hasher);
    }
    hasher.finish()
}

/// Server-owned read-only classification, used to grant tool memoization when
/// the caller says nothing. Deliberately an allow-list of canonical names: see
/// [`exploration_tool_name`] for why arbitrary `read_*` names are not trusted.
pub fn is_exploration_tool(name: &str) -> bool {
    crate::tool_names::classify(name, &serde_json::Value::Null, None).exploration()
}

/// Known read-only names may be qualified by an MCP server or client namespace.
/// Never classify arbitrary `read_*` tools as safe: names can hide mutations.
fn exploration_tool_name(name: &str) -> &str {
    let name = crate::tool_names::canonical_name(name);
    match name {
        "readFile" => "read_file",
        "fileSearch" => "file_search",
        "grepSearch" => "grep_search",
        "semanticSearch" => "semantic_search",
        "listDirectory" => "list_directory",
        _ => name,
    }
}

fn remember_exploration_source(
    call: &serde_json::Value,
    sources: &mut HashMap<String, String>,
    names: &mut HashMap<String, String>,
    profile: Option<&str>,
) {
    let Some(id) = call["id"].as_str() else {
        return;
    };
    let Some(name) = call["function"]["name"].as_str() else {
        return;
    };
    let arguments = &call["function"]["arguments"];
    let parsed = arguments
        .as_str()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
    let arguments = parsed.as_ref().unwrap_or(arguments);
    if !crate::tool_names::classify(name, arguments, profile).exploration() {
        names.insert(id.to_string(), "non_exploration".into());
        return;
    }
    let name = crate::tool_names::effective_name(name, arguments, profile);
    names.insert(id.to_string(), name.to_string());
    if let Some(source) = exploration_call_source(name, arguments) {
        sources.insert(id.to_string(), source);
    }
}

/// Identify what a read/search call is looking at.
///
/// File reads are keyed by path alone, deliberately ignoring the line range, so
/// that different slices of one file share an evidence set and an already-read
/// slice can be recognised. Searches are keyed by query *and* scope, so that
/// changing the query makes the result new evidence - three different searches
/// that each miss are three hypotheses ruled out, not one loop.
fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One canonical spelling per file: separators, `./` prefixes, and stray
/// whitespace must not split one file's repeat count into several buckets.
/// Windows-style paths (backslashes, drive letters) are case-folded because
/// the filesystem is; POSIX paths keep their case.
fn canonical_exploration_path(path: &str) -> String {
    let windows_style = path.contains('\\')
        || path.chars().nth(1).is_some_and(|c| c == ':')
            && path.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    let mut canonical = collapse_whitespace(&path.trim().replace('\\', "/"));
    while let Some(rest) = canonical.strip_prefix("./") {
        canonical = rest.to_string();
    }
    if windows_style {
        canonical = canonical.to_lowercase();
    }
    canonical
}

fn exploration_call_source(name: &str, arguments: &serde_json::Value) -> Option<String> {
    const PATH_KEYS: [&str; 8] = [
        "filePath",
        "file_path",
        "path",
        "filename",
        "uri",
        "notebook_path",
        "filepath",
        "dir_path",
    ];
    const QUERY_KEYS: [&str; 6] = ["query", "pattern", "regex", "search", "q", "glob"];
    let text = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| arguments[*key].as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let action = crate::tool_names::classify(name, arguments, None);
    let source = if action == crate::tool_names::Action::Read {
        if let Some(path) = text(&PATH_KEYS) {
            return Some(canonical_exploration_path(path));
        }
        // Some clients wrap file paths in a batch object. Keep the full scope
        // distinct rather than merging every unsupported shape into "unknown".
        serde_json::to_string(arguments).ok()?
    } else if action == crate::tool_names::Action::ReadMany {
        // Bulk scopes/globs are not exact file paths. Preserve their filters.
        serde_json::to_string(arguments).ok()?
    } else {
        // Kept human-readable: this string is also what the intervention
        // message shows the user as the thing being repeated.
        match (text(&QUERY_KEYS), text(&PATH_KEYS)) {
            (None, None) => serde_json::to_string(arguments).ok()?,
            (Some(query), Some(scope)) => format!(
                "{} in {}",
                collapse_whitespace(query),
                canonical_exploration_path(scope)
            ),
            (Some(query), None) => collapse_whitespace(query),
            (None, Some(scope)) => canonical_exploration_path(scope),
        }
    };
    let filters: serde_json::Map<String, serde_json::Value> = arguments
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| {
            !PATH_KEYS.contains(&key.as_str()) && !QUERY_KEYS.contains(&key.as_str())
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let source = if filters.is_empty() || action == crate::tool_names::Action::ReadMany {
        source
    } else {
        format!("{source} {}", serde_json::Value::Object(filters))
    };
    // Query backslashes are meaningful (especially in regex), not separators.
    Some(source)
}

/// What a transcript dedup pass changed, for logging and response metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolResultDedup {
    /// Tool results whose content was replaced by a pointer to the earlier copy.
    pub replaced: usize,
    /// Replacements belonging to the newest tool-result batch. Historical
    /// duplicates are still removed from the outbound prompt, but must not be
    /// reported again as a new health finding on every continuation.
    pub new_replacements: usize,
    /// Characters of duplicated tool output removed from the outgoing request.
    pub saved_chars: usize,
}

impl ToolResultDedup {
    pub fn is_empty(&self) -> bool {
        self.replaced == 0
    }
}

/// Advisory finding recording repetition that dedup absorbed.
///
/// Without this the most successful case leaves the least evidence: the loop
/// never reaches the detector, so nothing is written and the session looks
/// identical to one that never repeated anything.
pub fn redundant_lookup_finding(
    replaced: usize,
) -> Option<crate::agent_health::AgentHealthFinding> {
    use crate::agent_health::{AgentHealthFinding, AgentHealthKind, AgentHealthSeverity};
    if replaced == 0 {
        return None;
    }
    Some(AgentHealthFinding {
        kind: AgentHealthKind::RedundantLookup,
        severity: AgentHealthSeverity::Warning,
        message: format!(
            "replaced {replaced} repeated read/search result(s) with a pointer to the copy already in context"
        ),
        tool_name: None,
    })
}

/// Replace read/search results the transcript already contains with a short
/// pointer to the earlier copy.
///
/// This is the corrective half of loop handling, and the half that does the
/// real work. A model that re-reads a file is told, in the exact place it is
/// looking, that the content is already above - which both saves the tokens the
/// hard stop existed to protect and hands the model something it can act on.
/// Refusing the turn does neither.
///
/// Only read-only exploration tools are touched. A mutation's result is always
/// delivered verbatim, so a failed edit can never be hidden behind a pointer.
pub fn dedupe_repeated_tool_results(request: &mut ChatCompletionRequest) -> ToolResultDedup {
    // Tool results name only their call id, so the tool and its arguments have
    // to be recovered from the assistant messages that requested them.
    let mut calls = HashMap::<String, (String, Option<String>)>::new();
    let mut seen_ids = std::collections::HashSet::new();
    let mut ambiguous_ids = std::collections::HashSet::new();
    let mut result_ids = std::collections::HashSet::new();
    let profile = crate::tool_names::request_profile(request);
    for message in &request.messages {
        if message.role.eq_ignore_ascii_case("tool") {
            if let Some(id) = &message.tool_call_id {
                if !result_ids.insert(id.clone()) {
                    ambiguous_ids.insert(id.clone());
                }
            }
        }
        if !message.role.eq_ignore_ascii_case("assistant") {
            continue;
        }
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            // A reused id can link an earlier read to a later mutation. Never
            // rewrite any result whose execution identity is ambiguous.
            if !seen_ids.insert(call.id.clone()) {
                ambiguous_ids.insert(call.id.clone());
            }
            // Composite editors require their arguments before deciding
            // whether this particular call is read-only.
            if !is_exploration_tool(&call.function.name)
                && !crate::tool_names::is_composite_editor(&call.function.name)
            {
                continue;
            }
            let arguments = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or(serde_json::Value::Null);
            if !crate::tool_names::classify(&call.function.name, &arguments, profile).exploration()
            {
                continue;
            }
            let name = crate::tool_names::effective_name(&call.function.name, &arguments, profile);
            calls.insert(
                call.id.clone(),
                (name.to_string(), exploration_call_source(name, &arguments)),
            );
        }
    }

    let newest_result_start = request
        .messages
        .iter()
        .rposition(|message| {
            message.role.eq_ignore_ascii_case("assistant")
                && message
                    .tool_calls
                    .as_deref()
                    .is_some_and(|calls| !calls.is_empty())
        })
        .map(|index| index.saturating_add(1))
        .unwrap_or(request.messages.len());
    let mut first_seen = HashMap::<String, (usize, String)>::new();
    let mut step = 0usize;
    let mut stats = ToolResultDedup::default();
    for (message_index, message) in request.messages.iter_mut().enumerate() {
        if !message.role.eq_ignore_ascii_case("tool") {
            continue;
        }
        step += 1;
        let Some(id) = message.tool_call_id.clone() else {
            continue;
        };
        if ambiguous_ids.contains(&id) {
            continue;
        }
        let Some((name, source)) = calls.get(&id).cloned() else {
            continue;
        };
        // Belt and braces: `calls` holds only exploration tools, but never
        // replacing a mutation's result is a safety property worth a second
        // check rather than a distant invariant.
        if !is_exploration_tool(&name) {
            continue;
        }
        // Only plain-text results are compared; a multipart result may carry
        // image bytes this pass does not inspect, so it is left verbatim.
        let Some((fingerprint, length)) = message.content_str().and_then(|content| {
            (content.len() >= MIN_DEDUPE_CHARS).then(|| {
                (
                    StepHash::new(
                        exploration_tool_name(&name),
                        &format!("{}\0{content}", source.as_deref().unwrap_or_default()),
                    )
                    .0,
                    content.len(),
                )
            })
        }) else {
            continue;
        };
        let Some((earlier, label)) = first_seen.get(&fingerprint) else {
            first_seen.insert(
                fingerprint,
                (step, describe_exploration_call(&name, source)),
            );
            continue;
        };
        let pointer = format!(
            "[reincore] Identical to the result of {label} at step {earlier}, already in the context above. Content omitted to save tokens - reuse that earlier result instead of repeating this lookup."
        );
        stats.replaced += 1;
        if message_index >= newest_result_start {
            stats.new_replacements += 1;
        }
        stats.saved_chars += length.saturating_sub(pointer.len());
        message.content = Some(MessageContent::text(pointer));
    }
    stats
}

/// Render a call as the pointer text refers to it: the canonical tool name plus
/// the file path or query, so the model can find the earlier copy.
fn describe_exploration_call(name: &str, source: Option<String>) -> String {
    match source {
        Some(source) => format!("`{}` ({source})", exploration_tool_name(name)),
        None => format!("`{}`", exploration_tool_name(name)),
    }
}

fn exploration_line(line: &str) -> &str {
    // Recognize editor line labels while preserving code whitespace/content.
    let Some((prefix, rest)) = line.split_once(':') else {
        return line;
    };
    let digits = prefix.trim().strip_prefix('L').unwrap_or(prefix.trim());
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        rest.strip_prefix(' ').unwrap_or(rest)
    } else {
        line
    }
}
/// Strip per-call identifiers so an identical step hashes identically.
///
/// Providers mint a fresh `tool_calls[].id` for every invocation and clients
/// echo it back as `tool_call_id`. Hashing the raw message therefore produced a
/// unique hash on every turn, which silently blinded loop detection for exactly
/// the tool-calling agents it exists to protect.
fn normalize_step_message(value: &serde_json::Value) -> serde_json::Value {
    let mut value = value.clone();
    match &mut value {
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                *item = normalize_step_message(item);
            }
        }
        serde_json::Value::Object(fields) => {
            let is_user = fields
                .get("role")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|role| role.eq_ignore_ascii_case("user"));
            if is_user {
                if let Some(content) = fields
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .and_then(crate::user_content::session_user_request)
                    .map(str::to_string)
                {
                    fields.insert("content".to_string(), serde_json::Value::String(content));
                }
            }
            fields.remove("tool_call_id");
            // New thinking/signatures are not evidence of task progress.
            fields.remove("provider_state");
            if let Some(serde_json::Value::Array(calls)) = fields.get_mut("tool_calls") {
                for call in calls.iter_mut() {
                    if let Some(call) = call.as_object_mut() {
                        call.remove("id");
                        // Argument key order and spacing are client
                        // serialization details, not step differences:
                        // canonicalize the JSON string before hashing.
                        if let Some(arguments) = call
                            .get_mut("function")
                            .and_then(|function| function.get_mut("arguments"))
                        {
                            if let Some(text) = arguments.as_str() {
                                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text)
                                {
                                    if let Ok(canonical) = crate::cache::canonicalize_json(&parsed)
                                    {
                                        *arguments = serde_json::Value::String(canonical);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    value
}

fn step_hash_from_event(event: &SessionEvent) -> Option<StepHash> {
    let request_messages = event.payload.get("request_messages")?;
    // Hash only the newest turn input (the last request message). Agentic
    // clients append the transcript every turn, so hashing the full history
    // would make identical stuck steps hash differently and blind the
    // detector. The final message (latest user turn or tool result) plus the
    // response captures the actual (input, output) step.
    let last_message = request_messages
        .as_array()
        .and_then(|messages| messages.last())
        .unwrap_or(request_messages);
    let input = serde_json::to_string(&normalize_step_message(last_message)).ok()?;
    let output = serde_json::to_string(&normalize_step_message(
        event.payload.get("response_message")?,
    ))
    .ok()?;
    Some(StepHash::new(&input, &output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;

    #[test]
    fn step_normalization_ignores_argument_key_order_and_spacing() {
        let left = serde_json::json!({"role":"assistant","tool_calls":[{"id":"1","function":{"name":"read_file","arguments":"{\"a\":1,\"b\":2}"}}]});
        let right = serde_json::json!({"role":"assistant","tool_calls":[{"id":"9","function":{"name":"read_file","arguments":"{ \"b\" : 2, \"a\" : 1 }"}}]});
        assert_eq!(
            normalize_step_message(&left),
            normalize_step_message(&right)
        );
    }

    #[test]
    fn exploration_paths_canonicalize_separators_case_and_dot_prefix() {
        assert_eq!(
            canonical_exploration_path("Src\\A.rs"),
            canonical_exploration_path("src/a.rs")
        );
        assert_eq!(
            canonical_exploration_path("./src/a.rs"),
            canonical_exploration_path("src/a.rs")
        );
        assert_eq!(
            canonical_exploration_path("C:\\Repo\\Src\\A.rs"),
            "c:/repo/src/a.rs"
        );
        // POSIX paths keep case: distinct files stay distinct.
        assert_ne!(
            canonical_exploration_path("Src/A.rs"),
            canonical_exploration_path("src/a.rs")
        );
        assert_eq!(
            collapse_whitespace("auth  token\t search"),
            "auth token search"
        );
    }

    fn read_events(id: usize, path: &str, text: &str) -> Vec<SessionEvent> {
        let make = |event_type, payload| SessionEvent {
            session_id: Uuid::nil(),
            seq: id as i64 * 2,
            event_type,
            payload,
            cost_cents: None,
            created_at: chrono::Utc::now(),
        };
        vec![
            make(
                SessionEventType::ToolCall,
                serde_json::json!({
                    "request_group_id":"g",
                    "tool_call":{
                        "id":format!("{id}"),
                        "function":{
                            "name":"mcp__editor__read_file",
                            "arguments":
                                serde_json::json!({"filePath":path,"startLine":id}).to_string()
                        }
                    }
                }),
            ),
            make(
                SessionEventType::ToolResult,
                serde_json::json!({
                    "request_group_id":"g",
                    "result":{
                        "tool_call_id":format!("{id}"),
                        "name":"mcp__editor__read_file",
                        "content":text
                    }
                }),
            ),
        ]
    }

    #[tokio::test]
    async fn overlapping_ranges_nudge_then_stop_without_identical_outputs() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        let lines: Vec<_> = (1..=60)
            .map(|i| format!("L{i}: unique source line {i}"))
            .collect();
        let nudge_at = breaker.config().exploration_repetition_threshold as usize;
        let stop_at = breaker.config().exploration_hard_stop_threshold as usize;
        for round in 0..=stop_at {
            // Every changed range is a strict subset of the first read, so the
            // first read is clean and each later one scores `round + 1`.
            let text = lines[round..lines.len() - round].join("\n");
            for event in read_events(round, "src/schema.rs", &text) {
                store
                    .append_event(session, event.event_type, event.payload, None)
                    .await
                    .unwrap();
            }
            let score = round + 1;
            assert_eq!(
                breaker.inspect_and_log(&store, session).await.unwrap(),
                score >= nudge_at,
                "round {round}"
            );
            if score >= nudge_at {
                assert_eq!(
                    breaker
                        .pending_intervention(&store, session)
                        .await
                        .unwrap()
                        .unwrap()
                        .hard_stop,
                    score >= stop_at,
                    "round {round}"
                );
            }
        }
    }

    #[test]
    fn new_lines_and_mutations_reset_exploration_streak() {
        let mut events = read_events(1, "a.rs", "alpha\nbeta\ngamma\ndelta\nepsilon");
        events.extend(read_events(2, "a.rs", "beta\ngamma\ndelta"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 2);
        events.extend(read_events(3, "a.rs", "beta\nCHANGED\ndelta"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
        events.extend(read_events(4, "b.rs", "beta\ngamma\ndelta\nepsilon"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
        let mut mutation = events.last().unwrap().clone();
        mutation.payload["result"]["tool_call_id"] = serde_json::json!("edit");
        mutation.payload["result"]["name"] =
            serde_json::json!("mcp__editor__replace_string_in_file");
        events.push(mutation);
        assert!(repeated_exploration_results(&events).is_none());
        events.extend(read_events(5, "a.rs", "beta\ngamma\ndelta"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
    }

    #[test]
    fn composite_views_share_read_evidence_but_edits_are_never_deduped() {
        let mut events = read_events(1, "a.rs", "alpha\nbeta\ngamma\ndelta");
        events.extend(read_events(2, "a.rs", "beta\ngamma\ndelta"));
        for event in &mut events {
            if event.event_type == SessionEventType::ToolCall {
                event.payload["tool_call"]["function"]["name"] = serde_json::json!("file_editor");
                event.payload["tool_call"]["function"]["arguments"] =
                    serde_json::json!({"command":"view","path":"a.rs"})
                        .to_string()
                        .into();
            } else {
                event.payload["result"]["name"] = serde_json::json!("file_editor");
            }
        }
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 2);
        assert!(!is_exploration_tool("file_editor")); // Name-only memoization is unsafe.
        let content = "long observed content\n".repeat(100);
        let mut messages = Vec::new();
        for (id, command) in [
            ("r1", "view"),
            ("r2", "view"),
            ("w1", "insert"),
            ("w2", "insert"),
        ] {
            messages.push(serde_json::json!({"role":"assistant","tool_calls":[{"id":id,"type":"function","function":{"name":"file_editor","arguments":serde_json::json!({"command":command,"path":"a.rs"}).to_string()}}]}));
            messages.push(serde_json::json!({"role":"tool","tool_call_id":id,"content":content}));
        }
        let mut request =
            serde_json::from_value(serde_json::json!({"model":"test","messages":messages}))
                .unwrap();
        assert_eq!(dedupe_repeated_tool_results(&mut request).replaced, 1);
        assert_eq!(
            request.messages.last().unwrap().content_str(),
            Some(content.as_str())
        );
    }

    #[test]
    fn researched_search_scopes_and_orchestration_are_not_false_progress() {
        let a = exploration_call_source(
            "search_file_content",
            &serde_json::json!({"pattern":"auth","dir_path":"src","include_pattern":"*.rs"}),
        );
        let b = exploration_call_source(
            "search_file_content",
            &serde_json::json!({"pattern":"auth","dir_path":"src","include_pattern":"*.ts"}),
        );
        assert_ne!(a, b);
        assert!(exploration_call_source(
            "read_multiple_files",
            &serde_json::json!({"paths":["a.rs","b.rs"]})
        )
        .unwrap()
        .contains("b.rs"));
        assert!(!is_verification_call_for_profile(
            "execute",
            &serde_json::json!({"command":"cargo test"}),
            &CircuitBreakerConfig::default().verification_commands,
            Some("opencode")
        ));
        assert!(!is_verification_call(
            "Monitor",
            &serde_json::json!({"command":"cargo test"}),
            &CircuitBreakerConfig::default().verification_commands
        ));
        assert!(is_verification_call(
            "run_shell_command",
            &serde_json::json!({"command":"cargo test"}),
            &CircuitBreakerConfig::default().verification_commands
        ));
        let mut calls = read_events(1, "a.rs", "all tests passed");
        calls[0].payload["tool_call"]["function"]["name"] = serde_json::json!("execute");
        calls[0].payload["tool_call"]["function"]["arguments"] =
            serde_json::json!({"command":"cargo test"})
                .to_string()
                .into();
        calls[0].payload["delegation_profile"] = serde_json::json!("opencode");
        calls[1].payload["result"]["name"] = serde_json::json!("execute");
        calls[1].payload["result"]["is_error"] = serde_json::json!(false);
        let mut completion = calls[0].clone();
        completion.event_type = SessionEventType::LlmCall;
        completion.payload =
            serde_json::json!({"response_message":{"tool_calls":[calls[0].payload["tool_call"]]}});
        calls.insert(0, completion);
        // The execution receipt's profile must override an earlier unprofiled
        // model-call audit when both describe the same call.
        assert!(substantive_progress_result_ids(
            &calls,
            &CircuitBreakerConfig::default().verification_commands
        )
        .is_empty());
    }

    #[test]
    fn namespaced_read_names_do_not_classify_unknown_tools_as_read_only() {
        assert!(is_exploration_tool("mcp__editor__read_file"));
        assert!(is_exploration_tool("functions.grep_search"));
        assert!(!is_exploration_tool("mcp__editor__read_and_delete_file"));
    }

    #[test]
    fn agent_aliases_share_exploration_evidence_and_edits_reset_it() {
        for alias in [
            "read",
            "Read",
            "NotebookRead",
            "view",
            "search",
            "Grep",
            "Glob",
        ] {
            assert!(is_exploration_tool(alias), "{alias}");
            assert!(is_exploration_tool(&format!("mcp__client__{alias}")));
        }
        for alias in [
            "edit",
            "Edit",
            "MultiEdit",
            "Write",
            "execute",
            "shell",
            "Bash",
            "powershell",
            "read_and_delete_file",
        ] {
            assert!(!is_exploration_tool(alias), "{alias}");
        }
        assert_eq!(
            exploration_call_source(
                "NotebookRead",
                &serde_json::json!({"notebook_path":"a.ipynb"})
            ),
            Some("a.ipynb".into())
        );
        assert_eq!(
            exploration_call_source("Grep", &serde_json::json!({"pattern":"auth", "path":"src"})),
            Some("auth in src".into())
        );

        let mut events = read_events(1, "a.rs", "alpha\nbeta\ngamma\ndelta");
        events.extend(read_events(2, "a.rs", "beta\ngamma\ndelta"));
        for (event, alias) in events.iter_mut().zip(["read", "read", "view", "view"]) {
            if event.event_type == SessionEventType::ToolResult {
                event.payload["result"]["name"] = serde_json::json!(alias);
            } else {
                event.payload["tool_call"]["function"]["name"] = serde_json::json!(alias);
            }
        }
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 2);
        for alias in ["edit", "Edit", "MultiEdit", "Write", "str_replace"] {
            let mut changed = events.clone();
            let mut mutation = changed.last().unwrap().clone();
            mutation.payload["result"]["tool_call_id"] = serde_json::json!("edit");
            mutation.payload["result"]["name"] = serde_json::json!(alias);
            changed.push(mutation);
            assert!(repeated_exploration_results(&changed).is_none(), "{alias}");
        }
    }

    /// One agentic search step: the model requests `query`, the client returns
    /// `content`. The request tail is a tool result, as it is mid-loop - only a
    /// genuinely new user turn ends with a user message.
    async fn search_cycle(
        store: &crate::in_memory_stores::InMemoryEventStore,
        session: Uuid,
        turn: usize,
        query: &str,
        content: &str,
    ) {
        let id = format!("call-{turn}");
        // Match the compact audit used by both streaming and nonstreaming chat.
        store
            .append_event(
                session,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_group_id": "task-1",
                    "request_messages": [{"role":"tool", "content":"earlier result"}],
                    "response_message": {"role":"assistant", "tool_calls":[{
                        "id": id.clone(), "function":{"name":"grep_search",
                            "arguments": serde_json::json!({"query": query}).to_string()}
                    }]}
                }),
                None,
            )
            .await
            .unwrap();
        store
            .append_event(
                session,
                SessionEventType::ToolResult,
                serde_json::json!({
                    "request_group_id": "task-1",
                    "result": {"tool_call_id": id, "name":"grep_search",
                        "content": content, "is_error": false}
                }),
                None,
            )
            .await
            .unwrap();
    }

    /// Ruling out eight hypotheses is exploration, not repetition. The previous
    /// detector hashed every "no matches" alike regardless of query and refused
    /// the turn at six, which is the false stop this split exists to prevent.
    #[tokio::test]
    async fn changing_queries_that_each_miss_are_not_a_loop() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        for turn in 1..=8 {
            search_cycle(
                &store,
                session,
                turn,
                &format!("variation-{turn}"),
                "No matches found",
            )
            .await;
            assert!(
                !breaker.inspect_and_log(&store, session).await.unwrap(),
                "turn {turn} treated a new query as repetition"
            );
        }
    }

    /// The same failing lookup, on the other hand, is still a loop - just held
    /// to the negative-evidence budget rather than the ordinary one.
    #[tokio::test]
    async fn the_same_failing_search_repeated_is_still_a_loop() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        let nudge_at = (breaker.config.exploration_repetition_threshold
            * breaker.config.negative_result_threshold_multiplier) as usize;
        let stop_at = (breaker.config.exploration_hard_stop_threshold
            * breaker.config.negative_result_threshold_multiplier) as usize;
        for turn in 1..=stop_at {
            search_cycle(&store, session, turn, "sort models", "No matches found").await;
            assert_eq!(
                breaker.inspect_and_log(&store, session).await.unwrap(),
                turn >= nudge_at,
                "turn {turn}"
            );
            if turn >= nudge_at {
                let pending = breaker
                    .pending_intervention(&store, session)
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(pending.hard_stop, turn >= stop_at, "turn {turn}");
                breaker.clear_intervention(&store, session).await.unwrap();
            }
        }
    }

    /// Healthy exploration: every search hits something new. Nothing here is
    /// repetition and the detector must stay silent however long it runs.
    #[tokio::test]
    async fn distinct_findings_never_trigger_an_intervention() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        for turn in 1..=12 {
            search_cycle(
                &store,
                session,
                turn,
                &format!("symbol-{turn}"),
                &format!("src/module-{turn}.rs:{turn}: fn handler_{turn}()"),
            )
            .await;
            assert!(
                !breaker.inspect_and_log(&store, session).await.unwrap(),
                "turn {turn} flagged a distinct finding as repetition"
            );
        }
    }

    /// A written answer of real length is work the agent can build on, so the
    /// redundancy earned before it must not carry straight into a hard stop.
    #[tokio::test]
    async fn substantive_progress_decays_accumulated_redundancy() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        // Five identical reads: enough to be redundant, short of the stop.
        for turn in 1..=5 {
            for event in read_events(turn, "src/schema.rs", "alpha\nbeta\ngamma") {
                store
                    .append_event(session, event.event_type, event.payload, None)
                    .await
                    .unwrap();
            }
        }
        let before = repeated_exploration_results(
            &store
                .read_events(session, Some(1), Some(128))
                .await
                .unwrap(),
        )
        .unwrap()
        .count;

        store
            .append_event(
                session,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_group_id": "g",
                    "request_messages": [{"role":"tool", "content":"earlier result"}],
                    "response_message": {"role":"assistant", "content": "x".repeat(
                        SUBSTANTIVE_RESPONSE_CHARS + 1)}
                }),
                None,
            )
            .await
            .unwrap();
        let after = repeated_exploration_results(
            &store
                .read_events(session, Some(1), Some(128))
                .await
                .unwrap(),
        )
        .map(|found| found.count)
        .unwrap_or(0);

        assert!(before >= 5, "expected accumulated redundancy, got {before}");
        assert!(
            after < before,
            "redundancy {before} survived substantive progress as {after}"
        );
    }

    fn tool_message(id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "tool".to_string(),
            content: Some(MessageContent::text(content)),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            name: None,
        }
    }

    fn assistant_call(id: &str, name: &str, arguments: serde_json::Value) -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![crate::models::ToolCall {
                id: id.to_string(),
                call_type: "function".to_string(),
                function: crate::models::FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: Default::default(),
        }
    }

    /// The corrective half: a re-read is answered with a pointer to the copy
    /// already in context, which both saves the tokens and tells the model
    /// something it can act on. The first copy is always left verbatim.
    #[test]
    fn repeated_reads_are_replaced_by_a_pointer_to_the_first_copy() {
        let file = "fn handler() {}\n".repeat(MIN_DEDUPE_CHARS / 8);
        let mut req = request(vec![
            assistant_call("a", "read_file", serde_json::json!({"filePath":"src/a.rs"})),
            tool_message("a", &file),
            assistant_call(
                "b",
                "read_file",
                serde_json::json!({"filePath":"src/a.rs","startLine":40}),
            ),
            tool_message("b", &file),
        ]);
        let stats = dedupe_repeated_tool_results(&mut req);

        assert_eq!(stats.replaced, 1);
        assert_eq!(stats.new_replacements, 1);
        assert!(stats.saved_chars > 0);
        assert_eq!(req.messages[1].content_str(), Some(file.as_str()));
        let pointer = req.messages[3].full_text();
        assert!(pointer.starts_with("[reincore]"), "{pointer}");
        assert!(pointer.contains("src/a.rs"), "{pointer}");
        assert!(pointer.contains("step 1"), "{pointer}");
    }

    /// A mutation's result is never hidden behind a pointer: an edit that
    /// reports the same text twice may genuinely have run twice, and a failed
    /// write must reach the model verbatim.
    #[test]
    fn search_fingerprints_preserve_regex_filters_and_wrapped_read_scopes() {
        for (left, right) in [
            (
                serde_json::json!({"regex":"\\d","path":"src"}),
                serde_json::json!({"regex":"/d","path":"src"}),
            ),
            (
                serde_json::json!({"query":"auth","isRegexp":false}),
                serde_json::json!({"query":"auth","isRegexp":true}),
            ),
            (
                serde_json::json!({"query":"auth","includePattern":"*.rs"}),
                serde_json::json!({"query":"auth","includePattern":"*.ts"}),
            ),
            (
                serde_json::json!({"query":"auth","paths":["src"]}),
                serde_json::json!({"query":"auth","paths":["tests"]}),
            ),
            (
                serde_json::json!({"query":"auth","maxResults":10}),
                serde_json::json!({"query":"auth","maxResults":20}),
            ),
        ] {
            assert_ne!(
                exploration_call_source("grep_search", &left),
                exploration_call_source("grep_search", &right)
            );
        }
        assert_ne!(
            exploration_call_source("read_file", &serde_json::json!({"files":[{"path":"a.rs"}]})),
            exploration_call_source("read_file", &serde_json::json!({"files":[{"path":"b.rs"}]})),
        );
    }

    #[test]
    fn ambiguous_call_ids_cannot_hide_mutation_or_duplicate_results() {
        let text = "observed result\n".repeat(MIN_DEDUPE_CHARS);
        for mutation in ["Write", "file_editor", "custom_mutation"] {
            let mut req = request(vec![
                assistant_call("first", "Read", serde_json::json!({"path":"a.rs"})),
                tool_message("first", &text),
                assistant_call("reused", "Read", serde_json::json!({"path":"a.rs"})),
                tool_message("reused", &text),
                assistant_call(
                    "reused",
                    mutation,
                    serde_json::json!({"command":"insert","path":"a.rs"}),
                ),
                tool_message("reused", &text),
            ]);
            assert!(
                dedupe_repeated_tool_results(&mut req).is_empty(),
                "{mutation}"
            );
            assert_eq!(
                req.messages.last().unwrap().content_str(),
                Some(text.as_str())
            );
        }
        let mut req = request(vec![
            assistant_call("first", "Read", serde_json::json!({"path":"a.rs"})),
            tool_message("first", &text),
            assistant_call("duplicate", "Read", serde_json::json!({"path":"a.rs"})),
            tool_message("duplicate", &text),
            tool_message("duplicate", &text),
        ]);
        assert!(dedupe_repeated_tool_results(&mut req).is_empty());
    }

    #[test]
    fn mutation_results_and_short_results_are_left_verbatim() {
        let written = "wrote src/a.rs\n".repeat(MIN_DEDUPE_CHARS / 4);
        let mut req = request(vec![
            assistant_call("a", "replace_string_in_file", serde_json::json!({})),
            tool_message("a", &written),
            assistant_call("b", "replace_string_in_file", serde_json::json!({})),
            tool_message("b", &written),
            // Below the size floor: the pointer would cost more than the text.
            assistant_call("c", "grep_search", serde_json::json!({"query":"x"})),
            tool_message("c", "No matches found"),
            assistant_call("d", "grep_search", serde_json::json!({"query":"x"})),
            tool_message("d", "No matches found"),
        ]);
        assert!(dedupe_repeated_tool_results(&mut req).is_empty());
        assert_eq!(req.messages[3].content_str(), Some(written.as_str()));
        assert_eq!(req.messages[7].content_str(), Some("No matches found"));
    }

    /// The most successful case must not be the quietest one: repetition that
    /// stayed below the threshold leaves an advisory health finding, so a
    /// prevented loop is distinguishable from a session where nothing happened.
    #[tokio::test]
    async fn a_prevented_loop_is_recorded_without_intervening() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        // Two identical reads: redundant, but short of the threshold of 3.
        for turn in 1..=2 {
            for event in read_events(turn, "src/schema.rs", "alpha\nbeta\ngamma") {
                store
                    .append_event(session, event.event_type, event.payload, None)
                    .await
                    .unwrap();
            }
        }

        let observation = breaker
            .observe(&store, session)
            .await
            .unwrap()
            .expect("redundancy below the threshold is still observed");
        assert_eq!(observation.count, 2);
        assert_eq!(
            observation.threshold,
            breaker.config().exploration_repetition_threshold
        );
        assert!(!observation.intervenes);
        assert_eq!(observation.redundancy().tool, "read_file");
        assert_eq!(
            observation.redundancy().source.as_deref(),
            Some("src/schema.rs")
        );

        assert!(!breaker
            .log_observation(&store, session, &observation)
            .await
            .unwrap());
        // The turn is untouched: no intervention is waiting for the next call.
        assert!(breaker
            .pending_intervention(&store, session)
            .await
            .unwrap()
            .is_none());
        // The near-miss travels as a finding the caller merges into the turn's
        // own health event, so one turn never writes two of them.
        let finding = observation
            .health_finding()
            .expect("a near-miss reports itself");
        assert_eq!(
            finding.kind,
            crate::agent_health::AgentHealthKind::RedundantLookup
        );
        assert_eq!(
            finding.severity,
            crate::agent_health::AgentHealthSeverity::Warning
        );
        assert!(finding.message.contains("src/schema.rs"), "{finding:?}");
    }

    /// Scoring happens before the accounting write, so the turn being scored is
    /// not in the event log yet. Without it fed in, every LLM-derived signal
    /// lags exactly one turn - the regression this guards.
    #[tokio::test]
    async fn the_turn_being_scored_is_visible_to_the_detector() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        let turn = serde_json::json!({
            "request_messages": [{"role": "user", "content": "retry"}],
            "response_message": {"role": "assistant", "content": "still broken"}
        });
        // One short of the threshold once persisted.
        for _ in 0..breaker.config().repetition_threshold - 1 {
            store
                .append_event(session, SessionEventType::LlmCall, turn.clone(), None)
                .await
                .unwrap();
        }

        assert!(
            breaker.observe(&store, session).await.unwrap().is_none(),
            "persisted turns alone are below the threshold"
        );
        let observation = breaker
            .observe_with_pending(&store, session, Some(turn))
            .await
            .unwrap()
            .expect("the pending turn completes the loop");
        assert!(observation.intervenes);
        assert_eq!(
            observation.count,
            breaker.config().repetition_threshold,
            "the pending turn must count exactly once"
        );
    }

    #[tokio::test]
    async fn audit_compacted_user_message_does_not_reset_a_tool_continuation() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        let threshold = breaker.config().exploration_repetition_threshold as usize;

        for turn in 1..=threshold {
            for event in read_events(turn, "PROJECT.md", "alpha\nbeta\ngamma\ndelta") {
                store
                    .append_event(session, event.event_type, event.payload, None)
                    .await
                    .unwrap();
            }
            store
                .append_event(
                    session,
                    SessionEventType::LlmCall,
                    serde_json::json!({
                        "request_group_id": "g",
                        // Durable audit keeps the user request only.
                        "request_messages": [{"role":"user","content":"build it"}],
                        // The real request is a tool continuation.
                        "message_roles": ["system","user","assistant","tool"],
                        "response_message": {"role":"assistant","tool_calls":[{
                            "id":format!("next-{turn}"),
                            "function":{"name":"read_file","arguments":"{\"filePath\":\"PROJECT.md\"}"}
                        }]}
                    }),
                    None,
                )
                .await
                .unwrap();
        }

        let observation = breaker
            .observe(&store, session)
            .await
            .unwrap()
            .expect("continuations must accumulate repeated-read evidence");
        assert!(observation.intervenes);
        assert_eq!(observation.count, threshold as u32);
    }

    #[test]
    fn completed_todos_and_group_steps_have_bounded_follow_up_work() {
        fn llm(seq: i64, call: serde_json::Value) -> SessionEvent {
            SessionEvent {
                session_id: Uuid::nil(),
                seq,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({
                    "request_group_id":"g",
                    "message_roles":["system","user","assistant","tool"],
                    "request_messages":[{"role":"user","content":"build it"}],
                    "response_message":{"role":"assistant","tool_calls":[call]}
                }),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            }
        }
        fn success(seq: i64, id: &str, name: &str) -> SessionEvent {
            SessionEvent {
                session_id: Uuid::nil(),
                seq,
                event_type: SessionEventType::ToolResult,
                payload: serde_json::json!({"request_group_id":"g","result":{
                    "tool_call_id":id,"name":name,"content":"Completed"
                }}),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            }
        }

        let config = CircuitBreakerConfig {
            post_completion_threshold: 2,
            post_completion_hard_stop_threshold: 3,
            request_group_step_threshold: 10,
            request_group_hard_stop_threshold: 12,
            ..CircuitBreakerConfig::default()
        };
        let todo = serde_json::json!({
            "id":"todo",
            "function":{"name":"manage_todo_list","arguments":serde_json::json!({
                "todoList":[{"id":"1","status":"completed"}]
            }).to_string()}
        });
        let mut events = vec![llm(1, todo), success(2, "todo", "manage_todo_list")];
        for turn in 0..3 {
            let id = format!("read-{turn}");
            events.push(llm(
                3 + turn * 2,
                serde_json::json!({"id":id,"function":{"name":"read_file","arguments":"{}"}}),
            ));
            events.push(success(4 + turn * 2, &id, "read_file"));
        }
        let soft = request_group_stall(&events, &config).expect("completion stall");
        assert_eq!(soft.count, 2);
        assert!(!soft.hard_stop);

        events.push(llm(
            9,
            serde_json::json!({"id":"read-3","function":{"name":"read_file","arguments":"{}"}}),
        ));
        events.push(success(10, "read-3", "read_file"));
        assert!(request_group_stall(&events, &config).unwrap().hard_stop);

        let cap = CircuitBreakerConfig {
            request_group_step_threshold: 2,
            request_group_hard_stop_threshold: 3,
            ..CircuitBreakerConfig::default()
        };
        let uncapped_work = vec![
            llm(
                1,
                serde_json::json!({"id":"a","function":{"name":"read_file","arguments":"{}"}}),
            ),
            llm(
                2,
                serde_json::json!({"id":"b","function":{"name":"read_file","arguments":"{}"}}),
            ),
        ];
        let group_guard = request_group_stall(&uncapped_work, &cap).expect("group cap");
        assert_eq!(group_guard.count, 2);
        assert!(!group_guard.hard_stop);
    }

    #[test]
    fn a_group_warning_is_emitted_once_until_the_hard_threshold() {
        fn stalled_call(seq: i64, id: &str) -> SessionEvent {
            SessionEvent {
                session_id: Uuid::nil(),
                seq,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({
                    "request_group_id":"g",
                    "response_message":{"role":"assistant","tool_calls":[{
                        "id":id,"function":{"name":"read_file","arguments":"{}"}
                    }]}
                }),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            }
        }
        let config = CircuitBreakerConfig {
            request_group_step_threshold: 2,
            request_group_hard_stop_threshold: 4,
            ..CircuitBreakerConfig::default()
        };
        let mut events = vec![stalled_call(1, "a"), stalled_call(2, "b")];
        let warning = request_group_stall(&events, &config).expect("first warning");
        assert!(!warning.hard_stop);
        events.push(SessionEvent {
            session_id: Uuid::nil(),
            seq: 3,
            event_type: SessionEventType::LoopIntervention,
            payload: serde_json::json!({
                "repeated_hash":StepHash::new(warning.kind, "g").0,
                "injected_message":"finalize",
                "hard_stop":false
            }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        });
        events.push(stalled_call(4, "c"));

        assert!(request_group_stall(&events, &config).is_none());
        events.push(SessionEvent {
            session_id: Uuid::nil(),
            seq: 5,
            event_type: SessionEventType::LlmCall,
            payload: serde_json::json!({
                "request_group_id":"g",
                "response_message":{"role":"assistant","tool_calls":[{
                    "id":"create","function":{"name":"create_file","arguments":"{}"}
                }]}
            }),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        });
        events.push(SessionEvent {
            session_id: Uuid::nil(),
            seq: 6,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({"request_group_id":"g","result":{
                "tool_call_id":"create","name":"create_file","content":"Created"
            }}),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        });
        events.push(stalled_call(7, "d"));
        events.push(stalled_call(8, "e"));
        let next_episode =
            request_group_stall(&events, &config).expect("new warning after progress");
        assert!(!next_episode.hard_stop);
        events.push(stalled_call(9, "f"));
        events.push(stalled_call(10, "g"));
        assert!(
            request_group_stall(&events, &config).is_some_and(|observation| observation.hard_stop)
        );
    }

    #[test]
    fn a_productive_128_turn_build_does_not_trip_the_stall_guard() {
        fn llm(seq: i64, id: &str, name: &str, arguments: serde_json::Value) -> SessionEvent {
            SessionEvent {
                session_id: Uuid::nil(),
                seq,
                event_type: SessionEventType::LlmCall,
                payload: serde_json::json!({
                    "request_group_id":"g",
                    "response_message":{"role":"assistant","tool_calls":[{
                        "id":id,"function":{"name":name,"arguments":arguments.to_string()}
                    }]}
                }),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            }
        }
        fn success(seq: i64, id: &str, name: &str) -> SessionEvent {
            SessionEvent {
                session_id: Uuid::nil(),
                seq,
                event_type: SessionEventType::ToolResult,
                payload: serde_json::json!({"request_group_id":"g","result":{
                    "tool_call_id":id,"name":name,"content":"Completed"
                }}),
                cost_cents: None,
                created_at: chrono::Utc::now(),
            }
        }

        let mut events = Vec::new();
        for turn in 0..127 {
            let id = format!("create-{turn}");
            events.push(llm(
                turn * 2 + 1,
                &id,
                "create_file",
                serde_json::json!({"filePath":format!("src/{turn}.rs")}),
            ));
            events.push(success(turn * 2 + 2, &id, "create_file"));
        }
        events.push(llm(
            255,
            "todos",
            "manage_todo_list",
            serde_json::json!({"todoList":[{"id":"1","status":"completed"}]}),
        ));
        events.push(success(256, "todos", "manage_todo_list"));

        assert!(request_group_stall(&events, &CircuitBreakerConfig::default()).is_none());
    }

    #[tokio::test]
    async fn progress_after_an_armed_hard_stop_receives_a_finalization_turn() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::default_config();
        store
            .append_event(
                session,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_group_id":"g",
                    "response_message":{"role":"assistant","tool_calls":[{
                        "id":"todos","function":{"name":"manage_todo_list","arguments":serde_json::json!({
                            "todoList":[{"id":"1","status":"completed"}]
                        }).to_string()}
                    }]}
                }),
                None,
            )
            .await
            .unwrap();
        breaker
            .log_intervention(
                &store,
                session,
                &StepHash::new("request_group_absolute_limit", "g"),
                512,
                "Stopped at the emergency ceiling.",
                true,
            )
            .await
            .unwrap();
        store
            .append_event(
                session,
                SessionEventType::ToolResult,
                serde_json::json!({"request_group_id":"g","result":{
                    "tool_call_id":"todos","name":"manage_todo_list",
                    "content":"Successfully wrote todo list"
                }}),
                None,
            )
            .await
            .unwrap();

        let pending = breaker
            .pending_intervention(&store, session)
            .await
            .unwrap()
            .expect("one finalization turn");
        assert!(!pending.hard_stop);
        assert!(pending.retryable);
        assert!(pending.message.contains("made substantive progress"));
    }

    #[tokio::test]
    async fn a_terminal_response_at_the_emergency_ceiling_is_not_poisoned() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            request_group_step_threshold: 10,
            request_group_hard_stop_threshold: 20,
            request_group_absolute_hard_stop_threshold: 3,
            ..CircuitBreakerConfig::default()
        });
        for turn in 1..=2 {
            store
                .append_event(
                    session,
                    SessionEventType::LlmCall,
                    serde_json::json!({
                        "request_group_id":"g",
                        "message_roles":["user"],
                        "request_messages":[{"role":"user","content":"build"}],
                        "response_message":{"role":"assistant","tool_calls":[{
                            "id":format!("read-{turn}"),
                            "function":{"name":"read_file","arguments":"{}"}
                        }]}
                    }),
                    None,
                )
                .await
                .unwrap();
        }
        store
            .append_event(
                session,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "request_group_id":"g",
                    "message_roles":["assistant","tool"],
                    "request_messages":[{"role":"user","content":"build"}],
                    "response_message":{"role":"assistant","content":"The implementation is complete."}
                }),
                None,
            )
            .await
            .unwrap();

        assert!(breaker.observe(&store, session).await.unwrap().is_none());
    }

    #[test]
    fn absorbed_repetition_reports_itself_and_a_quiet_turn_does_not() {
        let finding = redundant_lookup_finding(3).expect("dedup work is reported");
        assert_eq!(
            finding.kind,
            crate::agent_health::AgentHealthKind::RedundantLookup
        );
        assert_eq!(
            finding.severity,
            crate::agent_health::AgentHealthSeverity::Warning
        );
        assert!(redundant_lookup_finding(0).is_none());
    }

    #[test]
    fn negative_results_are_recognised_but_real_content_is_not() {
        assert!(is_negative_result(&serde_json::json!("No matches found")));
        assert!(is_negative_result(&serde_json::json!("  ")));
        assert!(is_negative_result(&serde_json::json!("file not found")));
        assert!(!is_negative_result(&serde_json::json!(
            "src/main.rs:1: fn main() {}"
        )));
        // Long content is a real finding whatever words it contains.
        assert!(!is_negative_result(&serde_json::json!(format!(
            "no matches{}",
            "x".repeat(MAX_NEGATIVE_RESULT_CHARS)
        ))));
    }

    #[test]
    fn searches_are_keyed_by_query_and_reads_by_path_alone() {
        let source = |name, arguments| exploration_call_source(name, &arguments);
        // Line ranges are deliberately ignored so slices of one file share evidence.
        assert_eq!(
            source(
                "read_file",
                serde_json::json!({"filePath": "src/a.rs", "startLine": 1})
            ),
            source(
                "read_file",
                serde_json::json!({"filePath": "src/a.rs", "startLine": 90})
            )
        );
        assert_ne!(
            source("grep_search", serde_json::json!({"query": "alpha"})),
            source("grep_search", serde_json::json!({"query": "beta"}))
        );
        assert_ne!(
            source(
                "grep_search",
                serde_json::json!({"query": "alpha", "path": "src"})
            ),
            source(
                "grep_search",
                serde_json::json!({"query": "alpha", "path": "tests"})
            )
        );
    }

    #[test]
    fn exploration_evidence_resets_and_does_not_count_replayed_results() {
        let make = |seq, group: &str, id: &str, name: &str, content: &str| SessionEvent {
            session_id: Uuid::nil(),
            seq,
            event_type: SessionEventType::ToolResult,
            payload: serde_json::json!({"request_group_id":group, "result":{
                "tool_call_id":id, "name":name, "content":content
            }}),
            cost_cents: None,
            created_at: chrono::Utc::now(),
        };
        let mut events = vec![
            make(1, "a", "1", "read_file", "same"),
            make(2, "a", "2", "read_file", "same"),
        ];
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 2);
        events.push(make(3, "a", "2", "read_file", "same"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 2);
        events.push(make(4, "a", "3", "read_file", "new finding"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
        events.push(make(5, "a", "4", "edit_file", "done"));
        assert!(repeated_exploration_results(&events).is_none());
        events.push(make(6, "a", "5", "read_file", "same"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
        events.push(make(7, "b", "6", "read_file", "same"));
        assert_eq!(repeated_exploration_results(&events).unwrap().count, 1);
    }

    #[test]
    fn settings_layer_with_the_most_specific_source_winning() {
        let stored = LoopSettings {
            exploration_repetition_threshold: Some(8),
            enabled: Some(false),
            verification_commands: Some(vec!["stored verify".to_string()]),
            ..LoopSettings::default()
        };
        let request = LoopSettings::from_extra(
            &serde_json::json!({
                "reincore_loop_enabled": true,
                "reincore_loop_exploration_hard_stop_threshold": 20,
                "reincore_loop_post_completion_threshold": 9,
                "reincore_loop_request_group_hard_stop_threshold": 144,
                "reincore_loop_request_group_absolute_hard_stop_threshold": 600,
                "reincore_loop_verification_commands": [" ACME Verify ", "acme verify", "acme lint"],
            })
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        );
        let breaker = CircuitBreaker::new(stored.overlay(request).resolve());
        let config = breaker.config();

        // The request re-enabled what the stored setting had turned off.
        assert!(config.enabled);
        assert_eq!(config.exploration_hard_stop_threshold, 20);
        assert_eq!(config.post_completion_threshold, 9);
        assert_eq!(config.request_group_hard_stop_threshold, 144);
        assert_eq!(config.request_group_absolute_hard_stop_threshold, 600);
        assert_eq!(config.verification_commands, ["acme verify", "acme lint"]);
        // Untouched by either layer, so still the stored value.
        assert_eq!(config.exploration_repetition_threshold, 8);
        // Untouched by any layer, so still the default.
        assert_eq!(
            config.repetition_threshold,
            CircuitBreakerConfig::default().repetition_threshold
        );
    }

    #[test]
    fn common_and_configured_terminal_commands_count_as_verification() {
        let defaults = default_verification_commands();
        for command in [
            "cargo clippy --all-targets",
            "ruff check src",
            "bun test",
            "./gradlew check",
            "mix test",
            "terraform validate",
            "tsc",
        ] {
            assert!(
                is_verification_call(
                    "exec_command",
                    &serde_json::json!({"cmd": command}),
                    &defaults,
                ),
                "{command:?} should count as verification"
            );
        }

        let custom = vec!["acme verify".to_string()];
        assert!(is_verification_call(
            "powershell",
            &serde_json::json!({"command": "ACME VERIFY --all"}),
            &custom,
        ));
        assert!(!is_verification_call(
            "exec_command",
            &serde_json::json!({"cmd": "cargo test"}),
            &custom,
        ));
        // A purpose-built verification tool remains authoritative even when
        // terminal command recognition is explicitly disabled.
        assert!(is_verification_call(
            "workspace_check",
            &serde_json::json!({}),
            &[],
        ));
    }

    #[test]
    fn verification_commands_accept_json_or_comma_separated_request_overrides() {
        let extra = |value: serde_json::Value| {
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        };
        let from_array = LoopSettings::from_extra(&extra(
            serde_json::json!({"reincore_loop_verification_commands": ["one", "two"]}),
        ));
        assert_eq!(
            from_array.verification_commands.unwrap(),
            ["one".to_string(), "two".to_string()]
        );

        let from_string = LoopSettings::from_extra(&extra(
            serde_json::json!({"reincore_loop_verification_commands": "one, two"}),
        ));
        assert_eq!(
            from_string.verification_commands.unwrap(),
            ["one".to_string(), "two".to_string()]
        );
    }

    /// A compose file forwarding `${VAR:-}` passes an empty string. Reading
    /// that as `false` would silently disable the breaker for every deployment
    /// that lists the variable without setting it.
    #[test]
    fn a_blank_or_unrecognized_flag_leaves_the_layer_unset() {
        // SAFETY: single-threaded test, variables are unique to it.
        const KEY: &str = "REINCORE_LOOP_ENABLED";
        for (value, expected) in [
            ("", None),
            ("   ", None),
            ("maybe", None),
            ("true", Some(true)),
            ("ON", Some(true)),
            ("false", Some(false)),
            ("0", Some(false)),
        ] {
            std::env::set_var(KEY, value);
            assert_eq!(
                LoopSettings::read_env().enabled,
                expected,
                "value {value:?}"
            );
        }
        std::env::remove_var(KEY);
    }

    #[test]
    fn exploration_gets_a_larger_budget_than_a_repeated_step() {
        let config = CircuitBreakerConfig::default();
        assert!(config.exploration_repetition_threshold > config.repetition_threshold);
        assert!(config.exploration_hard_stop_threshold > config.hard_stop_threshold);
    }

    /// The off switch must stop refusals without also going blind: the score is
    /// still taken and still recorded, it just never intervenes.
    #[tokio::test]
    async fn the_off_switch_keeps_observing() {
        use crate::in_memory_stores::InMemoryEventStore;
        let store = InMemoryEventStore::default();
        let session = Uuid::new_v4();
        let breaker = CircuitBreaker::new(
            LoopSettings {
                enabled: Some(false),
                ..LoopSettings::default()
            }
            .resolve(),
        );
        // Far past every threshold.
        for turn in 1..=20 {
            for event in read_events(turn, "src/schema.rs", "alpha\nbeta\ngamma") {
                store
                    .append_event(session, event.event_type, event.payload, None)
                    .await
                    .unwrap();
            }
        }

        let observation = breaker
            .observe(&store, session)
            .await
            .unwrap()
            .expect("scoring continues while interventions are off");
        assert!(observation.count >= 20);
        assert!(!observation.intervenes);
        assert!(!observation.hard_stop);
        assert!(!breaker
            .log_observation(&store, session, &observation)
            .await
            .unwrap());
        assert!(breaker
            .pending_intervention(&store, session)
            .await
            .unwrap()
            .is_none());
        // Still reported, so an operator who switched interventions off can see
        // how much repetition the session is producing.
        assert!(observation.health_finding().is_some());
    }

    #[test]
    fn dedupe_is_switched_independently_of_interventions() {
        let file = "fn handler() {}\n".repeat(MIN_DEDUPE_CHARS / 8);
        let messages = || {
            vec![
                assistant_call("a", "read_file", serde_json::json!({"filePath":"src/a.rs"})),
                tool_message("a", &file),
                assistant_call("b", "read_file", serde_json::json!({"filePath":"src/a.rs"})),
                tool_message("b", &file),
            ]
        };
        // Interventions off, dedup left alone: the tokens are still saved.
        let breaker = CircuitBreaker::new(
            LoopSettings {
                enabled: Some(false),
                ..LoopSettings::default()
            }
            .resolve(),
        );
        assert_eq!(
            breaker
                .dedupe_tool_results(&mut request(messages()))
                .replaced,
            1
        );

        let breaker = CircuitBreaker::new(
            LoopSettings {
                dedupe_tool_results: Some(false),
                ..LoopSettings::default()
            }
            .resolve(),
        );
        assert!(breaker
            .dedupe_tool_results(&mut request(messages()))
            .is_empty());
    }

    #[test]
    fn historical_duplicates_are_compacted_without_being_reported_again() {
        let repeated = "same file content\n".repeat(MIN_DEDUPE_CHARS / 8);
        let fresh = "different content\n".repeat(MIN_DEDUPE_CHARS / 8);
        let mut req = request(vec![
            assistant_call("a", "read_file", serde_json::json!({"filePath":"src/a.rs"})),
            tool_message("a", &repeated),
            assistant_call("b", "read_file", serde_json::json!({"filePath":"src/a.rs"})),
            tool_message("b", &repeated),
            assistant_call("c", "read_file", serde_json::json!({"filePath":"src/b.rs"})),
            tool_message("c", &fresh),
        ]);

        let stats = dedupe_repeated_tool_results(&mut req);

        assert_eq!(stats.replaced, 1);
        assert_eq!(stats.new_replacements, 0);
    }

    #[test]
    fn the_message_names_the_evidence_not_just_a_count() {
        let repetition = |kind, source: &str| {
            ExplorationRepetition {
                hash: StepHash::new("a", "b"),
                count: 5,
                tool: "read_file".to_string(),
                source: Some(source.to_string()),
                negative: kind == RepetitionKind::EmptyResult,
                kind,
            }
            .describe()
        };

        let overlapping = repetition(RepetitionKind::OverlappingRead, "src/schema.rs");
        assert!(overlapping.contains("src/schema.rs"), "{overlapping}");
        assert!(overlapping.contains("read_file"), "{overlapping}");
        assert!(overlapping.contains("5 overlapping reads"), "{overlapping}");
        assert!(
            overlapping.contains("content already in context"),
            "{overlapping}"
        );

        assert!(
            repetition(RepetitionKind::IdenticalResult, "src/a.rs").contains("5 identical results")
        );
        let empty = repetition(RepetitionKind::EmptyResult, "handler");
        assert!(empty.contains("5 empty results"), "{empty}");
        assert!(empty.contains("finding nothing"), "{empty}");
    }

    #[test]
    fn invalid_config_is_normalized() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            repetition_threshold: 0,
            hard_stop_threshold: 0,
            exploration_repetition_threshold: 0,
            exploration_hard_stop_threshold: 0,
            negative_result_threshold_multiplier: 0,
            post_completion_threshold: 0,
            post_completion_hard_stop_threshold: 0,
            request_group_step_threshold: 0,
            request_group_hard_stop_threshold: 0,
            request_group_absolute_hard_stop_threshold: 0,
            window_size: 0,
            ..CircuitBreakerConfig::default()
        });
        assert_eq!(breaker.config.repetition_threshold, 1);
        assert_eq!(breaker.config.hard_stop_threshold, 1);
        assert_eq!(breaker.config.exploration_repetition_threshold, 1);
        assert_eq!(breaker.config.exploration_hard_stop_threshold, 1);
        assert_eq!(breaker.config.negative_result_threshold_multiplier, 1);
        assert_eq!(breaker.config.post_completion_threshold, 1);
        assert_eq!(breaker.config.post_completion_hard_stop_threshold, 1);
        assert_eq!(breaker.config.request_group_step_threshold, 1);
        assert_eq!(breaker.config.request_group_hard_stop_threshold, 1);
        assert_eq!(breaker.config.request_group_absolute_hard_stop_threshold, 1);
        assert_eq!(breaker.config.window_size, 1);
    }

    #[test]
    fn circuit_breaker_detects_loop() {
        let mut cb = CircuitBreaker::default_config();
        let input = "retry fix";
        let output = "still broken";

        assert!(cb.record_step(input, output).is_none());
        assert!(cb.record_step(input, output).is_none());
        let intervention = cb.record_step(input, output);
        assert!(intervention.is_some());
        assert!(intervention.unwrap().contains("repeating"));
    }

    #[tokio::test]
    async fn persisted_loop_schedules_nudge_for_next_turn() {
        use crate::event::EventStore;
        use crate::in_memory_stores::InMemoryEventStore;

        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let payload = serde_json::json!({
            "request_messages": [{"role": "user", "content": "retry"}],
            "response_message": {"role": "assistant", "content": "still broken"}
        });
        for _ in 0..3 {
            store
                .append_event(session_id, SessionEventType::LlmCall, payload.clone(), None)
                .await
                .unwrap();
        }

        let breaker = CircuitBreaker::default_config();
        assert!(breaker.inspect_and_log(&store, session_id).await.unwrap());
        assert!(breaker
            .pending_intervention(&store, session_id)
            .await
            .unwrap()
            .unwrap()
            .message
            .contains("repeating"));
        let events = store.read_events(session_id, None, None).await.unwrap();
        let record: crate::InterventionRecord =
            serde_json::from_value(events.last().unwrap().payload["intervention"].clone()).unwrap();
        assert_eq!(record.detector, "loop_detection");
        assert_eq!(record.action, crate::InterventionAction::RepairRequested);
        assert_eq!(record.added_input_tokens, 0);
        let metrics = crate::InterventionMetrics::from_events(&events, true, 10, 0);
        assert_eq!(metrics.successful_recoveries, 0);
        assert_eq!(metrics.added_input_tokens, 0);
    }

    #[tokio::test]
    async fn loop_is_detected_even_when_transcript_history_grows() {
        use crate::event::EventStore;
        use crate::in_memory_stores::InMemoryEventStore;

        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        // Simulate a real agent loop: each turn resends the full transcript
        // (which grows) but ends with the same failing input and produces the
        // same response.
        for turn in 0..3 {
            let mut messages = Vec::new();
            for previous in 0..turn {
                messages.push(serde_json::json!(
                    {"role": "assistant", "content": format!("attempt {previous}")}
                ));
            }
            messages.push(serde_json::json!({"role": "user", "content": "error: command failed"}));
            let payload = serde_json::json!({
                "request_messages": messages,
                "response_message": {"role": "assistant", "content": "running the same command"}
            });
            store
                .append_event(session_id, SessionEventType::LlmCall, payload, None)
                .await
                .unwrap();
        }

        let breaker = CircuitBreaker::default_config();
        assert!(breaker.inspect_and_log(&store, session_id).await.unwrap());
    }

    /// Regression: providers mint a unique `tool_calls[].id` per invocation and
    /// clients echo it as `tool_call_id`, so the raw message hashed differently
    /// every turn and no repeated tool call was ever detected.
    #[tokio::test]
    async fn repeated_tool_call_is_detected_despite_unique_call_ids() {
        use crate::event::EventStore;
        use crate::in_memory_stores::InMemoryEventStore;

        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        for turn in 0..3 {
            let call_id = format!("call_{turn}");
            let payload = serde_json::json!({
                "request_messages": [{
                    "role": "tool",
                    "content": "<html>Watoto</html>",
                    "tool_call_id": call_id,
                }],
                "response_message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "fetch",
                            "arguments": "{\"url\":\"https://watoto.com/\"}"
                        }
                    }]
                }
            });
            store
                .append_event(session_id, SessionEventType::LlmCall, payload, None)
                .await
                .unwrap();
        }

        let breaker = CircuitBreaker::default_config();
        assert!(breaker.inspect_and_log(&store, session_id).await.unwrap());
        let pending = breaker
            .pending_intervention(&store, session_id)
            .await
            .unwrap()
            .expect("nudge scheduled");
        assert!(pending.message.contains("repeating"));
        assert!(!pending.hard_stop);
    }

    #[tokio::test]
    async fn a_loop_that_survives_the_nudge_escalates_to_a_hard_stop() {
        use crate::event::EventStore;
        use crate::in_memory_stores::InMemoryEventStore;

        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let payload = serde_json::json!({
            "request_messages": [{"role": "user", "content": "retry"}],
            "response_message": {"role": "assistant", "content": "still broken"}
        });
        let breaker = CircuitBreaker::default_config();
        for _ in 0..breaker.config.hard_stop_threshold {
            store
                .append_event(session_id, SessionEventType::LlmCall, payload.clone(), None)
                .await
                .unwrap();
        }

        assert!(breaker.inspect_and_log(&store, session_id).await.unwrap());
        let pending = breaker
            .pending_intervention(&store, session_id)
            .await
            .unwrap()
            .expect("intervention scheduled");
        assert!(pending.hard_stop);
        assert!(pending.message.contains("Stopped"));
    }

    #[tokio::test]
    async fn clearing_a_hard_stop_allows_one_more_turn() {
        use crate::event::EventStore;
        use crate::in_memory_stores::InMemoryEventStore;

        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let payload = serde_json::json!({
            "request_messages": [{"role": "user", "content": "retry"}],
            "response_message": {"role": "assistant", "content": "still broken"}
        });
        let breaker = CircuitBreaker::default_config();
        for _ in 0..breaker.config.hard_stop_threshold {
            store
                .append_event(session_id, SessionEventType::LlmCall, payload.clone(), None)
                .await
                .unwrap();
        }
        breaker.inspect_and_log(&store, session_id).await.unwrap();
        assert!(breaker
            .pending_intervention(&store, session_id)
            .await
            .unwrap()
            .is_some_and(|item| item.hard_stop));

        breaker
            .clear_intervention(&store, session_id)
            .await
            .unwrap();

        assert!(breaker
            .pending_intervention(&store, session_id)
            .await
            .unwrap()
            .is_none());
    }

    #[test]
    fn normalization_strips_only_volatile_identifiers() {
        let message = serde_json::json!({
            "role": "assistant",
            "tool_call_id": "call_1",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "fetch", "arguments": "{\"url\":\"a\"}"}
            }]
        });
        let normalized = normalize_step_message(&message);
        assert!(normalized.get("tool_call_id").is_none());
        let call = &normalized["tool_calls"][0];
        assert!(call.get("id").is_none());
        assert_eq!(call["function"]["name"], "fetch");
        assert_eq!(call["function"]["arguments"], "{\"url\":\"a\"}");
    }
}
