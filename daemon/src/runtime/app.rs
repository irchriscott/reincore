use engine_core::{
    CheckpointManager, CostPredictor, EscalationClassifier, FrameworkAdapter, ProviderRegistry,
    Session, ToolExecutorRegistry, ToolStateRetrievalConfig,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::db::LocalDatabase;
use crate::provider::build_provider_registry;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<LocalDatabase>,
    pub sessions: Arc<Mutex<std::collections::HashMap<Uuid, Session>>>,
    pub checkpoints: Arc<CheckpointManager>,
    pub cost_predictor: Arc<CostPredictor>,
    pub provider_replay_enabled: bool,
    pub task_decisions_enabled: bool,
    pub claim_grounding: Arc<engine_core::claim_grounding::ClaimGroundingConfig>,
    /// Loop-detection settings from `config.toml`, layered under the
    /// environment, the settings store, and per-request overrides.
    pub loop_detection: Arc<engine_core::LoopSettings>,
    pub action_policy: Arc<engine_core::action_gate::ActionPolicy>,
    pub detection: Arc<engine_core::detection::DetectionConfig>,
    pub providers: Arc<ProviderRegistry>,
    pub tool_executors: Arc<ToolExecutorRegistry>,
    pub mcp_gateway: mcp_gateway::Gateway,
    pub classifier: Arc<EscalationClassifier>,
    pub framework: Arc<FrameworkAdapter>,
    pub default_project_name: Arc<String>,
    pub default_project_path: Arc<String>,
    pub verbose: bool,
}

#[derive(Clone)]
pub struct SingleProviderAppState {
    pub db: Arc<LocalDatabase>,
    pub sessions: Arc<Mutex<std::collections::HashMap<Uuid, Session>>>,
    pub checkpoints: Arc<CheckpointManager>,
    pub cost_predictor: Arc<CostPredictor>,
    pub provider_replay_enabled: bool,
    pub task_decisions_enabled: bool,
    pub claim_grounding: Arc<engine_core::claim_grounding::ClaimGroundingConfig>,
    /// Loop-detection settings from `config.toml`, layered under the
    /// environment, the settings store, and per-request overrides.
    pub loop_detection: Arc<engine_core::LoopSettings>,
    pub action_policy: Arc<engine_core::action_gate::ActionPolicy>,
    pub detection: Arc<engine_core::detection::DetectionConfig>,
    pub provider: std::sync::Arc<dyn engine_core::Provider>,
    pub api_mode: Arc<String>,
    pub tool_executors: Arc<ToolExecutorRegistry>,
    pub mcp_gateway: mcp_gateway::Gateway,
    pub classifier: Arc<EscalationClassifier>,
    pub framework: Arc<FrameworkAdapter>,
    pub default_project_name: Arc<String>,
    pub default_project_path: Arc<String>,
    pub verbose: bool,
}

impl SingleProviderAppState {
    pub fn new_with_project_path(
        db: LocalDatabase,
        provider: std::sync::Arc<dyn engine_core::Provider>,
        project_path: Option<&str>,
    ) -> Self {
        let config = crate::local_config::load_local_config();
        engine_core::privacy::set_process_default_policy(config.privacy.clone());
        let (default_project_name, default_project_path) = local_project_defaults(project_path);
        let tool_executors = ToolExecutorRegistry::new();
        let classifier = EscalationClassifier::new(
            std::env::var("REINCORE_LITE_MODEL").unwrap_or_else(|_| "gpt-5.6-luna".to_string()),
            std::env::var("REINCORE_PRO_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".to_string()),
            std::env::var("REINCORE_ULTRA_MODEL").ok(),
        );
        let checkpoint_manager = CheckpointManager::new(
            std::env::var("REINCORE_CHECKPOINT_SECRET")
                .unwrap_or_else(|_| "local-dev-secret".to_string()),
            60,
        );
        let provider_label = provider.label().to_string();
        // Free providers (e.g. local Ollama) have no known per-token pricing,
        // so the cost predictor reports zero-cost predictions and the budget
        // degradation state machine never advances tiers.
        let cost_predictor = CostPredictor::new()
            .with_provider(provider_label)
            .with_free_provider(provider.is_free());
        let db = Arc::new(db);
        let framework = FrameworkAdapter::new(checkpoint_manager.clone()).with_tool_state_store(
            db.clone() as Arc<dyn engine_core::ToolStateStore>,
            ToolStateRetrievalConfig::default(),
        );
        let provider_replay_enabled = config.provider_replay.enabled;
        let task_decisions_enabled = config.task_decisions.enabled;
        tracing::info!(
            provider_replay_enabled,
            task_decisions_enabled,
            "agent context configuration loaded"
        );
        Self {
            provider_replay_enabled,
            task_decisions_enabled,
            claim_grounding: Arc::new(config.claim_grounding),
            loop_detection: Arc::new(config.loop_detection),
            action_policy: Arc::new(config.action_policy),
            detection: Arc::new(config.detection),
            db,
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            checkpoints: Arc::new(checkpoint_manager),
            cost_predictor: Arc::new(cost_predictor),
            provider,
            api_mode: Arc::new("agentic".to_string()),
            tool_executors: Arc::new(tool_executors),
            mcp_gateway: mcp_gateway::Gateway::default(),
            classifier: Arc::new(classifier),
            framework: Arc::new(framework),
            default_project_name: Arc::new(default_project_name),
            default_project_path: Arc::new(default_project_path),
            verbose: false,
        }
    }

    pub fn with_api_mode(mut self, mode: &str) -> Self {
        self.api_mode = Arc::new(mode.to_string());
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl AppState {
    pub fn new_with_project_path(db: LocalDatabase, project_path: Option<&str>) -> Self {
        let config = crate::local_config::load_local_config();
        engine_core::privacy::set_process_default_policy(config.privacy.clone());
        let (default_project_name, default_project_path) = local_project_defaults(project_path);
        let providers = build_provider_registry();
        let provider_label = providers.iter().next().map(|p| p.label().to_string());
        let tool_executors = ToolExecutorRegistry::new();
        let classifier = EscalationClassifier::new(
            std::env::var("REINCORE_LITE_MODEL").unwrap_or_else(|_| "gpt-5.6-luna".to_string()),
            std::env::var("REINCORE_PRO_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".to_string()),
            std::env::var("REINCORE_ULTRA_MODEL").ok(),
        );
        let checkpoint_manager = CheckpointManager::new(
            std::env::var("REINCORE_CHECKPOINT_SECRET")
                .unwrap_or_else(|_| "local-dev-secret".to_string()),
            60,
        );
        let cost_predictor = if let Some(label) = provider_label {
            CostPredictor::new().with_provider(label)
        } else {
            CostPredictor::new()
        };
        let db = Arc::new(db);
        let framework = FrameworkAdapter::new(checkpoint_manager.clone()).with_tool_state_store(
            db.clone() as Arc<dyn engine_core::ToolStateStore>,
            ToolStateRetrievalConfig::default(),
        );
        let provider_replay_enabled = config.provider_replay.enabled;
        let task_decisions_enabled = config.task_decisions.enabled;
        tracing::info!(
            provider_replay_enabled,
            task_decisions_enabled,
            "agent context configuration loaded"
        );
        Self {
            provider_replay_enabled,
            task_decisions_enabled,
            claim_grounding: Arc::new(config.claim_grounding),
            loop_detection: Arc::new(config.loop_detection),
            action_policy: Arc::new(config.action_policy),
            detection: Arc::new(config.detection),
            db,
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            checkpoints: Arc::new(checkpoint_manager),
            cost_predictor: Arc::new(cost_predictor),
            providers: Arc::new(providers),
            tool_executors: Arc::new(tool_executors),
            mcp_gateway: mcp_gateway::Gateway::default(),
            classifier: Arc::new(classifier),
            framework: Arc::new(framework),
            default_project_name: Arc::new(default_project_name),
            default_project_path: Arc::new(default_project_path),
            verbose: false,
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

fn local_project_defaults(project_path: Option<&str>) -> (String, String) {
    let raw = project_path
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    // Canonicalize so the daemon default matches the normalization sessions
    // and MCP scoping apply (verbatim prefixes, separators, symlinks).
    let absolute = if raw.is_absolute() {
        raw
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&raw))
            .unwrap_or(raw)
    };
    let path = std::fs::canonicalize(&absolute).unwrap_or(absolute);
    let name = path
        .file_name()
        .and_then(|part| part.to_str())
        .filter(|part| !part.is_empty())
        .unwrap_or("reincore")
        .to_string();
    (name, path.to_string_lossy().into_owned())
}
