use config::{Config, File, FileFormat};
use engine_core::{
    MinimumQuality, ModelTier, ProviderModelMapping, ProviderStrategy, RoutingMode, RoutingPolicy,
};
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

/// The default `config.toml` content written on first run. Kept in sync with
/// `config.example.toml`. API keys intentionally absent.
const DEFAULT_CONFIG_TOML: &str = include_str!("../../config.example.toml");

/// Name of the home-directory folder holding every local ReinCore file
/// (config, credentials, profiles, SQLite database, daemon state).
pub const CONFIG_DIR_NAME: &str = ".reincore-io";

/// Absolute path of the local ReinCore home folder (`~/.reincore-io`).
pub fn config_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR_NAME)
}

/// Local-only daemon configuration. Credentials intentionally remain outside
/// this file (environment variables or the credential store).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct LocalConfig {
    #[serde(default)]
    pub action_policy: engine_core::action_gate::ActionPolicy,
    /// Local privacy defaults. An explicit REINCORE_PRIVACY_POLICY environment
    /// value takes precedence for this process.
    #[serde(default)]
    pub privacy: engine_core::PrivacyPolicy,
    #[serde(default)]
    pub provider_replay: ProviderReplayConfig,
    #[serde(default)]
    pub task_decisions: TaskDecisionsConfig,
    #[serde(default)]
    pub claim_grounding: engine_core::claim_grounding::ClaimGroundingConfig,
    /// Response modes for heuristic agent-health checks. Authorization and
    /// tenant isolation remain mandatory regardless of these values.
    #[serde(default)]
    pub interventions: engine_core::InterventionPolicy,
    /// Loop-detection thresholds and switches. Unset fields keep their
    /// defaults, so an absent section is "use the defaults" rather than
    /// "disable everything". `REINCORE_LOOP_*` still wins where it is set.
    #[serde(default, rename = "loop_detection")]
    pub loop_detection: engine_core::LoopSettings,
    /// Per-feature detection switches (`lexical` by default, `llm` opts into
    /// a Meta Muse `muse-wonder` model judgment with lexical fallback).
    #[serde(default)]
    pub detection: engine_core::detection::DetectionConfig,
    /// Which desktop toasts the person at this machine wants. Read by the
    /// app rather than the daemon: the daemon has no window to show one in.
    #[serde(default)]
    pub notifications: crate::notifications::LocalNotificationSettings,
    /// Only session/task naming; never conversation compaction.
    #[serde(default)]
    pub titles: TitleConfig,
    /// Global routing defaults applied to every provider unless a provider
    /// overrides them.
    #[serde(default)]
    routing: Option<PartialRoutingPolicy>,
    #[serde(default)]
    providers: HashMap<String, LocalProviderConfig>,
    /// OpenTelemetry OTLP export configuration. Environment variables still
    /// take precedence when explicitly set.
    #[serde(default)]
    otel: Option<TelemetryConfig>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ProviderReplayConfig {
    pub enabled: bool,
}

impl Default for ProviderReplayConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct TaskDecisionsConfig {
    pub enabled: bool,
}

impl Default for TaskDecisionsConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TitleConfig {
    #[serde(default)]
    pub summarizer: TitleSummarizer,
}

pub use engine_core::title_generation::TitleSummarizer;

/// OpenTelemetry export settings persisted in `~/.reincore-io/config.toml`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(rename = "exporter_otlp_endpoint", default)]
    pub endpoint: Option<String>,
    #[serde(rename = "service_name", default)]
    pub service_name: Option<String>,
    #[serde(default)]
    pub insecure: bool,
}

impl TelemetryConfig {
    /// Fill in the bundled Jaeger defaults for any unset fields. This is the
    /// shape the desktop app writes when the user toggles telemetry on.
    pub fn with_defaults(self) -> Self {
        Self {
            enabled: self.enabled,
            endpoint: self
                .endpoint
                .or_else(|| Some("http://localhost:4317".to_string())),
            service_name: self
                .service_name
                .or_else(|| Some("reincore-local".to_string())),
            insecure: self.insecure,
        }
    }
}

impl LocalConfig {
    /// Return a normalized view of OTEL settings. If OTLP is enabled and an
    /// endpoint is configured (or the bundled Jaeger default), the daemon
    /// will export traces. Service name defaults to `reincore-local`.
    pub fn otel(&self) -> TelemetryRuntimeConfig {
        let enabled = self
            .routing
            .as_ref()
            .and_then(|r| r.otel_enabled)
            .or(self.otel.as_ref().map(|o| o.enabled))
            .unwrap_or(false);
        let configured_endpoint = self
            .routing
            .as_ref()
            .and_then(|r| r.otel_exporter_otlp_endpoint.clone())
            .or_else(|| self.otel.as_ref().and_then(|o| o.endpoint.clone()));
        // When enabled but no endpoint is set, default to the bundled Jaeger
        // gRPC endpoint so `reincore optl enable` works out of the box.
        let endpoint = match configured_endpoint {
            Some(url) => Some(url),
            None if enabled => Some("http://localhost:4317".to_string()),
            None => None,
        };
        let service_name = self
            .routing
            .as_ref()
            .and_then(|r| r.otel_service_name.clone())
            .or_else(|| self.otel.as_ref().and_then(|o| o.service_name.clone()))
            .unwrap_or_else(|| "reincore-local".to_string());
        let insecure = self
            .routing
            .as_ref()
            .and_then(|r| r.otel_insecure)
            .or(self.otel.as_ref().map(|o| o.insecure))
            .unwrap_or(true);

        TelemetryRuntimeConfig {
            enabled,
            endpoint,
            service_name,
            insecure,
        }
    }

    /// Return the raw `[otel]` table as read from `~/.reincore-io/config.toml`,
    /// suitable for the desktop app settings page.
    pub fn otel_config(&self) -> TelemetryConfig {
        self.otel.clone().unwrap_or_default()
    }
}

/// Normalized OTEL settings handed off to the telemetry module.
#[derive(Clone, Debug)]
pub struct TelemetryRuntimeConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub service_name: String,
    pub insecure: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct LocalProviderConfig {
    /// User-facing name and API root for the custom OpenAI-compatible provider.
    name: Option<String>,
    base_url: Option<String>,
    #[serde(alias = "standard")]
    standard_model: Option<String>,
    #[serde(alias = "lite")]
    lite_model: Option<String>,
    #[serde(alias = "pro")]
    pro_model: Option<String>,
    #[serde(alias = "ultra")]
    ultra_model: Option<String>,
    /// Per-provider routing override. Fields omitted here inherit the
    /// global `[routing]` defaults (or the built-in `auto` policy).
    #[serde(default)]
    routing: Option<PartialRoutingPolicy>,
}

/// Routing policy as expressed in TOML, where every field is optional so a
/// section can pin just one knob (e.g. `mode = "locked"`) while inheriting
/// the rest from the global section or the built-in defaults.
#[derive(Clone, Debug, Default, Deserialize)]
struct PartialRoutingPolicy {
    mode: Option<RoutingMode>,
    provider_strategy: Option<ProviderStrategy>,
    minimum_quality: Option<MinimumQuality>,
    allow_upgrade: Option<bool>,
    allow_degradation: Option<bool>,
    #[serde(rename = "otel_enabled")]
    otel_enabled: Option<bool>,
    #[serde(rename = "otel_exporter_otlp_endpoint")]
    otel_exporter_otlp_endpoint: Option<String>,
    #[serde(rename = "otel_service_name")]
    otel_service_name: Option<String>,
    #[serde(rename = "otel_insecure")]
    otel_insecure: Option<bool>,
}

impl PartialRoutingPolicy {
    /// Layer `overlay` on top of `base`: each set field in `overlay` replaces
    /// the corresponding field in `base`; unset fields inherit `base`.
    fn merge_into(self, mut base: RoutingPolicy) -> RoutingPolicy {
        if let Some(mode) = self.mode {
            base.mode = mode;
        }
        if let Some(strategy) = self.provider_strategy {
            base.provider_strategy = strategy;
        }
        if let Some(quality) = self.minimum_quality {
            base.minimum_quality = quality;
        }
        if let Some(allow) = self.allow_upgrade {
            base.allow_upgrade = allow;
        }
        if let Some(allow) = self.allow_degradation {
            base.allow_degradation = allow;
        }
        base
    }

    /// Keep serde from complaining about unknown fields when legacy telemetry
    /// keys live under `[routing]`. They are still read by `LocalConfig::otel`.
    #[allow(dead_code)]
    fn has_telemetry_fields(&self) -> bool {
        self.otel_enabled.is_some()
            || self.otel_exporter_otlp_endpoint.is_some()
            || self.otel_service_name.is_some()
            || self.otel_insecure.is_some()
    }
}

#[derive(Clone, Copy)]
pub struct ModelEnvironment<'a> {
    pub standard: &'a str,
    pub lite: &'a str,
    pub pro: &'a str,
    pub ultra: &'a str,
}

#[derive(Clone, Debug)]
pub struct ResolvedModelPolicy {
    pub standard_model: Option<String>,
    pub models: Vec<ProviderModelMapping>,
    pub routing: RoutingPolicy,
}

pub fn local_config_path() -> PathBuf {
    std::env::var_os("REINCORE_CONFIG_PATH")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| config_home().join("config.toml"))
}

pub fn load_local_config() -> LocalConfig {
    let path = local_config_path();
    match Config::builder()
        .add_source(
            File::from(path.clone())
                .format(FileFormat::Toml)
                .required(false),
        )
        .build()
        .and_then(|config| config.try_deserialize::<LocalConfig>())
    {
        Ok(mut config) => {
            // Keep the app, daemon, and direct local-config consumers on the
            // same effective policy. This also removes invalid exact-secret
            // placeholders such as source-language type names from older
            // configuration files.
            config.privacy = config.privacy.normalized();
            if let Err(error) = engine_core::claim_grounding::validate_tool_contracts(
                &config.claim_grounding.tool_contracts,
            ) {
                // Invalid entries never establish evidence; they only mark it incomplete.
                tracing::warn!(path = %path.display(), %error, "invalid claim grounding tool contracts");
            }
            config
        }
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "ignoring invalid local configuration");
            // A malformed policy must never silently broaden tool permissions.
            let mut fallback = LocalConfig::default();
            fallback.action_policy.default_allow = false;
            fallback
        }
    }
}

/// Make the local TOML policy visible to shared engine/provider boundaries.
/// Environment configuration is resolved later by `current_policy` and wins.
pub fn install_privacy_policy() {
    let config = load_local_config();
    engine_core::privacy::set_process_default_policy(config.privacy);
    engine_core::set_process_default_intervention_policy(config.interventions);
}

/// Create `~/.reincore-io/config.toml` with default contents if it does not
/// already exist. Returns the path to the config file. This is called on
/// daemon startup and before any config-editing command so a fresh install
/// always has a writable config.
pub fn ensure_config_exists() -> std::io::Result<PathBuf> {
    let path = local_config_path();
    if path.exists() {
        return Ok(path);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, DEFAULT_CONFIG_TOML)?;
    Ok(path)
}

/// Load the config document for editing with `toml_edit`, preserving comments
/// and formatting. Returns a fresh document created from the default template
/// if the file is missing.
fn load_config_document() -> anyhow::Result<toml_edit::DocumentMut> {
    let path = local_config_path();
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|_| DEFAULT_CONFIG_TOML.to_string());
    let document = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|err| anyhow::anyhow!("failed to parse {}: {}", path.display(), err))?;
    Ok(document)
}

/// Ensure a top-level `[otel]` table exists in the document, creating it if needed.
fn ensure_otel_table(document: &mut toml_edit::DocumentMut) -> &mut toml_edit::Table {
    if !document.contains_key("otel") {
        let mut table = toml_edit::Table::new();
        table.set_implicit(true);
        document.insert("otel", toml_edit::Item::Table(table));
    }
    document["otel"].as_table_mut().expect("otel table")
}

/// Ensure a top-level `[privacy]` table exists in the document.
fn ensure_privacy_table(document: &mut toml_edit::DocumentMut) -> &mut toml_edit::Table {
    if !document.contains_key("privacy") {
        document.insert("privacy", toml_edit::Item::Table(toml_edit::Table::new()));
    }
    document["privacy"].as_table_mut().expect("privacy table")
}

/// Ensure a top-level `[notifications]` table exists in the document.
fn ensure_notifications_table(document: &mut toml_edit::DocumentMut) -> &mut toml_edit::Table {
    if !document.contains_key("notifications") {
        document.insert(
            "notifications",
            toml_edit::Item::Table(toml_edit::Table::new()),
        );
    }
    document["notifications"]
        .as_table_mut()
        .expect("notifications table")
}

/// Write a config document back to `~/.reincore-io/config.toml`.
fn save_config_document(document: &toml_edit::DocumentMut) -> anyhow::Result<()> {
    let path = local_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, document.to_string())?;
    Ok(())
}

/// Ensure the `[routing]` table exists in the document, creating it if needed.
fn ensure_routing_table(document: &mut toml_edit::DocumentMut) -> &mut toml_edit::Table {
    if !document.contains_key("routing") {
        let mut table = toml_edit::Table::new();
        table.set_implicit(true);
        document.insert("routing", toml_edit::Item::Table(table));
    }
    document["routing"].as_table_mut().expect("routing table")
}

/// Ensure the `[providers.<label>]` table exists, creating it if needed.
fn ensure_provider_table<'a>(
    document: &'a mut toml_edit::DocumentMut,
    provider: &'a str,
) -> &'a mut toml_edit::Table {
    if !document.contains_key("providers") {
        let mut table = toml_edit::Table::new();
        table.set_implicit(true);
        document.insert("providers", toml_edit::Item::Table(table));
    }
    let providers = document["providers"]
        .as_table_mut()
        .expect("providers table");
    if !providers.contains_key(provider) {
        providers.insert(provider, toml_edit::Item::Table(toml_edit::Table::new()));
    }
    providers[provider].as_table_mut().expect("provider table")
}

fn set_string_value(table: &mut toml_edit::Table, key: &str, value: &str) {
    table.insert(key, toml_edit::value(value));
}

/// Update a field in the global `[routing]` table. Writes the file back.
pub fn set_routing_field(key: &str, value: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let routing = ensure_routing_table(&mut document);
    set_string_value(routing, key, value);
    save_config_document(&document)
}

/// Remove a field from the global `[routing]` table if present.
pub fn clear_routing_field(key: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    if let Some(routing) = document.get_mut("routing").and_then(|r| r.as_table_mut()) {
        routing.remove(key);
    }
    save_config_document(&document)
}

/// Update a field in the top-level `[otel]` table. Writes the file back.
pub fn set_otel_field(key: &str, value: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let otel = ensure_otel_table(&mut document);
    set_string_value(otel, key, value);
    save_config_document(&document)
}

/// Remove a field from the top-level `[otel]` table if present.
pub fn clear_otel_field(key: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    if let Some(otel) = document.get_mut("otel").and_then(|o| o.as_table_mut()) {
        otel.remove(key);
    }
    save_config_document(&document)
}

/// Write the entire top-level `[otel]` table from a config value.
/// This is used by the desktop app to persist the user's telemetry toggle.
pub fn set_otel_config(config: &TelemetryConfig) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let otel = ensure_otel_table(&mut document);
    set_string_value(otel, "enabled", &config.enabled.to_string());
    if let Some(endpoint) = &config.endpoint {
        set_string_value(otel, "exporter_otlp_endpoint", endpoint);
    } else {
        otel.remove("exporter_otlp_endpoint");
    }
    if let Some(service_name) = &config.service_name {
        set_string_value(otel, "service_name", service_name);
    } else {
        otel.remove("service_name");
    }
    set_string_value(otel, "insecure", &config.insecure.to_string());
    save_config_document(&document)
}

fn secret_action_name(action: engine_core::SecretAction) -> &'static str {
    match action {
        engine_core::SecretAction::Allow => "allow",
        engine_core::SecretAction::Redact => "redact",
        engine_core::SecretAction::Block => "block",
    }
}

fn string_array(values: &[String]) -> toml_edit::Array {
    let mut array = toml_edit::Array::new();
    for value in values {
        array.push(value.as_str());
    }
    array
}

/// Persist the complete local privacy policy while preserving unrelated TOML
/// comments and settings. The daemon reads this table on its next restart.
pub fn set_privacy_config(config: &engine_core::PrivacyPolicy) -> anyhow::Result<()> {
    let config = config.clone().normalized();
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let privacy = ensure_privacy_table(&mut document);
    privacy.insert("enabled", toml_edit::value(config.enabled));
    privacy.insert(
        "provider_secret_action",
        toml_edit::value(secret_action_name(config.provider_secret_action)),
    );
    privacy.insert(
        "tool_secret_action",
        toml_edit::value(secret_action_name(config.tool_secret_action)),
    );
    privacy.insert(
        "persist_reasoning",
        toml_edit::value(config.persist_reasoning),
    );
    privacy.insert(
        "persist_provider_state",
        toml_edit::value(config.persist_provider_state),
    );
    privacy.insert(
        "retention_days",
        toml_edit::value(i64::from(config.retention_days)),
    );
    privacy.insert(
        "max_persisted_string_bytes",
        toml_edit::value(config.max_persisted_string_bytes as i64),
    );
    privacy.insert(
        "redaction_replacement",
        toml_edit::value(config.redaction_replacement.as_str()),
    );
    privacy.insert(
        "additional_secret_values",
        toml_edit::value(string_array(&config.additional_secret_values)),
    );
    privacy.insert(
        "approved_providers",
        toml_edit::value(string_array(&config.approved_providers)),
    );
    save_config_document(&document)
}

/// Persist desktop notification settings while preserving unrelated TOML
/// comments and settings.
///
/// Every field is written explicitly, including the ones that match the
/// default. These are toggles somebody set on purpose, and a default that
/// later changes must not silently move a switch they left alone.
pub fn set_notifications_config(
    config: &crate::notifications::LocalNotificationSettings,
) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let notifications = ensure_notifications_table(&mut document);
    notifications.insert("enabled", toml_edit::value(config.enabled));
    notifications.insert("run_outcomes", toml_edit::value(config.run_outcomes));
    notifications.insert("agent_health", toml_edit::value(config.agent_health));
    notifications.insert("budget", toml_edit::value(config.budget));
    save_config_document(&document)
}

/// Update a model field (`standard_model`/`lite_model`/`pro_model`/`ultra_model`)
/// for a provider. Writes the file back.
pub fn set_provider_model_field(provider: &str, field: &str, value: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let table = ensure_provider_table(&mut document, provider);
    let (field, alias) = canonical_model_field(field);
    if let Some(alias) = alias {
        table.remove(alias);
    }
    set_string_value(table, field, value);
    save_config_document(&document)
}

/// Persist a provider metadata field such as the custom provider's name or
/// base URL while preserving the rest of config.toml.
pub fn set_provider_detail_field(provider: &str, field: &str, value: &str) -> anyhow::Result<()> {
    set_provider_model_field(provider, field, value)
}

/// Remove a model field from a provider's table (e.g. to let Lite be
/// auto-derived again). No-op if the field is absent. Writes the file back.
pub fn clear_provider_model_field(provider: &str, field: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    if let Some(table) = document.get_mut("providers").and_then(|p| p.as_table_mut()) {
        if let Some(provider_table) = table.get_mut(provider).and_then(|p| p.as_table_mut()) {
            let (field, alias) = canonical_model_field(field);
            if let Some(alias) = alias {
                provider_table.remove(alias);
            }
            provider_table.remove(field);
        }
    }
    save_config_document(&document)
}

fn canonical_model_field(field: &str) -> (&str, Option<&str>) {
    match field {
        "standard" | "standard_model" => ("standard_model", Some("standard")),
        "lite" | "lite_model" => ("lite_model", Some("lite")),
        "pro" | "pro_model" => ("pro_model", Some("pro")),
        "ultra" | "ultra_model" => ("ultra_model", Some("ultra")),
        other => (other, None),
    }
}

/// Update a field in a provider's `[providers.<label>.routing]` table.
/// Writes the file back.
pub fn set_provider_routing_field(provider: &str, key: &str, value: &str) -> anyhow::Result<()> {
    ensure_config_exists().ok();
    let mut document = load_config_document()?;
    let provider_table = ensure_provider_table(&mut document, provider);
    if !provider_table.contains_key("routing") {
        let mut table = toml_edit::Table::new();
        table.set_implicit(true);
        provider_table.insert("routing", toml_edit::Item::Table(table));
    }
    let routing = provider_table["routing"]
        .as_table_mut()
        .expect("routing table");
    set_string_value(routing, key, value);
    save_config_document(&document)
}

/// A read-only snapshot of one provider's configured models, for display.
#[derive(Clone, Debug, Default)]
pub struct ProviderModelSummary {
    pub standard: Option<String>,
    pub lite: Option<String>,
    pub pro: Option<String>,
    pub ultra: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CustomProviderDetails {
    pub name: Option<String>,
    pub base_url: Option<String>,
}

pub fn read_custom_provider_details() -> CustomProviderDetails {
    let config = load_local_config();
    let Some(provider) = config
        .providers
        .iter()
        .find(|(label, _)| label.eq_ignore_ascii_case("custom"))
        .map(|(_, config)| config)
    else {
        return CustomProviderDetails::default();
    };
    CustomProviderDetails {
        name: provider.name.clone(),
        base_url: provider.base_url.clone(),
    }
}

/// Read the model fields for a provider from the TOML file (not env vars).
pub fn read_provider_models(provider: &str) -> ProviderModelSummary {
    let config = load_local_config();
    let Some(provider_config) = config
        .providers
        .iter()
        .find(|(label, _)| label.eq_ignore_ascii_case(provider))
        .map(|(_, config)| config)
    else {
        return ProviderModelSummary::default();
    };
    ProviderModelSummary {
        standard: provider_config.standard_model.clone(),
        lite: provider_config.lite_model.clone(),
        pro: provider_config.pro_model.clone(),
        ultra: provider_config.ultra_model.clone(),
    }
}

/// Read the merged routing policy for a provider (global + per-provider +
/// env), for display.
pub fn read_provider_routing(provider: &str) -> RoutingPolicy {
    let config = load_local_config();
    let provider_config = config
        .providers
        .iter()
        .find(|(label, _)| label.eq_ignore_ascii_case(provider))
        .map(|(_, config)| config);
    resolve_routing_policy(&config, provider_config)
}

/// Read the global routing policy (without per-provider or env overrides),
/// for display.
pub fn read_global_routing() -> RoutingPolicy {
    let config = load_local_config();
    let mut policy = RoutingPolicy::default();
    if let Some(global) = config.routing.as_ref() {
        policy = global.clone().merge_into(policy);
    }
    policy
}

/// Resolve the hybrid model policy for one provider.
///
/// The standard model is the routing anchor. Only explicitly configured tier
/// overrides are materialized so engine-core remains free to derive a cheaper
/// Lite sibling from the curated catalog. Environment variables have highest
/// precedence and the legacy tier variables remain explicit overrides.
pub fn resolve_model_policy(
    config: &LocalConfig,
    provider: &str,
    environment: ModelEnvironment<'_>,
    built_in_standard: &str,
) -> ResolvedModelPolicy {
    let provider_config = config
        .providers
        .iter()
        .find(|(label, _)| label.eq_ignore_ascii_case(provider))
        .map(|(_, config)| config);
    let standard_model = non_empty_env(environment.standard)
        .or_else(|| {
            provider_config
                .and_then(|config| non_empty(config.standard_model.as_deref()).map(str::to_owned))
        })
        .or_else(|| non_empty(Some(built_in_standard)).map(str::to_owned));

    let mut models = Vec::with_capacity(3);
    for (tier, env_name, configured) in [
        (
            ModelTier::Lite,
            environment.lite,
            provider_config.and_then(|config| config.lite_model.as_deref()),
        ),
        (
            ModelTier::Pro,
            environment.pro,
            provider_config.and_then(|config| config.pro_model.as_deref()),
        ),
        (
            ModelTier::Ultra,
            environment.ultra,
            provider_config.and_then(|config| config.ultra_model.as_deref()),
        ),
    ] {
        if let Some(provider_model) =
            non_empty_env(env_name).or_else(|| non_empty(configured).map(str::to_owned))
        {
            models.push(ProviderModelMapping {
                tier,
                provider_model,
            });
        }
    }

    let routing = resolve_routing_policy(config, provider_config);

    ResolvedModelPolicy {
        standard_model,
        models,
        routing,
    }
}

/// Merge routing policy sources: built-in `auto` default, global `[routing]`
/// section, per-provider `[providers.<label>.routing]` override, and the
/// `REINCORE_ROUTING_MODE` env var (highest precedence for the mode). Each
/// source layers on top of the previous so a per-provider section can pin
/// only `minimum_quality` while inheriting the global `mode`.
fn resolve_routing_policy(
    config: &LocalConfig,
    provider_config: Option<&LocalProviderConfig>,
) -> RoutingPolicy {
    let mut policy = RoutingPolicy::default();
    if let Some(global) = config.routing.as_ref() {
        policy = global.clone().merge_into(policy);
    }
    if let Some(provider) = provider_config.and_then(|c| c.routing.as_ref()) {
        policy = provider.clone().merge_into(policy);
    }
    if let Ok(mode) = std::env::var("REINCORE_ROUTING_MODE") {
        if let Ok(parsed) = mode.trim().parse() {
            policy.mode = parsed;
        }
    }
    policy
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .and_then(|value| non_empty(Some(value.as_str())).map(str::to_owned))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped example is written to `~/.reincore-io/config.toml` on first
    /// run, so every value in its `[loop_detection]` section has to parse and
    /// has to agree with the built-in defaults it claims to document.
    #[test]
    fn the_example_loop_section_matches_the_built_in_defaults() {
        let documented = parse(DEFAULT_CONFIG_TOML).loop_detection.resolve();
        assert_eq!(documented, engine_core::CircuitBreakerConfig::default());

        // An absent section means "use the defaults", never "disable".
        let absent = parse("").loop_detection.resolve();
        assert_eq!(absent, engine_core::CircuitBreakerConfig::default());

        // A partial section leaves everything it does not mention alone.
        let partial = parse("[loop_detection]\nexploration_hard_stop_threshold = 30")
            .loop_detection
            .resolve();
        assert_eq!(partial.exploration_hard_stop_threshold, 30);
        assert_eq!(
            partial.exploration_repetition_threshold,
            engine_core::CircuitBreakerConfig::default().exploration_repetition_threshold
        );
        assert!(partial.enabled);
    }

    #[test]
    fn intervention_modes_parse_and_default_without_weakening_boundaries() {
        let documented = parse(DEFAULT_CONFIG_TOML).interventions;
        assert_eq!(documented, engine_core::InterventionPolicy::default());
        assert_eq!(
            parse("").interventions,
            engine_core::InterventionPolicy::default()
        );
        let custom =
            parse("[interventions]\nloop_detection = \"observe\"\naction_validation = \"warn\"")
                .interventions;
        assert_eq!(custom.loop_detection, engine_core::HeuristicMode::Observe);
        assert_eq!(custom.action_validation, engine_core::HeuristicMode::Warn);
        assert_eq!(
            custom.mode_for(engine_core::AgentHealthKind::AuthorizationDenied),
            engine_core::HeuristicMode::Block
        );
    }

    #[test]
    fn detection_defaults_to_lexical_and_example_matches() {
        let documented = parse(DEFAULT_CONFIG_TOML).detection;
        assert_eq!(
            documented,
            engine_core::detection::DetectionConfig::default()
        );
        assert_eq!(
            parse("").detection,
            engine_core::detection::DetectionConfig::default()
        );
        assert!(!parse("").detection.uses_llm());

        let llm = parse("[detection.prompt_injection]\nmode = \"llm\"").detection;
        assert!(llm.prompt_injection.mode.is_llm());
        assert!(!llm.task_boundary.mode.is_llm());
        assert!(!llm.task_type.mode.is_llm());
        assert!(llm.uses_llm());
        assert_eq!(llm.llm_model, "muse-wonder-1.0");

        let typed = parse(
            "[detection.task_boundary]\nmode = \"llm\"\n[detection.task_type]\nmode = \"llm\"",
        )
        .detection;
        assert!(typed.task_boundary.mode.is_llm());
        assert!(typed.task_type.mode.is_llm());
        assert!(!typed.prompt_injection.mode.is_llm());
    }

    #[test]
    fn claim_grounding_defaults_and_optional_verifier_are_independent() {
        let defaults = parse(DEFAULT_CONFIG_TOML);
        assert!(defaults.claim_grounding.enabled);
        assert!(!defaults.claim_grounding.verifier_enabled);
        assert_eq!(defaults.claim_grounding.verifier_token_budget, 4096);
        let configured=parse("[claim_grounding]\nenabled = false\nverifier_enabled = true\nverifier_token_budget = 2048");
        assert!(!configured.claim_grounding.enabled);
        assert!(configured.claim_grounding.verifier_enabled);
        assert_eq!(configured.claim_grounding.verifier_token_budget, 2048);
        assert!(configured.task_decisions.enabled);
    }

    #[test]
    fn claim_grounding_tool_contracts_parse_from_toml() {
        assert!(parse(DEFAULT_CONFIG_TOML)
            .claim_grounding
            .tool_contracts
            .is_empty());
        let config = parse(
            r#"
            [[claim_grounding.tool_contracts]]
            tool = "gmail__send_email"
            action = "email.send"
            target_pointer = "/to"
            completion_template = "Email sent to {target}."
            confirmation_pointer = "/id"
            "#,
        );
        let contracts = &config.claim_grounding.tool_contracts;
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].tool, "gmail__send_email");
        assert_eq!(contracts[0].confirmation_pointer.as_deref(), Some("/id"));
        assert!(engine_core::claim_grounding::validate_tool_contracts(contracts).is_ok());
    }

    #[test]
    fn task_decisions_default_on_and_can_be_disabled_independently() {
        assert!(LocalConfig::default().task_decisions.enabled);
        for input in ["", "[task_decisions]", DEFAULT_CONFIG_TOML] {
            assert!(parse(input).task_decisions.enabled);
        }
        let disabled = parse("[task_decisions]\nenabled = false");
        assert!(!disabled.task_decisions.enabled);
        assert!(!disabled.provider_replay.enabled);
        assert!(disabled.action_policy.default_allow);
        let enabled = parse("[task_decisions]\nenabled = true\n[provider_replay]\nenabled = false");
        assert!(enabled.task_decisions.enabled);
        assert!(!enabled.provider_replay.enabled);
        assert!(Config::builder()
            .add_source(File::from_str(
                "[task_decisions]\nenabled = []",
                FileFormat::Toml
            ))
            .build()
            .unwrap()
            .try_deserialize::<LocalConfig>()
            .is_err());
    }

    #[test]
    fn provider_replay_defaults_off_and_accepts_explicit_boolean() {
        assert!(!LocalConfig::default().provider_replay.enabled);
        assert!(!parse("").provider_replay.enabled);
        assert!(!parse("[provider_replay]").provider_replay.enabled);
        assert!(!parse(DEFAULT_CONFIG_TOML).provider_replay.enabled);
        assert!(
            !parse("[provider_replay]\nenabled = false")
                .provider_replay
                .enabled
        );
        assert!(
            parse("[provider_replay]\nenabled = true")
                .provider_replay
                .enabled
        );
    }

    #[test]
    fn privacy_defaults_are_documented_and_local_overrides_parse() {
        let defaults = engine_core::PrivacyPolicy::default();
        let documented = parse(DEFAULT_CONFIG_TOML).privacy;
        assert!(documented.enabled);
        assert_eq!(
            documented.provider_secret_action,
            defaults.provider_secret_action
        );
        assert_eq!(documented.tool_secret_action, defaults.tool_secret_action);
        assert!(!documented.persist_reasoning);
        assert!(!documented.persist_provider_state);
        assert_eq!(documented.retention_days, 30);
        assert_eq!(documented.max_persisted_string_bytes, 262_144);
        assert_eq!(documented.redaction_replacement, "[REDACTED]");
        assert!(documented.additional_secret_values.is_empty());
        assert!(documented.approved_providers.is_empty());

        let configured = parse(
            r#"
            [privacy]
            provider_secret_action = "block"
            tool_secret_action = "allow"
            persist_reasoning = true
            retention_days = 7
            approved_providers = ["anthropic"]
            "#,
        )
        .privacy;
        assert_eq!(
            configured.provider_secret_action,
            engine_core::SecretAction::Block
        );
        assert_eq!(
            configured.tool_secret_action,
            engine_core::SecretAction::Allow
        );
        assert!(configured.persist_reasoning);
        assert_eq!(configured.retention_days, 7);
        assert_eq!(configured.approved_providers, ["anthropic"]);
    }

    /// The shipped example is written to `~/.reincore-io/config.toml` on first
    /// run, so what it documents has to be what the code actually defaults to.
    ///
    /// The partial case is the one worth guarding: each field carries its own
    /// serde default precisely so a table that sets one key leaves the other
    /// three where `Default` put them. A struct-level `#[serde(default)]`
    /// would instead put them at `false`, silently switching off two
    /// categories for anyone who only meant to turn one on.
    #[test]
    fn notification_defaults_are_documented_and_partial_tables_keep_the_rest() {
        let defaults = crate::notifications::LocalNotificationSettings::default();
        let documented = parse(DEFAULT_CONFIG_TOML).notifications;
        assert_eq!(documented.enabled, defaults.enabled);
        assert_eq!(documented.run_outcomes, defaults.run_outcomes);
        assert_eq!(documented.agent_health, defaults.agent_health);
        assert_eq!(documented.budget, defaults.budget);

        // An absent section means "use the defaults", never "show nothing".
        let absent = parse("").notifications;
        assert!(absent.enabled);
        assert!(absent.run_outcomes);
        assert!(absent.budget);
        assert!(!absent.agent_health);

        let partial = parse("[notifications]\nagent_health = true").notifications;
        assert!(partial.agent_health);
        assert!(partial.enabled, "enabled was not left at its default");
        assert!(
            partial.run_outcomes,
            "run_outcomes was not left at its default"
        );
        assert!(partial.budget, "budget was not left at its default");

        let off = parse("[notifications]\nenabled = false").notifications;
        assert!(!off.enabled);
        assert!(
            off.run_outcomes,
            "a master switch must not clear a category"
        );
    }

    fn parse(input: &str) -> LocalConfig {
        Config::builder()
            .add_source(File::from_str(input, FileFormat::Toml))
            .build()
            .unwrap()
            .try_deserialize()
            .unwrap()
    }

    fn unused_environment() -> ModelEnvironment<'static> {
        ModelEnvironment {
            standard: "REINCORE_TEST_UNUSED_STANDARD_MODEL_7F45",
            lite: "REINCORE_TEST_UNUSED_LITE_MODEL_7F45",
            pro: "REINCORE_TEST_UNUSED_PRO_MODEL_7F45",
            ultra: "REINCORE_TEST_UNUSED_ULTRA_MODEL_7F45",
        }
    }

    #[test]
    fn standard_model_is_separate_from_explicit_tier_overrides() {
        let config = parse(
            r#"
                [providers.openai]
                standard_model = "gpt-standard"
                lite_model = "gpt-small"
            "#,
        );
        let policy = resolve_model_policy(&config, "OPENAI", unused_environment(), "built-in-pro");

        assert_eq!(policy.standard_model.as_deref(), Some("gpt-standard"));
        assert_eq!(policy.models.len(), 1);
        assert_eq!(policy.models[0].tier, ModelTier::Lite);
        assert_eq!(policy.models[0].provider_model, "gpt-small");
    }

    #[test]
    fn built_in_pro_becomes_standard_without_explicit_configuration() {
        let policy = resolve_model_policy(
            &LocalConfig::default(),
            "openai",
            unused_environment(),
            "built-in-pro",
        );

        assert_eq!(policy.standard_model.as_deref(), Some("built-in-pro"));
        assert!(policy.models.is_empty());
    }

    #[test]
    fn short_tier_aliases_are_accepted() {
        let config = parse(
            r#"
                [providers.anthropic]
                standard = "claude-standard"
                ultra = "claude-largest"
            "#,
        );
        let policy = resolve_model_policy(&config, "anthropic", unused_environment(), "pro");

        assert_eq!(policy.standard_model.as_deref(), Some("claude-standard"));
        assert_eq!(policy.models.len(), 1);
        assert_eq!(policy.models[0].tier, ModelTier::Ultra);
        assert_eq!(policy.models[0].provider_model, "claude-largest");
    }

    #[test]
    fn legacy_tier_environment_variable_remains_an_explicit_override() {
        const LITE_ENV: &str = "REINCORE_TEST_LEGACY_LITE_MODEL_91D2";
        std::env::set_var(LITE_ENV, "env-lite");
        let environment = ModelEnvironment {
            standard: "REINCORE_TEST_LEGACY_STANDARD_MODEL_91D2",
            lite: LITE_ENV,
            pro: "REINCORE_TEST_LEGACY_PRO_MODEL_91D2",
            ultra: "REINCORE_TEST_LEGACY_ULTRA_MODEL_91D2",
        };
        let config = parse(
            r#"
                [providers.openai]
                standard = "toml-standard"
                lite = "toml-lite"
            "#,
        );

        let policy = resolve_model_policy(&config, "openai", environment, "built-in-pro");
        std::env::remove_var(LITE_ENV);

        assert_eq!(policy.standard_model.as_deref(), Some("toml-standard"));
        assert_eq!(policy.models.len(), 1);
        assert_eq!(policy.models[0].provider_model, "env-lite");
    }

    #[test]
    fn global_routing_section_sets_default_mode() {
        let config = parse(
            r#"
                [routing]
                mode = "locked"
                minimum_quality = "high"
            "#,
        );
        let policy = resolve_model_policy(&config, "openai", unused_environment(), "pro");

        assert_eq!(policy.routing.mode, engine_core::RoutingMode::Locked);
        assert_eq!(
            policy.routing.minimum_quality,
            engine_core::MinimumQuality::High
        );
    }

    #[test]
    fn per_provider_routing_overrides_global_section() {
        let config = parse(
            r#"
                [routing]
                mode = "auto"
                minimum_quality = "economy"

                [providers.openai.routing]
                mode = "custom"
            "#,
        );
        let policy = resolve_model_policy(&config, "openai", unused_environment(), "pro");

        assert_eq!(policy.routing.mode, engine_core::RoutingMode::Custom);
        // Inherited from the global section because the per-provider override
        // did not set it.
        assert_eq!(
            policy.routing.minimum_quality,
            engine_core::MinimumQuality::Economy
        );
    }

    #[test]
    fn per_provider_allow_degradation_overrides_global() {
        let config = parse(
            r#"
                [routing]
                allow_degradation = false

                [providers.openai.routing]
                allow_degradation = true
            "#,
        );
        let policy = resolve_model_policy(&config, "openai", unused_environment(), "pro");
        assert!(policy.routing.allow_degradation);

        // A provider that does not set it inherits the global value.
        let config = parse(
            r#"
                [routing]
                allow_degradation = false
            "#,
        );
        let policy = resolve_model_policy(&config, "anthropic", unused_environment(), "pro");
        assert!(!policy.routing.allow_degradation);
    }

    #[test]
    fn default_routing_policy_is_auto() {
        let policy = resolve_model_policy(
            &LocalConfig::default(),
            "openai",
            unused_environment(),
            "pro",
        );
        assert_eq!(policy.routing.mode, engine_core::RoutingMode::Auto);
        assert!(policy.routing.allow_degradation);
    }

    /// Isolate config-file mutations behind a unique REINCORE_CONFIG_PATH so
    /// parallel tests do not clobber each other's files. The env var is
    /// removed on drop. A process-wide mutex serializes these tests because
    /// they all share the single `REINCORE_CONFIG_PATH` env var.
    struct ConfigPathGuard {
        var: &'static str,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    static CONFIG_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    impl ConfigPathGuard {
        fn new(suffix: &str) -> Self {
            let lock = CONFIG_TEST_MUTEX
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let var = "REINCORE_CONFIG_PATH";
            let dir = std::env::temp_dir().join(format!(
                "reincore-config-test-{}-{suffix}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("config.toml");
            // Start from a clean slate so prior test runs don't leak.
            let _ = std::fs::remove_file(&path);
            std::env::set_var(var, &path);
            Self { var, _guard: lock }
        }

        fn path(&self) -> PathBuf {
            local_config_path()
        }
    }

    impl Drop for ConfigPathGuard {
        fn drop(&mut self) {
            std::env::remove_var(self.var);
        }
    }

    #[test]
    fn ensure_config_exists_writes_default_template() {
        let guard = ConfigPathGuard::new("ensure");
        let path = guard.path();
        assert!(!path.exists());

        ensure_config_exists().unwrap();
        assert!(path.exists());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("[routing]"));
        assert!(contents.contains("mode = \"auto\""));
        assert!(contents.contains("[privacy]"));
        assert!(contents.contains("retention_days = 30"));

        // Calling again is a no-op (does not overwrite).
        std::fs::write(&path, "# custom marker\n").unwrap();
        ensure_config_exists().unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("# custom marker"));
    }

    #[test]
    fn set_routing_field_updates_and_persists() {
        let guard = ConfigPathGuard::new("routing");
        ensure_config_exists().unwrap();

        set_routing_field("mode", "locked").unwrap();
        set_routing_field("minimum_quality", "high").unwrap();

        let policy = read_global_routing();
        assert_eq!(policy.mode, engine_core::RoutingMode::Locked);
        assert_eq!(policy.minimum_quality, engine_core::MinimumQuality::High);

        // The file on disk reflects the change.
        let contents = std::fs::read_to_string(guard.path()).unwrap();
        assert!(contents.contains("mode = \"locked\""));
        assert!(contents.contains("minimum_quality = \"high\""));
    }

    #[test]
    fn set_privacy_config_round_trips_every_app_setting() {
        let guard = ConfigPathGuard::new("privacy");
        ensure_config_exists().unwrap();
        let policy = engine_core::PrivacyPolicy {
            enabled: false,
            provider_secret_action: engine_core::SecretAction::Block,
            tool_secret_action: engine_core::SecretAction::Allow,
            persist_reasoning: true,
            persist_provider_state: true,
            retention_days: 14,
            max_persisted_string_bytes: 8192,
            redaction_replacement: "[PRIVATE]".to_string(),
            additional_secret_values: vec!["string".to_string(), "exact-secret".to_string()],
            approved_providers: vec!["anthropic".to_string(), "openai".to_string()],
        };

        set_privacy_config(&policy).unwrap();

        let saved = load_local_config().privacy;
        assert!(!saved.enabled);
        assert_eq!(
            saved.provider_secret_action,
            engine_core::SecretAction::Block
        );
        assert_eq!(saved.tool_secret_action, engine_core::SecretAction::Allow);
        assert!(saved.persist_reasoning);
        assert!(saved.persist_provider_state);
        assert_eq!(saved.retention_days, 14);
        assert_eq!(saved.max_persisted_string_bytes, 8192);
        assert_eq!(saved.redaction_replacement, "[PRIVATE]");
        assert_eq!(saved.additional_secret_values, ["exact-secret"]);
        assert_eq!(saved.approved_providers, ["anthropic", "openai"]);

        let contents = std::fs::read_to_string(guard.path()).unwrap();
        assert!(contents.contains("# Copy this file"));
        assert!(contents.contains("provider_secret_action = \"block\""));
        assert!(contents.contains("additional_secret_values = [\"exact-secret\"]"));
    }

    #[test]
    fn set_provider_model_field_creates_provider_section() {
        let guard = ConfigPathGuard::new("model");
        ensure_config_exists().unwrap();

        set_provider_model_field("openai", "standard_model", "gpt-team-standard").unwrap();
        set_provider_model_field("openai", "lite_model", "gpt-team-lite").unwrap();

        let models = read_provider_models("openai");
        assert_eq!(models.standard.as_deref(), Some("gpt-team-standard"));
        assert_eq!(models.lite.as_deref(), Some("gpt-team-lite"));

        let contents = std::fs::read_to_string(guard.path()).unwrap();
        assert!(contents.contains("standard_model = \"gpt-team-standard\""));
    }

    #[test]
    fn custom_provider_details_round_trip() {
        let _guard = ConfigPathGuard::new("custom-provider");
        ensure_config_exists().unwrap();
        set_provider_detail_field("custom", "name", "OpenRouter").unwrap();
        set_provider_detail_field("custom", "base_url", "https://openrouter.ai/api/v1").unwrap();

        let details = read_custom_provider_details();
        assert_eq!(details.name.as_deref(), Some("OpenRouter"));
        set_provider_model_field("custom", "standard_model", "vendor/original").unwrap();
        set_provider_model_field("custom", "standard", "vendor/updated").unwrap();
        assert_eq!(
            read_provider_models("custom").standard.as_deref(),
            Some("vendor/updated")
        );
        assert_eq!(
            details.base_url.as_deref(),
            Some("https://openrouter.ai/api/v1")
        );
    }

    #[test]
    fn clear_provider_model_field_removes_override() {
        let _guard = ConfigPathGuard::new("clear");
        ensure_config_exists().unwrap();
        set_provider_model_field("openai", "lite_model", "pinned-lite").unwrap();
        assert_eq!(
            read_provider_models("openai").lite.as_deref(),
            Some("pinned-lite")
        );

        clear_provider_model_field("openai", "lite_model").unwrap();
        assert!(read_provider_models("openai").lite.is_none());
    }

    #[test]
    fn set_provider_routing_field_creates_nested_table() {
        let guard = ConfigPathGuard::new("provider-routing");
        ensure_config_exists().unwrap();

        set_provider_routing_field("openai", "mode", "custom").unwrap();

        let policy = read_provider_routing("openai");
        assert_eq!(policy.mode, engine_core::RoutingMode::Custom);

        let contents = std::fs::read_to_string(guard.path()).unwrap();
        assert!(contents.contains("[providers.openai.routing]"));
        assert!(contents.contains("mode = \"custom\""));
    }

    #[test]
    fn edits_preserve_unrelated_comments_and_sections() {
        let guard = ConfigPathGuard::new("preserve");
        ensure_config_exists().unwrap();
        // The default template ships with commented examples and an
        // [providers.anthropic] section. Edits to openai must not drop them.
        set_provider_model_field("openai", "standard_model", "gpt-edited").unwrap();

        let contents = std::fs::read_to_string(guard.path()).unwrap();
        assert!(contents.contains("[providers.anthropic]"));
        assert!(contents.contains("# Copy this file"));
    }
}
