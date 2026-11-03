//! OpenTelemetry pipeline tracing for the ReinCore engine.
//!
//! This module wires the tracing → OpenTelemetry bridge and emits a span
//! for every stage of the chat-completion pipeline:
//!
//! ```text
//! reincore.request (root span - the HTTP handler)
//! ├── reincore.stage.resolve    (session / file / task resolution)
//! ├── reincore.stage.classify   (escalation classifier)
//! ├── reincore.stage.predict    (pre-flight cost prediction)
//! ├── reincore.stage.route      (budget degradation + model-tier routing)
//! ├── reincore.stage.compact    (context compression / rollover / handoff)
//! ├── reincore.stage.provider   (upstream request until stream is opened)
//! ├── reincore.stage.stream     (provider response stream consumption)
//! ├── reincore.stage.health     (agent-health scanning post-response)
//! └── reincore.stage.account    (event-log write, budget ledger, calibration)
//! ```
//!
//! ## Activation
//!
//! Tracing is **opt-in** and controlled by environment variables. When none
//! are set the OTEL layer is **not** installed and all spans are no-ops
//! (sub-microsecond `tracing` macro calls with a disabled layer).
//!
//! | Variable                        | Effect                                               |
//! |---------------------------------|------------------------------------------------------|
//! | `OTEL_EXPORTER_OTLP_ENDPOINT`   | gRPC endpoint (enables the exporter).               |
//! | `OTEL_SERVICE_NAME`             | Overrides `service.name` resource attribute.         |
//! | `REINCORE_OTEL_INSECURE`        | `"true"` / `"1"` → `http://`, else `https://`.       |
//!
//! The OTLP exporter uses gRPC with `tonic` transport (default port 4317).
//!
//! ## Usage
//!
//! ```rust,ignore
//! // At process start (before any handler):
//! let _guard = engine_core::telemetry::maybe_init("reincore-local")?;
//!
//! // In a handler:
//! let request_span = engine_core::telemetry::pipeline_root_span("chat_completions", session_id);
//! let _root_guard = request_span.enter();
//! engine_core::stage!(Resolve);
//! // ... do work ...
//! // `_stage_guard` drops at end of block → span closes → OTLP export
//! ```

use futures::Stream;
use opentelemetry::{propagation::Extractor, trace::TraceContextExt};
use std::{
    pin::Pin,
    task::{Context as TaskContext, Poll},
};
use tracing::{field, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::prelude::*;

/// Canonical pipeline stage identifiers emitted as `reincore.pipeline.stage`
/// span attributes. These are stable strings - dashboards may key on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Session / file / task and request-group resolution.
    Resolve,
    /// Escalation classifier (prompt classification → model-tier).
    Classify,
    /// Pre-flight cost prediction (token counts + price range).
    Predict,
    /// Budget degradation, provider selection, model-tier resolution.
    Route,
    /// Context compression, rollover detection, session handoff.
    Compact,
    /// Upstream provider call (sync or streaming first chunk).
    Provider,
    /// Consume the provider's streaming response until its terminal chunk.
    Stream,
    /// Post-response agent-health and citation scanning.
    Health,
    /// Event-log writes, budget ledger updates, calibration metrics.
    Account,
}

impl Stage {
    /// Stable short name for span naming (`reincore.stage.resolve`, ...).
    pub fn span_name(self) -> &'static str {
        match self {
            Stage::Resolve => "reincore.stage.resolve",
            Stage::Classify => "reincore.stage.classify",
            Stage::Predict => "reincore.stage.predict",
            Stage::Route => "reincore.stage.route",
            Stage::Compact => "reincore.stage.compact",
            Stage::Provider => "reincore.stage.provider",
            Stage::Stream => "reincore.stage.stream",
            Stage::Health => "reincore.stage.health",
            Stage::Account => "reincore.stage.account",
        }
    }
}

/// Create the root span for an incoming chat-completion request.
///
/// The returned span is the trace root for the entire pipeline. All stage
/// spans created with [`stage_span`] become children of whichever span is
/// entered when they are created - typically this root.
///
/// `handler` is a short human name (e.g. `"chat_completions"`). `session_id`
/// is recorded as the `session_id` attribute so traces can be correlated
/// with the ReinCore session / event-log views.
pub fn pipeline_root_span(handler: &str, session_id: Option<uuid::Uuid>) -> Span {
    let session_id = session_id.map(|id| id.to_string()).unwrap_or_default();
    tracing::info_span!(
        "reincore.request",
        otel.name = %format!("reincore.{}", handler),
        otel.kind = ?opentelemetry::trace::SpanKind::Server,
        otel.status_code = field::Empty,
        otel.status_message = field::Empty,
        session_id = %session_id,
        handler = handler,
        http.request.method = field::Empty,
        http.route = field::Empty,
        http.response.status_code = field::Empty,
        error.type = field::Empty,
        reincore.request_group_id = field::Empty,
    )
}

/// Create a child span for a single pipeline stage.
///
/// The span is created within the *current* ambient span context, so
/// calling this while the root request span is entered links it as a
/// child. Stage-specific attributes can be recorded later via
/// [`Span::record`] - do not record large payloads (they will be exported
/// into the trace backend).
pub fn stage_span(stage: Stage) -> Span {
    tracing::info_span!(
        "reincore.pipeline.stage",
        otel.name = stage.span_name(),
        otel.kind = ?opentelemetry::trace::SpanKind::Internal,
        otel.status_code = field::Empty,
        otel.status_message = field::Empty,
        reincore.pipeline.stage = stage.span_name(),
        reincore.session_id = field::Empty,
        reincore.request_group_id = field::Empty,
        reincore.classified_tier = field::Empty,
        reincore.budget_tier = field::Empty,
        reincore.final_tier = field::Empty,
        reincore.rolled_over = field::Empty,
        reincore.prefix_hash = field::Empty,
        reincore.predicted_cost_high_cents = field::Empty,
        reincore.predicted_input_tokens = field::Empty,
        reincore.provider_ms = field::Empty,
        reincore.provider_first_chunk_ms = field::Empty,
        reincore.provider_stream_ms = field::Empty,
        reincore.provider_stream_upstream_wait_ms = field::Empty,
        reincore.provider_stream_downstream_wait_ms = field::Empty,
        reincore.provider_stream_chunks = field::Empty,
        reincore.provider_stream_content_chars = field::Empty,
        reincore.provider_stream_reasoning_chars = field::Empty,
        reincore.provider_stream_tool_calls = field::Empty,
        reincore.provider = field::Empty,
        reincore.model = field::Empty,
        reincore.health_findings = field::Empty,
        reincore.actual_cost_cents = field::Empty,
        reincore.cost_nano_usd = field::Empty,
        reincore.cost_status = field::Empty,
        reincore.pricing_version = field::Empty,
        reincore.actual_to_predicted_high_ratio_bps = field::Empty,
        gen_ai.usage.input_tokens = field::Empty,
        gen_ai.usage.output_tokens = field::Empty,
        error.type = field::Empty,
    )
}

/// Convenience: create a stage span and return the entered guard.
///
/// Dropping the guard closes the span and exports it if OTEL is active.
/// This helper is only safe for synchronous scopes. Do not hold the returned
/// guard across `.await`; use `tracing::Instrument` for asynchronous work.
#[must_use]
pub fn enter_stage(stage: Stage) -> tracing::span::EnteredSpan {
    let span = stage_span(stage);
    span.entered()
}

/// Extract the W3C `traceparent` value from the *current* span context.
///
/// Useful when the engine must propagate the OTEL trace ID to downstream
/// callers (e.g. storing the trace ID alongside an event for cross-referencing
/// in the Trace Explorer, or injecting it into a provider request header
/// when the upstream supports distributed tracing).
///
/// Returns `None` when the current span is not part of an OTEL trace
/// (e.g. OTEL is disabled or the span was sampled out).
pub fn current_traceparent() -> Option<String> {
    let span = Span::current();
    let ctx = span.context();
    let otel_ctx = ctx.span().span_context().clone();
    if otel_ctx.is_valid() {
        Some(format!(
            "00-{:032x}-{:016x}-{:02x}",
            otel_ctx.trace_id(),
            otel_ctx.span_id(),
            otel_ctx.trace_flags() & opentelemetry::trace::TraceFlags::SAMPLED
        ))
    } else {
        None
    }
}

struct TraceHeaderCarrier<'a> {
    traceparent: Option<&'a str>,
    tracestate: Option<&'a str>,
}

impl Extractor for TraceHeaderCarrier<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        if key.eq_ignore_ascii_case("traceparent") {
            self.traceparent
        } else if key.eq_ignore_ascii_case("tracestate") {
            self.tracestate
        } else {
            None
        }
    }

    fn keys(&self) -> Vec<&str> {
        let mut keys = Vec::with_capacity(2);
        if self.traceparent.is_some() {
            keys.push("traceparent");
        }
        if self.tracestate.is_some() {
            keys.push("tracestate");
        }
        keys
    }
}

/// Attach an incoming W3C trace context to a newly-created server span.
pub fn set_parent_from_trace_headers(
    span: &Span,
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) {
    let carrier = TraceHeaderCarrier {
        traceparent,
        tracestate,
    };
    opentelemetry::global::get_text_map_propagator(|propagator| {
        let parent = propagator.extract(&carrier);
        if parent.span().span_context().is_valid() {
            let _ = span.set_parent(parent);
        }
    });
}

/// Record HTTP completion semantics on a request span.
pub fn record_http_status(span: &Span, status: u16) {
    span.record("http.response.status_code", status as i64);
    if status >= 500 {
        span.record("otel.status_code", "error");
        span.record("error.type", status.to_string());
    }
}

/// Mark a span as failed with a low-cardinality error type.
pub fn record_error(span: &Span, error_type: &str, message: &str) {
    span.record("otel.status_code", "error");
    span.record("otel.status_message", message);
    span.record("error.type", error_type);
}

/// A stream wrapper that enters `poll_span` only while the stream is polled.
/// Retained spans stay open until the response stream completes or is dropped.
pub struct InstrumentedStream<S> {
    inner: Pin<Box<S>>,
    poll_span: Span,
    _retained_spans: Vec<Span>,
}

pub fn instrument_stream<S>(
    stream: S,
    poll_span: Span,
    retained_spans: Vec<Span>,
) -> InstrumentedStream<S>
where
    S: Stream,
{
    InstrumentedStream {
        inner: Box::pin(stream),
        poll_span,
        _retained_spans: retained_spans,
    }
}

impl<S> Stream for InstrumentedStream<S>
where
    S: Stream,
{
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let _guard = this.poll_span.enter();
        this.inner.as_mut().poll_next(cx)
    }
}

/// Initialize tracing + OpenTelemetry if environment configuration is present.
///
/// Returns a guard that must be kept alive for the lifetime of the process
/// (or the OTLP exporter may shut down and drop in-flight spans). Returns
/// `Ok(None)` when OTEL is not configured - in that case only the std-tracing
/// layer is activated.
///
/// This function never fails hard - misconfiguration of the OTLP endpoint is
/// logged as a warning and falls back to plain std-tracing so the engine can
/// always start.
///
/// `service_name` is used for the `service.name` resource attribute and is
/// overridden by the standard `OTEL_SERVICE_NAME` env var when set.
///
/// Note: `install_batch` requires the caller to be inside a Tokio runtime.
pub fn maybe_init(service_name: &str) -> Option<TelemetryGuard> {
    // Resolve config from env, keeping precedence: explicit env vars > config.
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let insecure = std::env::var("REINCORE_OTEL_INSECURE")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true);
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| service_name.to_string());
    maybe_init_with_config(service_name, endpoint, insecure)
}

/// Initialize tracing + OpenTelemetry with explicit values, bypassing env vars.
///
/// This is used by the local daemon after reading OTLP settings from
/// `~/.reincore-io/config.toml`. Pass `None` as `endpoint` to disable OTEL export
/// and only install the fmt layer.
///
/// When `endpoint` is set and `health_check` is `true`, the function first
/// verifies the OTLP backend is reachable (and, for the bundled Jaeger
/// endpoint, that the container is running). If the check fails, the OTLP
/// exporter is not installed and plain tracing is used instead. This avoids
/// repeated gRPC "service unavailable" / "tcp connect error" log spam.
pub fn maybe_init_with_config(
    service_name: String,
    endpoint: Option<String>,
    insecure: bool,
) -> Option<TelemetryGuard> {
    maybe_init_with_config_inner(service_name, endpoint, insecure, true)
}

/// Same as [`maybe_init_with_config`] but allows disabling the pre-flight health
/// check. Used by callers that already verified reachability (e.g. after
/// `reincore optl start` returned successfully).
pub fn maybe_init_with_config_skip_health_check(
    service_name: String,
    endpoint: Option<String>,
    insecure: bool,
) -> Option<TelemetryGuard> {
    maybe_init_with_config_inner(service_name, endpoint, insecure, false)
}

fn maybe_init_with_config_inner(
    service_name: String,
    endpoint: Option<String>,
    insecure: bool,
    health_check: bool,
) -> Option<TelemetryGuard> {
    let service_name = if service_name.trim().is_empty() {
        "reincore-local".to_string()
    } else {
        service_name
    };
    let endpoint = endpoint
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // The Tauri desktop app links the daemon in-process (dev mode) and
    // installs its own subscriber via `tracing_subscriber::fmt::init()`
    // before the daemon starts. Trying to install another global subscriber
    // then fails with a scary warning ("a global default trace dispatcher has
    // already been set"). When a subscriber already exists, keep it - the
    // daemon's spans then flow through the host's subscriber, which is
    // exactly what the dev-mode console output expects.
    if tracing::dispatcher::has_been_set() {
        tracing::info!(
            service = %service_name,
            "global tracing subscriber already installed - reusing it"
        );
        return None;
    }

    // Build the base fmt layer (always present)
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_filter(env_filter);

    // When no endpoint is configured we install only the fmt layer and
    // return early - this keeps OTEL spans as sub-microsecond no-ops.
    let Some(endpoint) = endpoint else {
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        // Best-effort: if another subscriber already exists (e.g. the Tauri
        // desktop app in dev mode), keep going without OTEL.
        if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
            eprintln!("warning: failed to install tracing subscriber: {error}");
        }
        tracing::info!("OpenTelemetry disabled: no OTLP endpoint configured");
        return None;
    };

    // Pre-flight health check: only install the OTLP exporter if the backend
    // is actually reachable. This prevents noisy gRPC errors when a user has
    // enabled OTEL but the collector/jaeger container is not running.
    if health_check {
        let health = check_otel_health_without_nesting_runtime(endpoint.clone());
        if !health.is_healthy() {
            tracing::warn!("{}", health.message());
            let subscriber = tracing_subscriber::registry().with(fmt_layer);
            let _ = tracing::subscriber::set_global_default(subscriber);
            return None;
        }
    }

    // Build the OTLP exporter pipeline
    let url = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint
    } else if insecure {
        format!("http://{endpoint}")
    } else {
        format!("https://{endpoint}")
    };

    use opentelemetry_otlp::WithExportConfig;
    let exporter_builder = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(url.clone());

    let deployment_environment = std::env::var("OTEL_DEPLOYMENT_ENVIRONMENT")
        .or_else(|_| std::env::var("DEPLOYMENT_ENVIRONMENT"))
        .unwrap_or_else(|_| "local".to_string());
    let service_version = std::env::var("REINCORE_SERVICE_VERSION")
        .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string());
    let resource = opentelemetry_sdk::Resource::new(vec![
        opentelemetry::KeyValue::new("service.name", service_name.clone()),
        opentelemetry::KeyValue::new("service.version", service_version),
        opentelemetry::KeyValue::new("deployment.environment.name", deployment_environment),
        opentelemetry::KeyValue::new(
            "service.instance.id",
            format!("{}-{}", service_name, std::process::id()),
        ),
    ]);

    // Config::default honors OTEL_TRACES_SAMPLER and
    // OTEL_TRACES_SAMPLER_ARG; its default is parent-based always-on.
    let trace_config = opentelemetry_sdk::trace::Config::default().with_resource(resource);

    let provider_result = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter_builder)
        .with_trace_config(trace_config)
        .install_batch(opentelemetry_sdk::runtime::Tokio);

    let provider = match provider_result {
        Ok(p) => p,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "failed to build OTLP exporter; continuing without OpenTelemetry"
            );
            let subscriber = tracing_subscriber::registry().with(fmt_layer);
            let _ = tracing::subscriber::set_global_default(subscriber);
            return None;
        }
    };

    let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, service_name.clone());
    opentelemetry::global::set_text_map_propagator(
        opentelemetry_sdk::propagation::TraceContextPropagator::new(),
    );

    // Keep exporter and HTTP/2 internals out of the trace pipeline. RUST_LOG
    // continues to control console output independently.
    let otel_filter = tracing_subscriber::EnvFilter::try_from_env("REINCORE_OTEL_FILTER")
        .unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("off,engine_core=info,reincore=info")
        });
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_error_events_to_status(true)
        .with_error_events_to_exceptions(true)
        .with_filter(otel_filter);

    let subscriber = tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_layer);
    if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("warning: failed to install OpenTelemetry subscriber: {error}");
        return None;
    }
    tracing::info!(
        endpoint = %url,
        service = %service_name,
        "OpenTelemetry tracing enabled"
    );

    Some(TelemetryGuard { provider })
}

/// Run the asynchronous OTLP pre-flight check without ever calling
/// `Handle::block_on` on a thread that is already driving Tokio. The public
/// initializer is synchronous, so a dedicated current-thread runtime keeps
/// it safe for both synchronous binaries and async daemon startup paths.
fn check_otel_health_without_nesting_runtime(
    endpoint: String,
) -> crate::otel_health::OpenTelemetryHealth {
    let is_bundled_jaeger = endpoint == "http://localhost:4317";
    let fallback_endpoint = endpoint.clone();
    std::thread::Builder::new()
        .name("reincore-otel-health".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(runtime.block_on(crate::otel_health::check(
                crate::otel_health::OpenTelemetryHealthConfig {
                    endpoint,
                    is_bundled_jaeger,
                },
            )))
        })
        .and_then(|thread| {
            thread.join().map_err(|_| {
                std::io::Error::other("OpenTelemetry health-check thread panicked")
            })
        })
        .ok()
        .and_then(Result::ok)
        .unwrap_or_else(|| crate::otel_health::OpenTelemetryHealth::Unhealthy {
            reason: format!(
                "OpenTelemetry endpoint {fallback_endpoint} could not be checked; trace export disabled"
            ),
            needs_docker_check: is_bundled_jaeger,
        })
}

/// RAII guard for the OTLP exporter.
///
/// Dropping the guard shuts down the batch exporter and flushes any
/// in-flight spans. Keep the returned value alive for the lifetime of the
/// process - typically assign it to a `let _telemetry_guard = ...` in `main`.
#[must_use]
pub struct TelemetryGuard {
    provider: opentelemetry_sdk::trace::TracerProvider,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Best-effort flush - individual exporter failures are logged but
        // do not panic.
        for result in self.provider.force_flush() {
            if let Err(e) = result {
                eprintln!(
                    "warning: failed to flush OpenTelemetry spans on shutdown: {}",
                    e
                );
            }
        }
    }
}

/// Shorthand macro: create + enter a stage span in one line.
///
/// Expands to `let _stage_guard = telemetry::enter_stage(Stage::$variant);`
/// so the span stays open until the end of the enclosing block. The
/// underscore-prefixed variable makes it clear this is a hold-until-drop
/// guard, not a value to be used later.
#[macro_export]
macro_rules! stage {
    ($variant:ident) => {
        let _stage_guard = $crate::telemetry::enter_stage($crate::telemetry::Stage::$variant);
    };
    // Allow overriding the auto-generated binding name (useful when two
    // consecutive stages must live inside the same block).
    ($variant:ident as $guard:ident) => {
        let $guard = $crate::telemetry::enter_stage($crate::telemetry::Stage::$variant);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_health_check_is_safe_inside_tokio_runtime() {
        let health = check_otel_health_without_nesting_runtime("http://127.0.0.1:1".to_string());
        assert!(!health.is_healthy());
    }
}

/// Attribute estimates explicitly; unknown usage is never emitted as free spend.
pub fn record_cost(span: &tracing::Span, cost: &crate::pricing::CostEstimate) {
    if let Some(amount) = cost.amount_nano_usd {
        span.record("reincore.cost_nano_usd", amount);
        span.record(
            "reincore.actual_cost_cents",
            amount as f64 / crate::pricing::NANO_USD_PER_CENT as f64,
        );
    }
    span.record(
        "reincore.cost_status",
        match cost.status {
            crate::pricing::CostStatus::Estimated => "estimated",
            crate::pricing::CostStatus::Unknown => "unknown",
        },
    );
    span.record("reincore.pricing_version", cost.pricing_version.as_str());
}
