//! Bounded, process-local operational metrics. No request or tenant labels.
use std::{
    collections::BTreeMap,
    fmt::Write,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

const BUCKETS: [f64; 12] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 30.0, 120.0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Metric {
    ChatHandler,
    GatewayPrepare,
    GatewayFinalize,
    ProviderOperation,
    ProviderStreamUpstreamWait,
    ProviderStreamDownstreamWait,
    ProviderFirstOutput,
    GatewayFirstOutput,
    DatabaseProbe,
    OutboxSample,
    RelayPublish,
}

impl Metric {
    fn name(self) -> &'static str {
        match self {
            Self::ChatHandler => "reincore_chat_handler_seconds",
            Self::GatewayPrepare => "reincore_gateway_prepare_seconds",
            Self::GatewayFinalize => "reincore_gateway_finalize_seconds",
            Self::ProviderOperation => "reincore_provider_operation_seconds",
            Self::ProviderStreamUpstreamWait => "reincore_provider_stream_upstream_wait_seconds",
            Self::ProviderStreamDownstreamWait => {
                "reincore_provider_stream_downstream_wait_seconds"
            }
            Self::ProviderFirstOutput => "reincore_provider_first_output_seconds",
            Self::GatewayFirstOutput => "reincore_gateway_first_output_seconds",
            Self::DatabaseProbe => "reincore_database_probe_seconds",
            Self::OutboxSample => "reincore_outbox_sample_seconds",
            Self::RelayPublish => "reincore_relay_publish_seconds",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Counter {
    RequestsShed,
    ChatRequests,
    ChatErrors,
    ProviderStreamErrors,
    RelayPublished,
    RelayRetries,
    RelayClaimErrors,
    RelayAckErrors,
    RelayReleaseErrors,
    RelayCleanupErrors,
    RelayInvalidEvents,
}

impl Counter {
    fn name(self) -> &'static str {
        match self {
            Self::RequestsShed => "reincore_requests_shed_total",
            Self::ChatRequests => "reincore_chat_requests_total",
            Self::ChatErrors => "reincore_chat_errors_total",
            Self::ProviderStreamErrors => "reincore_provider_stream_errors_total",
            Self::RelayPublished => "reincore_relay_published_total",
            Self::RelayRetries => "reincore_relay_retry_batches_total",
            Self::RelayClaimErrors => "reincore_relay_claim_errors_total",
            Self::RelayAckErrors => "reincore_relay_ack_errors_total",
            Self::RelayReleaseErrors => "reincore_relay_release_errors_total",
            Self::RelayCleanupErrors => "reincore_relay_cleanup_errors_total",
            Self::RelayInvalidEvents => "reincore_relay_invalid_events_total",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gauge {
    DatabaseUp,
    DatabaseObserved,
    RelayUp,
    RelaySubscriberUp,
    RelayPublisherUp,
    RelayConfigured,
    OutboxPending,
    OutboxOldestAge,
    OutboxRetried,
    OutboxObserved,
    OutboxSampleUp,
}

impl Gauge {
    fn name(self) -> &'static str {
        match self {
            Self::DatabaseUp => "reincore_database_up",
            Self::DatabaseObserved => "reincore_database_observed_timestamp_seconds",
            Self::RelayUp => "reincore_relay_connected",
            Self::RelaySubscriberUp => "reincore_relay_subscriber_up",
            Self::RelayPublisherUp => "reincore_relay_publisher_up",
            Self::RelayConfigured => "reincore_relay_configured",
            Self::OutboxPending => "reincore_outbox_pending_sample",
            Self::OutboxOldestAge => "reincore_outbox_oldest_sample_age_seconds",
            Self::OutboxRetried => "reincore_outbox_retried_pending_sample",
            Self::OutboxObserved => "reincore_outbox_observed_timestamp_seconds",
            Self::OutboxSampleUp => "reincore_outbox_sample_up",
        }
    }
}

#[derive(Clone, Default)]
struct Histogram {
    buckets: [u64; 12],
    count: u64,
    sum: f64,
}

impl Histogram {
    fn observe(&mut self, seconds: f64) {
        self.count += 1;
        self.sum += seconds;
        for (count, bound) in self.buckets.iter_mut().zip(BUCKETS) {
            if seconds <= bound {
                *count += 1;
            }
        }
    }
}

#[derive(Clone)]
struct Registry {
    histograms: BTreeMap<Metric, Histogram>,
    counters: BTreeMap<Counter, u64>,
    gauges: BTreeMap<Gauge, f64>,
}

impl Default for Registry {
    fn default() -> Self {
        use Counter::*;
        Self {
            histograms: BTreeMap::new(),
            gauges: BTreeMap::new(),
            counters: [
                RequestsShed,
                ChatRequests,
                ChatErrors,
                ProviderStreamErrors,
                RelayPublished,
                RelayRetries,
                RelayClaimErrors,
                RelayAckErrors,
                RelayReleaseErrors,
                RelayCleanupErrors,
                RelayInvalidEvents,
            ]
            .into_iter()
            .map(|counter| (counter, 0))
            .collect(),
        }
    }
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn observe(metric: Metric, duration: Duration) {
    registry()
        .histograms
        .entry(metric)
        .or_default()
        .observe(duration.as_secs_f64());
}

pub fn increment(counter: Counter, count: u64) {
    let mut r = registry();
    let value = r.counters.entry(counter).or_default();
    *value = value.saturating_add(count);
}

pub fn gauge(metric: Gauge, value: Option<f64>) {
    gauges(&[(metric, value)]);
}

/// Publish a coherent observation (values, health and timestamp) in one update.
pub fn gauges(values: &[(Gauge, Option<f64>)]) {
    let mut r = registry();
    for &(metric, value) in values {
        if let Some(value) = value.filter(|v| v.is_finite()) {
            r.gauges.insert(metric, value);
        } else {
            r.gauges.remove(&metric);
        }
    }
}

pub fn timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub fn database_observation(healthy: Option<bool>) {
    gauges(&[
        (Gauge::DatabaseUp, healthy.map(f64::from)),
        (Gauge::DatabaseObserved, Some(timestamp())),
    ]);
}

/// None invalidates the values while retaining the last successful timestamp.
pub fn outbox_observation(sample: Option<(i64, i64, f64)>) {
    let mut values = vec![
        (Gauge::OutboxPending, sample.map(|s| s.0 as f64)),
        (Gauge::OutboxRetried, sample.map(|s| s.1 as f64)),
        (Gauge::OutboxOldestAge, sample.map(|s| s.2)),
        (Gauge::OutboxSampleUp, Some(f64::from(sample.is_some()))),
    ];
    if sample.is_some() {
        values.push((Gauge::OutboxObserved, Some(timestamp())));
    }
    gauges(&values);
}

/// Prometheus text format. Scraping never performs dependency I/O.
pub fn render() -> String {
    let snapshot = registry().clone();
    snapshot.render()
}

impl Registry {
    fn render(&self) -> String {
        let mut text = String::new();
        for (metric, histogram) in &self.histograms {
            let name = metric.name();
            writeln!(text, "# TYPE {name} histogram").unwrap();
            for (bound, count) in BUCKETS.iter().zip(histogram.buckets) {
                writeln!(text, "{name}_bucket{{le=\"{bound}\"}} {count}").unwrap();
            }
            writeln!(
                text,
                "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_count {}\n{name}_sum {}",
                histogram.count, histogram.count, histogram.sum
            )
            .unwrap();
        }
        for (metric, value) in &self.counters {
            let name = metric.name();
            writeln!(text, "# TYPE {name} counter\n{name} {value}").unwrap();
        }
        for (metric, value) in &self.gauges {
            let name = metric.name();
            writeln!(text, "# TYPE {name} gauge\n{name} {value}").unwrap();
        }
        text
    }
}

/// Lifetime measurement includes errors and cancellation; it is not a success counter.
pub struct Timer {
    metric: Metric,
    started: Instant,
}

impl Timer {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            started: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        observe(self.metric, self.started.elapsed());
    }
}

/// Meaningful output excludes role, usage, finish and empty deltas. Gateway
/// timing stops before serializing/yielding the first outgoing fragment, not
/// when a client receives it.
pub struct OutputTiming {
    request: Instant,
    provider: Instant,
    provider_seen: bool,
    gateway_seen: bool,
}

impl OutputTiming {
    pub fn new(request: Instant, provider: Instant) -> Self {
        Self {
            request,
            provider,
            provider_seen: false,
            gateway_seen: false,
        }
    }

    pub fn output(&mut self, provider_output: bool, forwarded: bool) {
        if provider_output && !self.provider_seen {
            observe(Metric::ProviderFirstOutput, self.provider.elapsed());
            self.provider_seen = true;
        }
        if forwarded && !self.gateway_seen {
            observe(Metric::GatewayFirstOutput, self.request.elapsed());
            self.gateway_seen = true;
        }
    }

    pub fn chunk(&mut self, chunk: &crate::provider::ChatCompletionStreamChunk) {
        if self.provider_seen && self.gateway_seen {
            return;
        }
        let nonempty = |s: &Option<String>| s.as_ref().is_some_and(|s| !s.is_empty());
        let tool = chunk.delta_tool_calls.as_ref().is_some_and(|deltas| {
            deltas
                .iter()
                .any(|d| nonempty(&d.function_name) || nonempty(&d.function_arguments))
        });
        let forwarded = nonempty(&chunk.delta_content) || tool;
        self.output(
            forwarded || nonempty(&chunk.delta_reasoning_content),
            forwarded,
        );
    }
}

pub struct MetricsServer(tokio::task::JoinHandle<()>);

/// Task liveness returns to zero on normal exit, panic or cancellation.
pub struct TaskPresence(Gauge);
impl TaskPresence {
    pub fn new(metric: Gauge) -> Self {
        gauge(metric, Some(1.0));
        Self(metric)
    }
}
impl Drop for TaskPresence {
    fn drop(&mut self) {
        gauge(self.0, Some(0.0));
    }
}

impl Drop for MetricsServer {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Background sampler owned by the server lifetime; missed ticks do not queue.
pub fn sampler<F, Fut>(mut sample: F) -> MetricsServer
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    MetricsServer(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            sample().await;
        }
    }))
}

pub fn metrics_address(value: &str) -> anyhow::Result<std::net::SocketAddr> {
    let address: std::net::SocketAddr = value.parse()?;
    anyhow::ensure!(
        address.ip().is_loopback(),
        "REINCORE_METRICS_ADDR must be a loopback IP address"
    );
    Ok(address)
}

/// Optional, separate operator listener: never expose global metrics on tenant APIs.
pub async fn start_from_env() -> anyhow::Result<Option<MetricsServer>> {
    let Some(value) = std::env::var_os("REINCORE_METRICS_ADDR") else {
        return Ok(None);
    };
    let address = metrics_address(
        value
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid metrics address"))?,
    )?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    Ok(Some(serve_metrics(listener)))
}

fn serve_metrics(listener: tokio::net::TcpListener) -> MetricsServer {
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(|| async {
            (
                [
                    (
                        axum::http::header::CONTENT_TYPE,
                        "text/plain; version=0.0.4; charset=utf-8",
                    ),
                    (axum::http::header::CACHE_CONTROL, "no-store"),
                ],
                render(),
            )
        }),
    );
    MetricsServer(tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::error!(%error, "metrics listener stopped");
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn histogram_uses_cumulative_buckets_and_exact_sum() {
        let mut r = Registry::default();
        let h = r.histograms.entry(Metric::DatabaseProbe).or_default();
        h.observe(0.005);
        h.observe(0.2);
        h.observe(130.0);
        assert_eq!(h.count, 3);
        assert_eq!(h.buckets[0], 0);
        assert_eq!(h.buckets[1], 1);
        assert_eq!(h.buckets[6], 2);
        assert!((h.sum - 130.205).abs() < 1e-9);
        assert!(r
            .render()
            .contains("reincore_database_probe_seconds_bucket{le=\"+Inf\"} 3"));
    }

    #[test]
    fn concurrent_scrapes_never_observe_half_a_gauge_update() {
        gauges(&[
            (Gauge::OutboxPending, Some(0.0)),
            (Gauge::OutboxSampleUp, Some(0.0)),
        ]);
        let writer = std::thread::spawn(|| {
            for i in 0..2000 {
                let value = Some((i % 2) as f64);
                gauges(&[
                    (Gauge::OutboxPending, value),
                    (Gauge::OutboxSampleUp, value),
                ]);
            }
        });
        for _ in 0..500 {
            let snapshot = registry().clone();
            assert_eq!(
                snapshot.gauges[&Gauge::OutboxPending],
                snapshot.gauges[&Gauge::OutboxSampleUp]
            );
            assert!(snapshot.render().contains("reincore_outbox_sample_up"));
        }
        writer.join().unwrap();
    }

    #[tokio::test]
    async fn aborted_task_does_not_leave_liveness_healthy() {
        let presence = TaskPresence::new(Gauge::RelayPublisherUp);
        let task = tokio::spawn(async move {
            let _presence = presence;
            std::future::pending::<()>().await;
        });
        task.abort();
        let _ = task.await;
        assert_eq!(registry().gauges[&Gauge::RelayPublisherUp], 0.0);
    }
    #[test]
    fn operator_listener_rejects_public_addresses() {
        assert!(metrics_address("0.0.0.0:9091").is_err());
        assert!(metrics_address("[::]:9091").is_err());
        assert!(metrics_address("127.0.0.1:9091").is_ok());
        assert!(metrics_address("[::1]:9091").is_ok());
    }
    #[test]
    fn empty_chunks_do_not_set_first_output_and_reasoning_is_not_forwarded() {
        use crate::provider::ChatCompletionStreamChunk;
        let mut timing = OutputTiming::new(Instant::now(), Instant::now());
        timing.chunk(&ChatCompletionStreamChunk {
            provider_state: None,
            delta_content: Some(String::new()),
            finish_reason: Some("stop".into()),
            ..Default::default()
        });
        assert!(!timing.provider_seen && !timing.gateway_seen);
        timing.chunk(&ChatCompletionStreamChunk {
            provider_state: None,
            delta_reasoning_content: Some("thinking".into()),
            ..Default::default()
        });
        assert!(timing.provider_seen && !timing.gateway_seen);
        timing.chunk(&ChatCompletionStreamChunk {
            provider_state: None,
            delta_content: Some("answer".into()),
            ..Default::default()
        });
        assert!(timing.gateway_seen);
    }

    #[tokio::test]
    async fn metrics_listener_serves_text_without_dependency_access() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = serve_metrics(listener);
        let response = reqwest::get(format!("http://{address}/metrics"))
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain; version=0.0.4"));
        assert_eq!(response.headers()["cache-control"], "no-store");
        assert!(response
            .text()
            .await
            .unwrap()
            .contains("# TYPE reincore_chat_requests_total counter"));
        assert_eq!(
            reqwest::get(format!("http://{address}/v1/sessions"))
                .await
                .unwrap()
                .status()
                .as_u16(),
            404
        );
        let client = reqwest::Client::new();
        let mut durations = Vec::new();
        for _ in 0..100 {
            let started = Instant::now();
            let response = client
                .get(format!("http://{address}/metrics"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), 200);
            let _ = response.text().await.unwrap();
            durations.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        durations.sort_by(f64::total_cmp);
        // Informational smoke measurement, never a hardware-dependent test gate.
        eprintln!(
            "100 loopback metric scrapes: p95={:.3}ms, max={:.3}ms",
            durations[94], durations[99]
        );
        drop(server);
    }
}

/// Live callback/task signals are maintained even when metrics export is disabled.
pub fn relay_ready() -> bool {
    let r = registry();
    [
        Gauge::RelayUp,
        Gauge::RelaySubscriberUp,
        Gauge::RelayPublisherUp,
    ]
    .iter()
    .all(|g| r.gauges.get(g) == Some(&1.0))
}
