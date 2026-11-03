//! Runtime observability: metrics, tracing, and logging.
//!
//! Bounded process-local operational metrics, the OpenTelemetry pipeline
//! with its health check, and shared terminal logging helpers. No request
//! or tenant labels leave the process.

pub mod log;
pub mod operational;
pub mod otel_health;
pub mod telemetry;
