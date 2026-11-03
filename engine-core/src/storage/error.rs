//! Canonical error types for the ReinCore engine.
//!
//! Defines [`EngineError`] (via `thiserror`) and the [`EngineResult`] alias
//! used throughout `engine-core` and `daemon`.
//! Includes a helper to distinguish non-retryable authentication/authorization
//! failures from transient network or rate-limit errors.

use thiserror::Error;

/// The canonical error type for the ReinCore engine.
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("session not found: {0}")]
    SessionNotFound(uuid::Uuid),

    #[error("checkpoint not found or expired: {0}")]
    CheckpointNotFound(String),

    #[error("checkpoint token has already been redeemed")]
    CheckpointAlreadyRedeemed,

    #[error("invalid budget: {0}")]
    InvalidBudget(String),

    #[error("invalid model tier: {0}")]
    InvalidModelTier(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("{provider} model '{model}' does not support {capability}")]
    UnsupportedCapability {
        provider: String,
        model: String,
        capability: String,
    },

    #[error("request requires {required} input tokens but the model limit is {limit}")]
    ContextWindowExceeded { required: u64, limit: u64 },

    #[error("normalization error: {0}")]
    Normalization(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("tool execution error: {0}")]
    ToolExecution(String),

    #[error("streaming not supported: {0}")]
    StreamingNotSupported(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

/// Describe a transport failure, preserving the underlying cause.
///
/// `reqwest`'s own `Display` stops at `error sending request for url (...)`,
/// which hides whether the call timed out, was refused, or failed DNS -
/// exactly the detail that says what to fix. Walk the source chain and label
/// the failure kind so the log is actionable.
pub fn describe_request_error(error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "could not connect"
    } else if error.is_decode() {
        "could not decode response"
    } else {
        "request failed"
    };

    let mut causes = Vec::new();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let text = cause.to_string();
        if !causes.contains(&text) {
            causes.push(text);
        }
        source = cause.source();
    }

    let url = error
        .url()
        .map(|url| format!(" ({url})"))
        .unwrap_or_default();
    let detail = if causes.is_empty() {
        error.to_string()
    } else {
        causes.join(": ")
    };

    let hint = if error.is_timeout() {
        " - raise the provider's timeout_seconds if the model legitimately needs longer"
    } else if error.is_connect() {
        " - check the provider is running and the base_url is correct"
    } else {
        ""
    };

    format!("{kind}{url}: {detail}{hint}")
}

impl EngineError {
    /// Build a provider error from a transport failure, preserving the cause.
    pub fn from_request_error(error: reqwest::Error) -> Self {
        EngineError::Provider(describe_request_error(&error))
    }

    /// Returns `true` if the provider call exceeded its timeout rather than
    /// failing outright, so callers can answer 504 instead of 502.
    pub fn is_timeout_error(&self) -> bool {
        match self {
            EngineError::Provider(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("timed out") || lower.contains("timeout")
            }
            _ => false,
        }
    }
}

impl EngineError {
    /// Returns `true` if the error is caused by an authentication or
    /// authorization failure that will not succeed on retry.
    ///
    /// Provider errors that mention "401", "403", "Unauthorized",
    /// "invalid_api_key", or "Forbidden" are considered non-retryable.
    /// All other errors (network timeouts, 429 rate limits, 500/502/503
    /// server errors) are retryable.
    pub fn is_auth_error(&self) -> bool {
        match self {
            EngineError::Provider(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("401")
                    || lower.contains("403")
                    || lower.contains("unauthorized")
                    || lower.contains("forbidden")
                    || lower.contains("invalid_api_key")
                    || lower.contains("invalid api key")
                    || lower.contains("incorrect api key")
            }
            _ => false,
        }
    }

    /// Returns `true` if the request was rejected for exceeding the model's
    /// context window, either by our own pre-flight check or by the provider.
    ///
    /// Providers word this differently and often return it as a plain 400, so
    /// the message has to be sniffed. The caller's remedy is the same in every
    /// case: compact the conversation and continue in a fresh session.
    pub fn is_context_overflow_error(&self) -> bool {
        match self {
            EngineError::ContextWindowExceeded { .. } => true,
            EngineError::Provider(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("context_length_exceeded")
                    || lower.contains("context length")
                    || lower.contains("context window")
                    || lower.contains("maximum context")
                    || lower.contains("context limit")
                    || lower.contains("too many tokens")
                    || lower.contains("prompt is too long")
                    || lower.contains("input is too long")
                    || lower.contains("reduce the length of the messages")
                    || lower.contains("request too large")
            }
            _ => false,
        }
    }

    /// Returns `true` if the error is transient and a retry might succeed.
    ///
    /// Auth errors, invalid input, and invalid model tiers are
    /// non-retryable. Everything else (network errors, rate limits, server
    /// errors) is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            EngineError::Provider(msg) => {
                let lower = msg.to_lowercase();
                // Auth errors are never retryable.
                if lower.contains("401")
                    || lower.contains("403")
                    || lower.contains("unauthorized")
                    || lower.contains("forbidden")
                    || lower.contains("invalid_api_key")
                    || lower.contains("invalid api key")
                    || lower.contains("incorrect api key")
                {
                    return false;
                }
                // Retrying the same oversized payload cannot succeed; the
                // caller has to compact it first.
                if self.is_context_overflow_error() {
                    return false;
                }
                // 400 Bad Request / 404 Not Found are not retryable.
                if lower.contains("400") || lower.contains("404") {
                    return false;
                }
                // Everything else (429, 500, 502, 503, network errors) is retryable.
                true
            }
            EngineError::CheckpointAlreadyRedeemed
            | EngineError::ContextWindowExceeded { .. }
            | EngineError::InvalidInput(_)
            | EngineError::UnsupportedCapability { .. }
            | EngineError::InvalidModelTier(_) => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_refused_connection_names_the_cause_and_the_url() {
        // Port 1 on loopback is reserved and refuses immediately, so this
        // exercises the real reqwest error path without touching the network.
        let error = reqwest::Client::new()
            .post("http://127.0.0.1:1/api/chat")
            .send()
            .await
            .expect_err("connection to port 1 must fail");

        let engine_error = EngineError::from_request_error(error);
        let message = engine_error.to_string();

        assert!(
            message.contains("could not connect"),
            "expected a connect failure, got: {message}"
        );
        assert!(
            message.contains("127.0.0.1:1"),
            "expected the url in the message, got: {message}"
        );
        // The bare reqwest Display was the whole problem: it stopped here.
        assert!(
            message != "error sending request",
            "cause chain was dropped: {message}"
        );
        assert!(engine_error.is_retryable());
        assert!(!engine_error.is_auth_error());
    }

    #[test]
    fn a_timeout_is_distinguished_from_a_plain_failure() {
        let timeout = EngineError::Provider("timed out (http://x/api/chat): ...".to_string());
        assert!(timeout.is_timeout_error());

        let refused = EngineError::Provider("could not connect (http://x): refused".to_string());
        assert!(!refused.is_timeout_error());
    }
}
