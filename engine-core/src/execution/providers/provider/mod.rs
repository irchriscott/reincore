//! Provider abstraction and concrete adapters.

// The internal module contains the provider trait and shared registry while this
// parent module groups provider adapters. Keeping the established public paths
// avoids a needless API-breaking rename.
#[allow(clippy::module_inception)]
mod provider;

pub mod batch;
pub(crate) mod replay;
pub use replay::{prepare_request_replay, retain_request_group_state, ProviderReplay};
pub mod anthropic;
pub mod azure;
pub mod copilot;
pub mod deepseek;
pub mod gemini;
pub mod groq;
pub mod meta;
pub mod ollama;
pub mod openai;
pub mod xai;

pub use anthropic::AnthropicProvider;
pub use azure::AzureOpenAiProvider;
pub use copilot::{get_global_oauth_token, CopilotModelFamily, CopilotProvider};
pub use deepseek::DeepSeekProvider;
pub use gemini::GeminiProvider;
pub use groq::GroqProvider;
pub use meta::MetaProvider;
pub use ollama::{OllamaMode, OllamaModelInfo, OllamaProvider};
pub use openai::OpenAiProvider;
pub(crate) use provider::{capture_response_headers, openai_compatible_stream};
pub use provider::{
    chat_message_has_visible_output, extract_provider_reasoning, is_rate_limit_response_header,
    provider_response_has_visible_output, retry_empty_provider_response, BatchSubmission,
    ChatCompletionResponse, ChatCompletionStreamChunk, Provider, ProviderConfig, ProviderModel,
    ProviderModelMapping, ProviderRegistry, ProviderResponseAttempts, ProviderResponseHeaders,
    ProviderStream, RateLimitRemaining, TokenUsage, ToolCallDelta, ToolExecutor,
    ToolExecutorRegistry, EMPTY_PROVIDER_RESPONSE_RETRY_LIMIT,
};
pub use xai::XaiProvider;
