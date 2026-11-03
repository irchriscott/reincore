//! Normalized request/response models for the OpenAI/Anthropic-compatible
//! chat-completion surface.
//!
//! Defines the shared [`ChatMessage`], [`ToolCall`], [`ToolDefinition`], and
//! [`ChatCompletionRequest`]/[`ChatCompletionResponse`] types that all provider
//! adapters translate to and from. Keeping a single canonical shape lets the
//! routing, cost, cache, and memory subsystems operate on provider-agnostic
//! data regardless of whether the outbound call targets OpenAI, Anthropic,
//! Gemini, or a local Ollama server.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A single content part in a multi-part message (OpenAI format).
///
/// `text` parts carry plain text; `image_url` parts carry an image URL
/// or data URL (e.g. `data:image/png;base64,iVBOR...`); `file` parts
/// carry a file URL or data URL for PDFs and other document types.
/// This mirrors the OpenAI chat completions multi-part content schema
/// with extensions for document attachments.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    File { file_url: FileUrl },
}

impl ContentPart {
    /// Return the text content if this is a `Text` part.
    pub fn text_ref(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Return the image URL if this is an `ImageUrl` part.
    pub fn image_url_ref(&self) -> Option<&ImageUrl> {
        match self {
            ContentPart::ImageUrl { image_url } => Some(image_url),
            _ => None,
        }
    }

    /// Return the file URL if this is a `File` part.
    pub fn file_url_ref(&self) -> Option<&FileUrl> {
        match self {
            ContentPart::File { file_url } => Some(file_url),
            _ => None,
        }
    }
}

/// The URL (or data URL) for an image content part.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
    /// Optional detail level: "auto", "low", or "high".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The URL (or data URL) for a file content part (PDFs, documents, etc.).
///
/// Follows the same pattern as [`ImageUrl`]: the `url` field can be a
/// remote `https://...` URL, a `data:application/pdf;base64,...` data
/// URL, or a local `file://...` path that will be resolved to a data
/// URL before being forwarded to the provider.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileUrl {
    pub url: String,
}

/// Message content that can be either a plain string or an array of
/// structured content parts (text + images), following the OpenAI
/// multi-part content schema.
///
/// Serialized with `#[serde(untagged)]` so that:
/// - A simple string deserializes as `MessageContent::Text`
/// - An array of content parts deserializes as `MessageContent::Parts`
/// This provides full backward compatibility with existing clients that
/// send `content` as a string.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Create a simple text content.
    pub fn text<S: Into<String>>(s: S) -> Self {
        MessageContent::Text(s.into())
    }

    /// Return the content as a plain string. For multi-part content, all
    /// text parts are concatenated and image/file parts are skipped.
    /// Returns an empty string when the content is `None`-equivalent.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                    ContentPart::File { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Return a reference to the text if this is `Text` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(_) => None,
        }
    }

    /// Return the full text content as an owned `String`. For multi-part
    /// content, all text parts are concatenated. Unlike [`as_str`] this
    /// never returns `None` - multipart messages produce their text parts
    /// joined together, ensuring they are visible to token estimators,
    /// metrics, and classification.
    pub fn text_content(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Approximate the total character length of the content, including
    /// file and image data URL payloads. For multipart messages, text
    /// parts contribute their full length, while `ImageUrl` and `File`
    /// parts contribute the length of their `url` field (which for data
    /// URLs includes the full base64-encoded payload). This ensures token
    /// estimators and context-window enforcers see the real size of
    /// binary attachments instead of treating them as zero.
    pub fn content_len(&self) -> usize {
        match self {
            MessageContent::Text(s) => s.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { image_url } => image_url.url.len(),
                    ContentPart::File { file_url } => file_url.url.len(),
                })
                .sum(),
        }
    }

    /// True when the content is a plain text string (not multi-part).
    pub fn is_text(&self) -> bool {
        matches!(self, MessageContent::Text(_))
    }

    /// True when the content is multi-part (contains image parts).
    pub fn is_multipart(&self) -> bool {
        matches!(self, MessageContent::Parts(_))
    }

    /// Return the content parts if this is a `Parts` variant.
    pub fn parts(&self) -> Option<&[ContentPart]> {
        match self {
            MessageContent::Parts(parts) => Some(parts.as_slice()),
            MessageContent::Text(_) => None,
        }
    }

    /// True when any part is an image.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
            MessageContent::Text(_) => false,
        }
    }

    /// True when any part is a file (PDF, document, etc.).
    pub fn has_files(&self) -> bool {
        match self {
            MessageContent::Parts(parts) => {
                parts.iter().any(|p| matches!(p, ContentPart::File { .. }))
            }
            MessageContent::Text(_) => false,
        }
    }

    /// True when any part is a binary attachment (image or file).
    pub fn has_attachments(&self) -> bool {
        match self {
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. } | ContentPart::File { .. })),
            MessageContent::Text(_) => false,
        }
    }

    /// Resolve local file paths to base64 data URLs in-place.
    ///
    /// For each `ImageUrl` or `File` part whose `url` is a `file://...`
    /// path or a bare local filesystem path, reads the file from disk
    /// and replaces the URL with a `data:{mime};base64,{data}` URL.
    /// Remote `https://...` URLs and existing `data:...` URLs are left
    /// unchanged.
    ///
    /// Returns `Ok(())` if all local files were resolved successfully,
    /// or an error describing which file could not be read.
    pub fn resolve_local_files(&mut self) -> Result<(), String> {
        if let MessageContent::Parts(parts) = self {
            for part in parts.iter_mut() {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        if let Some(data_url) = resolve_url_to_data_url(&image_url.url)? {
                            image_url.url = data_url;
                        }
                    }
                    ContentPart::File { file_url } => {
                        if let Some(data_url) = resolve_url_to_data_url(&file_url.url)? {
                            file_url.url = data_url;
                        }
                    }
                    ContentPart::Text { .. } => {}
                }
            }
        }
        Ok(())
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// Resolve a URL to a base64 data URL if it points to a local file.
///
/// Handles three cases:
/// - `data:...` URLs → returned as-is (already base64)
/// - `file://...` paths → read from disk, converted to data URL
/// - Bare local paths (e.g. `/path/to/file.pdf` or `C:\path\to\file.pdf`)
///   → if the path exists on disk, read and convert to data URL
/// - `https://...` / `http://...` URLs → returned as-is (remote)
///
/// Returns `Ok(None)` when the URL is already a data URL or is a remote
/// URL that doesn't need resolution. Returns `Ok(Some(data_url))` when
/// a local file was successfully read and encoded.
/// Returns `Err(message)` when a local file path exists but cannot be
/// read.
fn resolve_url_to_data_url(url: &str) -> Result<Option<String>, String> {
    // Already a data URL - no resolution needed.
    if url.starts_with("data:") {
        return Ok(None);
    }

    // Remote URLs - no resolution needed.
    if url.starts_with("https://") || url.starts_with("http://") {
        return Ok(None);
    }

    // Extract the file path from the URL.
    let file_path = if let Some(stripped) = url.strip_prefix("file://") {
        // On Windows, file:///C:/path → C:/path
        // On Unix, file:///path → /path
        let cleaned = stripped.trim_start_matches('/');
        if cleaned.len() >= 2 && cleaned.as_bytes()[1] == b':' {
            // Windows drive letter: C:/path
            cleaned.to_string()
        } else {
            // Unix absolute path
            format!("/{}", cleaned)
        }
    } else {
        // Bare path - check if it exists on disk.
        url.to_string()
    };

    let path = Path::new(&file_path);
    if !path.exists() {
        // If it doesn't exist, treat it as a remote/unresolvable URL.
        return Ok(None);
    }

    // Read the file and determine its MIME type.
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read file {}: {}", file_path, e))?;

    let mime = guess_mime_type(path);
    let data = base64_encode(&bytes);

    Ok(Some(format!("data:{};base64,{}", mime, data)))
}

/// Guess the MIME type from a file extension.
fn guess_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("txt") => "text/plain",
        Some("json") => "application/json",
        Some("csv") => "text/csv",
        Some("html") | Some("htm") => "text/html",
        Some("xml") => "application/xml",
        Some("md") | Some("markdown") => "text/markdown",
        Some("yaml") | Some("yml") => "application/x-yaml",
        Some("toml") => "application/toml",
        Some("js") | Some("mjs") | Some("cjs") => "application/javascript",
        Some("ts") => "application/typescript",
        Some("sh") | Some("bash") | Some("zsh") => "application/x-sh",
        Some("sql") => "application/sql",
        Some("rb") => "application/x-ruby",
        Some("py") => "application/x-python-code",
        Some("css") => "text/css",
        Some("scss") | Some("sass") => "text/x-scss",
        Some("less") => "text/x-less",
        Some("svg") => "image/svg+xml",
        Some("rtf") => "application/rtf",
        // Microsoft Word (.docx)
        Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        // Legacy Microsoft Word (.doc)
        Some("doc") => "application/msword",
        // OpenDocument Text (.odt)
        Some("odt") => "application/vnd.oasis.opendocument.text",
        _ => "application/octet-stream",
    }
}

/// Base64-encode raw bytes without an external dependency.
/// Uses a lookup table for the standard base64 alphabet.
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };

        result.push(TABLE[(b0 >> 2) as usize] as char);
        result.push(TABLE[((b0 & 0x03) << 4 | b1 >> 4) as usize] as char);

        if chunk.len() > 1 {
            result.push(TABLE[((b1 & 0x0f) << 2 | b2 >> 6) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(TABLE[(b2 & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// A chat message in the OpenAI/Anthropic-compatible surface.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct ChatMessage {
    /// Opaque state returned by the same provider/model; never display as text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<crate::provider::ProviderReplay>,
    pub role: String,
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Get the content as a plain string. For multi-part content, text
    /// parts are concatenated and image parts are skipped. Returns `None`
    /// when content is `None`.
    pub fn content_text(&self) -> Option<String> {
        self.content.as_ref().map(|c| c.as_text())
    }

    /// Get the content as a `&str` when it is plain text. Returns `None`
    /// for multi-part content or when content is absent.
    pub fn content_str(&self) -> Option<&str> {
        self.content.as_ref().and_then(|c| c.as_str())
    }

    /// Get the full text content as an owned `String`. For multi-part
    /// content, all text parts are concatenated. Returns an empty string
    /// when content is absent. Unlike [`content_str`], this never returns
    /// `None` for multipart messages, ensuring token estimators and
    /// metrics see the text portion of multi-part messages.
    ///
    /// [`content_str`]: ChatMessage::content_str
    pub fn full_text(&self) -> String {
        self.content
            .as_ref()
            .map(|c| c.text_content())
            .unwrap_or_default()
    }

    /// Approximate the character length of the message content, including
    /// file and image data URL payloads. Returns 0 when content is absent.
    /// See [`MessageContent::content_len`].
    pub fn content_len(&self) -> usize {
        self.content.as_ref().map(|c| c.content_len()).unwrap_or(0)
    }

    /// Check whether this message has meaningful user content - either a
    /// non-empty text string, or multi-part content with at least one
    /// text part or attachment. Used by conversation projection and
    /// compaction to decide whether a user message should be retained
    /// without dropping multi-part messages that `content_str()` would
    /// miss.
    pub fn has_user_content(&self) -> bool {
        match &self.content {
            Some(MessageContent::Text(s)) => !s.trim().is_empty(),
            Some(MessageContent::Parts(parts)) => parts.iter().any(|p| match p {
                ContentPart::Text { text } => !text.trim().is_empty(),
                ContentPart::ImageUrl { .. } | ContentPart::File { .. } => true,
            }),
            None => false,
        }
    }

    /// True when the message has multi-part content (images, etc.).
    pub fn is_multipart(&self) -> bool {
        self.content.as_ref().is_some_and(|c| c.is_multipart())
    }

    /// Resolve local file paths in the message content to base64 data URLs.
    /// See [`MessageContent::resolve_local_files`] for details.
    pub fn resolve_local_files(&mut self) -> Result<(), String> {
        if let Some(content) = &mut self.content {
            content.resolve_local_files()?;
        }
        Ok(())
    }

    /// Return the data URLs of all image and file attachments in this
    /// message. Used by session-level file deduplication to detect when
    /// the same file is attached across multiple messages in a compiled
    /// session. Returns an empty vector for plain-text messages.
    pub fn attachment_urls(&self) -> Vec<String> {
        match &self.content {
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    ContentPart::File { file_url } => Some(file_url.url.clone()),
                    ContentPart::Text { .. } => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// A single tool call issued by an LLM.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Derive the OpenAI-compatible `finish_reason` for an assistant message.
/// Agentic clients rely on `"tool_calls"` to know they must execute tools
/// before continuing, so it must never be reported as a plain `"stop"`.
pub fn finish_reason_for(message: &ChatMessage) -> &'static str {
    if message
        .tool_calls
        .as_deref()
        .map(|calls| !calls.is_empty())
        .unwrap_or(false)
    {
        "tool_calls"
    } else {
        "stop"
    }
}

/// Reconcile a provider's finish reason with the normalized response message.
///
/// Some OpenAI-compatible providers report `"stop"` even when their streamed
/// response contains tool calls. The normalized tool calls are authoritative:
/// clients must execute them, and request grouping must remain open for the
/// resulting tool-response turn. Other provider reasons such as `"length"`
/// and `"content_filter"` are preserved when no tool call is present.
pub fn effective_finish_reason(
    message: &ChatMessage,
    provider_finish_reason: Option<&str>,
) -> String {
    if finish_reason_for(message) == "tool_calls" {
        "tool_calls".to_string()
    } else {
        provider_finish_reason.unwrap_or("stop").to_string()
    }
}

/// JSON variant of [`finish_reason_for`] for cached responses that are stored
/// as raw `serde_json::Value` messages.
pub fn finish_reason_for_json(message: &serde_json::Value) -> &'static str {
    if message
        .get("tool_calls")
        .and_then(|calls| calls.as_array())
        .map(|calls| !calls.is_empty())
        .unwrap_or(false)
    {
        "tool_calls"
    } else {
        "stop"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// The result of executing a tool call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub tool_call_id: String,
    /// OpenAI-compatible tool results always carry `role: "tool"`, but
    /// some producers omit it. Default to `"tool"` so deserialization
    /// tolerates the omission rather than silently dropping the result.
    #[serde(default = "default_tool_role")]
    pub role: String,
    pub name: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

fn default_tool_role() -> String {
    "tool".to_string()
}

/// A file attached to the context, e.g. via a `read_file` or `edit_file` tool.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileContextAttachment {
    /// Absolute/workspace-relative file path or source URL.
    pub path: String,
    /// Monotonically increasing version for this path. The compactor and
    /// cache manager only keep the latest version.
    pub version: u64,
    /// Optional content hash for fast equality checks.
    pub content_hash: Option<String>,
    /// Optional snippet of the file content (may be omitted for large files).
    pub content: Option<String>,
}

/// Definition of a tool exposed to the agent.
///
/// Supports both the flat format (`{name, description, parameters}`) and
/// the OpenAI nested format (`{type: "function", function: {name, ...}}`).
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ToolDefinition {
    /// Host declaration used only for action-receipt grounding, never authorization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_contract: Option<crate::action_receipts::ActionContract>,
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Whether this tool is idempotent and read-only. Only safe tools are
    /// eligible for memoization by default.
    #[serde(default)]
    pub safe_to_memoize: bool,
    /// Optional memoization policy. If omitted, the engine uses a default
    /// time-based TTL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memoization_policy: Option<String>,
}

/// Deserialize a `ToolDefinition` from either the flat format or the OpenAI
/// nested format (`{type: "function", function: {name, description, parameters}}`).
impl<'de> serde::Deserialize<'de> for ToolDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;

        // Check for the OpenAI nested format: {"type": "function", "function": {...}}
        if let Some(function) = value.get("function") {
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let description = function
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            // Absent means "caller did not say", not "caller said no": fall
            // back to the server's own read-only classification.
            let safe_to_memoize = function
                .get("safe_to_memoize")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| crate::circuit::is_exploration_tool(&name));
            let memoization_policy = function
                .get("memoization_policy")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            return Ok(ToolDefinition {
                action_contract: function
                    .get("action_contract")
                    .filter(|v| !v.is_null())
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(serde::de::Error::custom)?,
                name,
                description,
                parameters,
                safe_to_memoize,
                memoization_policy,
            });
        }

        // Flat format: {name, description, parameters, ...}
        let name = value
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let description = value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let parameters = value
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        // Absent means "caller did not say", not "caller said no": fall back
        // to the server's own read-only classification.
        let safe_to_memoize = value
            .get("safe_to_memoize")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| crate::circuit::is_exploration_tool(&name));
        let memoization_policy = value
            .get("memoization_policy")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ToolDefinition {
            action_contract: value
                .get("action_contract")
                .filter(|v| !v.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(serde::de::Error::custom)?,
            name,
            description,
            parameters,
            safe_to_memoize,
            memoization_policy,
        })
    }
}

/// OpenAI/Anthropic-compatible chat completion request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Extra parameters that ReinCore uses but that are not part of the
    /// standard completion surface.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ChatCompletionRequest {
    /// Resolve all local file paths in all messages to base64 data URLs.
    /// See [`MessageContent::resolve_local_files`] for details.
    pub fn resolve_local_files(&mut self) -> Result<(), String> {
        for msg in &mut self.messages {
            msg.resolve_local_files()?;
        }
        Ok(())
    }

    /// Deduplicate file and image attachments across all messages in the
    /// session. When the same data URL appears in multiple messages (or
    /// multiple times within a single message), only the first occurrence
    /// is kept; subsequent duplicates are removed from their `Parts` vec.
    /// This operates at the session compilation level, ensuring that when
    /// historical messages are projected alongside the current request,
    /// each unique file is sent to the provider only once.
    ///
    /// Returns the number of duplicate attachment parts removed.
    pub fn deduplicate_attachments(&mut self) -> usize {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut removed = 0usize;

        for msg in &mut self.messages {
            if let Some(MessageContent::Parts(parts)) = &mut msg.content {
                let mut filtered = Vec::with_capacity(parts.len());
                for part in parts.drain(..) {
                    match &part {
                        ContentPart::ImageUrl { image_url } => {
                            if seen.insert(image_url.url.clone()) {
                                filtered.push(part);
                            } else {
                                removed += 1;
                            }
                        }
                        ContentPart::File { file_url } => {
                            if seen.insert(file_url.url.clone()) {
                                filtered.push(part);
                            } else {
                                removed += 1;
                            }
                        }
                        ContentPart::Text { .. } => {
                            filtered.push(part);
                        }
                    }
                }
                // If only text parts remain, collapse to Text variant for
                // efficiency. If no parts remain at all, set content to None.
                if filtered.is_empty() {
                    msg.content = None;
                } else if filtered
                    .iter()
                    .all(|p| matches!(p, ContentPart::Text { .. }))
                {
                    let text = filtered
                        .into_iter()
                        .filter_map(|p| match p {
                            ContentPart::Text { text } => Some(text),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    msg.content = Some(MessageContent::Text(text));
                } else {
                    *parts = filtered;
                }
            }
        }

        removed
    }
}

/// Classification of model cost/quality tier used by the escalation ladder.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Lite, fast model for routine classification/tool-selection work.
    #[default]
    Lite,
    /// Pro model for reasoning/synthesis.
    Pro,
    /// Ultra model for complex or high-stakes synthesis.
    Ultra,
}

impl ModelTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelTier::Lite => "lite",
            ModelTier::Pro => "pro",
            ModelTier::Ultra => "ultra",
        }
    }
}

impl std::str::FromStr for ModelTier {
    type Err = crate::error::EngineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "lite" => Ok(Self::Lite),
            "pro" => Ok(Self::Pro),
            "ultra" => Ok(Self::Ultra),
            _ => Err(crate::error::EngineError::InvalidModelTier(
                value.to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant_with_tool_call() -> ChatMessage {
        ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"src/main.rs"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn tool_calls_override_an_incorrect_provider_stop_reason() {
        assert_eq!(
            effective_finish_reason(&assistant_with_tool_call(), Some("stop")),
            "tool_calls"
        );
    }

    #[test]
    fn non_tool_provider_finish_reasons_are_preserved() {
        let message = ChatMessage {
            provider_state: None,
            role: "assistant".to_string(),
            content: Some(crate::MessageContent::text("partial answer".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        assert_eq!(effective_finish_reason(&message, Some("length")), "length");
        assert_eq!(effective_finish_reason(&message, None), "stop");
    }

    // Multi-part content tests

    #[test]
    fn message_content_text_serializes_as_plain_string() {
        let content = MessageContent::text("Hello, world!");
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json, serde_json::json!("Hello, world!"));
    }

    #[test]
    fn message_content_parts_serializes_as_array() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "What's in this image?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/cat.png".to_string(),
                    detail: None,
                },
            },
        ]);
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                { "type": "text", "text": "What's in this image?" },
                { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } }
            ])
        );
    }

    #[test]
    fn message_content_deserializes_string_as_text() {
        let json = serde_json::json!("Hello!");
        let content: MessageContent = serde_json::from_value(json).unwrap();
        assert!(content.is_text());
        assert_eq!(content.as_str(), Some("Hello!"));
    }

    #[test]
    fn message_content_deserializes_array_as_parts() {
        let json = serde_json::json!([
            { "type": "text", "text": "Describe this" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBOR=", "detail": "high" } }
        ]);
        let content: MessageContent = serde_json::from_value(json).unwrap();
        assert!(content.is_multipart());
        assert!(!content.is_text());
    }

    #[test]
    fn message_content_as_text_concats_text_parts_skips_images() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Hello ".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".to_string(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "world!".to_string(),
            },
        ]);
        assert_eq!(content.as_text(), "Hello world!");
    }

    #[test]
    fn message_content_as_str_returns_text_only() {
        let text = MessageContent::text("plain text");
        assert_eq!(text.as_str(), Some("plain text"));

        let parts = MessageContent::Parts(vec![ContentPart::Text {
            text: "part text".to_string(),
        }]);
        assert_eq!(parts.as_str(), None);
    }

    #[test]
    fn message_content_is_text_and_is_multipart() {
        let text = MessageContent::text("hello");
        assert!(text.is_text());
        assert!(!text.is_multipart());

        let parts = MessageContent::Parts(vec![ContentPart::Text {
            text: "hello".to_string(),
        }]);
        assert!(!parts.is_text());
        assert!(parts.is_multipart());
    }

    #[test]
    fn message_content_parts_returns_slice() {
        let parts = vec![
            ContentPart::Text {
                text: "a".to_string(),
            },
            ContentPart::Text {
                text: "b".to_string(),
            },
        ];
        let content = MessageContent::Parts(parts.clone());
        assert_eq!(content.parts(), Some(parts.as_slice()));

        let text = MessageContent::text("hello");
        assert_eq!(text.parts(), None);
    }

    #[test]
    fn message_content_has_images_detects_image_parts() {
        let with_image = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "look".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".to_string(),
                    detail: None,
                },
            },
        ]);
        assert!(with_image.has_images());

        let text_only = MessageContent::Parts(vec![ContentPart::Text {
            text: "no images".to_string(),
        }]);
        assert!(!text_only.has_images());

        let plain_text = MessageContent::text("plain");
        assert!(!plain_text.has_images());
    }

    #[test]
    fn message_content_from_string_and_str() {
        let from_string: MessageContent = "hello".to_string().into();
        assert_eq!(from_string, MessageContent::Text("hello".to_string()));

        let from_str: MessageContent = "world".into();
        assert_eq!(from_str, MessageContent::Text("world".to_string()));
    }

    #[test]
    fn message_content_default_is_empty_text() {
        let default = MessageContent::default();
        assert!(default.is_text());
        assert_eq!(default.as_str(), Some(""));
    }

    #[test]
    fn image_url_serializes_with_detail_when_present() {
        let url = ImageUrl {
            url: "https://example.com/img.png".to_string(),
            detail: Some("high".to_string()),
        };
        let json = serde_json::to_value(&url).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "url": "https://example.com/img.png",
                "detail": "high"
            })
        );
    }

    #[test]
    fn image_url_skips_detail_when_none() {
        let url = ImageUrl {
            url: "https://example.com/img.png".to_string(),
            detail: None,
        };
        let json = serde_json::to_value(&url).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "url": "https://example.com/img.png"
            })
        );
    }

    #[test]
    fn chat_message_content_str_returns_text_for_plain_text() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::text("hello")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.content_str(), Some("hello"));
    }

    #[test]
    fn chat_message_content_str_returns_none_for_multipart() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "hello".to_string(),
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.content_str(), None);
    }

    #[test]
    fn chat_message_content_text_concats_parts() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello ".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/img.png".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "world!".to_string(),
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.content_text(), Some("Hello world!".to_string()));
    }

    #[test]
    fn chat_message_content_text_returns_none_when_content_is_none() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.content_text(), None);
    }

    #[test]
    fn chat_message_is_multipart_true_for_parts() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "hello".to_string(),
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert!(msg.is_multipart());
    }

    #[test]
    fn chat_message_is_multipart_false_for_text() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::text("hello")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert!(!msg.is_multipart());
    }

    #[test]
    fn chat_message_is_multipart_false_for_none() {
        let msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert!(!msg.is_multipart());
    }

    #[test]
    fn content_part_text_serializes_with_snake_case_tag() {
        let part = ContentPart::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json, serde_json::json!({ "type": "text", "text": "hello" }));
    }

    #[test]
    fn content_part_image_url_serializes_with_snake_case_tag() {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/img.png".to_string(),
                detail: Some("auto".to_string()),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/img.png",
                    "detail": "auto"
                }
            })
        );
    }

    #[test]
    fn message_content_roundtrip_text() {
        let original = MessageContent::text("roundtrip text");
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn message_content_roundtrip_parts() {
        let original = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "part 1".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".to_string(),
                    detail: Some("high".to_string()),
                },
            },
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn message_content_empty_string_deserializes_as_text() {
        let json = serde_json::json!("");
        let content: MessageContent = serde_json::from_value(json).unwrap();
        assert!(content.is_text());
        assert_eq!(content.as_str(), Some(""));
    }

    #[test]
    fn message_content_data_url_image_part() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Analyze this image".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==".to_string(),
                    detail: None,
                },
            },
        ]);
        assert!(content.is_multipart());
        assert!(content.has_images());
        assert_eq!(content.as_text(), "Analyze this image");
    }

    #[test]
    fn chat_completion_request_with_multipart_content_serializes() {
        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "What's in this image?".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/cat.png".to_string(),
                            detail: None,
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: std::collections::HashMap::new(),
        };
        let json = serde_json::to_value(&request).unwrap();
        let content = &json["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What's in this image?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    // File / PDF content tests

    #[test]
    fn content_part_file_serializes_with_snake_case_tag() {
        let part = ContentPart::File {
            file_url: FileUrl {
                url: "https://example.com/doc.pdf".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "file",
                "file_url": {
                    "url": "https://example.com/doc.pdf"
                }
            })
        );
    }

    #[test]
    fn content_part_file_deserializes_from_json() {
        let json = serde_json::json!({
            "type": "file",
            "file_url": { "url": "data:application/pdf;base64,JVBERi0=" }
        });
        let part: ContentPart = serde_json::from_value(json).unwrap();
        match part {
            ContentPart::File { file_url } => {
                assert_eq!(file_url.url, "data:application/pdf;base64,JVBERi0=");
            }
            _ => panic!("expected File variant"),
        }
    }

    #[test]
    fn file_url_serializes_without_optional_fields() {
        let fu = FileUrl {
            url: "https://example.com/doc.pdf".to_string(),
        };
        let json = serde_json::to_value(&fu).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "url": "https://example.com/doc.pdf" })
        );
    }

    #[test]
    fn message_content_with_file_part_is_multipart() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Analyze this PDF".to_string(),
            },
            ContentPart::File {
                file_url: FileUrl {
                    url: "data:application/pdf;base64,JVBERi0=".to_string(),
                },
            },
        ]);
        assert!(content.is_multipart());
        assert!(!content.has_images());
        assert!(content.has_files());
        assert!(content.has_attachments());
        assert_eq!(content.as_text(), "Analyze this PDF");
    }

    #[test]
    fn message_content_with_image_and_file_has_attachments() {
        let content = MessageContent::Parts(vec![
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,iVBOR=".to_string(),
                    detail: None,
                },
            },
            ContentPart::File {
                file_url: FileUrl {
                    url: "data:application/pdf;base64,JVBERi0=".to_string(),
                },
            },
        ]);
        assert!(content.has_images());
        assert!(content.has_files());
        assert!(content.has_attachments());
        assert_eq!(content.as_text(), "");
    }

    #[test]
    fn message_content_has_files_false_for_text() {
        let text = MessageContent::text("plain text");
        assert!(!text.has_files());

        let image_only = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/img.png".to_string(),
                detail: None,
            },
        }]);
        assert!(!image_only.has_files());
    }

    #[test]
    fn message_content_roundtrip_with_file_part() {
        let original = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Read this document".to_string(),
            },
            ContentPart::File {
                file_url: FileUrl {
                    url: "https://example.com/report.pdf".to_string(),
                },
            },
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn message_content_with_pdf_data_url() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Summarize this PDF".to_string(),
            },
            ContentPart::File {
                file_url: FileUrl {
                    url: "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MK".to_string(),
                },
            },
        ]);
        assert!(content.is_multipart());
        assert!(content.has_files());
        assert_eq!(content.as_text(), "Summarize this PDF");
    }

    #[test]
    fn chat_completion_request_resolve_local_files_is_noop_for_data_urls() {
        let mut request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Analyze this".to_string(),
                    },
                    ContentPart::File {
                        file_url: FileUrl {
                            url: "data:application/pdf;base64,JVBERi0=".to_string(),
                        },
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,iVBOR=".to_string(),
                            detail: None,
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: std::collections::HashMap::new(),
        };
        // Data URLs should remain unchanged.
        request.resolve_local_files().unwrap();
        let parts = request.messages[0]
            .content
            .as_ref()
            .unwrap()
            .parts()
            .unwrap();
        assert_eq!(
            parts[1].file_url_ref().unwrap().url,
            "data:application/pdf;base64,JVBERi0="
        );
        assert_eq!(
            parts[2].image_url_ref().unwrap().url,
            "data:image/png;base64,iVBOR="
        );
    }

    #[test]
    fn chat_completion_request_resolve_local_files_is_noop_for_remote_urls() {
        let mut request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                provider_state: None,
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.png".to_string(),
                            detail: None,
                        },
                    },
                    ContentPart::File {
                        file_url: FileUrl {
                            url: "https://example.com/doc.pdf".to_string(),
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: None,
            extra: std::collections::HashMap::new(),
        };
        // Remote URLs should remain unchanged.
        request.resolve_local_files().unwrap();
        let parts = request.messages[0]
            .content
            .as_ref()
            .unwrap()
            .parts()
            .unwrap();
        assert_eq!(
            parts[0].image_url_ref().unwrap().url,
            "https://example.com/image.png"
        );
        assert_eq!(
            parts[1].file_url_ref().unwrap().url,
            "https://example.com/doc.pdf"
        );
    }

    #[test]
    fn resolve_local_files_reads_file_from_disk() {
        // Create a temporary file and verify it gets resolved to a data URL.
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("reincore_test_file.txt");
        std::fs::write(&test_file, b"Hello, world!").unwrap();

        let url = format!("file://{}", test_file.display());
        let mut content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Read this".to_string(),
            },
            ContentPart::File {
                file_url: FileUrl { url },
            },
        ]);

        content.resolve_local_files().unwrap();

        let parts = content.parts().unwrap();
        let resolved_url = parts[1].file_url_ref().unwrap().url.clone();
        assert!(resolved_url.starts_with("data:text/plain;base64,"));
        // Verify the base64 decodes to the original content.
        let b64 = resolved_url
            .strip_prefix("data:text/plain;base64,")
            .unwrap();
        // We can't easily decode base64 in tests without a dependency, but
        // we can check that it's non-empty and valid base64 characters.
        assert!(!b64.is_empty());
        assert!(b64
            .chars()
            .all(|c| c.is_alphanumeric() || c == '+' || c == '/' || c == '='));

        // Clean up.
        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn resolve_local_files_reads_image_from_disk() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("reincore_test_img.png");
        // Minimal PNG header bytes.
        std::fs::write(&test_file, b"\x89PNG\r\n\x1a\n").unwrap();

        let url = format!("file://{}", test_file.display());
        let mut content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl { url, detail: None },
        }]);

        content.resolve_local_files().unwrap();

        let resolved_url = content.parts().unwrap()[0]
            .image_url_ref()
            .unwrap()
            .url
            .clone();
        assert!(resolved_url.starts_with("data:image/png;base64,"));

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn resolve_local_files_reads_pdf_from_disk() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("reincore_test_doc.pdf");
        // Fake PDF header.
        std::fs::write(&test_file, b"%PDF-1.4\n%test\n").unwrap();

        let url = format!("file://{}", test_file.display());
        let mut content = MessageContent::Parts(vec![ContentPart::File {
            file_url: FileUrl { url },
        }]);

        content.resolve_local_files().unwrap();

        let resolved_url = content.parts().unwrap()[0]
            .file_url_ref()
            .unwrap()
            .url
            .clone();
        assert!(resolved_url.starts_with("data:application/pdf;base64,"));

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn resolve_local_files_handles_nonexistent_path_gracefully() {
        let mut content = MessageContent::Parts(vec![ContentPart::File {
            file_url: FileUrl {
                url: "file:///nonexistent/path/to/file.pdf".to_string(),
            },
        }]);

        // Nonexistent files are treated as unresolvable URLs (no error).
        let result = content.resolve_local_files();
        assert!(result.is_ok());
        // URL should be unchanged since the file doesn't exist.
        assert_eq!(
            content.parts().unwrap()[0].file_url_ref().unwrap().url,
            "file:///nonexistent/path/to/file.pdf"
        );
    }

    #[test]
    fn resolve_local_files_handles_bare_path() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("reincore_bare_test.txt");
        std::fs::write(&test_file, b"test content").unwrap();

        // Use a bare path (no file:// prefix).
        let bare_path = test_file.display().to_string();
        let mut content = MessageContent::Parts(vec![ContentPart::File {
            file_url: FileUrl { url: bare_path },
        }]);

        content.resolve_local_files().unwrap();

        let resolved_url = content.parts().unwrap()[0]
            .file_url_ref()
            .unwrap()
            .url
            .clone();
        assert!(resolved_url.starts_with("data:text/plain;base64,"));

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn resolve_local_files_handles_text_content_noop() {
        let mut content = MessageContent::text("plain text");
        let result = content.resolve_local_files();
        assert!(result.is_ok());
        assert_eq!(content.as_str(), Some("plain text"));
    }

    #[test]
    fn base64_encode_encodes_empty_data() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_encode_encodes_simple_data() {
        // "Hello" in base64 is "SGVsbG8="
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        // "Hello, world!" in base64
        assert_eq!(base64_encode(b"Hello, world!"), "SGVsbG8sIHdvcmxkIQ==");
    }

    #[test]
    fn base64_encode_encodes_binary_data() {
        // PNG header bytes
        let png_header = b"\x89PNG\r\n\x1a\n";
        let encoded = base64_encode(png_header);
        // Verify it round-trips by checking known value
        assert_eq!(encoded, "iVBORw0KGgo=");
    }

    #[test]
    fn guess_mime_type_for_common_extensions() {
        use std::path::PathBuf;
        // Images
        assert_eq!(
            guess_mime_type(&PathBuf::from("doc.pdf")),
            "application/pdf"
        );
        assert_eq!(guess_mime_type(&PathBuf::from("img.png")), "image/png");
        assert_eq!(guess_mime_type(&PathBuf::from("img.JPG")), "image/jpeg");
        assert_eq!(guess_mime_type(&PathBuf::from("img.jpeg")), "image/jpeg");
        assert_eq!(guess_mime_type(&PathBuf::from("anim.gif")), "image/gif");
        assert_eq!(guess_mime_type(&PathBuf::from("img.webp")), "image/webp");
        assert_eq!(guess_mime_type(&PathBuf::from("logo.svg")), "image/svg+xml");
        // Text / data formats
        assert_eq!(guess_mime_type(&PathBuf::from("doc.txt")), "text/plain");
        assert_eq!(
            guess_mime_type(&PathBuf::from("data.json")),
            "application/json"
        );
        assert_eq!(guess_mime_type(&PathBuf::from("data.csv")), "text/csv");
        assert_eq!(guess_mime_type(&PathBuf::from("page.html")), "text/html");
        assert_eq!(guess_mime_type(&PathBuf::from("page.htm")), "text/html");
        assert_eq!(
            guess_mime_type(&PathBuf::from("data.xml")),
            "application/xml"
        );
        // Markdown
        assert_eq!(
            guess_mime_type(&PathBuf::from("README.md")),
            "text/markdown"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("README.markdown")),
            "text/markdown"
        );
        // Config / structured formats
        assert_eq!(
            guess_mime_type(&PathBuf::from("config.yaml")),
            "application/x-yaml"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("config.yml")),
            "application/x-yaml"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("Cargo.toml")),
            "application/toml"
        );
        // Code
        assert_eq!(
            guess_mime_type(&PathBuf::from("app.js")),
            "application/javascript"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("app.mjs")),
            "application/javascript"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("app.ts")),
            "application/typescript"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("script.sh")),
            "application/x-sh"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("script.bash")),
            "application/x-sh"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("query.sql")),
            "application/sql"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("app.rb")),
            "application/x-ruby"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("app.py")),
            "application/x-python-code"
        );
        assert_eq!(guess_mime_type(&PathBuf::from("style.css")), "text/css");
        assert_eq!(guess_mime_type(&PathBuf::from("style.scss")), "text/x-scss");
        // Word documents
        assert_eq!(
            guess_mime_type(&PathBuf::from("report.docx")),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("legacy.doc")),
            "application/msword"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("doc.odt")),
            "application/vnd.oasis.opendocument.text"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("doc.rtf")),
            "application/rtf"
        );
        // Unknown / missing extensions
        assert_eq!(
            guess_mime_type(&PathBuf::from("file.unknownext")),
            "application/octet-stream"
        );
        assert_eq!(
            guess_mime_type(&PathBuf::from("no_extension")),
            "application/octet-stream"
        );
    }

    #[test]
    fn resolve_url_to_data_url_passes_data_urls_through() {
        let result = resolve_url_to_data_url("data:image/png;base64,iVBOR=").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_url_to_data_url_passes_https_through() {
        let result = resolve_url_to_data_url("https://example.com/file.pdf").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_url_to_data_url_passes_http_through() {
        let result = resolve_url_to_data_url("http://example.com/file.pdf").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_url_to_data_url_returns_none_for_nonexistent_file() {
        let result = resolve_url_to_data_url("file:///nonexistent/path/file.txt").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn chat_message_resolve_local_files_method() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("reincore_msg_test.txt");
        std::fs::write(&test_file, b"file content").unwrap();

        let mut msg = ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Read this file".to_string(),
                },
                ContentPart::File {
                    file_url: FileUrl {
                        url: format!("file://{}", test_file.display()),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        msg.resolve_local_files().unwrap();

        let parts = msg.content.as_ref().unwrap().parts().unwrap();
        assert!(parts[1]
            .file_url_ref()
            .unwrap()
            .url
            .starts_with("data:text/plain;base64,"));

        let _ = std::fs::remove_file(&test_file);
    }

    fn tool_from(value: serde_json::Value) -> ToolDefinition {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn known_read_only_tools_are_granted_memoization_when_the_caller_is_silent() {
        // Standard clients never send the non-standard `safe_to_memoize`
        // field, so without this the whole memoization path stays dormant.
        for name in ["read_file", "grep_search", "list_dir"] {
            assert!(
                tool_from(serde_json::json!({"name": name, "description": "d"})).safe_to_memoize,
                "{name} should be memoizable by default"
            );
            assert!(
                tool_from(serde_json::json!({
                    "type": "function",
                    "function": {"name": name, "description": "d"}
                }))
                .safe_to_memoize,
                "{name} should be memoizable in the nested format too"
            );
        }
    }

    #[test]
    fn unknown_and_mutating_tools_are_not_granted_memoization() {
        for name in ["write_file", "run_command", "read_secrets_and_delete"] {
            assert!(
                !tool_from(serde_json::json!({"name": name, "description": "d"})).safe_to_memoize,
                "{name} must not be memoizable by default"
            );
        }
    }

    #[test]
    fn an_explicit_caller_flag_still_wins_in_both_directions() {
        // The caller may always opt out of a grant the server would have made.
        assert!(
            !tool_from(serde_json::json!({
                "name": "read_file", "description": "d", "safe_to_memoize": false
            }))
            .safe_to_memoize
        );
        // An explicit opt-in is honored as before.
        assert!(
            tool_from(serde_json::json!({
                "name": "custom_lookup", "description": "d", "safe_to_memoize": true
            }))
            .safe_to_memoize
        );
    }
}
