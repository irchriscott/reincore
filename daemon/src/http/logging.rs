use super::request_group::{
    print_request_group_rollup, request_group_totals, response_to_request_group,
    session_active_request_groups, RequestGroupSnapshot,
};
use super::*;

pub(super) fn cents_range_dollars(low_cents: i64, high_cents: i64) -> String {
    format!(
        "${:.2} - ${:.2}",
        low_cents as f64 / 100.0,
        high_cents as f64 / 100.0
    )
}

/// Print all in-memory request-group totals (e.g. on daemon shutdown) and
/// clear the tracking maps.
pub fn print_cumulative_cost() {
    tokio::spawn(async move {
        let groups: Vec<RequestGroupSnapshot> = {
            let mut totals = request_group_totals().lock().await;
            totals
                .drain()
                .map(|(group_id, totals)| RequestGroupSnapshot { group_id, totals })
                .collect()
        };
        session_active_request_groups().lock().await.clear();
        response_to_request_group().lock().await.clear();
        for group in groups {
            if group.totals.steps > 0 {
                print_request_group_rollup(&group);
            }
        }
    });
}

/// Format a timestamp as `HH:MM:SS` for log line prefixes.
pub(super) fn log_timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

/// Print a provider-tagged log line directly to stdout, bypassing `tracing`
/// so the colored output isn't mangled by the tracing subscriber's own
/// formatting (which adds timestamps, log levels, and module paths around
/// the message, breaking the color escape codes).
///
/// Format: `[HH:MM:SS] [provider] message`
pub(super) fn log_provider(provider: &str, message: &str) {
    println!(
        "{}  {} {}",
        log_timestamp().dimmed(),
        format!("[{}]", provider).magenta().bold(),
        message,
    );
}

/// Format a byte count as a compact human-readable string (e.g. "12.3 KB").
pub(super) fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= MB {
        format!("{:.1} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes / KB)
    } else {
        format!("{} B", bytes as u64)
    }
}

/// Print a context-reduction summary when request-group projection or
/// configured compaction removed content. Reports message, token, and byte
/// reductions plus the mode that was applied.
pub(super) fn log_compression(provider: &str, adjustment: &engine_core::ContextAdjustment) {
    if adjustment.compressed_message_count == 0 {
        return;
    }
    let mode = adjustment
        .compression_mode
        .as_ref()
        .map(engine_core::CompressionMode::as_str)
        .unwrap_or("request_groups");
    let removed_tokens = adjustment
        .original_token_count
        .saturating_sub(adjustment.final_token_count);
    let removed_bytes = adjustment
        .original_byte_count
        .saturating_sub(adjustment.final_byte_count);
    log_provider(
        provider,
        &format!(
            "{}  mode={}, group-history: {} removed, active-tools: {} removed, messages: {} -> {} ({} removed), tokens: {} -> {} ({} removed), bytes: {} -> {} ({} removed)",
            "Context compacted".cyan().bold(),
            mode,
            adjustment.request_group_compacted_message_count,
            adjustment.active_tool_compacted_message_count,
            adjustment.original_message_count,
            adjustment.final_message_count,
            adjustment.compressed_message_count,
            adjustment.original_token_count,
            adjustment.final_token_count,
            removed_tokens,
            format_bytes(adjustment.original_byte_count),
            format_bytes(adjustment.final_byte_count),
            format_bytes(removed_bytes),
        ),
    );
}

/// Map an `EngineError` from a provider call to the appropriate HTTP status
/// code. Auth errors (401/403) are mapped to `UNAUTHORIZED` so the client
/// knows not to retry. Timeouts map to `GATEWAY_TIMEOUT`; other provider
/// errors map to `BAD_GATEWAY`.
pub(super) fn provider_error_to_status(e: &engine_core::EngineError) -> StatusCode {
    if matches!(
        e,
        engine_core::EngineError::InvalidInput(_)
            | engine_core::EngineError::UnsupportedCapability { .. }
    ) {
        eprintln!(
            "{}  {} {}",
            log_timestamp().dimmed(),
            "✗ INVALID REQUEST".red().bold(),
            e.to_string().red(),
        );
        StatusCode::BAD_REQUEST
    } else if e.is_auth_error() {
        eprintln!(
            "{}  {} {}",
            log_timestamp().dimmed(),
            "✗ AUTH ERROR".red().bold(),
            e.to_string().red(),
        );
        StatusCode::UNAUTHORIZED
    } else {
        eprintln!(
            "{}  {} {}",
            log_timestamp().dimmed(),
            "✗ PROVIDER ERROR".red().bold(),
            e.to_string().red(),
        );
        if e.is_timeout_error() {
            StatusCode::GATEWAY_TIMEOUT
        } else {
            StatusCode::BAD_GATEWAY
        }
    }
}
