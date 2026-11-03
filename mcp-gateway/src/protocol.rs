use super::*;

pub(super) fn supported_version(version: &str) -> bool {
    matches!(version, "2025-03-26" | "2025-06-18")
}

pub(super) fn validate_headers(headers: &HeaderMap) -> Result<(), StatusCode> {
    reject_origin(headers)?;
    for name in [
        "content-type",
        "mcp-protocol-version",
        "x-reincore-tool-cache",
    ] {
        if headers.get_all(name).iter().count() > 1 {
            return Err(StatusCode::BAD_REQUEST);
        }
    }
    if let Some(version) = headers.get("mcp-protocol-version") {
        if !version.to_str().ok().is_some_and(supported_version) {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    let content_type = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();

    if content_type != "application/json" {
        return Err(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    let accept = headers
        .get_all("accept")
        .iter()
        .filter_map(|h| h.to_str().ok())
        .collect::<Vec<_>>()
        .join(",");

    let types: Vec<_> = accept
        .split(',')
        .filter(|s| {
            !s.split(';').skip(1).any(|p| {
                p.trim()
                    .strip_prefix("q=")
                    .and_then(|q| q.parse::<f32>().ok())
                    == Some(0.0)
            })
        })
        .map(|s| s.split(';').next().unwrap_or("").trim())
        .collect();

    if !types.contains(&"application/json") || !types.contains(&"text/event-stream") {
        return Err(StatusCode::NOT_ACCEPTABLE);
    }
    Ok(())
}

pub(super) fn valid_id(id: &Value) -> bool {
    id.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 128)
        || id.as_i64().is_some()
        || id.as_u64().is_some()
}

pub(super) fn error(id: Value, code: i32, message: &str) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }))
        .into_response()
}

pub(super) fn result(id: Value, result: Value) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

pub(super) fn parse_call(params: Value, headers: &HeaderMap) -> Result<Call, ()> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= 256)
        .ok_or(())?
        .to_owned();

    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    if !arguments.is_object() {
        return Err(());
    }

    let meta = params.get("_meta");
    if meta.is_some_and(|v| !v.is_object()) {
        return Err(());
    }

    let consent = meta.and_then(|v| v.get("reincore/safe_to_memoize"));
    if consent.is_some_and(|v| !v.is_boolean()) {
        return Err(());
    }

    let group = meta.and_then(|v| v.get("reincore/request_group_id"));
    let group = match group {
        Some(Value::String(s)) if !s.trim().is_empty() && s.len() <= 128 => {
            Some(s.trim().to_owned())
        }
        None => None,
        _ => return Err(()),
    };

    let default_consent = match headers.get("x-reincore-tool-cache") {
        None => false,
        Some(value) if value == "true" => true,
        Some(value) if value == "false" => false,
        _ => return Err(()),
    };

    Ok(Call {
        name,
        arguments,
        safe_to_memoize: consent.and_then(Value::as_bool).unwrap_or(default_consent),
        request_group_id: group,
    })
}

pub(super) fn tool_result(envelope: Value) -> Option<Value> {
    let result: engine_core::ToolResult =
        serde_json::from_value(envelope.get("result")?.clone()).ok()?;

    let mut payload = serde_json::from_str::<Value>(&result.content)
        .ok()
        .filter(|v| v.get("content").is_some_and(Value::is_array))
        .unwrap_or_else(|| json!({ "content": [{ "type": "text", "text": result.content }] }));

    payload["isError"] = json!(result.is_error.unwrap_or(false));

    if !payload.get("_meta").is_some_and(Value::is_object) {
        payload["_meta"] = json!({});
    }

    payload["_meta"]["reincore"] = json!({
        "session_id": envelope.get("session_id"),
        "cache_hit": envelope.get("cache_hit"),
        "cache_hit_type": envelope.get("cache_hit_type"),
        "memoization_enabled": envelope.get("memoization_enabled"),
        "tool_call_id": result.tool_call_id,
    });

    Some(payload)
}
