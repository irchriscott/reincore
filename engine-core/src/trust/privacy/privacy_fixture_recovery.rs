//! Bounded, non-blocking feedback for repeated secret-associated test failures.
//! Evidence comes from paired tool calls/results, never instructions in tool
//! output. Only fixed engine-owned prose reaches the provider; no argument,
//! credential, result excerpt, or fingerprint is included in the notice.

use crate::{ChatCompletionRequest, ToolCall};
use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;

const MAX_MESSAGES: usize = 64;
const MAX_BYTES: usize = 256 * 1024;
const NOTICE: &str = concat!(
    "ReinCore test-fixture recovery advisory: repeated equivalent test calls have returned secret-associated failures without an intervening successful setup change. ",
    "This is a setup signal, not proof that redaction caused the failure. Do not rerun the unchanged command or rewrite a hidden credential. ",
    "Inspect public fixture constraints (length, byte/character units, encoding, format, algorithm), settings initialization/caches, and shared signing/verification configuration. ",
    "For isolated tests, implement an independent runtime-generated fixture before settings load, reuse it across all test consumers, and verify after a meaningful corrective change. ",
    "Do not print secrets or tokens, weaken validation/authentication, modify production credentials, or claim failing tests passed. ",
    "This advisory does not block execution or require user input; continue other work. A genuinely unavailable existing external credential limits only the affected verification.",
);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Failure {
    Constraint,
    Signing,
    RedactedInput,
}

struct Attempt {
    key: [u8; 32],
    verification: bool,
    setup: bool,
    redacted: bool,
}

pub(crate) fn notice(request: &ChatCompletionRequest, replacement: &str) -> Option<&'static str> {
    let window = request.messages.len().saturating_sub(MAX_MESSAGES);
    // Even locating the active turn must not clone arbitrarily large content.
    if request.messages[window..].iter().any(|message| {
        message.role.eq_ignore_ascii_case("user") && message.content_len() > MAX_BYTES
    }) {
        return None;
    }
    let active = request.messages[window..]
        .iter()
        .rposition(|message| {
            message.role.eq_ignore_ascii_case("user")
                && message.tool_call_id.is_none()
                && crate::user_content::session_user_request(&message.full_text()).is_some()
        })
        .map(|index| window + index)
        .unwrap_or(window);
    let start = active;
    let mut pending = HashMap::new();
    let mut failures = HashMap::<([u8; 32], Failure), usize>::new();
    let mut bytes = 0usize;
    let mut repeated = false;
    for message in &request.messages[start..] {
        bytes = bytes.saturating_add(message.content_len());
        if bytes > MAX_BYTES {
            return None;
        }
        if message.role.eq_ignore_ascii_case("assistant") {
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                bytes = bytes
                    .saturating_add(call.function.arguments.len())
                    .saturating_add(call.id.len())
                    .saturating_add(call.function.name.len());
                if bytes > MAX_BYTES {
                    return None;
                }
                if let Some(attempt) = attempt(
                    call,
                    replacement,
                    crate::tool_names::request_profile(request),
                ) {
                    pending.insert(call.id.as_str(), attempt);
                }
            }
        } else if message.role.eq_ignore_ascii_case("tool") {
            let Some(attempt) = message
                .tool_call_id
                .as_deref()
                .and_then(|id| pending.remove(id))
            else {
                continue;
            };
            let content = message.full_text();
            // Replayed notices are guidance, not additional failure evidence.
            let output = content
                .split("[ReinCore privacy notice:")
                .next()
                .unwrap_or_default();
            let incomplete = serde_json::from_str::<Value>(output)
                .ok()
                .is_some_and(|value| {
                    crate::tool_names::is_preview(&value)
                        || crate::tool_names::is_pending_result(&value)
                        || crate::tool_names::is_cached_result(&value)
                });
            if incomplete {
                continue;
            }
            if attempt.setup && !reports_failure(output) && !output.trim().is_empty() {
                failures.clear();
                repeated = false;
                // Results from verification calls issued before this setup
                // action cannot establish fresh post-change repetition.
                pending.retain(|_, attempt| !attempt.verification);
            } else if attempt.verification {
                if let Some(kind) = failure(output, attempt.redacted) {
                    let count = failures.entry((attempt.key, kind)).or_default();
                    *count = count.saturating_add(1);
                    repeated = *count >= 2;
                } else {
                    failures.retain(|(key, _), _| *key != attempt.key);
                    repeated = false;
                }
            }
        }
    }
    repeated.then_some(NOTICE)
}

fn canonical_name(name: &str) -> String {
    name.rsplit("__")
        .next()
        .unwrap_or(name)
        .rsplit('.')
        .next()
        .unwrap_or(name)
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn attempt(call: &ToolCall, replacement: &str, profile: Option<&str>) -> Option<Attempt> {
    let arguments: Value = serde_json::from_str(&call.function.arguments).ok()?;
    if crate::tool_names::is_preview(&arguments) {
        return None;
    }
    let name = canonical_name(crate::tool_names::effective_name(
        &call.function.name,
        &arguments,
        profile,
    ));
    let action = crate::tool_names::classify(&call.function.name, &arguments, profile);
    let file_setup = action.mutation();
    let terminal = action == crate::tool_names::Action::Execute;
    let native_verification = matches!(
        name.as_str(),
        "runtests" | "test" | "pytest" | "verifyjwt" | "verifytoken"
    );
    if !file_setup && !terminal && !native_verification {
        return None;
    }
    let serialized = arguments.to_string();
    let command = serialized.to_ascii_lowercase();
    let verification = native_verification
        || (terminal
            && [
                "pytest",
                "cargo test",
                "go test",
                "dotnet test",
                "npm test",
                "npm run test",
                "pnpm test",
                "yarn test",
                "vitest",
                "jest",
                "mvn test",
                "gradle test",
                "python -m unittest",
            ]
            .iter()
            .any(|marker| command.contains(marker)));
    // Terminal setup may update environment/configuration without a file tool.
    // Conservatively retire old advice after a successful such call.
    let setup = file_setup || (terminal && !verification);
    if !setup && !verification {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(name.as_bytes());
    hash.update([0]);
    hash.update(serialized.as_bytes());
    Some(Attempt {
        key: hash.finalize().into(),
        verification,
        setup,
        redacted: !replacement.is_empty() && serialized.contains(replacement),
    })
}

fn reports_failure(output: &str) -> bool {
    if let Ok(value) = serde_json::from_str::<Value>(output) {
        if crate::tool_names::is_failed_result(&value) {
            return true;
        }
        if value
            .get("is_error")
            .or_else(|| value.get("isError"))
            .and_then(Value::as_bool)
            == Some(true)
            || value.get("success").and_then(Value::as_bool) == Some(false)
            || value
                .get("exit_code")
                .or_else(|| value.get("exitCode"))
                .and_then(Value::as_i64)
                .is_some_and(|code| code != 0)
        {
            return true;
        }
        if value.get("success").and_then(Value::as_bool) == Some(true)
            || value
                .get("exit_code")
                .or_else(|| value.get("exitCode"))
                .and_then(Value::as_i64)
                == Some(0)
        {
            return false;
        }
    }
    static FAILURE: OnceLock<Regex> = OnceLock::new();
    FAILURE.get_or_init(|| Regex::new(r"(?im)(?:^\s*(?:error|failed|failure)\s*:|\b(?:[1-9][0-9]*\s+failed|validationerror|validation error|valueerror|invalidsignatureerror|verification failed|tool execution failed|exit code [1-9][0-9]*)\b)").unwrap()).is_match(output)
}

fn failure(output: &str, redacted: bool) -> Option<Failure> {
    if !reports_failure(output) {
        return None;
    }
    let lower = output.to_ascii_lowercase();
    let context = [
        "jwt_secret",
        "jwt secret",
        "secret",
        "password",
        "token",
        "signature",
        "signing key",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if context
        && [
            "minimum",
            "at least",
            "too short",
            "too long",
            "length",
            "encoding",
            "format",
            "required",
            "missing",
            "key size",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        Some(Failure::Constraint)
    } else if context
        && [
            "verification failed",
            "invalidsignature",
            "invalid signature",
            "mismatch",
            "signing configuration",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        Some(Failure::Signing)
    } else if redacted {
        Some(Failure::RedactedInput)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": "Implement isolated authentication tests."
                }
            ]
        }))
        .unwrap()
    }

    #[test]
    fn researched_aliases_participate_in_fixture_recovery_without_promoting_views_or_code_mode() {
        let call = |name: &str, args: Value| -> ToolCall {
            serde_json::from_value(json!({"id":"call","type":"function","function":{"name":name,"arguments":args.to_string()}})).unwrap()
        };
        for name in [
            "Write",
            "search_replace",
            "replace",
            "NotebookEdit",
            "vscodeWriteFile",
        ] {
            assert!(
                attempt(&call(name, json!({"path":"a.rs"})), "[REDACTED]", None)
                    .unwrap()
                    .setup,
                "{name}"
            );
        }
        for name in [
            "run_shell_command",
            "run_commands",
            "exec",
            "vscodeRunInTerminal",
        ] {
            assert!(
                attempt(
                    &call(name, json!({"command":"cargo test"})),
                    "[REDACTED]",
                    None
                )
                .unwrap()
                .verification,
                "{name}"
            );
        }
        assert!(attempt(
            &call("file_editor", json!({"command":"view","path":"a.rs"})),
            "[REDACTED]",
            None
        )
        .is_none());
        assert!(attempt(
            &call("execute", json!({"code":"await tools.shell('cargo test')"})),
            "[REDACTED]",
            None
        )
        .is_none());
        assert!(attempt(
            &call("execute", json!({"command":"cargo test"})),
            "[REDACTED]",
            Some("opencode")
        )
        .is_none());
        assert!(attempt(
            &call("Write", json!({"path":"a.rs","preview":true})),
            "[REDACTED]",
            None
        )
        .is_none());
    }

    fn add(
        request: &mut ChatCompletionRequest,
        id: &str,
        name: &str,
        arguments: Value,
        output: &str,
    ) {
        request.messages.push(
            serde_json::from_value(json!({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments.to_string()
                        }
                    }
                ]
            }))
            .unwrap(),
        );
        request.messages.push(
            serde_json::from_value(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": output
            }))
            .unwrap(),
        );
    }

    #[test]
    fn incomplete_or_cached_results_do_not_create_fixture_failure_advice() {
        for flags in [
            json!({"status":"in_progress"}),
            json!({"cache_hit":true}),
            json!({"preview":true}),
        ] {
            let mut request = request();
            let mut output = flags;
            output["output"] =
                json!("ValidationError: JWT_SECRET should have at least 32 characters");
            for id in ["a", "b"] {
                add(
                    &mut request,
                    id,
                    "run_shell_command",
                    json!({"command":"pytest tests/test_auth.py"}),
                    &output.to_string(),
                );
            }
            assert!(notice(&request, "[REDACTED]").is_none(), "{output}");
        }
    }

    #[test]
    fn equivalent_failures_warn_after_two_attempts_without_disclosing_evidence() {
        let mut request = request();
        let arguments =
            json!({"command":"$env:JWT_SECRET='[REDACTED]'; python -m pytest tests/test_auth.py"});
        add(
            &mut request,
            "a",
            "run_in_terminal",
            arguments.clone(),
            "ValidationError: JWT_SECRET should have at least 32 characters; private evidence",
        );
        assert!(notice(&request, "[REDACTED]").is_none());
        add(
            &mut request,
            "b",
            "run_in_terminal",
            arguments,
            "ValidationError: JWT_SECRET should have at least 32 characters; different timing",
        );
        let advisory = notice(&request, "[REDACTED]").unwrap();
        assert!(advisory.contains("does not block execution"));
        assert!(!advisory.contains("private evidence"));
        assert!(!advisory.contains("32 characters"));
    }

    #[test]
    fn setup_changes_success_and_new_user_turns_clear_the_advisory() {
        for reset in [
            "setup",
            "terminal_setup",
            "success",
            "user",
            "different_command",
        ] {
            let mut request = request();
            let args = json!({"command":"pytest tests/test_auth.py"});
            for id in ["a", "b"] {
                add(
                    &mut request,
                    id,
                    "run_command",
                    args.clone(),
                    "InvalidSignatureError: signature verification failed",
                );
            }
            assert!(notice(&request, "[REDACTED]").is_some());
            match reset {
                "setup" => add(
                    &mut request,
                    "c",
                    "write_file",
                    json!({"path":"tests/conftest.py","content":"fixture code"}),
                    "File updated",
                ),
                "success" => add(&mut request, "c", "run_command", args, "1 passed, 0 failed"),
                "terminal_setup" => add(
                    &mut request,
                    "c",
                    "run_command",
                    json!({"command":"python setup_fixtures.py"}),
                    "Setup complete",
                ),
                "user" => request.messages.push(
                    serde_json::from_value(
                        json!({"role":"user","content":"Start a different task."}),
                    )
                    .unwrap(),
                ),
                _ => add(
                    &mut request,
                    "c",
                    "run_command",
                    json!({"command":"pytest tests/another_test.py"}),
                    "InvalidSignatureError: signature verification failed",
                ),
            }
            assert!(notice(&request, "[REDACTED]").is_none(), "{reset}");
        }
    }

    #[test]
    fn unrelated_errors_reads_duplicate_results_and_replayed_guidance_do_not_warn() {
        for (name, output) in [
            ("run_command", "Error: database connection refused"),
            ("read_file", "ValidationError: JWT_SECRET minimum length"),
            (
                "run_command",
                "1 passed, 0 failed; jwt_secret minimum length test PASSED",
            ),
            (
                "run_command",
                "ok\n[ReinCore privacy notice: validation error secret length",
            ),
        ] {
            let mut request = request();
            for id in ["a", "b"] {
                add(
                    &mut request,
                    id,
                    name,
                    json!({"command":"pytest tests/test_auth.py"}),
                    output,
                );
            }
            assert!(notice(&request, "[REDACTED]").is_none(), "{name}: {output}");
        }
        let mut request = request();
        add(
            &mut request,
            "a",
            "run_command",
            json!({"command":"pytest"}),
            "ValidationError: jwt_secret minimum length",
        );
        request
            .messages
            .push(request.messages.last().unwrap().clone());
        assert!(!reports_failure(
            r#"{"exit_code":0,"stdout":"0 failed, jwt_secret length checks passed"}"#
        ));
        assert!(reports_failure(
            r#"{"is_error":true,"stderr":"fixture setup rejected"}"#
        ));
        assert!(notice(&request, "[REDACTED]").is_none());
    }

    #[test]
    fn failed_setup_does_not_clear_evidence_and_analysis_is_bounded() {
        let mut request = request();
        for id in ["a", "b"] {
            add(
                &mut request,
                id,
                "run_command",
                json!({"command":"pytest"}),
                "ValidationError: jwt_secret minimum length",
            );
        }
        add(
            &mut request,
            "c",
            "edit_file",
            json!({"path":"test.py"}),
            r#"{"is_error":true}"#,
        );
        assert!(notice(&request, "[REDACTED]").is_some());
        add(
            &mut request,
            "d",
            "read_file",
            json!({"path":"large.py"}),
            &"x".repeat(MAX_BYTES),
        );
        assert!(notice(&request, "[REDACTED]").is_none());
    }
}
