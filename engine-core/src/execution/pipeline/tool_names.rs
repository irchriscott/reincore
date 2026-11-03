//! Semantic tool classification. Original names/arguments remain on receipts.
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Read,
    ReadMany,
    List,
    Search,
    Create,
    Write,
    Edit,
    Patch,
    Move,
    Delete,
    Execute,
    Orchestrate,
    Output,
    Unknown,
}

impl Action {
    pub(crate) fn exploration(self) -> bool {
        matches!(
            self,
            Self::Read | Self::ReadMany | Self::List | Self::Search
        )
    }
    pub(crate) fn mutation(self) -> bool {
        matches!(
            self,
            Self::Create | Self::Write | Self::Edit | Self::Patch | Self::Move | Self::Delete
        )
    }
}

/// The name after any agent namespace prefix (`muse.edit_file` -> `edit_file`).
pub(crate) fn leaf(name: &str) -> &str {
    name.rsplit("__")
        .next()
        .unwrap_or(name)
        .rsplit('.')
        .next()
        .unwrap_or(name)
}

fn key(name: &str) -> String {
    let key = leaf(name).replace('_', "").to_ascii_lowercase();
    key.strip_prefix("vscode").unwrap_or(&key).to_string()
}

pub(crate) fn is_composite_editor(name: &str) -> bool {
    matches!(
        key(name).as_str(),
        "fileeditor" | "strreplacebasededittool" | "strreplaceeditor" | "editor"
    )
}

/// How a proposed tool name resolved against the supplied tool contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractNameMatch {
    /// The proposed name is declared in the contract verbatim.
    Exact,
    /// The proposed name carries an agent namespace prefix
    /// (`muse.read_file`); its canonical form is declared.
    NamespaceAlias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractNameError {
    Unknown,
    Ambiguous,
}

/// Resolve a proposed tool name against contract definitions.
/// Exact matches win. Otherwise the proposed name is canonicalized and
/// compared against the canonical form of each contract entry, so agents
/// that address tools through their own namespace (`muse.powershell` for a
/// contract entry declared as `powershell`, both canonical
/// `execute_command`) validate without weakening the contract. Unknown
/// names — and aliases matching zero or several entries — fail closed.
pub(crate) fn match_contract_tool<'a>(
    definitions: &'a [crate::ToolDefinition],
    proposed: &str,
) -> Result<(&'a crate::ToolDefinition, ContractNameMatch), ContractNameError> {
    let mut exact = definitions.iter().filter(|tool| tool.name == proposed);
    match (exact.next(), exact.next()) {
        (Some(tool), None) => return Ok((tool, ContractNameMatch::Exact)),
        (Some(_), Some(_)) => return Err(ContractNameError::Ambiguous),
        (None, _) => {}
    }
    let canonical = canonical_name(proposed);
    let mut aliased = definitions
        .iter()
        .filter(|tool| canonical_name(tool.name.as_str()) == canonical);
    match (aliased.next(), aliased.next()) {
        (Some(tool), None) => Ok((tool, ContractNameMatch::NamespaceAlias)),
        (Some(_), Some(_)) => Err(ContractNameError::Ambiguous),
        (None, _) => Err(ContractNameError::Unknown),
    }
}

pub(crate) fn canonical_name(name: &str) -> &str {
    let name = leaf(name);
    let key = key(name);
    match key.as_str() {
        "read"
        | "readfile"
        | "notebookread"
        | "view"
        | "readtextfile"
        | "readmediafile"
        | "readcurrentlyopenfile" => "read_file",
        "readmanyfiles" | "readmultiplefiles" | "readfiles" => "read_many_files",
        "listdir"
        | "listdirectory"
        | "listdirectorywithsizes"
        | "listfiles"
        | "ls"
        | "directorytree"
        | "viewrepomap"
        | "viewsubdirectory"
        | "getfileinfo"
        | "listalloweddirectories"
        | "viewdiff" => "list_directory",
        "grep" | "grepsearch" | "searchfilecontent" | "textsearch" => "grep_search",
        "glob" | "globsearch" | "filesearch" => "file_search",
        "semanticsearch" | "codebasesearch" | "searchcodebase" | "codebasetool" => {
            "semantic_search"
        }
        // Roo's content search and MCP's glob-path search share this name.
        "searchfiles" => "search_files",
        "search" => "search",
        "edit"
        | "editfile"
        | "multiedit"
        | "strreplace"
        | "replace"
        | "replaceinfile"
        | "searchreplace"
        | "editexistingfile"
        | "notebookedit"
        | "replacestringinfile"
        | "multireplacestringinfile"
        | "replacefilecontent" => "edit_file",
        "write" | "writefile" | "writetofile" | "savefile" => "write_file",
        "createfile" | "createnewfile" => "create_file",
        "applypatch" | "patch" | "patchfile" | "applydiff" => "apply_patch",
        "movefile" => "move_file",
        "deletefile" | "removefile" => "delete_file",
        "execute"
        | "shell"
        | "bash"
        | "powershell"
        | "cmd"
        | "exec"
        | "execcommand"
        | "executecommand"
        | "runcommand"
        | "runcommands"
        | "runshellcommand"
        | "runterminalcommand"
        | "runinterminal"
        | "executeterminalcommand"
        | "runterminalcmd"
        | "terminal" => "execute_command",
        "readcommandoutput" | "process" | "monitor" => "command_output",
        _ => name,
    }
}

pub(crate) fn classify(name: &str, args: &Value, profile: Option<&str>) -> Action {
    let key = key(name);
    if matches!(key.as_str(), "createdirectory" | "makedirectory") {
        return Action::Edit;
    }
    if key == "monitor" && args.get("command").is_some() {
        return Action::Orchestrate;
    }
    if is_composite_editor(name) {
        if args.get("command").is_some_and(|value| !value.is_string()) {
            return Action::Unknown;
        }
        return match args.get("command").and_then(Value::as_str) {
            Some("view") => Action::Read,
            Some("create") => Action::Create,
            Some("str_replace" | "insert" | "undo_edit" | "edit" | "replace") | None => {
                Action::Edit
            }
            // Unknown subcommands establish neither read safety nor a write.
            _ => Action::Unknown,
        };
    }
    if key == "execute" {
        let profile = profile
            .unwrap_or_default()
            .replace(['_', '-', ' '], "")
            .to_ascii_lowercase();
        if profile.starts_with("opencode") || args.get("code").is_some() {
            return Action::Orchestrate;
        }
    }
    if key == "terminal"
        && !["command", "cmd"].iter().any(|key| {
            args.get(*key).is_some_and(|value| {
                value.is_string()
                    || value.as_array().is_some_and(|items| {
                        !items.is_empty() && items.iter().all(Value::is_string)
                    })
            })
        })
    {
        return Action::Output;
    }
    match canonical_name(name) {
        "read_file" => Action::Read,
        "read_many_files" => Action::ReadMany,
        "list_directory" => Action::List,
        "grep_search" | "file_search" | "semantic_search" | "search_files" | "search" => {
            Action::Search
        }
        "create_file" => Action::Create,
        "write_file" => Action::Write,
        "edit_file" => Action::Edit,
        "apply_patch" => Action::Patch,
        "move_file" => Action::Move,
        "delete_file" => Action::Delete,
        "execute_command" => Action::Execute,
        "command_output" => Action::Output,
        _ => Action::Unknown,
    }
}

pub(crate) fn effective_name<'a>(name: &'a str, args: &Value, profile: Option<&str>) -> &'a str {
    match classify(name, args, profile) {
        Action::Read => "read_file",
        Action::Create => "create_file",
        Action::Edit => "edit_file",
        Action::Orchestrate => "agent_orchestration",
        Action::Output => "command_output",
        Action::Search if canonical_name(name) == "search_files" => {
            let profile = profile.unwrap_or_default().to_ascii_lowercase();
            if profile.starts_with("roo") || args.get("regex").is_some() {
                "grep_search"
            } else if name.to_ascii_lowercase().contains("filesystem")
                || args.get("pattern").is_some()
            {
                "file_search"
            } else {
                "search_files"
            }
        }
        _ => canonical_name(name),
    }
}

pub(crate) fn request_profile(request: &crate::models::ChatCompletionRequest) -> Option<&str> {
    ["reincore_delegation_profile", "reincore_client_id"]
        .iter()
        .find_map(|key| {
            request
                .extra
                .get(*key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
}

pub(crate) fn patch_text(args: &Value) -> Option<&str> {
    ["patch", "input", "patchText", "diff"]
        .iter()
        .find_map(|key| args[*key].as_str())
}

pub(crate) fn is_preview(value: &Value) -> bool {
    ["dry_run", "dryRun", "preview"]
        .iter()
        .any(|key| value[*key].as_bool() == Some(true))
}

pub(crate) fn is_cached_result(value: &Value) -> bool {
    value["cache_hit"].as_bool() == Some(true) || value["cacheHit"].as_bool() == Some(true)
}

pub(crate) fn is_pending_result(value: &Value) -> bool {
    let status = value["status"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(
        status.as_str(),
        "pending" | "queued" | "running" | "building" | "started" | "in_progress" | "in-progress"
    ) || ["session_id", "sessionId"]
        .iter()
        .any(|key| value.get(*key).is_some_and(|v| !v.is_null()))
        && value
            .get("exit_code")
            .or_else(|| value.get("exitCode"))
            .and_then(Value::as_i64)
            .is_none()
}

pub(crate) fn is_failed_result(value: &Value) -> bool {
    let status = value["status"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    value["is_error"].as_bool() == Some(true)
        || value["isError"].as_bool() == Some(true)
        || value["success"].as_bool() == Some(false)
        || matches!(
            status.as_str(),
            "failed" | "failure" | "error" | "cancelled" | "canceled"
        )
        || value
            .get("exit_code")
            .or_else(|| value.get("exitCode"))
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0)
        || value
            .get("error")
            .is_some_and(|v| !v.is_null() && v != &Value::Bool(false) && v.as_str() != Some(""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_agent_aliases_without_renaming_unknown_tools() {
        for (aliases, expected) in [
            (vec!["read", "Read", "NotebookRead", "view"], "read_file"),
            (
                vec!["edit", "Edit", "MultiEdit", "str_replace"],
                "edit_file",
            ),
            (vec!["Write", "write"], "write_file"),
            (
                vec!["execute", "shell", "Bash", "powershell"],
                "execute_command",
            ),
            (vec!["search"], "search"),
            (vec!["Grep"], "grep_search"),
            (vec!["Glob"], "file_search"),
        ] {
            for alias in aliases {
                assert_eq!(canonical_name(alias), expected);
                assert_eq!(canonical_name(&format!("mcp__editor__{alias}")), expected);
                assert_eq!(canonical_name(&format!("functions.{alias}")), expected);
            }
        }
        for name in [
            "read_file",
            "read_secrets_and_delete",
            "preview",
            "search_and_replace",
        ] {
            assert_eq!(canonical_name(name), name);
        }
    }

    #[test]
    fn researched_aliases_are_exact_namespaced_and_keep_action_distinctions() {
        for (aliases, expected) in [
            (
                vec!["read_text_file", "read_currently_open_file"],
                Action::Read,
            ),
            (
                vec!["read_many_files", "read_multiple_files", "read_files"],
                Action::ReadMany,
            ),
            (
                vec![
                    "list_files",
                    "ls",
                    "directory_tree",
                    "list_directory_with_sizes",
                ],
                Action::List,
            ),
            (
                vec![
                    "glob_search",
                    "codebase_search",
                    "search_codebase",
                    "search_file_content",
                ],
                Action::Search,
            ),
            (
                vec![
                    "replace",
                    "search_replace",
                    "edit_existing_file",
                    "NotebookEdit",
                ],
                Action::Edit,
            ),
            (vec!["write_to_file"], Action::Write),
            (vec!["create_new_file"], Action::Create),
            (vec!["apply_diff", "apply_patch", "patch"], Action::Patch),
            (
                vec![
                    "exec",
                    "run_commands",
                    "run_shell_command",
                    "run_terminal_command",
                ],
                Action::Execute,
            ),
            (
                vec!["read_command_output", "process", "Monitor"],
                Action::Output,
            ),
        ] {
            for alias in aliases {
                assert_eq!(classify(alias, &Value::Null, None), expected, "{alias}");
                assert_eq!(
                    classify(&format!("mcp__client__{alias}"), &Value::Null, None),
                    expected
                );
                assert_eq!(
                    classify(&format!("functions.{alias}"), &Value::Null, None),
                    expected
                );
            }
        }
        assert_eq!(
            classify("run_slash_command", &Value::Null, None),
            Action::Unknown
        );
    }

    #[test]
    fn composite_editors_and_collisions_use_arguments_and_profile() {
        use serde_json::json;
        for editor in [
            "file_editor",
            "str_replace_based_edit_tool",
            "str_replace_editor",
            "editor",
        ] {
            assert_eq!(
                classify(editor, &json!({"command":"view"}), None),
                Action::Read
            );
            assert_eq!(
                classify(editor, &json!({"command":"create"}), None),
                Action::Create
            );
            assert_eq!(
                classify(editor, &json!({"command":"insert"}), None),
                Action::Edit
            );
            assert!(!classify(editor, &Value::Null, None).exploration());
            assert_eq!(
                classify(editor, &json!({"command":"unknown"}), None),
                Action::Unknown
            );
            assert_eq!(
                classify(editor, &json!({"command":["insert"]}), None),
                Action::Unknown
            );
        }
        assert_eq!(
            classify("execute", &json!({"command":"npm test"}), None),
            Action::Execute
        );
        assert_eq!(
            classify("execute", &json!({"command":"npm test"}), Some("opencode")),
            Action::Orchestrate
        );
        assert_eq!(
            classify("execute", &json!({"code":"await tools.shell(...)"}), None),
            Action::Orchestrate
        );
        assert_eq!(
            effective_name("search_files", &json!({"regex":"abc"}), Some("roo-code")),
            "grep_search"
        );
        assert_eq!(
            effective_name(
                "mcp__filesystem__search_files",
                &json!({"pattern":"*.rs"}),
                None
            ),
            "file_search"
        );
        assert_eq!(
            classify("terminal", &json!({"action":"poll","sessionId":"s"}), None),
            Action::Output
        );
        assert_eq!(
            classify(
                "terminal",
                &json!({"command":["bash","-lc","cargo test"]}),
                None
            ),
            Action::Execute
        );
        for (name, expected) in [
            ("vscodeReadFile", Action::Read),
            ("vscodeWriteFile", Action::Write),
            ("vscodeRunInTerminal", Action::Execute),
            ("vscodeFileEditor", Action::Read),
        ] {
            assert_eq!(
                classify(name, &json!({"command":"view"}), None),
                expected,
                "{name}"
            );
        }
    }
}
