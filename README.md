<h1 align="center">ReinCore</h1>

<p align="center">
  <strong>Guided, never stopped.</strong><br>
  Run your agents through ReinCore.
</p>

## What is ReinCore?

ReinCore is a session-aware gateway for AI agents. It sits between an agent framework (Claude Code, Codex, LangGraph, CrewAI, AutoGen, custom MCP-based agents) and the LLM providers and MCP tool servers those agents call.

Existing gateways optimize the call. ReinCore optimizes the task. It understands that ten or two hundred calls belong to one session working toward one goal. It predicts cost before a run, degrades gracefully instead of hard-stopping, caches tool calls across session history, and resumes where it left off when a budget or connection is interrupted.

Its rule is simple: guided, never stopped. Budgets lower the cost profile of later steps instead of destroying progress. Loop detection redirects the agent instead of killing the session. State is serialized so work is resumable.

### For consumers (people/companies that use agents)

- Run agents cheaper. Prompt-prefix normalization and tool-result memoization cut repeated token and tool spend.
- Set a budget without fear. Sessions degrade to cheaper models instead of stopping mid-task, and checkpoints let you pause and resume.
- Keep one local setup. A single daemon works with Muse Code, Claude Code, Codex, Cursor, Gemini CLI, Aider, and other OpenAI-compatible clients.
- See what you spent. Per-session usage, versioned cost receipts, and Prometheus metrics show where money went.

### For developers who build agents

- Drop it in with no rewrite. The daemon exposes an OpenAI-compatible HTTP API.
- Control routing per task. Global routing policies plus per-request escape hatches select provider and model.
- Manage long sessions. Sessions, checkpoints, compaction, and durable completion accounting are built in.
- Debug agent behavior. Outcome diagnostics separate completion from success, with prediction error and cache-utilization fields, plus request logging and framework adapters for LangGraph, CrewAI, and AutoGen.

### For LLM providers who train LLMs

- See task-level behavior, not isolated calls. Session-scoped traces show how models act across multi-step goals.
- Get structured outcome signals. Completion, explicit success reports, prediction error, and token-weighted cache utilization are recorded as distinct diagnostic fields.
- Observe real cost and reliability pressure. Routing, budget-degradation, and cache-hit metrics reveal where models waste tokens, loop, or fail under constraints.

## Security, privacy, and agent health

### Security

- Every tool call passes a deterministic action gate. Admin-owned rules and JSON-schema checks decide what runs; batches are capped at 64 calls and 1 MB of arguments per call.
- Mandatory authorization boundaries cannot be downgraded. A withheld action reports why it was stopped instead of failing silently.
- MCP servers must use HTTPS. Plain HTTP is accepted only on numeric loopback for local servers.
- Provider keys and OAuth tokens live in `~/.reincore-io/credentials.json`, restricted to the current user (0600 on Unix).

### Privacy

- Local-first by default. The daemon and its SQLite database run on your laptop and bind to `127.0.0.1`. The optional Prometheus endpoint is loopback-only.
- Secrets are redacted before they cross a trust boundary. One policy covers provider requests, tool execution, logs, and stored events. Placeholders are marked as non-credentials, and execution continues instead of stopping.
- PII is masked as well. Free text is scanned for emails, phone numbers, valid US SSNs, IP addresses, and Luhn-checked card numbers. Names, addresses, and birth dates are covered through structured argument keys instead of patterns, so code rarely trips a false match. Each sanitized event records its PII catch count for the activity timeline.
- Retention is operator-controlled. Session payloads can be pruned while ledger rows survive for accounting.

### Agent health

- Every turn is scanned for tool loops, unknown tools, malformed or duplicate calls, prompt injection in tool output, and ungrounded citations. Findings are graded Warning or Critical, and a session pauses before the next provider call when needed.
- The circuit breaker is progress-aware. Already-satisfied reads are answered with a pointer to the existing transcript, genuine progress decays the redundancy score, and loops trigger escalation or a corrective message instead of a kill.
- Completion and success are recorded separately, with prediction error and cache utilization as distinct fields, plus an event log and action receipts for audit.
- Overload is handled with fail-fast admission and a retry signal, and checkpoints with resumable tokens preserve work across budget or connection interruptions.

## Built with

ReinCore was built with Muse Code, the agentic coding CLI, running Meta's Muse models. Muse Spark and Muse Wonder from that same lineup also ship in the product: both are listed in the provider model catalog, Muse Wonder (`muse-wonder-1.0`) is the default judge for opt-in LLM detection mode, and `reincore launch muse` routes Muse Code itself through the daemon.

The Rust workspace that powers ReinCore. It contains five crates:

Usage costs are [versioned estimates with sub-cent precision](#usage-costs-and-pricing-receipts). Models and prices can be extended without rebuilding through the [provider model catalog](#provider-model-catalog).

Completion, explicit success reports, prediction error, and token-weighted cache utilization have distinct [diagnostic definitions and API fields](docs/outcome-diagnostics.md).

Operational metrics and proposed scale budgets are documented in [F10 assessment and reference targets](../FEATURE_ASSESSMENT.md). Set `REINCORE_METRICS_ADDR=127.0.0.1:9091` to enable a separate loopback-only Prometheus endpoint; the [Python client timing runner](scripts/README.md) measures meaningful SSE output independently of provider timing.

| Crate | Purpose |
|-------|---------|
| `engine-core` | Engine logic in feature modules (`sessions`, `memory`, `execution`, `trust`, `storage`): providers, routing, budget, sessions, cache, context compaction, checkpoints. |
| `daemon` | Local CLI and daemon backed by SQLite. This is what you run on your laptop. |
| `mcp-gateway` | Local MCP gateway and upstream tool execution. |
| `mcp-mock` | Mock MCP server used by tests. |
| `app` | Desktop app (Tauri) linking the engine in-process. |

## Table of contents

- [Requirements](#requirements)
- [Quick start (local)](#quick-start-local)
- [CLI reference](#cli-reference)
  - [Daemon commands](#daemon-commands)
    - [`start`](#start) - start the local daemon
    - [`status`](#status) - check if the daemon is running
    - [`stop`](#stop) - stop the running daemon
  - [Provider commands](#provider-commands)
    - [`provider list`](#provider-list) - list configured providers
    - [`provider add`](#provider-add) - add or update a provider's credentials
    - [`provider remove`](#provider-remove) - remove a provider's stored credentials
    - [`provider test`](#provider-test) - test a provider
    - [`provider models`](#provider-models) - list available models
  - [App integration commands](#app-integration-commands)
    - [`launch`](#launch) - launch an app through a reincore proxy
    - [`setup`](#setup) - write persistent config without launching
    - [`restore`](#restore) - revert a reincore profile
    - [`logout`](#logout) - remove stored provider credentials

  - [Session & budget commands](#session--budget-commands)
    - [`budget`](#budget) - show or set budget limits
    - [`sessions`](#sessions) - list sessions
    - [`export`](#export) - export a session timeline as JSON
    - [`pause`](#pause) - pause a session
    - [`resume`](#resume) - resume from a checkpoint
    - [`compact`](#compact) - compact session messages
  - [Configuration commands](#configuration-commands)
    - [`config`](#config) - configure compaction settings
    - [`routing`](#routing) - show or update global routing policy
    - [`model`](#model) - show or update a provider's model mappings
  - [Database commands](#database-commands)
    - [`migrate`](#migrate) - run database migrations
  - [Special integrations](#special-integrations)
    - [GitHub Copilot CLI](#github-copilot-cli)
    - [GitHub Copilot in VS Code (BYOK)](#github-copilot-in-vs-code-byok)
    - [Muse Code](#muse-code)
    - [Ollama (local + cloud)](#ollama-local--cloud)
- [Provider configuration](#provider-configuration)
  - [Routing modes](#routing-modes)
  - [Per-request escape hatches](#per-request-escape-hatches)
  - [Managing `config.toml` from the CLI](#managing-configtoml-from-the-cli)
  - [Provider model catalog](#provider-model-catalog)
- [Using the local daemon](#using-the-local-daemon)
  - [Request logging](#request-logging)
  - [API endpoints](#api-endpoints)
- [Durable completion accounting](#durable-completion-accounting)
  - [Usage costs and pricing receipts](#usage-costs-and-pricing-receipts)

- [Framework adapters (LangGraph / CrewAI / AutoGen)](#framework-adapters-langgraph--crewai--autogen)
- [Agent tool aliases](#agent-tool-aliases)
- [Running tests](#running-tests)
- [Building a release binary](#building-a-release-binary)

## Agent tool aliases

ReinCore classifies client tool executions by their names, arguments, and available client profile. The original screenshot aliases remain supported, alongside these researched names:

| Category | Compatible names and aliases | Action |
|---|---|---|
| File read | `read`, `Read`, `NotebookRead`, `view`, `read_file`, `read_text_file`, `read_media_file`, `read_currently_open_file` | Read contents; binary output cannot supply text evidence. |
| Bulk read | `read_many_files`, `read_multiple_files`, `read_files` | Read multiple files; preserve batch scope. |
| Directory/metadata read | `list_directory`, `list_directory_with_sizes`, `list_files`, `ls`, `directory_tree`, `get_file_info`, `list_allowed_directories`, `view_repo_map`, `view_subdirectory`, `view_diff` | Explore without recording a file write. |
| Search | `search`, `Grep`, `Glob`, `grep_search`, `glob_search`, `search_file_content`, `file_search`, `codebase_search`, `search_codebase`, `codebase_tool` | Keep text, file and semantic search identities distinct. |
| Partial edit | `edit`, `Edit`, `MultiEdit`, `str_replace`, `replace`, `replace_in_file`, `search_replace`, `edit_existing_file`, `NotebookEdit` | Invalidate evidence; do not hash submitted replacement text as a complete file. |
| Full write/create | `Write`, `write`, `write_file`, `write_to_file`, `create_new_file` | Verify exact submitted bytes before reusing evidence. |
| Patch | `apply_patch`, `apply_diff`, `patch` | Extract paths from `patch`, `input`, `patchText`, or `diff`; never fabricate full-file contents. |
| Move/delete | `move_file`, `delete_file`, `remove_file` | Record affected source/destination paths and invalidate old evidence. |
| Shell execution | `execute`, `shell`, `Bash`, `powershell`, `exec`, `exec_command`, `execute_command`, `run_shell_command`, `run_commands`, `run_terminal_command` | Invalidate file evidence; completion requires actual execution proof. |
| Command output/lifecycle | `read_command_output`, `process`, `Monitor` | Not file reads or automatic proof of successful shell execution. |
| Combined editor | `file_editor`, `str_replace_based_edit_tool`, legacy `str_replace_editor`, `editor` | `command=view` reads, `create` writes, recognized mutations edit. Unknown subcommands cannot prove a write. Never memoize these tools based on name alone. |

Matching is case-insensitive and supports existing camelCase spellings and namespaces such as `functions.view` and `mcp__editor__Write`. Unknown names are not promoted to read-only tools. These aliases classify operations; they do not rename executable tools, grant permissions, or replay calls.

`search_files` is Roo's regex-content search but MCP filesystem's glob-path search. The profile, namespace, and `regex`/`pattern` arguments disambiguate it; otherwise its identity stays generic. OpenCode's `execute` is JavaScript orchestration, not shell execution. `reincore_delegation_profile` or `reincore_client_id` identifies the client; persisted external execution receipts also retain `x-reincore-client`. A `code` argument independently identifies orchestration. `Monitor` with a command invalidates evidence but does not certify command completion.

Classification applies to reuse, loop detection, continuation, file history, citations, fixture recovery, and completion verification. Search query/scope/filter changes remain distinct evidence. Failed, pending, cached, and dry-run/preview results cannot certify fresh completion. Background `exec` results stay unverified until an unambiguous, correlated `process` poll reports completion using the originating client's profile. Reused call IDs cannot supply deduplication pointers or write citations. Shell execution invalidates all retained file evidence even if its arguments contain a path. The extension captures at most 24 exact, explicitly requested paths per pass, verifies returned/submitted text against current disk bytes, and never expands bulk-read globs to discover extra files. Literal Gemini `include` paths are supported; nested metadata paths and failed individual bulk-read entries cannot supply file evidence. Reads without an explicit path or path-correlated returned text are classified but cannot supply file evidence.

Research references: [Gemini CLI](https://github.com/google-gemini/gemini-cli/blob/main/docs/reference/tools.md), [Cline](https://docs.cline.bot/tools-reference/all-cline-tools), [Roo Code](https://github.com/RooCodeInc/Roo-Code-Docs/blob/main/docs/advanced-usage/available-tools/tool-use-overview.md), [Continue](https://docs.continue.dev/ide-extensions/agent/how-it-works), [Claude Code](https://code.claude.com/docs/en/tools-reference), [OpenCode](https://opencode.ai/docs/tools/), [OpenCode v2](https://opencode.ai/v2/docs/tools/), [OpenClaw](https://docs.openclaw.ai/tools/exec/index.html), [MCP filesystem](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/README.md), [Anthropic text editor](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool), [OpenHands file editor](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-tools/openhands/tools/file_editor/definition.py).

## Requirements

- [Rust](https://rustup.rs/) 1.78 or newer (workspace uses `rust-version = "1.78"`).
- For `daemon`: nothing else; SQLite is bundled.

## Quick start (local)

```bash
# Build everything
cargo build --workspace

# Run the local daemon on the default port 8989
cargo run --bin reincore -- start

# Or bind to a custom port
cargo run --bin reincore -- start --port 8788

# The same binary is also available as `rc`
```

Once installed, you run the CLI directly as `reincore` or `rc`:

```bash
reincore start
rc start --port 8788
```

The daemon exposes an OpenAI-compatible HTTP API at `http://127.0.0.1:8989` (or the chosen port).

## CLI reference

The installed CLI binary is named `reincore` (with `rc` as a short alias). While developing from source, use `cargo run --bin reincore -- <command>`. Every command and subcommand supports `--help`.

```bash
# Top-level help
cargo run --bin reincore -- --help

# Help for a specific command
cargo run --bin reincore -- start --help
cargo run --bin reincore -- provider --help
cargo run --bin reincore -- provider add --help

```

Commands are organized into the following groups:

| Group | Commands |
|-------|----------|
| [Daemon](#daemon-commands) | `start`, `status`, `stop` |
| [Provider](#provider-commands) | `provider list`, `provider add`, `provider remove`, `provider test`, `provider models` |
| [App integration](#app-integration-commands) | `launch`, `setup`, `restore`, `logout` |

| [Session & budget](#session--budget-commands) | `budget`, `sessions`, `export`, `pause`, `resume`, `compact` |
| [Configuration](#configuration-commands) | `config`, `routing`, `model` |
| [Database](#database-commands) | `migrate` |

---

### Daemon commands

#### `start`

Start the local ReinCore daemon.

By default, `start` runs in **multi-provider mode** - all configured providers are available and the escalation classifier routes each request. Pass `--provider` to restrict the daemon to a single provider (this replaces the old `provider start-proxy` subcommand and supports the same options as `launch`).

```bash
# Multi-provider mode (default)
cargo run --bin reincore -- start
cargo run --bin reincore -- start --host 0.0.0.0 --port 8788 --db-path ~/.reincore-io/reincore.db --verbose

# Single-provider mode (replaces `provider start-proxy`)
cargo run --bin reincore -- start --provider openai --port 8789
cargo run --bin reincore -- start --provider ollama --cloud --model gpt-oss:120b
cargo run --bin reincore -- start --provider groq --mode chat
```

Options:

- `--provider` - restrict the daemon to a single provider (e.g. `openai`, `ollama`, `groq`). When omitted, all configured providers are available.
- `--mode` - API mode: `agentic` (default, uses `/v1/responses`) or `chat` (uses `/v1/chat/completions`). Only meaningful with `--provider`.
- `--host` - host address to bind to (default: `127.0.0.1`).
- `--port` / `-p` - port to listen on (default: `8989`).
- `--db-path` / `-d` - path to the local SQLite database (default: `~/.reincore-io/reincore.db`).
- `--path` - working directory (accepted for parity with `launch`; not currently used by the daemon).
- `--verbose` / `-v` - enable verbose logging.
- `--cloud` - use the cloud variant of a provider that supports both local and cloud deployment (currently only `ollama`). See [Ollama (local + cloud)](#ollama-local--cloud).
- `--model` - model name for providers that take a user-specified model (e.g. `ollama --cloud`). If omitted or invalid, you'll be prompted to enter one interactively. See [Ollama (local + cloud)](#ollama-local--cloud).

#### `status`

Show whether the daemon is running and on which host/port.

```bash
cargo run --bin reincore -- status
```

#### `stop`

Stop the running local daemon.

```bash
cargo run --bin reincore -- stop
```

---

### Provider commands

Inspect, test, add, or remove LLM providers. Provider credentials are stored in `~/.reincore-io/credentials.json` and model routing is configured in `~/.reincore-io/config.toml` (see [Provider configuration](#provider-configuration)).

> **Note:** To start a single-provider proxy, use `start --provider <label>` (see [`start`](#start) above). The old `provider start-proxy` subcommand has been removed.

#### `provider list`

List configured providers and their tier mappings.

```bash
cargo run --bin reincore -- provider list
```

For each supported provider, shows whether it's configured (via env var or saved credential), the source of the API key, and the current tier model mappings. For Ollama, reports both local (binary installed, server running) and cloud (API key set) status.

#### `provider add`

Add a provider by entering its API key (or, for Copilot, running the OAuth device flow). The credential is saved to `~/.reincore-io/credentials.json` so the provider is available in all future daemon starts without re-entering the key.

```bash
# Add OpenAI (prompts for API key interactively)
cargo run --bin reincore -- provider add openai

# Add with the key on the command line
cargo run --bin reincore -- provider add openai --key sk-...

# Add GitHub Copilot (runs OAuth device flow)
cargo run --bin reincore -- provider add copilot

# Add Ollama cloud (prompts for API key + model)
cargo run --bin reincore -- provider add ollama --cloud
```

Supported provider labels: `openai`, `anthropic`, `gemini`, `azure`, `groq`, `xai`, `deepseek`, `meta`, `copilot`, `ollama`.

**For API-key providers** (openai, anthropic, gemini, azure, groq, xai, deepseek, meta, ollama cloud): the key is prompted interactively (or accepted via `--key`) and saved to `~/.reincore-io/credentials.json`. After saving the key, you're shown the current model configuration and can optionally update the standard model by picking from a numbered list fetched live from the provider's API.

**For Copilot**: runs the GitHub OAuth device flow and stores the resulting OAuth token. See [GitHub Copilot CLI](#github-copilot-cli) for details.

**For Ollama**: in local mode (default), shows the current standard model from `~/.reincore-io/config.toml` and lets you pick a new one from installed models. In cloud mode (`--cloud`), prompts for the Ollama Cloud API key and a default cloud model. See [Ollama (local + cloud)](#ollama-local--cloud).

Options:

- `--key` / `-k` - API key for non-OAuth providers. If omitted, you'll be prompted interactively. Ignored for Copilot.
- `--cloud` - use the cloud variant of a provider (currently only `ollama`). Fetches models from the Ollama Cloud API instead of the local server.

#### `provider remove`

Remove a provider's stored credentials from `~/.reincore-io/credentials.json`.

```bash
cargo run --bin reincore -- provider remove openai
cargo run --bin reincore -- provider remove copilot
cargo run --bin reincore -- provider remove ollama
```

This only removes the saved credential - it does not unset environment variables or remove model mappings from `~/.reincore-io/config.toml`. If no stored credential exists for the provider, a message is printed and the command exits.

> **Tip:** `reincore logout <provider>` is an alias for `provider remove` - see [`logout`](#logout).

#### `provider test`

Send a minimal chat completion to test that a provider is working.

```bash
cargo run --bin reincore -- provider test openai
cargo run --bin reincore -- provider test anthropic --prompt "Hello, world!"
```

Options:

- `--prompt` / `-p` - optional prompt to send. Defaults to a simple hello.

#### `provider models`

Fetch and display available models from one provider or all configured providers.

```bash
# List models from all configured providers
cargo run --bin reincore -- provider models

# List models from a specific provider
cargo run --bin reincore -- provider models openai
```

---

### App integration commands

#### `launch`

Launch a coding/agent app and automatically route its requests through a reincore single-provider proxy. The app is spawned in a new terminal tab/window so reincore can keep running in the original terminal and the user can see proxy logs.

For apps with a known config mechanism (Codex, Claude Code, Aider, Gemini CLI, Open Code, OpenClaw), reincore also persistently updates the app's own config file so it keeps routing through reincore even when launched outside of `reincore launch` - until you run `reincore <app> --restore`. Muse Code takes its endpoint exclusively via CLI flags, so it is always launched through `reincore launch muse` with no persistent profile.

```bash
# Launch Claude Code with the default Anthropic provider
cargo run --bin reincore -- launch claude

# Launch Codex with the default OpenAI provider (agentic mode)
cargo run --bin reincore -- launch codex

# Launch an app with a different provider
cargo run --bin reincore -- launch muse --provider ollama

# Use a custom port
cargo run --bin reincore -- launch cursor --port 8789

# Launch the app in a specific working directory
cargo run --bin reincore -- launch claude --path ~/projects/my-app

# Restore an app's original config (revert reincore profile)
cargo run --bin reincore -- launch codex --restore
```

When `--path` is given, the app is launched from that directory. If the directory doesn't exist, you'll be prompted to create it. When `--path` is omitted, the app launches in the current working directory where `reincore launch` was invoked.

Supported apps and their default providers:

| App | Default provider | Env vars injected | Persistent profile |
|-----|----------------|-------------------|--------------------|
| `claude` | `anthropic` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_MODEL`, `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `~/.claude/settings.json` + VS Code `claudeCode.environmentVariables` |
| `codex`, `codex-app` | `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `~/.codex/config.toml` |
| `open-code`, `opencode` | `openai` | - (config-file driven; see profile) | `~/.config/opencode/opencode.json` (`provider.reincore`) |
| `openclaw`, `open-claw` | `anthropic` | - (config-file driven; see profile) | `~/.openclaw/openclaw.json` (`models.providers.reincore`) |
| `copilot` | `copilot` | `COPILOT_PROVIDER_TYPE`, `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_API_KEY`, `COPILOT_MODEL` | - (OAuth device flow, no config file) |
| `gemini`, `gemini-cli` | `gemini` | `GOOGLE_GEMINI_BASE_URL`, `GEMINI_API_KEY` | `~/.gemini/.env` |
| `cursor` | `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | - |
| `windsurf` | `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | - |
| `aider` | `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `~/.aider.conf.yml` |
| `continue` | `openai` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | - |
| `muse` | `meta` | - (flag driven: `--provider meta --base-url <proxy> --model reincore-auto`) | - (no config file; pass `--base-url` every time) |
| `ollama` | `ollama` | - (provider, not a client app) | - (see [Ollama (local + cloud)](#ollama-local--cloud)) |

##### API modes (`--mode`)

ReinCore's proxy supports two API wire formats:

| Mode | `--mode` value | Wire API | Endpoint | Description |
|------|---------------|----------|----------|-------------|
| **Agentic** (default) | `agentic` | `responses` | `POST /v1/responses` | OpenAI Responses API - designed for agentic workflows with tool use, reasoning, and state management. Used by Codex and similar agentic coding tools. |
| **Chat** | `chat` | `chat` | `POST /v1/chat/completions` | Standard OpenAI Chat Completions API - the universal format supported by every OpenAI-compatible provider. |

When you launch an app with `--mode agentic` (the default), reincore configures the app to use `/v1/responses` and the proxy translates the Responses API format into the internal request pipeline (classification, cost prediction, budget tracking, semantic caching) and converts the response back to the Responses API output format.

When you launch with `--mode chat`, the app uses the standard `/v1/chat/completions` endpoint directly.

Both modes run through the same reincore engine - the only difference is the wire format the client app uses to talk to the proxy.

> **Note:** Codex/Open Code/Muse Code integrations are pinned to the Responses API. For these apps, reincore always uses the `responses` wire API regardless of the `--mode` flag. The `--mode chat` option is available for apps that still support the Chat Completions API. For Open Code, the wire mode selects the AI SDK package written into `opencode.json` (`@ai-sdk/openai` for `responses`, `@ai-sdk/openai-compatible` for `chat`). OpenClaw only speaks the OpenAI Chat Completions wire (`api: "openai-completions"`), so it is always launched in `chat` mode.

##### Persistent profiles

For Codex, Open Code, OpenClaw, Claude Code, Gemini CLI, and Aider, `reincore launch` writes a persistent profile to the app's own config file so the app routes through reincore even when launched independently. The original config is backed up in `~/.reincore-io/profiles/` and can be restored with `--restore`:

```bash
# Restore Codex to its original configuration
cargo run --bin reincore -- launch codex --restore

# Restore Claude Code
cargo run --bin reincore -- launch claude --restore

# Restore Aider
cargo run --bin reincore -- launch aider --restore

# Restore Gemini CLI
cargo run --bin reincore -- launch gemini --restore
```

Options:

- `--provider` - override the default provider for this app.
- `--mode` - API mode: `agentic` (default, uses `/v1/responses`) or `chat` (uses `/v1/chat/completions`).
- `--host` - host address to bind to (default: `127.0.0.1`).
- `--port` / `-p` - port to listen on (default: `8989`).
- `--db-path` / `-d` - path to the local SQLite database.
- `--path` - working directory where the app should be launched. If the directory doesn't exist, you'll be prompted to create it. Defaults to the current working directory.
- `--verbose` / `-v` - enable verbose logging. When enabled, all incoming HTTP requests are logged with full headers and body.
- `--restore` / `-r` - restore the app's original config and exit (no proxy, no launch).
- `--cloud` - use the cloud variant of a provider that supports both local and cloud deployment (currently only `ollama`). When set, the user is prompted for an API key and a default cloud model name instead of picking from locally installed models. See [Ollama (local + cloud)](#ollama-local--cloud).
- `--model` - model name for providers that take a user-specified model (e.g. `ollama --cloud`). If omitted or invalid, you'll be prompted to enter one interactively. See [Ollama (local + cloud)](#ollama-local--cloud).

#### `setup`

Set up reincore with an app without launching it. Writes the persistent config so the app routes through reincore when launched manually from any directory. No proxy is started and no app is launched - the user simply runs the app themselves afterward.

```bash
# Set up Claude Code to route through reincore
reincore setup claude

# Set up Codex with a custom provider and port
reincore setup codex --provider groq --port 8789

# Set up Claude Code on a custom port
reincore setup claude --port 8788
```

After running `setup`, launch the app manually (e.g. `claude` or `codex`) from any directory - it will route through reincore at the configured host and port.

Options:

- `--provider` - override the default provider for this app.
- `--mode` - API mode: `agentic` (default) or `chat`.
- `--host` - host address for the proxy written into the app's config (default: `127.0.0.1`).
- `--port` / `-p` - port for the proxy written into the app's config (default: `8989`).

#### `restore`

Restore an app's original config (revert a reincore profile) without launching the proxy. This is a convenience shortcut for `launch <app> --restore`.

```bash
# Restore Codex to its original configuration
cargo run --bin reincore -- restore codex

# Restore Claude Code
cargo run --bin reincore -- restore claude

# Restore Aider
cargo run --bin reincore -- restore aider

# Restore Gemini CLI
cargo run --bin reincore -- restore gemini
```

Only apps with a persistent reincore profile (Codex, Claude Code, Gemini CLI, Aider) can be restored. For Claude Code, both `~/.claude/settings.json` and the VS Code extension's `claudeCode.environmentVariables` setting are restored. If the app has no active profile, the command prints a message and exits.

#### `logout`

Remove a provider's stored credentials from `~/.reincore-io/credentials.json`. This is an alias for [`provider remove`](#provider-remove).

```bash
reincore logout openai
reincore logout copilot
reincore logout ollama
```

If no stored credential exists for the provider, a message is printed and the command exits. This only removes the saved credential - it does not unset environment variables or remove model mappings from `~/.reincore-io/config.toml`.

---

### Session & budget commands

#### `budget`

Show or set budget limits.

```bash
# Show usage for the current session
cargo run --bin reincore -- budget

# Show usage for a specific session
cargo run --bin reincore -- budget --scope session --key <session-uuid> --usage

# Set a $5.00 budget for the current session
cargo run --bin reincore -- budget --limit 5.00

# Show budget history
cargo run --bin reincore -- budget --history
```

Options:

- `--scope` / `-s` - `session`, `project`, `day`, or `month` (default: `session`).
- `--key` / `-k` - scope key such as a session UUID or project name.
- `--limit` / `-l` - set the budget limit in dollars (e.g. `5.00`).
- `--history` - show history of budget updates.
- `--usage` - show spend and remaining budget.

#### `sessions`

List sessions, optionally filtered by project path.

```bash
cargo run --bin reincore -- sessions
cargo run --bin reincore -- sessions --path . --checkpoints
```

Options:

- `--path` / `-p` - project path to filter by; use `.` for the current directory.
- `--checkpoints` - include checkpoint tokens in the output.

Local clients can register logical projects through the local-only API:

- `GET /v1/local-projects` lists registered projects and their directories.
- `GET /v1/local-projects/resolve?path=...` checks a working directory.
- `POST /v1/local-projects` creates a project with `{ "name", "project_path" }`.
- `POST /v1/local-projects/:id/directories` assigns another working directory.

Chat clients should send `reincore_project_path` with each request. The engine
normalizes that path, resolves its durable project id, and scopes external
session lookup to that project. A project may own multiple directories.

#### `export`

Export a session's timeline (tasks, requests, tokens, spend) as JSON. Prints to stdout, or writes to a file with `--output`. Requires the daemon to be running.

```bash
cargo run --bin reincore -- export --session <session-uuid>
cargo run --bin reincore -- export --session <session-uuid> --output timeline.json
```

Options:

- `--session` / `-s` - session UUID to export.
- `--output` / `-o` - write to this file instead of stdout.

#### `pause`

Pause a session and return a resumable checkpoint token.

```bash
cargo run --bin reincore -- pause --session <session-uuid>
cargo run --bin reincore -- pause --session <session-uuid> --reason "lunch-break"
```

Options:

- `--session` / `-s` - session UUID to pause.
- `--reason` / `-r` - optional reason for the pause (default: `user-pause`).

#### `resume`

Resume a session from a checkpoint token.

```bash
cargo run --bin reincore -- resume --token <checkpoint-token>
```

Options:

- `--token` / `-t` - resumable checkpoint token.

#### `compact`

Compact previous chat messages for a session and persist a summary event to the session log. This is an offline counterpart to the automatic compaction that runs during each chat completion: in-request compaction is transient (it only shrinks the outbound request), while `compact` writes a durable summary that future session reloads can reuse. The former `compress` command remains available as a compatibility alias.

```bash
cargo run --bin reincore -- compact --session <session-uuid>
cargo run --bin reincore -- compact --session <session-uuid> --mode max --json
```

Options:

- `--session` / `-s` - session UUID to compact.
- `--mode` / `-m` - compaction mode: `low`, `mid`, or `max`. Omit to use the mode configured via `reincore config --compaction-mode` (default `low`).
- `--json` - print the retained messages as JSON.

---

### Configuration commands

#### `config`

Configure global context compaction settings. These are stored in the local database and applied to every chat completion by the workflow. Compaction is **off by default** (`mode = none`): the workflow skips full semantic compaction until you set a mode. Setting `--compaction-mode` both picks the level and enables compaction; `--compaction false` clears the mode (off). Tier-3 budget pressure still forces `max` compaction regardless of this setting. The former `--compression` and `--compression-mode` spellings remain compatibility aliases.

```bash
# Enable compaction at a specific mode (also turns it on)
cargo run --bin reincore -- config --compaction-mode low

# Or enable compaction at the currently configured mode (defaults to low)
cargo run --bin reincore -- config --compaction true

# Turn compaction off
cargo run --bin reincore -- config --compaction false
```

Options:

- `--compaction <true|false>` - enable (at the configured mode, or `low`) or disable (clear the mode) automatic compaction. Default: off.
- `--compaction-mode <MODE>` - set the compaction level: `low`, `mid`, or `max`. Setting a mode enables compaction.

#### `routing`

Show or update the global routing policy in `~/.reincore-io/config.toml`. Run with no flags to print the current policy.

```bash
# Show the global routing policy
reincore routing

# Update global routing fields (any subset of flags)
reincore routing --mode locked --minimum-quality high --allow-degradation false
```

Options:

| Flag | Values |
|------|--------|
| `--mode` | `auto` \| `custom` \| `locked` |
| `--provider-strategy` | `same_provider` |
| `--minimum-quality` | `economy` \| `balanced` \| `high` |
| `--allow-upgrade` | `true` \| `false` |
| `--allow-degradation` | `true` \| `false` |

See [Routing modes](#routing-modes) for details on each mode.

#### `model`

Show or update a provider's model mappings and per-provider routing override in `~/.reincore-io/config.toml`. Run with no model flags to print the current configuration for the provider.

```bash
# Show a provider's model mappings and effective routing policy
reincore model openai

# Set model mappings (standard/lite/pro/ultra). Pass an empty string to clear
# a tier override, e.g. to let Lite be auto-derived again.
reincore model openai --standard gpt-5.6-terra --lite gpt-5.6-luna
reincore model openai --lite ""

# Set a per-provider routing override (inherits global [routing] for omitted fields)
reincore model openai --routing-mode custom --routing-minimum-quality balanced
```

Options:

| Flag | Values |
|------|--------|
| `--standard` | standard (anchor) model id |
| `--lite` / `--pro` / `--ultra` | tier model id (empty string clears) |
| `--routing-mode` | `auto` \| `custom` \| `locked` |
| `--routing-minimum-quality` | `economy` \| `balanced` \| `high` |

See [`daemon/config.example.toml`](daemon/config.example.toml) for a complete example, including the default `[privacy]` and `[interventions]` policies. Local users can edit those sections and restart the engine. `REINCORE_PRIVACY_POLICY` and `REINCORE_INTERVENTION_POLICY` remain explicit process-level JSON overrides. API keys should not be placed in this TOML file.

Intervention modes are `observe`, `warn`, `repair_once`, `block`, and `pause`. The event log records the detector evidence, configured mode, response taken, recovery result, latency, and extra tokens. Session, task, and request-group cost/outcome projections expose aggregate intervention metrics, including false alarms, repeated questions, successful recoveries, task completion, and cost per completed task. Authorization and tenant isolation always block and are not affected by heuristic mode settings.

---

### Database commands

#### `migrate`

Run database migrations and exit. This creates or upgrades the local SQLite database schema.

```bash
cargo run --bin reincore -- migrate
cargo run --bin reincore -- migrate --db-path ~/.reincore-io/reincore.db
```

Options:

- `--db-path` / `-d` - path to the local SQLite database (default: `~/.reincore-io/reincore.db`).

---

### Special integrations

#### GitHub Copilot CLI

ReinCore supports two modes for launching GitHub Copilot CLI:

##### Copilot endpoint mode (default)

By default, `reincore launch copilot` authenticates via the **GitHub OAuth Device Flow** - no PAT or API key needed. ReinCore ships with built-in OAuth App credentials, so it works out of the box.

When you run `reincore launch copilot`, you'll be prompted to select your GitHub account type:

```
┌─── Select GitHub Account Type ─────────────────────────
│  1. GitHub (github.com)
│  2. GitHub Enterprise Cloud with data residency (*.ghe.com)
└────────────────────────────────────────────────────────
```

If you select GitHub Enterprise, you'll be asked for your instance URL (e.g. `https://enterprise.ghe.com`).

Then the OAuth device flow begins:

```
┌─── GitHub Copilot Login ──────────────────────────────
│ To authenticate with GitHub Copilot:
│  1. Open: https://github.com/login/device
│  2. Enter the code: ABCD-1234
│ Waiting for authorization...
└────────────────────────────────────────────────────────
```

Open the URL in your browser, enter the code, and authorize the app. The OAuth token is cached for the session.

**Individual vs Business/Enterprise**: ReinCore automatically detects your Copilot plan type:
- **Individual subscribers** (free or paid): The raw OAuth token is used directly with `api.individual.githubcopilot.com` - no token exchange needed.
- **Business/Enterprise subscribers**: The OAuth token is exchanged for a short-lived Copilot token via `copilot_internal/v2/token`, then used with `api.githubcopilot.com`.

**Free tier**: If your GitHub account is on the free Copilot tier, all model tiers (Lite/Pro/Ultra) resolve to the same limited model (`claude-haiku-4-5`). No model selection is needed.

**Paid tier**: If your account has a paid Copilot subscription, you'll be prompted to select a model family:
- **OpenAI**: `gpt-5.4-mini` (lite), `gpt-5.4` (pro), `gpt-5.5` (ultra)
- **Anthropic**: `claude-4.5-haiku` (lite), `claude-5-sonnet` (pro), `claude-4.8-opus` (ultra)

You can skip the prompt by setting `COPILOT_MODEL_FAMILY=openai` or `COPILOT_MODEL_FAMILY=anthropic` as an env var.

**OAuth App override**: ReinCore ships with built-in OAuth App credentials. To use your own OAuth App, set:
```bash
export COPILOT_CLIENT_ID=Ov23li...
export COPILOT_CLIENT_SECRET=...
```
Create an OAuth App at https://github.com/settings/developers (callback URL: `http://localhost:8989/callback`).

**GitHub Enterprise instance override**: Set `COPILOT_GITHUB_INSTANCE` to skip the account type prompt:
```bash
export COPILOT_GITHUB_INSTANCE=https://enterprise.ghe.com
```

##### BYOK mode

If you don't have a Copilot subscription, you can use BYOK (Bring Your Own Key) mode. The Copilot CLI is pointed at reincore's proxy, which routes to your own provider:

```bash
# Launch Copilot CLI in BYOK mode (uses your own API keys)
reincore launch copilot --provider openai
reincore launch copilot --provider groq
```

#### GitHub Copilot in VS Code (BYOK)

To use ReinCore with Copilot **inside VS Code** (rather than the CLI), register the proxy as a custom model via VS Code's **Bring Your Own Key (BYOK)** feature. You don't use `reincore launch copilot` here - you run the ReinCore daemon and point VS Code at it.

1. **Start ReinCore as a standalone proxy in `chat` mode.** VS Code's BYOK "OpenAI Compatible" provider speaks OpenAI Chat Completions, so use `--mode chat` (not the default agentic/responses):

   ```bash
   reincore start --provider openai --mode chat --host 127.0.0.1 --port 8989
   ```

   This serves `POST http://127.0.0.1:8989/v1/chat/completions` and `GET http://127.0.0.1:8989/v1/models`. Omit `--provider` to keep all configured providers available. Keep this process running while you use Copilot.

2. **Register it in Copilot Chat.** Open the Copilot Chat view → click the model picker dropdown → **Manage Models...** → choose **OpenAI Compatible** (or **OpenAI** with a custom base URL) and enter:
   - **Base URL:** `http://127.0.0.1:8989/v1`
   - **API key:** `reincore` (any non-empty string - ReinCore uses your own configured provider key upstream)
   - **Model ID:** `reincore-auto` - ReinCore picks the actual model itself (tier classification + routing), so this is just a stable handle. You can also pass a specific id from `curl http://127.0.0.1:8989/v1/models` to pin one.

3. **Select the model** in the Copilot Chat model dropdown.

**Caveats:**
- BYOK custom models work in the Copilot **Chat / Ask / Edit / Agent** panel only. Inline gray-text completions always go through GitHub's own models and can't be redirected to a custom endpoint - this is a Copilot limitation, not ReinCore.
- Use `--mode chat`. The default `agentic` mode serves `/v1/responses`, which the BYOK OpenAI-compatible client won't call.

#### Muse Code

Like the other coding agents, Muse Code only needs a base URL and the `reincore-auto` model - all provider configuration lives in ReinCore:

```bash
muse exec --provider meta --base-url http://127.0.0.1:8989 --model reincore-auto 'Reply with the single word: ok'
```

`reincore launch muse` starts a single-provider proxy for Muse's default provider (Meta) and opens Muse in a new terminal with those flags already set. Muse Code has no persistent base-URL setting, so outside a launched terminal pass `--base-url` on every invocation.

Routing without headers: Muse Code cannot send custom request headers, so ReinCore resolves the provider from the model id instead. `reincore-auto` always routes to the default provider; a model id owned by a configured provider (e.g. `muse-spark-1.3`) routes straight to that provider - so once Meta is configured below, Muse traffic defaults to Muse models. An explicit `X-ReinCore-Provider` header still wins when present; requests naming no owned model still require the header in multi-provider mode.

To serve Muse Code with Muse models, configure the Meta provider (needs a `MODEL_API_KEY` from the Meta developer console):

```bash
# Store the key, then map the tiers to Muse models
reincore provider add meta --key <MODEL_API_KEY>
reincore model meta --standard muse-spark-1.3 --pro muse-spark-1.3 --lite muse-spark-1.3 --ultra muse-wonder-1.0

# Restart the daemon so it registers Meta, then run Muse Code against it
reincore start --port 8989
muse exec --provider meta --base-url http://127.0.0.1:8989 --model reincore-auto 'Reply with the single word: ok'
```

Without `META_API_KEY`, Meta is not registered and Muse requests fall back to the first configured provider (e.g. Ollama).

Notes:
- `--provider local` does **not** work for this: in Muse Code `local` means on-device Glimmer (macOS-only, 64 GiB RAM), not a generic base-URL client. Use `meta` with `--base-url`.
- Muse Code speaks its own Meta-gateway dialect: `GET {base_url}/muse-code/models` for the catalog and `POST {base_url}/responses` (no `/v1` prefix) for turns. ReinCore serves both alongside the standard `/v1/models` and `/v1/responses`.
- Every Responses SSE frame carries a `sequence_number` (required by Muse Code; other clients ignore it), output items carry stable `msg_`/`fc_` ids, usage is reported with both `prompt_tokens`/`completion_tokens` and `input_tokens`/`output_tokens` names, and Muse Code's `{type: "namespace", tools: [...]}` tool group is expanded into individual function definitions upstream.
- Responses output contains only standard fields; ReinCore diagnostics stay in the chat-pipeline payload, since Muse Code rejects unknown top-level fields.

#### Ollama (local + cloud)

ReinCore supports two deployment modes for Ollama: **local** (default) and **cloud**. Ollama is treated as a **free provider** - per-token pricing for community Ollama models isn't tracked in the model catalog, so the cost predictor reports zero-cost predictions and the budget-degradation state machine never advances tiers. There is no model degradation: all tiers (Lite/Pro/Ultra) resolve to the single user-selected model.

##### Local mode (default)

`reincore launch ollama` (or `reincore launch ollama --cloud=false`) sets up a self-hosted Ollama server:

1. **Checks for the `ollama` binary** on PATH. If not found, prints an install hint (`winget install Ollama.Ollama` / `brew install ollama` / `curl -fsSL https://ollama.com/install.sh | sh`) and aborts.
2. **Verifies the server is running** at `OLLAMA_BASE_URL` (default `http://localhost:11434`) by polling `/api/tags`. If unreachable, tells you to start it with `ollama serve` and aborts.
3. **Fetches installed models** from `/api/tags` and presents a numbered list:
   ```
   Select the Ollama model to use:
     1. llama3.1:8b
     2. llama3.1:70b
     3. qwen2.5-coder:14b
   Enter choice [1-3]:
   ```
   You can enter either the number or the model name.
4. **Starts the proxy** with all three tiers mapped to the selected model.

```bash
# Local mode (default)
cargo run --bin reincore -- launch ollama

# Point at a non-default local Ollama server
OLLAMA_BASE_URL=http://localhost:11435 cargo run --bin reincore -- launch ollama
```

If no models are installed, the command aborts with a hint to run `ollama pull <model>` (browse available models at https://ollama.com/library).

##### Cloud mode (`--cloud`)

`reincore launch ollama --cloud` sets up the Ollama Cloud API at `https://ollama.com` (see https://docs.ollama.com/cloud):

1. **Reads the API key** from a saved credential, the `OLLAMA_API_KEY` env var, or an interactive prompt (with a link to https://ollama.com/settings/keys). The key is stored for future launches.
2. **Resolves the model**: if `--model` was provided and is valid, uses it directly; if `--model` was omitted or the model isn't available, prompts for a default cloud model name (e.g. `gpt-oss:120b`) and re-prompts on invalid input until a valid model is entered.
3. **Validates the model** against `https://ollama.com/api/tags` - fails fast at config time if the model isn't available, rather than on the first request.
4. **Starts the proxy** with the API key sent as `Authorization: Bearer <key>` on every request.

```bash
# Cloud mode (prompt for model)
cargo run --bin reincore -- launch ollama --cloud

# Cloud mode with --model (skip the model prompt)
cargo run --bin reincore -- launch ollama --cloud --model gpt-oss:120b

# Pre-set the API key to skip the prompt
export OLLAMA_API_KEY=your_api_key
cargo run --bin reincore -- launch ollama --cloud --model gpt-oss:120b
```

If `--model` is provided but the model isn't available at the endpoint, reincore prints a warning and falls back to the interactive prompt so you can pick a valid one.

The API key is saved to `~/.reincore-io/credentials.json` under the `ollama` provider label and reused on subsequent cloud launches. Run `reincore logout ollama` to remove it.

##### No model degradation

Because Ollama models have no known per-token pricing in the ReinCore catalog, the provider is marked as **free**:

- `CostPredictor::predict()` returns a zero-cost prediction (`$0.00 - $0.00`).
- The progressive budget state machine records zero spend per request, so it never advances from Tier1 → Tier2 → Tier3 → Tier4.
- All three model tiers (Lite/Pro/Ultra) resolve to the same user-selected model - the escalation classifier still runs for logging/auditing, but no tier-based model switch ever happens.

## Provider configuration

Provider credentials remain in environment variables or ReinCore's local credential store. Model routing is configured separately in `~/.reincore-io/config.toml` (or the path in `REINCORE_CONFIG_PATH`) so normal routing choices do not need to live in environment variables.

| Provider | Required env var | Optional env vars |
|----------|------------------|-------------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL`, `OPENAI_ORG_ID` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Gemini | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_BASE_URL` |
| Groq | `GROQ_API_KEY` | `GROQ_BASE_URL` |
| xAI Grok | `XAI_API_KEY` | `XAI_BASE_URL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| Meta Muse | `META_API_KEY` | `META_BASE_URL` (default: `https://api.meta.ai/v1`) |
| GitHub Copilot | OAuth (built-in) | `COPILOT_CLIENT_ID`, `COPILOT_CLIENT_SECRET`, `COPILOT_GITHUB_INSTANCE`, `COPILOT_MODEL_FAMILY` |
| Ollama (local) | - (no auth) | `OLLAMA_BASE_URL` (default: `http://localhost:11434`) |
| Ollama (cloud) | `OLLAMA_API_KEY` (prompted at launch) | - (fixed base URL: `https://ollama.com`) |

The recommended hybrid configuration sets one standard model per provider and pins only tiers that need a specific override:

```toml
# ~/.reincore-io/config.toml

# Global routing defaults (optional - omitted fields use the recommended
# `auto` defaults).
[routing]
mode = "auto"                       # auto | custom | locked
provider_strategy = "same_provider" # same_provider (cross-provider is opt-in)
minimum_quality = "balanced"        # economy | balanced | high
allow_upgrade = true                # route to a stronger tier when justified
allow_degradation = true            # let budget pressure lower the tier

[providers.openai]
standard_model = "gpt-5.6-terra"
# Optional overrides. Omit Lite to let ReinCore choose a safe cheaper model.
# lite_model = "gpt-5.6-luna"
ultra_model = "gpt-5.6-sol"
# Per-provider routing override (inherits global [routing] for omitted fields).
# [providers.openai.routing]
# mode = "locked"

[providers.anthropic]
standard_model = "claude-sonnet-5"
# Short aliases are accepted too: standard, lite, pro, ultra.
```

### Routing modes

ReinCore builds a routing ladder around the standard model and picks a tier per request. Three modes control how that ladder is used:

| Mode | Behavior |
|------|----------|
| `auto` (default) | User selects a standard model. ReinCore derives a safe cheaper Lite sibling from the catalog and keeps Pro/Ultra at the standard unless explicitly overridden. |
| `custom` | User explicitly configures LITE, PRO, and ULTRA. ReinCore never derives a model - an unmapped tier falls back to the standard model. Preserves the original three-model behavior. |
| `locked` | Always use the standard model. Budget monitoring still runs and may checkpoint an exhausted session, but ReinCore never switches tiers without stopping or asking. |

Resolution order is: explicit tier override, safe automatic Lite degradation from the standard model (in `auto` mode only), then the standard model itself. Pro and Ultra stay at the user-selected standard unless explicitly overridden, so automatic routing never upgrades above the chosen ceiling. Existing provider-specific variables such as `OPENAI_LITE_MODEL`, `OPENAI_PRO_MODEL`, and `OPENAI_ULTRA_MODEL` remain supported as highest-precedence explicit overrides; `OPENAI_STANDARD_MODEL` (and the equivalent provider-prefixed variable) overrides the TOML standard model. `REINCORE_ROUTING_MODE` overrides the configured mode for all providers.

### Per-request escape hatches

A client can override routing for a single call by sending fields in the request body (OpenAI-style extra fields, forwarded through the Responses and Messages APIs too):

| Field | Effect |
|-------|--------|
| `reincore_routing_mode` | `auto`, `custom`, or `locked` for this request only. |
| `reincore_minimum_tier` | Never route below `lite`, `pro`, or `ultra`. |
| `reincore_maximum_tier` | Never route above the given tier. |

The selected tier, the routing reason, and any routing adjustment are returned in the response as `reincore_routing_reason` and `reincore_routing_adjustment`.

### Detection switches (lexical or LLM)

Prompt-injection scanning, task-boundary detection, and task typing run on cheap local heuristics by default. Each can be flipped to a Meta Muse `muse-wonder` model judgment in `~/.reincore-io/config.toml`:

```toml
[detection.prompt_injection]
mode = "llm" # lexical (default) | llm
[detection.task_boundary]
mode = "lexical" # lexical | llm
[detection.task_type]
mode = "lexical" # lexical | llm
```

In `llm` mode the engine sends a bounded analysis prompt to `muse-wonder-1.0` (configurable via `llm_model`, with `llm_timeout_seconds` and `llm_max_input_chars` bounds) and maps the verdict back onto the same agent-health and task-assignment results, so downstream behavior is identical. Any judge failure - missing Meta credentials, timeout, API error, unparseable verdict - falls back to the lexical check with a warning. `llm` mode needs the `meta` provider configured (`META_API_KEY`) and adds latency plus usage cost on hot paths such as the pre-tool-call gate.

### Managing `config.toml` from the CLI

On first run (e.g. `reincore start`), ReinCore creates `~/.reincore-io/config.toml` from [`config.example.toml`](daemon/config.example.toml) if it does not already exist, so a fresh install has a writable routing/model configuration immediately. You can also point `REINCORE_CONFIG_PATH` at another file.

Two CLI commands read and update the file without editing it by hand. Edits preserve your comments and unrelated sections. See [`routing`](#routing) and [`model`](#model) in the CLI reference above for full flag details.

```bash
# Show the global routing policy.
reincore routing

# Update global routing fields (any subset of flags).
reincore routing --mode locked --minimum-quality high --allow-degradation false

# Show a provider's model mappings and effective routing policy.
reincore model openai

# Set model mappings (standard/lite/pro/ultra). Pass an empty string to clear
# a tier override, e.g. to let Lite be auto-derived again.
reincore model openai --standard gpt-5.6-terra --lite gpt-5.6-luna
reincore model openai --lite ""

# Set a per-provider routing override (inherits global [routing] for omitted
# fields).
reincore model openai --routing-mode custom --routing-minimum-quality balanced
```

### Provider model catalog

Local and cloud engines share [the JSON model catalog](engine-core/catalog/models.json). It contains model IDs, explicit aliases, context/output limits, modalities, routing rankings and versioned pricing schedules. Rankings are curated routing judgments, not benchmark scores.

To add a compatible model or update prices without rebuilding, copy the catalog to an operator-managed JSON file, edit complete records in its `models` array, keep `"schema_version": 1`, and set this variable before starting the engine:

```powershell
$env:REINCORE_MODEL_CATALOG = 'C:\ReinCore\models.json'
```

Restart an existing daemon to load changes. Records replace bundled entries by provider and exact model ID; unlisted models remain available. Provider names are normalized to lowercase. Aliases must be explicit, and copied entries for new models should omit storage IDs and timestamps. Set `auto_select: false` to exclude a model from automatic selection; this does not restrict explicit access. New models still require a compatible provider adapter and provider-account access.

Each `pricing` object records its currency (`USD`), version, verification date, source URLs and effective periods. Periods contain inclusive input-token bands and UTC minute-of-day windows (start inclusive, end exclusive). Rates separately cover input, output, cache reads, ordinary/5-minute and one-hour cache writes, and their batch equivalents. Missing rates or `pricing: null` mean unknown, not free. The calculator does not infer prices from model names.

Bundled rates use `rate_scale: 1000`: multiply a published USD-per-million-token price by **1,000,000** to obtain the stored integer. For example, `$0.0375 / million` becomes `37500`. The legacy `input_price` and `output_price` fields remain preflight fallback heuristics; actual accounting uses the pricing schedules.

Invalid configured files fail initialization, including duplicate aliases, invalid rates or overlapping pricing bands. Local migration **1** persists pricing, aliases, selection flags and notes. Startup seeds the loaded catalog; editing database rows alone does not update the in-process prices. Keep the operator file under version control and review its source prices regularly. There is no automatic price synchronization or hot reload.

## Using the local daemon

Once the daemon is running, send OpenAI-compatible requests to it:

```bash
curl http://127.0.0.1:8989/v1/models

curl http://127.0.0.1:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ReinCore-Provider: openai" \
  -d '{
    "model": "reincore-pro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The daemon also supports the OpenAI Responses API for agentic clients:

```bash
curl http://127.0.0.1:8989/v1/responses \
  -H "Content-Type: application/json" \
  -H "X-ReinCore-Provider: openai" \
  -d '{
    "model": "reincore-auto",
    "instructions": "You are a helpful coding assistant.",
    "input": "Explain how Rust ownership works."
  }'
```

When the local daemon has more than one provider registered, `X-ReinCore-Provider` is required on `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`. Missing, empty, or unknown provider headers return `400 Bad Request`. A daemon started with `--provider <label>` remains single-provider and does not require the header.

The `/v1/responses` endpoint translates the Responses API format (`input` + `instructions`) into the internal request pipeline and converts the response back to the Responses API output format. Both endpoints run through the same engine - classification, cost prediction, budget tracking, and semantic caching work identically regardless of which API format the client uses.

#### Semantic response-cache safety

Response reuse is scoped to a resolved workspace and project and requires an
exact fingerprint match for the provider, resolved model, stable system
prefix, tool schema, generation parameters, application/config version, and
source-data version. Entries expire after 10 minutes by default; callers may
set `reincore_semantic_cache_ttl_seconds` to a shorter value (values are capped
at one hour), set `reincore_safe_to_reuse: false` to opt out, or set
`reincore_semantic_cache: false` to disable lookup and storage.

Automatic answer reuse requires similarity of at least `0.97`. Lower matches
are returned only as explicit candidate objects (payload plus provenance) in
`reincore_semantic_retrieval_candidates`; they are never placed in `choices`
or silently substituted as the answer. Requests with tools, tool traffic,
multipart attachments, assistant/conversation history, multiple substantive
user turns, or `reincore_live_state: true` are not reusable, even when the
caller supplies a `reincore_embedding`.

An application can enable a narrowly scoped immutable tool workflow by adding
its workflow id to the server-side comma-separated
`REINCORE_SEMANTIC_CACHE_IMMUTABLE_WORKFLOWS` allow-list and sending both
`reincore_immutable_workflow` and a non-empty
`reincore_source_data_version`. Changing that version invalidates reuse by
changing the request fingerprint. Set `REINCORE_CONFIG_VERSION` when deploying
a behavior-changing application configuration so old cache rows cannot match.

The daemon also supports the Anthropic Messages API format at `/v1/messages`, which is what Claude Code uses when `ANTHROPIC_BASE_URL` is set. This endpoint translates the Messages API format (`system` + `messages` with content blocks) into the internal request pipeline and converts the response back to the Messages API format. A `/v1/messages/count_tokens` endpoint is also available for token counting.

```bash
curl http://127.0.0.1:8989/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

All three API formats (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) run through the same reincore engine - the only difference is the wire format the client uses to talk to the proxy.

### Provider rate-limit metadata

ReinCore preserves upstream provider response headers on
`ChatCompletionResponse::response_headers`. Rust callers can read an exact
provider header with `response.header(name)` or use
`response.rate_limit_remaining()` for the common request, token, input-token,
and output-token remaining values.

The local and cloud HTTP gateways also forward rate-limit response headers on
`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and the Gemini
compatibility endpoints, including streamed responses. Header names stay in
the provider's native format-for example,
`x-ratelimit-remaining-requests` for OpenAI-compatible providers and
`anthropic-ratelimit-requests-remaining` for Anthropic. `retry-after` is also
forwarded. Other upstream headers such as cookies are not forwarded.

Missing headers mean the provider did not report remaining capacity (or the
response was served from a ReinCore cache); ReinCore does not estimate or
invent a value. Gemini, DeepSeek, xAI, Meta, Ollama, and Copilot do not currently
have a documented remaining-capacity header contract, but any recognized
rate-limit headers they return in the future will be preserved and forwarded.

### OpenTelemetry tracing

The engine can export structured request
traces over OTLP. Tracing is **opt-in** and disabled by default: when no
OTLP endpoint is configured the tracing layer is not installed and the
instrumentation is a near-zero-cost no-op.

For the engine, the easiest way to get started is the bundled Jaeger
stack. Run one of the following from anywhere:

```bash
# Start the bundled Jaeger container
reincore otel start

# Enable telemetry: start Jaeger and persist OTLP settings in ~/.reincore-io/config.toml
reincore otel enable

# Disable telemetry: remove OTLP settings from ~/.reincore-io/config.toml
reincore otel disable

# Open the Jaeger UI
reincore otel open

# Stop the container
reincore otel stop

# Check whether the container is running
reincore otel status
```

`reincore otel enable` writes an `[otel]` section to `~/.reincore-io/config.toml`
and restarts the daemon to begin exporting traces automatically. After that,
plain `reincore start` will export to the bundled Jaeger container until you
run `reincore otel disable`.

You can also configure telemetry manually by editing `~/.reincore-io/config.toml`:

```toml
[otel]
enabled = true
exporter_otlp_endpoint = "http://localhost:4317"
service_name = "reincore-local"
```

Environment variables still override the config file when explicitly set:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
OTEL_SERVICE_NAME=reincore-local \
reincore start
```

Set the following environment variables to control trace export:

| Variable | Effect |
|----------|--------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | gRPC endpoint (default port 4317). Leave unset to disable tracing. |
| `OTEL_SERVICE_NAME` | Override the `service.name` resource attribute. |
| `REINCORE_OTEL_INSECURE` | `true` (default when unset) uses `http://`; set `false` for `https://`. |

The Compose stack uses Jaeger v2 with a persistent Badger store in the Docker
named volume `reincore-telemetry_reincore-jaeger-data`. Traces are retained for
30 days by default and survive daemon restarts, Jaeger container recreation,
and `reincore otel stop` / `reincore otel start`. Set `JAEGER_TRACE_TTL` on the
environment before starting the stack to use a different retention duration.
Only explicitly removing the Docker volume (for example, Compose
`down --volumes`) purges the stored traces.

The Compose and Jaeger configuration files live at
`daemon/resources/jaeger.yml` and
`daemon/resources/jaeger-config.yml`; both are embedded in the CLI binary
as a fallback for production builds.

Each chat-completion request emits a `reincore.request` root span with nested
pipeline stage spans:

```text
reincore.request
├── reincore.stage.resolve   (session / file / task resolution)
├── reincore.stage.classify  (escalation classifier)
├── reincore.stage.predict   (pre-flight cost prediction)
├── reincore.stage.route     (provider selection + budget degradation)
├── reincore.stage.compact   (context compression / rollover / handoff)
├── reincore.stage.provider  (upstream LLM call)
├── reincore.stage.health    (agent-health / citation scanning)
└── reincore.stage.account   (event-log + ledger + calibration writes)
```

Spans are flushed when the process exits via an RAII guard returned by the
`engine_core::telemetry::maybe_init` call in each engine entry point.

### Request logging

When the daemon is started with `--verbose`, all incoming HTTP requests are logged with the full method, URI, headers (with sensitive headers like `Authorization` redacted), and body. Without `--verbose`, no request information is logged.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check. |
| `GET` | `/v1/models` | List available models. |
| `POST` | `/v1/chat/completions` | Standard OpenAI Chat Completions API. |
| `POST` | `/v1/responses` | OpenAI Responses API (agentic mode). |
| `POST` | `/v1/messages` | Anthropic Messages API (used by Claude Code). |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting endpoint. |
| `POST` | `/v1/sessions` | Create a session. |
| `GET` | `/v1/sessions` | List sessions. |
| `GET` | `/v1/sessions/:id` | Get a session. |
| `GET` | `/v1/sessions/:id/events` | Get session events. |
| `GET` | `/v1/sessions/:id/tasks` | Get request-group/task cost and outcome rollups. |
| `POST` | `/v1/sessions/:id/pause` | Pause a session. |
| `POST` | `/v1/sessions/:id/complete` | Mark a session complete and optionally record an explicit outcome report. |
| `POST` | `/v1/checkpoints/:token/resume` | Resume from a checkpoint. |
| `POST` | `/mcp` | Native MCP gateway; requires `X-ReinCore-Session`. |
| `GET` | `/v1/tools?session_id=<uuid>` | Discover configured tools authorized for the session. |
| `POST` | `/v1/tool-calls` | Execute a tool call (with exact and cloud team-semantic memoization). |

The daemon routes each request to the appropriate provider based on the escalation classifier and your configured model tiers.

In the cloud edition, tool reuse requires an explicit server-owned policy from
the registered executor. Results remain partitioned by workspace and project;
private reuse also includes the authenticated principal and key scope. Semantic
reuse additionally requires the same request group. Forward the `reincore_request_group_id` returned by the
chat response as `request_group_id` in the tool-call body or as the
`x-reincore-request-group-id` header:

```json
{
  "session_id": "00000000-0000-0000-0000-000000000000",
  "tool_name": "search_code",
  "arguments": { "query": "find the authentication middleware" },
  "safe_to_memoize": true,
  "request_group_id": "shared-agent-request"
}
```

`safe_to_memoize` is caller consent, not permission. The default executor policy
is no caching. Anonymous/unscoped requests, mutations, errors and unversioned
policies are never cached by the cloud handler. Legacy `reincore_embedding` input
is accepted but ignored: the server derives vectors from the actual arguments.
Only query-only natural-language arguments are eligible for approximate reuse;
structured filters, identifiers and path-like queries remain exact-only.

### Server-owned tool reuse policy

A registered `ToolExecutor` may implement `cache_policy(tool_call, context)` and
return `engine_core::tool_cache::ToolCachePolicy`. Policy must describe the
executor's effective tool configuration, data snapshot and credential identity;
never copy version or sharing claims directly from client arguments. Required
fields are `read_only`, `sharing`, `semantic_reuse`, `ttl_seconds`,
`policy_version`, `tool_version`, `data_namespace`, `data_version`,
`credential_identity`, and `credential_version`. Version/identity strings must be
nonblank and at most 256 bytes; TTL must be 1–600 seconds. Credential identity is
an opaque non-secret identifier, not an API key or its raw hash.

- `ToolCacheSharing::Private` isolates reuse by principal and API-key scope.
- `ToolCacheSharing::Workspace` explicitly permits cross-user reuse only inside
  the same project, for workspace keys whose caller can access the source.
  Personal and project keys always remain privately partitioned, even with this
  policy. Cross-project reuse is never enabled by this setting.
- `semantic_reuse` is a separate server opt-in. Enable it only where approximate
  query equivalence is safe; lexical similarity does not establish equal answers.
- The gateway resolves the executor once for policy and execution. Override
  `execute_in_context` for tenant-specific credentials/data; it receives the same
  trusted `ToolCacheContext`. Do not pass context between calls through mutable
  executor-global state. The gateway re-checks policy after execution and skips
  storage if the effective versions changed.

Every exact or semantic hit validates a versioned provenance envelope and reads
the source session's current authorization attributes. Deleted/reassigned or
inaccessible sources become misses. Exact keys include the full policy scope;
semantic SQL queries require the same fingerprint, which includes the embedding
model/version (`reincore-fnv1a-char-trigram-384:v1:query-only`). Similarity is
recomputed from the stored originating arguments before returning a candidate.
Cache payloads carry integrity checks, source session/owner/actor, canonical
request hash and request group. Both hit types return `source_session_id`; semantic
hits also return `cache_hit_type: "semantic_team"` and `semantic_similarity`.

Migration 21 adds an indexed semantic scope fingerprint. Existing rows remain
NULL/untrusted; legacy exact keys/payloads also cannot satisfy the new contract.
No legacy results are promoted or backfilled into trusted entries. Deploy the
updated engine to every instance to enforce the new handler behavior. Data or
credential rotation must change its server policy version/identity so future
requests miss old entries. Existing scoped invalidation APIs remain available.
Both engines populate their executor registry from the MCP configuration below.
Without configuration, no upstream tools are exposed. ReinCore connects to existing
MCP servers; it does not install or launch them.

### MCP operational limits

Local startup and cloud discovery connect to at most eight configured servers concurrently per operation,
with a global limit of 32 initializations and a 125-second initialization deadline.
Discovery responses are limited to 4 MiB in aggregate across pages. Gateway admission,
cancellation and its 125-second deadline include tool discovery, not only dispatch.

Failed calls are never automatically replayed.
Local upstream metadata changes still require restarting the daemon.

Read-only projects may use explicitly configured cache policies; operators must advance data versions when external writers change the underlying data.

### Configured MCP runtime

Engine embeds `daemon/resources/mcp.json`, an empty `mcp.json` (`{"servers": []}`) and installs it at
`$HOME/.reincore-io/mcp.json` on daemon startup or when MCP settings are opened.
Existing files are never overwritten. Set the local file path in
`~/.reincore-io/config.toml` (or the file selected by `REINCORE_CONFIG_PATH`):

```toml
[mcp]
enabled = false
config_path = "~/.reincore-io/mcp.json"
```

Path precedence is `REINCORE_MCP_CONFIG`, then `[mcp].config_path`, then
`$HOME/.reincore-io/mcp.json` when the setting is omitted. `~/` expands to the home
directory; relative paths (including environment overrides) resolve against the
directory containing `config.toml`. Blank paths are rejected. Only the default
file is installed automatically; create a custom JSON file before using its path.
`reincore mcp status` and desktop Settings display the effective path and indicate
when the environment override is active. Restart the daemon after changing the path.

Local MCP is disabled by default. Enable or disable it from **Settings → MCP tools**
in the desktop app, or from the CLI:

```sh
reincore mcp enable
reincore mcp disable
reincore mcp status
```

These commands save `[mcp] enabled = true|false` in `~/.reincore-io/config.toml`
(or `REINCORE_CONFIG_PATH`). Restart the daemon to apply changes. When disabled,
engine neither connects to configured upstream servers nor exposes `/mcp`.
Add your upstream servers to the JSON file before enabling tool discovery.

Local configuration and environment references remain supported. Discovery failures reject the affected operation; names are exposed as `<server>__<tool>`.
Only configured tools are exposed, and discovery verifies every allowed tool exists.

#### What belongs in `server.json`?

The input is **one server object**, not the `{"servers": [...]}` file wrapper,
not an MCP client configuration using `mcpServers`, and not a database record.
Do not include the `id` returned by cloud list/get responses in a save body.

For local servers, choose exactly one scope in the form or JSON:

- **Project IDs** (recommended):
  `"access": { "mode": "local", "project_ids": ["bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "dddddddd-dddd-4ddd-8ddd-dddddddddddd"] }`.
  One connection covers sessions whose persisted `project_id` is in this list,
  across all their paths. The form provides searchable multi-project selection using project names and directories. Legacy `project_id`
  remains supported and is converted to a one-element list when edited in the form.
- **Local directory**: `"access": { "mode": "local", "project_path": "C:/work/my-project" }`.
  Existing path-based configuration remains supported and matches the canonical directory.

Choose exactly one of `project_ids`, legacy `project_id`, or `project_path`.
Project lists require 1–256 unique non-nil UUIDs. A missing scope or invalid path is rejected.
Project scope controls tool availability; it does not change the upstream server's
working directory or merge cached results across paths/sessions. Configure the upstream
server to support the project's directories. Restart the local daemon after changes.

Local example (the absolute project directory must exist, and the upstream
server must expose the allowed tool):

```json
{
  "name": "repository",
  "url": "http://127.0.0.1:9000/mcp",
  "allow_loopback_http": true,
  "timeout_ms": 30000,
  "access": {
    "mode": "local",
    "project_path": "C:/work/my-project"
  },
  "tools": {
    "read_file": { "read_only": true }
  }
}
```

For an authenticated local upstream, add `"bearer_token": "replace-with-upstream-token"`
to the server object above. The desktop form, local API and CLI all persist this
value in the server entry inside `~/.reincore-io/mcp.json` (or the configured path).
Do not include the `Bearer ` prefix. Existing `bearer_token_env` configurations
remain supported; configure only one credential source per server.

{
  "name": "repository",
  "url": "https://mcp.example.com/mcp",
  "bearer_token": "replace-with-upstream-token",
  "enabled": true,
  "timeout_ms": 30000,
  "access": {
    "mode": "cloud",
    "workspace_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "project_ids": ["bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "dddddddd-dddd-4ddd-8ddd-dddddddddddd"],
    "principals": ["cccccccc-cccc-4ccc-8ccc-cccccccccccc"]
  },
  "tools": {
    "read_file": { "read_only": true },
    "write_file": { "read_only": false }
  }
}
```

| Field | Contract |
|---|---|
| `name` | Required, 1–32 ASCII letters/digits/underscores; cannot contain `__`. Becomes the prefix in `repository__read_file`. Names are unique in the file. |
| `url` | HTTPS or explicitly allowed numeric loopback HTTP. Userinfo, queries and fragments are forbidden. |
| `bearer_token_env` | **Legacy alternative.** Environment variable name. Cannot be combined with `bearer_token`. |
| `allow_loopback_http` | Optional boolean, default false. |
| `bearer_token` | Token value (1–16384 visible ASCII bytes, without the `Bearer ` prefix). Stored directly in `mcp.json`; required for non-loopback endpoints unless using the legacy env reference. Omit for unauthenticated loopback servers. |

| `timeout_ms` | Required integer, 100–120000 milliseconds. |
| `access` | Required: `mode` and `project_ids` or legacy `project_id` or absolute existing `project_path` (exactly one). |
| `tools` | Required object containing 1–256 upstream tool names. Each policy requires a `read_only` boolean. Names must match upstream discovery. |
| `tools.<name>.cache` | Optional/null; omitted means no caching. A cache policy is allowed only for read-only tools. The full versioned policy example appears below. |

Unknown fields are rejected. Saving validates the structure and local path,
without connecting upstream. Restart performs discovery.

#### Manage saved servers

Local commands edit the effective JSON file resolved using the precedence above:

```sh
reincore mcp list
reincore mcp add server.json
reincore mcp update repository server.json
reincore mcp delete repository
```

Files are optional. Local add/update accept `--stdin` instead of
a file argument. Exactly one input source is required:

```powershell
Get-Content -Raw .\server.json | reincore mcp add --stdin
Get-Content -Raw .\server.json | reincore mcp update repository --stdin
```

A program can write the same JSON directly to stdin; no temporary file is needed.
CLI input is limited to 1 MiB. Local update replaces the entire server and
requires the same name. Add rejects duplicates; update/delete reject
missing names. Invalid configuration does not replace the saved file. Local
writes and the enabled toggle use a lock and atomic replacement, preserving the target file permissions and resolving symlinks; after a crashed CLI operation, remove
its adjacent `.lock` file (for example, `mcp.json.lock` or `config.toml.lock`) only once no other management command is running.
These commands do not enable MCP automatically or connect to the upstream server.
Enable MCP separately and restart the daemon to perform discovery and apply changes.

#### Desktop form and JSON editor

Open **Settings → MCP tools → Configured servers** and choose **Add MCP server**
or **Edit** to open the right-side configuration panel. Add or edit a server using
**Form** or **Raw JSON**; the form supports name, URL, project IDs or a project path, credential
variable, timeout, loopback permission and tool allowlists/read-only policies.
Use Raw JSON for advanced cache policies. The editor manages local servers and
works while MCP is disabled. Delete requires confirmation. Saves use the same
validation, lock and atomic file replacement as the CLI. The Tauri commands are
`list_mcp_servers`, `save_mcp_server` (server object plus optional `originalName`),
and `delete_mcp_server` (name). **Reload servers** retries a failed load or refreshes the list without discarding the editor draft. Form/JSON switching preserves cache policies and rejects unsupported fields instead of dropping them. Restart the daemon to apply changes.

Example `mcp.json` for a local project (replace the path, endpoint and tool names
with those of your MCP server):

```json
{
  "servers": [{
    "name": "repository",
    "url": "http://127.0.0.1:9000/mcp",
    "allow_loopback_http": true,
    "timeout_ms": 30000,
    "access": { "mode": "local", "project_path": "C:/work/my-project" },
    "tools": {
      "read_file": {
        "read_only": true,
        "cache": {
          "read_only": true,
          "sharing": "private",
          "semantic_reuse": false,
          "ttl_seconds": 60,
          "policy_version": "1",
          "tool_version": "1",
          "data_namespace": "my-project",
          "data_version": "snapshot-1",
          "credential_identity": "repository-reader",
          "credential_version": "1"
        }
      }
    }
  }]
}
```

Edit `~/.reincore-io/mcp.json`, run `reincore mcp enable`, then start or restart your
usual local daemon. For a custom path in PowerShell, set
`$env:REINCORE_MCP_CONFIG = 'C:/config/mcp.json'` before starting the daemon. Use an existing session whose project path resolves
to the configured directory. `GET /v1/tools?session_id=<uuid>` returns
`{"data":[...],"session_id":"..."}`. Send `POST /v1/tool-calls` with that session,
`"tool_name":"repository__read_file"`, the tool's JSON arguments and
`"safe_to_memoize":true`. Repeating an eligible call returns `cache_hit:true`.
The existing `result.content` string contains the complete MCP CallToolResult
JSON, preserving content blocks, structured content and `isError`.

Configure each upstream service account with only the data/operations authorized for that project: the engine's scope check is not a sandbox for the upstream server.

Local remote endpoints require HTTPS and a `bearer_token` value stored in `mcp.json`
(or a legacy `bearer_token_env` reference), loaded at startup. The desktop form accepts
the token directly and masks the field. Local JSON contains the plaintext token,
including when viewed through Raw JSON or local configuration commands. Explicitly
enabled loopback HTTP may omit authentication. The client disables

redirects/environment proxies and redacts raw upstream errors and
credentials from diagnostics.

**Mutation/cache contract:** unlisted tools are denied. Upstream annotations and
caller memoization flags cannot grant permission. To authorize a mutation, add its
name with `{"read_only":false}`; never attach a cache policy to it. This is explicit
administrator authorization, not an interactive per-call approval flow. A mutation
invalidates this executor's cache generation before and after dispatch, including
error results. While a mutation is in flight, reuse is disabled. An interrupted,
malformed or timed-out mutation disables reuse for that executor until restart;
reconcile the upstream outcome before restarting. Mutations are never retried
automatically. Do not manually retry an ambiguous mutation without checking its
outcome or using an upstream idempotency key.

Each runtime boot has a unique cache namespace, so restarts and configuration or
credential reloads cannot resurrect old entries. Invalidation applies to this
executor/process; external writers or another engine process require coordinated
data-version changes/restarts, or caching must be omitted. Only enable caching
for data whose stability you can guarantee over the configured TTL. This runtime
does not claim distributed transactional cache coherence. Local reuse remains
session-private.

**Transport profile and limits:** the client proposes MCP `2025-06-18`, accepts negotiated `2025-06-18` or `2025-03-26`, and implements
initialization, initialized notification, paginated tools/list, tools/call and
Streamable HTTP POST JSON/SSE responses with session and negotiated protocol-version headers. See the official
[MCP transport contract](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
and [initialization contract](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).
Requests are bounded by `timeout_ms` (100–120000 ms), responses by 4 MiB, discovery
by 32 pages/4096 discovered tools, and configuration by 32 servers/256 allowlisted tools per server.
Timeouts or dropped execution futures send best-effort cancellation notifications;
upstream cancellation does not guarantee rollback and cannot survive forced process
termination. There is no automatic tool-call retry or synchronous fallback.

This is a configured HTTP tool client, not every MCP capability: stdio/process
launching, legacy HTTP+SSE, OAuth acquisition/refresh, resources/prompts, sampling,
elicitation, background subscription streams, resumable streams and hot discovery
are not implemented. Server pings receive replies; unsupported server requests receive method-not-found.
An observed tools/list_changed notification disables dispatch/reuse until restart.
Locally, restart to reconnect an expired upstream MCP session or reload tools/credentials.
Cloud reconnects on the next request and reloads configuration dynamically. Tools are discoverable through the HTTP API and the inbound MCP gateway below.
The native MCP client can supply them to its model and manage its tool loop; the
engine does not run an autonomous tool loop itself.

### MCP gateway for coding clients

The `mcp-gateway` workspace crate exposes **`/mcp` on the existing engine
listener**, in local daemon and local single-provider modes. It
adapts MCP requests to the same discovery/execution services as `/v1/tools` and
`/v1/tool-calls`; authorization, loop protection, event recording and cache
invalidation therefore apply to both interfaces.

The two MCP directions are separate:

```text
Coding client → ReinCore /mcp gateway → existing tool services → upstream MCP servers
```

Local gateway access requires `reincore mcp enable` followed by a daemon restart.
The upstream configuration defaults to `~/.reincore-io/mcp.json` locally;
`REINCORE_MCP_CONFIG` overrides it.
Configure your coding client's **Streamable HTTP** connection with these settings:

| Setting | Value |
|---|---|
| URL | `http://127.0.0.1:8989/mcp` (or your configured host and port) |
| `X-ReinCore-Session` header | UUID of an existing ReinCore session, on every POST including initialization |

| `X-ReinCore-Tool-Cache` header | Optional literal `true` to consent to policy-approved reuse; defaults to `false` |

Use the actual listener port if you changed it. The client must support custom
HTTP headers. Create the ReinCore session through the existing session API/UI
first. scoped to the configured upstream project directory.
Tool arguments
cannot override this binding. The engine follows an active successor in the
session's existing lineage, as the HTTP tool API does.

The gateway offers `initialize`, `ping`, `tools/list`, `tools/call`,
`notifications/initialized` and `notifications/cancelled`. It negotiates
`2025-06-18` or `2025-03-26`; other proposed versions receive `2025-06-18` so the
client can accept or disconnect. Subsequent requests should carry the negotiated
`MCP-Protocol-Version` header. POST bodies are JSON-RPC 2.0 single messages with
`Content-Type: application/json` and `Accept: application/json, text/event-stream`.
Responses use JSON; accepted notifications return empty HTTP 202. Unsupported
versions, missing session headers and invalid HTTP content negotiation are rejected.

Example `tools/call` body after initialization (the native client normally builds
these protocol messages itself):

```json
{
  "jsonrpc": "2.0",
  "id": "read-1",
  "method": "tools/call",
  "params": {
    "name": "repository__read_file",
    "arguments": { "path": "README.md" },
    "_meta": {
      "reincore/safe_to_memoize": true,
      "reincore/request_group_id": "task-1"
    }
  }
}
```

Per-call `reincore/safe_to_memoize` overrides the static cache-consent header.
Neither grants cache permission: executor policy remains authoritative, and
mutations/errors cannot cache. `reincore/request_group_id` enables eligible cloud
semantic reuse under existing scope checks; it does not enable local semantic
reuse. The response preserves upstream content blocks, `structuredContent` and
`isError`; `_meta.reincore` adds session, cache and tool-event identifiers. Other
engine executors returning plain text are wrapped as MCP text content. Internal
errors are redacted; loop blocks and execution failures become tool error results.

**Cancellation:** send `notifications/cancelled` with `params.requestId` equal to
the in-flight request ID, using the same session and identity headers. Cancellation
is bound to the verified principal/key scope and resolved session, drops the
execution future, and triggers the upstream runtime's best-effort cancellation.
The cancelled request closes with empty HTTP 204, without a JSON-RPC response.
The gateway admits at most 1024 active tool calls per instance, rejects duplicate
in-flight IDs within a binding and applies a 125-second execution deadline.
Request IDs are not durable idempotency keys. There is no automatic replay or retry;
inspect ambiguous mutation outcomes before retrying. A client disconnect or process
termination does not guarantee upstream cancellation or rollback. Cloud deployments
need request affinity for cancellation to reach the instance executing the call.

**Deliberate limits:** the gateway is stateless at the MCP transport level and
issues no `Mcp-Session-Id`. Every request carries its ReinCore session and is
reauthorized. GET subscriptions and DELETE return HTTP 405; no notification stream,
resumption, hot discovery, resources/prompts, sampling, elicitation, stdio, OAuth
flow or automatic client-profile installation is provided. Tools/list returns the
configured authorized list in one response and rejects pagination cursors. Browser
requests carrying any `Origin` are rejected to prevent browser-origin/DNS-rebinding
access; use a native HTTP client and keep the engine bound to loopback.
Payloads are limited to 2 MiB.

This follows the [MCP Streamable HTTP transport profile](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
with a ReinCore session header.

### Local tool cache boundaries

Local `POST /v1/tool-calls` uses the same `ToolCachePolicy` contract through
`ToolExecutor::local_cache_policy(tool_call, LocalToolCacheContext)`, defaulting
to no reuse. Implement `execute_local_in_context` to execute with that same
context. The effective repository and working directory must be represented by
the executor's `data_namespace`; change data/credential versions when they
change. Tool-definition flags and HTTP `safe_to_memoize` cannot grant permission.

Local exact reuse is session-private, including when a policy requests workspace
sharing. Keys bind the persisted session/client/project identity, canonical
absolute project path, arguments and complete policy. Missing, relative or
unresolvable project paths disable caching. This intentionally avoids inventing
cloud identities for local sessions. Semantic tool reuse is not enabled locally.

The core `ExactToolCacheEnvelope` validates scope, content integrity, request,
result and TTL. The local handler revalidates persisted session context before
replay/storage, excludes error results, rechecks policy after execution, and
rebinds replayed results to the new tool-call ID. Legacy entries become cold
misses without a SQLite migration. The legacy `MemoizationCache::lookup/store`
implementation on `LocalDatabase` now returns misses/skips writes because it
cannot carry an authorized scope; integrations must use the hardened executor
path. File invalidation still works and checks nested JSON strings literally.

## Durable completion accounting

Local chat (including completed streams) stages an immutable
`CompletionAccounting` record before settling usage. Settlement is awaited before
reporting successful completion. An accounting failure returns an HTTP error, or
an `accounting_failed` SSE error after streaming has started, instead of silently
acknowledging incomplete persistence.

SQLite uses an immediate transaction. One transaction commits the LLM event and projections,
a ledger entry with a unique accounting ID, session spend/token/calibration/tier
updates, tier-change events, and the journal's settled marker. Notifications happen only after commit.
Accounting never writes session status, task title, or other lifecycle fields.
General session upserts preserve accounting-owned values after the first settlement commits.
The ledger retains the original recording date when recovery happens later.

`AccountingStore::stage_accounting`, `settle_accounting`, and `recover_accounting`
allow safe retries using the same immutable ID and record. Reusing an ID with
changed usage is rejected. A new provider invocation is a new charge; this is
settlement idempotency, not HTTP request deduplication or free provider retries.
Local migration **1** creates the durable journal and
unique ledger reference. Existing historical rows are not guessed or backfilled.
Upgrade all engine instances before relying on this behavior.

The engine recovers pending staged usage during startup before serving.
Settlement failures retain their journal records for recovery; startup fails if
recovery cannot safely finish. No detached accounting tasks need to be drained.
A crash before usage reaches the durable journal, or an interrupted provider stream
without final usage, still needs provider-side reconciliation; the engine cannot
reconstruct usage it never received. Budgets remain soft progressive limits;
this change does not introduce reservations or hard spend caps. Tier-triggered
compaction remains best-effort maintenance after the financial transaction.

SQLite tests cover concurrent/duplicate settlement, rollback, lifecycle isolation,
and killing a child process after staging then reopening the database twice.
```

### Usage costs and pricing receipts

Accounting stores integer **nano-USD**: one cent is 10,000,000 units. A 0.01-cent call contributes 100,000 units, so 100 such calls total exactly one cent. Scaled token-rate products are summed before conversion to ledger units; only a fraction of one nano-dollar is rounded up. Whole-cent fields are compatibility/display values and must not be summed per call.

Chat responses and terminal chat SSE events expose `reincore_cost`, containing the amount (or `null`), `estimated`/`unknown` status, provider/model, currency, pricing fingerprint, selected rates, normalized usage and batch-submission flag. Recovery uses the saved receipt instead of repricing usage. Cached tokens and cache writes are priced separately, reasoning is counted once within output, and batch prices apply only to submitted batches. Title and prewarm calls also contribute durable usage accounting.

The baseline migration carries precise session and ledger fields. Unknown costs increment `unknown_cost_calls`; totals represent the known estimated portion, and unknown or zero denominators suppress efficiency scores. Explicitly free providers have zero rates. Missing usage, unknown model/category prices, uncovered schedule intervals, invalid counts and overflow remain unknown rather than falling back to tier prices.

These are standard direct-provider text-token estimates, not reconciled invoices. Non-text tariffs, cache-storage duration, paid server tools, regional/service-tier premiums and negotiated prices need additional attribution and reconciliation. Schedule selection uses accounting time, so calls crossing price boundaries may require adjustment. Historical cent-only records cannot recover discarded precision. Budgets remain soft limits.

## Framework adapters (LangGraph / CrewAI / AutoGen)

ReinCore exposes its resumable checkpoints and tiered memory manager through a set of framework-friendly HTTP endpoints. Frameworks can store the checkpoint object in their own graph state and resume execution later by posting the checkpoint token back to ReinCore.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/sessions/:id/framework-checkpoint` | Pause the session and return a `FrameworkCheckpoint` with a resumable token. |
| `POST` | `/v1/framework-checkpoints/:token/resume` | Resume a session from a checkpoint token. |
| `POST` | `/v1/sessions/:id/framework-memory` | Return the tiered memory snapshot for the session. |
| `POST` | `/v1/sessions/:id/framework-turn` | Record a framework-generated turn in the session event log. |

### Checkpoint a session

```bash
curl -X POST http://127.0.0.1:8989/v1/sessions/SESSION_ID/framework-checkpoint \
  -H "Content-Type: application/json" \
  -d '{"reason": "langgraph-node-boundary", "metadata": {"node": "research"}}'
```

Response:

```json
{
  "checkpoint": {
    "version": 1,
    "checkpoint_token": "SESSION_ID:SEQ:EXPIRY:SIGNATURE",
    "session_id": "SESSION_ID",
    "session_status": "checkpointed",
    "task_summary": "...",
    "budget_cents": 500,
    "spend_cents": 120,
    "tier": 1,
    "last_event_seq": 7,
    "expires_at": "2026-07-06T12:00:00Z",
    "metadata": {"node": "research"}
  }
}
```

### Resume from a checkpoint

```bash
curl -X POST http://127.0.0.1:8989/v1/framework-checkpoints/TOKEN/resume
```

Checkpoint tokens are single-use until their signed expiry. A successful resume appends a `checkpoint_redeemed` event containing a token fingerprint; reusing the same token returns `409 Conflict`.

### Read tiered memory

```bash
curl -X POST http://127.0.0.1:8989/v1/sessions/SESSION_ID/framework-memory \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a helpful coding assistant"}'
```

Response:

```json
{
  "snapshot": {
    "version": 1,
    "session_id": "SESSION_ID",
    "working_messages": [...],
    "episodic_summary": "[Context summary ...]",
    "tool_state_messages": [],
    "attached_files": [...],
    "captured_at": "2026-07-06T11:59:00Z"
  }
}
```

### Record a framework turn

```bash
curl -X POST http://127.0.0.1:8989/v1/sessions/SESSION_ID/framework-turn \
  -H "Content-Type: application/json" \
  -d '{"role": "assistant", "content": "I will search the codebase."}'
```

### LangGraph example

```python
import requests

REINCORE = "http://127.0.0.1:8989"

# After each graph node, checkpoint through ReinCore.
def checkpoint_node(state):
    r = requests.post(
        f"{REINCORE}/v1/sessions/{state['session_id']}/framework-checkpoint",
        json={"reason": "post-node", "metadata": {"node": state["node"]}}
    )
    state["reincore_checkpoint"] = r.json()["checkpoint"]
    return state

# On graph resume, restore from the stored token.
def resume_node(state):
    token = state["reincore_checkpoint"]["checkpoint_token"]
    requests.post(f"{REINCORE}/v1/framework-checkpoints/{token}/resume")
    return state
```

## Running tests

```bash
cargo test --workspace
```

## Building a release binary

```bash
cargo build --release
```

The binary will be at `target/release/reincore`.

## Batch API

The engine persists session-bound batch jobs for OpenAI, Anthropic and
Gemini. Gemini uses its native resumable Files API for upload, native file download
and cancellation, and the OpenAI-compatible batch create/status endpoints, as
required by [Google's documentation](https://ai.google.dev/gemini-api/docs/openai#batch-api).

Supply an existing `session_id` and a client-generated UUID `idempotency_key` when
creating work. Reuse that key with the same session, credentials and requests after
a timeout. The returned `id` is an **engine UUID**, not a provider job ID. This is a
change from the earlier provider-proxy API: the updated Python, TypeScript, Go and Rust SDKs send these fields through
`CreateBatchOptions` and recognize the statuses below. Older clients must migrate.

| Method | Endpoint | Request / response |
|---|---|---|
| POST | `/v1/batches` | `{"session_id":"<session UUID>","idempotency_key":"<new UUID>","requests":[{"custom_id":"one","body":{"messages":[{"role":"user","content":"Hello"}]}}]}` → `Batch` |
| GET | `/v1/batches/{id}` | Refresh the durable job snapshot |
| POST | `/v1/batches/{id}/cancel` | Persist cancellation intent; reconcile any partial results |
| GET | `/v1/batches/{id}/results` | `{"data":[BatchResult, ...],"batch":Batch}` after accounting finishes; otherwise 409 |

Batch routes use the same omitted/default, `auto`, and explicit-label selection,
but `auto` pins a newly submitted durable job to its primary provider rather than
retrying an ambiguous submission across providers. Cloud routes require authentication,
authorize access to the job's session, and use the user's provider credentials. Jobs are bound to a hash of the original provider endpoint
and credentials; rotated or different keys cannot fetch their results. Keys are
not stored in job records. Omit per-item `model`: provider policy chooses the
configured Pro/default model. Inputs require 1–50,000 unique nonempty custom IDs,
non-streaming messages, and fit within the engine's 32 MiB request limit and
provider limits. Per-item session/budget overrides are rejected.

`Batch` contains `id`, `provider`, `status`, `request_counts`, and `raw` metadata
with `session_id`, `provider_batch_id`, `next_poll_at`, `error` and
`accounting_complete`. Results retain provider JSON and normalized completions,
map by `custom_id`, and preserve individual failures. Unknown or duplicate result
IDs block accounting; missing items become explicit failures with unknown cost.
Terminal engine statuses are `completed`, `partial_failed`, `failed`, `cancelled`
and `expired`. Terminal status means durable receipt settlement has finished,
not that every item succeeded or that a provider invoice was reconciled.

Workers poll at 30-second intervals with database revision checks. Intent, remote
IDs, downloaded results and immutable per-item usage receipts survive restarts.
Local startup resumes jobs for configured credentials. Cloud polling resumes when
an authorized GET/results/cancel request supplies the original credentials after
a restart; unattended cloud recovery requires an external caller to do this.
Cancellation intent is durable and retried; successful items still incur usage.
The existing accounting journal settles each item once into ledger, events and
session totals, using explicit catalog batch/cache rates. Missing usage or rates
remain unknown. Budgets remain soft; these endpoints do not reserve funds,
execute tools or inject session memory.

`submission_unknown` means the provider may have accepted a request whose response
was lost; `submitting` can also indicate interrupted submission. The engine never
resubmits that job or falls back to synchronous chat automatically. Inspect the
provider before creating new work; Gemini input files carry the engine job UUID
in their display name. There is currently no automatic provider-ID recovery for
this ambiguity. Supplying an idempotency key is essential to recover a lost engine
response; omitting it generates a new ID on every create request. Automatic chat
batch requests accept `reincore_batch_idempotency_key` for the same protection.

Migration **3** adds the durable job table. The
mock tests exercise native endpoints, headers, JSONL, restart, partial failures,
cancellation, credential binding and idempotent accounting. To explicitly run the
paid sandbox lifecycle test, set `GEMINI_API_KEY` and
`REINCORE_BATCH_SANDBOX_MODEL` to a supported, catalog-priced model, then run:

```sh
cargo test gemini_batch_sandbox_upload_submit_poll_download_and_accounting -- --ignored --nocapture
```

The sandbox retains its recovery database in the system temporary directory and
prints its path/job ID. `REINCORE_BATCH_SANDBOX_TIMEOUT_SECONDS` defaults to 1800;
a timeout requests cancellation but still requires checking the provider job.

### Batch agent events

The SDKs orchestrate batched agents in rounds and execute tools locally. Tracked
runs write append-only `batch` events through:

| Method | Endpoint | Request / response |
|---|---|---|
| POST | `/v1/sessions/{id}/batch-events` | `{ "batch_id": "job", "provider": "label", "custom_id": "run", "phase": "submitted", "data": {} }` → `{ "seq": 1, "event_type": "batch", "session_id": "..." }` |

Supported phases are `submitted`, `completed`, `failed`, and `tools`. Data contains
request messages, the individual model response with usage, errors, or tool
executions respectively. `completed` describes a model request, not an entire
agent run. These SDK-reported events include `source: "sdk"` and
`accounting: "unsettled"`; they do not charge session budgets or claim to be a
reconciled provider ledger. The endpoint verifies session existence and, in cloud
mode, session ownership. Read them through the normal session-events endpoints.

Durable `/v1/batches` jobs produce accounting events for their required session.
SDK-reported batch events remain optional observability and never settle usage a
second time. The updated SDKs use the durable contract and partition agent rounds by session.

Local MCP tool discovery in the desktop app: enter the endpoint and optional token,
then choose **Connect & load tools**. Search by name or description and check the
tools to enable (up to 256). **Select all shown** selects the filtered results only
when the combined selection fits that limit. **Refresh tools** preserves existing
selections and permissions; newly discovered tools remain unselected. Missing
selected tools must be removed or rediscovered before saving. Read-only status is
an explicit user decision; upstream annotations do not enable caching. Discovery
only initializes the MCP connection and lists tools; it neither calls tools nor
saves configuration. Raw JSON remains available for advanced configuration.

### Data lifecycle and recovery

Retention is an opt-in operator command; nothing is deleted automatically. Preview candidates with a past RFC3339 `--before` cutoff, `--retention-days N`, or the privacy policy's `retention_days` value (30 by default), and a limit of 1-100 sessions (default 100). Only completed, unlinked sessions qualify. Active/subagent/linked work, unsettled accounting, active batches, live checkpoints and referenced conversation bindings are protected. Expired checkpoints can be removed with an eligible session.

```text
rc data retain --source <database.db> --before 2026-01-01T00:00:00Z --limit 10
rc data retain --source <database.db> --retention-days 14 --limit 10
rc data backup --source <database.db> --output <new-snapshot.db>
rc data verify --source <new-snapshot.db>
rc data restore --source <new-snapshot.db> --output <new-restored.db>
```

Before applying retention, stop **the daemon and any background workers** using the database. The flags acknowledge operator actions; they do not detect or stop running processes. SQLite apply requires a new pre-deletion backup, created before the deletion transaction:

```text
rc data retain --source <database.db> --before 2026-01-01T00:00:00Z --limit 10 --apply --offline --backup <new-predelete.db>
```

Local migration **2** adds parent-session and parent-binding indexes for retention eligibility checks. Each batch is atomic and repeatable; selection is rechecked on apply. The session count limit does not bound an individual session's history, total scan cost or transaction duration. Use small batches in a maintenance window.

Deletion removes the selected sessions and their events, projections, titles, expired checkpoints, bindings, settled accounting staging records and batch records. The durable usage ledger is deliberately preserved. Shared caches, external files/logs and existing backups remain; this is not a complete privacy-erasure operation. There is no automatic compaction or secure physical erasure. Manage backup expiry separately, since restoring an older snapshot can bring back deleted data.

SQLite backup uses the WAL-aware backup API and refuses to overwrite an existing destination. Destinations with existing SQLite WAL, SHM or journal sidecars are rejected. Verification checks SQLite integrity, foreign keys and supported schema. The copy loop has a 60-second deadline; integrity checks can take longer on large databases. Restore always creates a fresh path. Before switching the engine to it, verify representative session/event content, budgets and pending-accounting recovery, restart again to check idempotence, and check `/readyz`. Preserve the original database for rollback. Recovery correctness depends on also retaining the required configuration and encryption keys.

Compare restored counts, representative payloads, workspace isolation and ledger totals against the source snapshot, then start an isolated engine and check accounting recovery and readiness before any operator-directed cutover. Automated SQLite tests exercise WAL content, rollback, protected sessions, vector cleanup and exactly-once accounting after restore/restart. Reference-scale deletion timing and measured recovery-time/recovery-point targets remain release gates; they have not been certified by the unit tests.

### Runtime resilience

`GET /health` retains its compatibility response. Use public `GET /livez` for process liveness and `GET /readyz` for dependency readiness. Readiness returns 200 only when the session schema is accessible; otherwise it returns 503. No provider request is made. Readiness responses contain only dependency booleans, with no credentials or database errors, and are not cacheable.

Database probes have a 750ms deadline and allow only one outstanding probe per process. The schema probe executes a read bounded to one session row. SQLite uses a nonblocking connection-mutex check; contention conservatively reports not ready. Its blocking probe retains the probe slot after a timeout until the worker exits, preventing a pileup of detached probes. These checks establish schema/dependency accessibility, not write durability, provider health or a successful end-to-end relay delivery.

`REINCORE_MAX_IN_FLIGHT` sets HTTP request capacity (1-4096; default 32). Requests exceeding capacity receive 503 with `Retry-After: 1` immediately, without an admission queue. Clients should retry with backoff/jitter and their normal idempotency rules. The permit lasts through the response body, including SSE, and releases on completion/error/disconnect. Long-lived event streams share capacity with other API requests. `/health`, `/livez` and `/readyz` bypass admission so overload remains observable. Invalid configuration prevents server startup. These are per-process HTTP limits, not tenant quotas or bounds on separately scheduled background work.

The optional metrics endpoint exposes `reincore_requests_shed_total`, alongside existing DB observations. Readiness failure does not automatically cancel existing requests or prohibit every new request: route traffic using readiness at the deployment layer. There is no hard durable-outbox size cap in this stage.

### Desktop hardening

The packaged webview uses an explicit CSP: bundled scripts/fonts/images, data images, Tauri IPC and the fixed `http://127.0.0.1:8989` daemon are permitted. Remote scripts, fonts and images, dynamic script evaluation, frames, plugins and form submission are blocked. Inline styles remain permitted because React components use style attributes; production inline scripts are not broadly permitted. Tauri injects its required asset hashes/nonces. Development has a separate policy allowing Vite's inline bootstrap and WebSocket on port 1420; it does not widen the packaged policy.

Only the local `main` window receives explicit application-command permissions. The build manifest enumerates every registered command, and a regression test checks that these lists stay aligned. Browser access to the generic opener/dialog/core plugin defaults has been removed; directory selection still runs through the Rust `pick_directory` command. Custom commands still validate their own inputs: capabilities are a window boundary, not authorization between application features. See Tauri's [capability model](https://v2.tauri.app/security/capabilities/) and [CSP documentation](https://tauri.app/security/csp/).

The main window is created after daemon startup with navigation guards already installed. Only the bundled app origin (and the exact Vite origin in debug builds) may navigate; external pages and popups cannot replace/open a privileged webview. External transcript links/images are consequently not an embedded browsing surface. Release builds no longer enable Tauri's DevTools feature or install the debug error-logging script. The daemon remains a separate process; closing the window retains the existing daemon lifecycle.

Windows packaging remains per-user. Installer PATH registration/removal now matches whole entries, preserves expandable environment references and includes hooks after their required definitions. `makensis app/installer/check-hooks.nsi` (run from the installer directory as `makensis check-hooks.nsi`) compiles a hook harness; **do not execute its output**. It is a syntax check, not an install test. Generated command permissions are recreated by `app/build.rs` and ignored by Git.

Verification commands: `cargo test -p reincore-app --lib --features tauri/custom-protocol`, plus `npm run build` and `npm run lint` in `app/frontend`. Before release, exercise the actual WebView2/WebKit app: session loading/SSE, fonts/icons, directory picker, provider/launch/settings commands, blocked remote navigation/popups and release DevTools. In a disposable Windows user profile, verify clean installation, upgrade, CLI PATH registration, similarly named PATH entries and uninstall while retaining user data. Signed release packaging, these interactive native checks and macOS/Linux installer coverage remain open acceptance gates; compilation and CSP assertions do not establish them.

The [feature assessment](../FEATURE_ASSESSMENT.md) records load, freeze/recovery and logical-restore results. Reference-scale and crash-durability limits are listed in the report.
