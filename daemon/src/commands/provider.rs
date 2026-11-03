use anyhow::Result;
use colored::Colorize;
use engine_core::{ChatCompletionRequest, ProviderConfig, ProviderRegistry};

use crate::cli::ProviderCommands;
use crate::credentials;
use crate::daemon::start_single_provider_daemon;
use crate::local_config::{
    load_local_config, read_custom_provider_details, read_provider_models, resolve_model_policy,
    set_provider_model_field, LocalConfig, ModelEnvironment, ResolvedModelPolicy,
};

pub(crate) fn configured_model_policy(config: &LocalConfig, provider: &str) -> ResolvedModelPolicy {
    let (environment, built_in_standard) = match provider {
        "openai" => (
            ModelEnvironment {
                standard: "OPENAI_STANDARD_MODEL",
                lite: "OPENAI_LITE_MODEL",
                pro: "OPENAI_PRO_MODEL",
                ultra: "OPENAI_ULTRA_MODEL",
            },
            "gpt-5.6-terra",
        ),
        "anthropic" => (
            ModelEnvironment {
                standard: "ANTHROPIC_STANDARD_MODEL",
                lite: "ANTHROPIC_LITE_MODEL",
                pro: "ANTHROPIC_PRO_MODEL",
                ultra: "ANTHROPIC_ULTRA_MODEL",
            },
            "claude-sonnet-5",
        ),
        "gemini" => (
            ModelEnvironment {
                standard: "GEMINI_STANDARD_MODEL",
                lite: "GEMINI_LITE_MODEL",
                pro: "GEMINI_PRO_MODEL",
                ultra: "GEMINI_ULTRA_MODEL",
            },
            "gemini-3.6-flash",
        ),
        "azure" => (
            ModelEnvironment {
                standard: "AZURE_OPENAI_STANDARD_DEPLOYMENT",
                lite: "AZURE_OPENAI_LITE_DEPLOYMENT",
                pro: "AZURE_OPENAI_PRO_DEPLOYMENT",
                ultra: "AZURE_OPENAI_ULTRA_DEPLOYMENT",
            },
            "gpt-5.6-terra",
        ),
        "groq" => (
            ModelEnvironment {
                standard: "GROQ_STANDARD_MODEL",
                lite: "GROQ_LITE_MODEL",
                pro: "GROQ_PRO_MODEL",
                ultra: "GROQ_ULTRA_MODEL",
            },
            "openai/gpt-oss-120b",
        ),
        "xai" => (
            ModelEnvironment {
                standard: "XAI_STANDARD_MODEL",
                lite: "XAI_LITE_MODEL",
                pro: "XAI_PRO_MODEL",
                ultra: "XAI_ULTRA_MODEL",
            },
            "grok-4.3",
        ),
        "deepseek" => (
            ModelEnvironment {
                standard: "DEEPSEEK_STANDARD_MODEL",
                lite: "DEEPSEEK_LITE_MODEL",
                pro: "DEEPSEEK_PRO_MODEL",
                ultra: "DEEPSEEK_ULTRA_MODEL",
            },
            "deepseek-v4-pro",
        ),
        "meta" => (
            ModelEnvironment {
                standard: "META_STANDARD_MODEL",
                lite: "META_LITE_MODEL",
                pro: "META_PRO_MODEL",
                ultra: "META_ULTRA_MODEL",
            },
            "muse-spark-1.3",
        ),
        "copilot" => (
            ModelEnvironment {
                standard: "COPILOT_STANDARD_MODEL",
                lite: "COPILOT_LITE_MODEL",
                pro: "COPILOT_PRO_MODEL",
                ultra: "COPILOT_ULTRA_MODEL",
            },
            "gpt-4o",
        ),
        "custom" => (
            ModelEnvironment {
                standard: "CUSTOM_PROVIDER_STANDARD_MODEL",
                lite: "CUSTOM_PROVIDER_LITE_MODEL",
                pro: "CUSTOM_PROVIDER_PRO_MODEL",
                ultra: "CUSTOM_PROVIDER_ULTRA_MODEL",
            },
            "",
        ),
        "ollama" => (
            ModelEnvironment {
                standard: "OLLAMA_STANDARD_MODEL",
                lite: "OLLAMA_LITE_MODEL",
                pro: "OLLAMA_PRO_MODEL",
                ultra: "OLLAMA_ULTRA_MODEL",
            },
            "",
        ),
        _ => {
            return ResolvedModelPolicy {
                standard_model: None,
                models: Vec::new(),
                routing: engine_core::RoutingPolicy::default(),
            }
        }
    };
    resolve_model_policy(config, provider, environment, built_in_standard)
}

/// Resolve which model family (OpenAI or Anthropic) to use for the Copilot
/// provider on the paid tier. Checks the `COPILOT_MODEL_FAMILY` env var first,
/// then prompts the user interactively.
fn resolve_copilot_model_family() -> engine_core::provider::CopilotModelFamily {
    // Check env var first - if set, use it without prompting.
    if let Ok(family) = std::env::var("COPILOT_MODEL_FAMILY") {
        if let Ok(f) = family.parse::<engine_core::provider::CopilotModelFamily>() {
            return f;
        }
    }

    // Default to OpenAI without prompting. The actual prompt happens in
    // `provider_start_proxy` after the tier is detected, so we only ask
    // if the user is on the paid tier.
    engine_core::provider::CopilotModelFamily::OpenAI
}

/// Prompt the user to select a model family for the Copilot paid tier.
pub fn prompt_copilot_model_family() -> engine_core::provider::CopilotModelFamily {
    use colored::Colorize;
    println!(
        "\n┌─── {} ─────────────────────────",
        "Select Model Family".cyan().bold()
    );
    println!(
        "│  1. {} {}",
        "OpenAI".white(),
        "(gpt-5.6-luna, gpt-5.6-terra, gpt-5.6-sol)".dimmed()
    );
    println!(
        "│  2. {} {}",
        "Anthropic".white(),
        "(claude-haiku-4-5, claude-sonnet-5, claude-fable-5)".dimmed()
    );
    println!("└────────────────────────────────────────────────────────");
    print!("Enter choice [1 or 2, default 1]: ");
    use std::io::{self, Write};
    io::stdout().flush().ok();

    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    match input.trim() {
        "2" | "anthropic" | "claude" => engine_core::provider::CopilotModelFamily::Anthropic,
        _ => engine_core::provider::CopilotModelFamily::OpenAI,
    }
}

fn prompt_for_api_key(provider_label: &str) -> Result<String> {
    use std::io::{self, Write};
    let env_var = api_key_env_var(provider_label);
    print!(
        "Enter API key for provider '{}': ",
        provider_display_name(provider_label)
    );
    io::stdout().flush()?;
    let mut key = String::new();
    io::stdin().read_line(&mut key)?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(anyhow::anyhow!(
            "API key is required for provider '{}' (set {} or enter one at the prompt)",
            provider_label,
            env_var
        ));
    }
    std::env::set_var(&env_var, &key);
    Ok(key)
}

pub fn api_key_env_var(provider_label: &str) -> String {
    match provider_label.to_lowercase().as_str() {
        "openai" => "OPENAI_API_KEY".to_string(),
        "anthropic" => "ANTHROPIC_API_KEY".to_string(),
        "gemini" => "GEMINI_API_KEY".to_string(),
        "azure" => "AZURE_OPENAI_API_KEY".to_string(),
        "groq" => "GROQ_API_KEY".to_string(),
        "xai" => "XAI_API_KEY".to_string(),
        "deepseek" => "DEEPSEEK_API_KEY".to_string(),
        "meta" => "META_API_KEY".to_string(),
        "copilot" => "COPILOT_CLIENT_ID".to_string(),
        // Ollama cloud mode uses an API key; local mode ignores it. The
        // actual setup is handled in `launch::setup_ollama_cloud`, not here.
        "ollama" => "OLLAMA_API_KEY".to_string(),
        "custom" => "CUSTOM_PROVIDER_API_KEY".to_string(),
        _ => format!("{}_API_KEY", provider_label.to_uppercase()),
    }
}

pub fn rebuild_provider_with_key(
    provider_label: &str,
    api_key: &str,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    let label = provider_label.to_lowercase();
    let local_config = load_local_config();
    let policy = configured_model_policy(&local_config, &label);
    let config = match label.as_str() {
        "openai" => ProviderConfig {
            label: "openai".to_string(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "anthropic" => ProviderConfig {
            label: "anthropic".to_string(),
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "gemini" => ProviderConfig {
            label: "gemini".to_string(),
            base_url: std::env::var("GEMINI_BASE_URL").unwrap_or_else(|_| {
                "https://generativelanguage.googleapis.com/v1beta/openai".to_string()
            }),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "azure" => ProviderConfig {
            label: "azure".to_string(),
            base_url: std::env::var("AZURE_OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://example.openai.azure.com".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "groq" => ProviderConfig {
            label: "groq".to_string(),
            base_url: std::env::var("GROQ_BASE_URL")
                .unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "xai" => ProviderConfig {
            label: "xai".to_string(),
            base_url: std::env::var("XAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.x.ai/v1".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "deepseek" => ProviderConfig {
            label: "deepseek".to_string(),
            base_url: std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "meta" => ProviderConfig {
            label: "meta".to_string(),
            base_url: std::env::var("META_BASE_URL")
                .unwrap_or_else(|_| "https://api.meta.ai/v1".to_string()),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "copilot" => ProviderConfig {
            label: "copilot".to_string(),
            // The Copilot provider ignores base_url - it always talks to
            // api.githubcopilot.com internally. We store the GitHub token
            // endpoint here for reference.
            base_url: "https://api.githubcopilot.com".to_string(),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        "custom" => ProviderConfig {
            label: "custom".to_string(),
            base_url: read_custom_provider_details().base_url.unwrap_or_default(),
            api_key: api_key.to_string(),
            standard_model: policy.standard_model.clone(),
            models: policy.models.clone(),
            routing: policy.routing.clone(),
            ..Default::default()
        },
        _ => {
            return Err(anyhow::anyhow!(
                "interactive API key prompt not supported for provider '{}'",
                provider_label
            ));
        }
    };

    let provider: std::sync::Arc<dyn engine_core::Provider> = match label.as_str() {
        "openai" => std::sync::Arc::new(engine_core::provider::OpenAiProvider::new(config)?),
        "anthropic" => std::sync::Arc::new(engine_core::provider::AnthropicProvider::new(config)?),
        "gemini" => std::sync::Arc::new(engine_core::provider::GeminiProvider::new(config)?),
        "azure" => std::sync::Arc::new(engine_core::provider::AzureOpenAiProvider::new(config)?),
        "groq" => std::sync::Arc::new(engine_core::provider::GroqProvider::new(config)?),
        "xai" => std::sync::Arc::new(engine_core::provider::XaiProvider::new(config)?),
        "deepseek" => std::sync::Arc::new(engine_core::provider::DeepSeekProvider::new(config)?),
        "meta" => std::sync::Arc::new(engine_core::provider::MetaProvider::new(config)?),
        "copilot" => {
            let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
            let client_secret = engine_core::provider::CopilotProvider::resolve_client_secret();
            let github_instance = engine_core::provider::CopilotProvider::resolve_github_instance();
            std::sync::Arc::new(engine_core::provider::CopilotProvider::new(
                client_id,
                client_secret,
                github_instance,
                config,
                resolve_copilot_model_family(),
            )?)
        }
        "custom" => std::sync::Arc::new(engine_core::provider::OpenAiProvider::new(config)?),
        _ => unreachable!(),
    };
    Ok(provider)
}

pub async fn handle_provider_command(command: ProviderCommands) -> Result<()> {
    match command {
        ProviderCommands::List => provider_list().await,
        ProviderCommands::Add {
            provider,
            key,
            cloud,
        } => provider_add(provider, key, cloud).await,
        ProviderCommands::Remove { provider } => provider_remove(provider),
        ProviderCommands::Test { provider, prompt } => provider_test(provider, prompt).await,
        ProviderCommands::Models { provider } => provider_models(provider).await,
    }
}

/// `reincore provider add <label>` - store or update a provider's credentials
/// so the daemon can use it on future starts without an env var.
///
/// For API-key providers (openai, anthropic, gemini, azure, groq, xai,
/// deepseek, ollama cloud), the key is prompted interactively (or accepted
/// via `--key`) and saved to `~/.reincore-io/credentials.json`.
///
/// For Ollama, this also lets you view and update the saved standard model in
/// `~/.reincore-io/config.toml` (local mode) or the cloud API key (cloud mode).
///
/// For Copilot, this runs the OAuth device flow and stores the resulting
/// GitHub OAuth token.
async fn provider_add(provider: String, key: Option<String>, cloud: bool) -> Result<()> {
    let label = provider.to_lowercase();

    if !supported_providers().contains(&label) {
        return Err(anyhow::anyhow!(
            "Unknown provider '{}'. Supported: openai, anthropic, gemini, azure, groq, xai, deepseek, meta, copilot, ollama.",
            provider
        ));
    }

    if label == "copilot" {
        // Copilot uses OAuth device flow - build the provider (which triggers
        // the flow), then extract and store the OAuth token.
        println!("Starting GitHub Copilot OAuth device flow...");
        let local_config = load_local_config();
        let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
        let client_secret = engine_core::provider::CopilotProvider::resolve_client_secret();
        let github_instance = engine_core::provider::CopilotProvider::resolve_github_instance();
        let policy = configured_model_policy(&local_config, "copilot");
        let config = ProviderConfig {
            label: "copilot".to_string(),
            base_url: "https://api.githubcopilot.com".to_string(),
            api_key: String::new(),
            standard_model: policy.standard_model,
            models: policy.models,
            routing: policy.routing,
            ..Default::default()
        };
        // CopilotProvider::new succeeds without a token; the device flow
        // happens when the first API call triggers get_oauth_token(). We
        // force it by calling check_tier().
        let copilot = engine_core::provider::CopilotProvider::new(
            client_id,
            client_secret,
            github_instance,
            config,
            resolve_copilot_model_family(),
        )?;

        // Block on the OAuth flow - this prints the device code and waits.
        let rt = tokio::runtime::Runtime::new()?;
        let oauth_token = rt.block_on(async { copilot.get_oauth_token().await })?;

        credentials::store_oauth_token("copilot", &oauth_token)?;
        println!(
            "{}",
            "GitHub Copilot credentials saved to ~/.reincore-io/credentials.json."
                .green()
                .bold()
        );
        return Ok(());
    }

    if label == "ollama" {
        return provider_add_ollama(key, cloud).await;
    }

    // Standard API-key providers.
    let env_var = api_key_env_var(&label);

    // Show current configuration.
    let current_models = read_provider_models(&label);
    let saved_cred = credentials::get_credential(&label);
    let env_key = std::env::var(&env_var)
        .ok()
        .filter(|k| !k.trim().is_empty());
    let has_key = key.is_some() || saved_cred.is_some() || env_key.is_some();

    println!(
        "\n{}",
        format!("Current {} configuration:", provider_display_name(&label))
            .cyan()
            .bold()
    );
    println!(
        "  {}: {}",
        "Standard model".white(),
        current_models
            .standard
            .as_deref()
            .unwrap_or("(not set)")
            .cyan()
    );
    if let Some(m) = &current_models.lite {
        println!("  {}: {}", "Lite model".white(), m.cyan());
    }
    if let Some(m) = &current_models.pro {
        println!("  {}: {}", "Pro model".white(), m.cyan());
    }
    if let Some(m) = &current_models.ultra {
        println!("  {}: {}", "Ultra model".white(), m.cyan());
    }
    if let Some(ref _k) = env_key {
        println!("  {}: {}", "API key (env)".white(), "set".green());
    } else if saved_cred.is_some() {
        println!("  {}: {}", "API key (saved)".white(), "set".green());
    } else {
        println!("  {}: {}", "API key".white(), "(none)".dimmed());
    }

    // If --key provided, save it immediately.
    if let Some(ref k) = key {
        if !k.trim().is_empty() {
            let _provider = rebuild_provider_with_key(&label, k)?;
            credentials::store_api_key(&label, k)?;
            println!(
                "{}",
                format!(
                    "API key for '{}' saved to ~/.reincore-io/credentials.json.",
                    provider_display_name(&label)
                )
                .green()
                .bold()
            );
        }
    }

    // Interactive update loop.
    use std::io::{self, Write};
    fn read_line() -> Result<String> {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_string())
    }

    loop {
        print!(
            "\n{} ",
            "Update [k=API key, s=standard model, q=quit]:"
                .dimmed()
                .bold()
        );
        io::stdout().flush()?;
        let choice = read_line()?;

        match choice.to_lowercase().as_str() {
            "k" | "key" => {
                let k = if let Some(ref passed) = key {
                    if !passed.trim().is_empty() {
                        passed.clone()
                    } else {
                        print!(
                            "{} '{}': ",
                            "Enter API key for".dimmed(),
                            provider_display_name(&label).white()
                        );
                        io::stdout().flush()?;
                        read_line()?
                    }
                } else {
                    print!(
                        "{} '{}': ",
                        "Enter API key for".dimmed(),
                        provider_display_name(&label).white()
                    );
                    io::stdout().flush()?;
                    read_line()?
                };
                if k.is_empty() {
                    println!("{}", "API key cannot be empty. Skipped.".yellow());
                } else {
                    let _provider = rebuild_provider_with_key(&label, &k)?;
                    credentials::store_api_key(&label, &k)?;
                    println!(
                        "{}",
                        format!(
                            "API key for '{}' saved to ~/.reincore-io/credentials.json.",
                            provider_display_name(&label)
                        )
                        .green()
                        .bold()
                    );
                }
                break;
            }
            "s" | "standard" => {
                if !has_key && key.is_none() {
                    println!(
                        "{}",
                        "API key is required before fetching models. Set it first with 'k'."
                            .yellow()
                    );
                    continue;
                }
                // Resolve the API key to use for the model fetch.
                let api_key = key
                    .as_ref()
                    .filter(|k| !k.is_empty())
                    .cloned()
                    .or_else(|| env_key.clone())
                    .or_else(|| saved_cred.map(|c| c.value))
                    .unwrap_or_default();

                match prompt_select_provider_model(&label, &api_key).await {
                    Ok(Some(model)) => {
                        set_provider_model_field(&label, "standard", &model)?;
                        println!(
                            "{}",
                            format!(
                                "Standard model for '{}' set to '{}' in ~/.reincore-io/config.toml.",
                                provider_display_name(&label),
                                model
                            )
                            .green()
                            .bold()
                        );
                    }
                    Ok(None) => {
                        println!("{}", "No models available or selection cancelled.".dimmed());
                    }
                    Err(e) => {
                        println!(
                            "{}",
                            format!("Could not fetch models: {}. You can set it manually with `reincore model {} --standard <model>`.", e, label).yellow()
                        );
                    }
                }
                break;
            }
            "q" | "quit" | "" => break,
            _ => println!("{}", "Invalid choice. Enter k, s or q.".yellow().bold()),
        }
    }

    Ok(())
}

/// Fetch the list of available models from a provider and prompt the user to
/// select one by number. Returns `Ok(Some(model_id))` on selection, or
/// `Ok(None)` if no models are available.
async fn prompt_select_provider_model(label: &str, api_key: &str) -> Result<Option<String>> {
    use std::io::{self, Write};

    let provider = rebuild_provider_with_key(label, api_key)?;
    let models = provider
        .list_models()
        .await
        .map_err(|e| anyhow::anyhow!("failed to list models: {}", e))?;

    if models.is_empty() {
        return Ok(None);
    }

    println!(
        "\n{}",
        format!("Available models for {}:", provider_display_name(label))
            .cyan()
            .bold()
    );
    for (i, m) in models.iter().enumerate() {
        let display = m.display_name.as_deref().unwrap_or("");
        if display.is_empty() {
            println!("  {}. {}", format!("{}", i + 1).yellow(), m.id.cyan());
        } else {
            println!(
                "  {}. {} ({})",
                format!("{}", i + 1).yellow(),
                display.cyan(),
                m.id.dimmed()
            );
        }
    }

    loop {
        print!(
            "\n{} ",
            format!(
                "Select model [1-{}] (or press Enter to cancel):",
                models.len()
            )
            .dimmed()
            .bold()
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        let input = input.trim();
        if input.is_empty() {
            return Ok(None);
        }
        if let Ok(idx) = input.parse::<usize>() {
            if idx >= 1 && idx <= models.len() {
                return Ok(Some(models[idx - 1].id.clone()));
            }
        }
        // Also accept exact model id.
        if let Some(m) = models.iter().find(|m| m.id == input) {
            return Ok(Some(m.id.clone()));
        }
        println!(
            "{}",
            "Invalid selection. Enter a number from the list or the exact model id."
                .red()
                .bold()
        );
    }
}

/// `reincore provider add ollama` - update the saved Ollama configuration.
///
/// Shows the current standard model (from `~/.reincore-io/config.toml`) and API
/// key status (saved credential or `OLLAMA_API_KEY` env var), then asks which
/// value to update. Passing `--key <key>` on the command line skips the
/// API-key prompt and stores the key immediately.
///
/// When updating the standard model, a numbered list of installed models
/// (fetched from the local Ollama server or the Ollama Cloud API) is
/// presented so the user can pick by number instead of typing the name.
async fn provider_add_ollama(key: Option<String>, cloud: bool) -> Result<()> {
    use std::io::{self, Write};

    fn mask_key(key: &str) -> String {
        if key.len() <= 8 {
            "********".to_string()
        } else {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        }
    }

    fn read_line() -> Result<String> {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
        Ok(input.trim().to_string())
    }

    let models = read_provider_models("ollama");
    let saved_cred = credentials::get_credential("ollama");
    let env_key = std::env::var("OLLAMA_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());

    let current_standard = models.standard.as_deref().unwrap_or("(not set)");
    let (key_source, key_masked) = if let Some(ref k) = env_key {
        ("env var", Some(mask_key(k)))
    } else if let Some(ref c) = &saved_cred {
        ("saved", Some(mask_key(&c.value)))
    } else {
        ("none", None)
    };

    println!("\n{}", "Current Ollama configuration:".cyan().bold());
    println!(
        "  {}: {}",
        "Standard model".white(),
        current_standard.cyan()
    );
    if let Some(masked) = key_masked {
        println!(
            "  {}: {}",
            format!("API key ({})", key_source).white(),
            masked.cyan()
        );
    } else {
        println!(
            "  {}: {}",
            "API key".white(),
            "(none - local mode)".dimmed()
        );
    }

    // If a key was passed on the command line, store it immediately and then
    // offer to update the standard model as well.
    let mut api_key_updated = false;
    if let Some(k) = key {
        if !k.trim().is_empty() {
            credentials::store_api_key("ollama", &k)?;
            println!(
                "{}",
                "Ollama Cloud API key saved to ~/.reincore-io/credentials.json."
                    .green()
                    .bold()
            );
            api_key_updated = true;
        }
    }

    // Determine the model source based on the --cloud flag:
    //   --cloud      → fetch from the Ollama Cloud API (requires an API key).
    //   no --cloud   → fetch from the local Ollama server.
    let local_base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| crate::launch::default_ollama_base_url().to_string());
    let has_cloud_key = env_key.is_some() || saved_cred.is_some();

    let (base_url, api_key_for_fetch) = if cloud {
        if !has_cloud_key {
            println!(
                "{}",
                "--cloud was given but no Ollama Cloud API key is set. Enter it now or run `reincore provider add ollama --key <key> --cloud`."
                    .yellow()
            );
        }
        (
            crate::launch::ollama_cloud_base_url().to_string(),
            env_key
                .as_deref()
                .map(|k| k.to_string())
                .or_else(|| saved_cred.as_ref().map(|c| c.value.clone())),
        )
    } else {
        (local_base_url, None)
    };

    /// Fetch models and prompt the user to pick one by number.
    async fn pick_model(base_url: &str, api_key: Option<&str>) -> Result<Option<String>> {
        let models = crate::launch::fetch_ollama_models_pub(base_url, api_key).await;
        if models.is_empty() {
            println!(
                "{}",
                "No models available from the Ollama server. Start it with `ollama serve` or pull a model with `ollama pull <model>`."
                    .yellow()
            );
            return Ok(None);
        }
        let selected = crate::launch::prompt_select_ollama_model_pub(
            &models,
            "Select the standard Ollama model to use:",
        )
        .await?;
        Ok(Some(selected))
    }

    loop {
        print!(
            "\n{} ",
            "Update [s=standard model, k=API key, b=both, q=quit]:"
                .dimmed()
                .bold()
        );
        io::stdout().flush()?;
        let choice = read_line()?;

        match choice.to_lowercase().as_str() {
            "s" | "standard" => {
                if let Some(model) = pick_model(&base_url, api_key_for_fetch.as_deref()).await? {
                    set_provider_model_field("ollama", "standard", &model)?;
                    println!(
                        "{}",
                        format!(
                            "Ollama standard model set to '{}' in ~/.reincore-io/config.toml.",
                            model
                        )
                        .green()
                        .bold()
                    );
                } else {
                    println!("{}", "Standard model not updated.".dimmed());
                }
                if api_key_updated {
                    break;
                }
                continue;
            }
            "k" | "key" => {
                print!(
                    "{} ",
                    "Enter Ollama Cloud API key (leave empty for local mode):".dimmed()
                );
                io::stdout().flush()?;
                let k = read_line()?;
                if k.is_empty() {
                    println!("{}", "No API key entered - local mode unchanged.".dimmed());
                } else {
                    credentials::store_api_key("ollama", &k)?;
                    println!(
                        "{}",
                        "Ollama Cloud API key saved to ~/.reincore-io/credentials.json."
                            .green()
                            .bold()
                    );
                }
                break;
            }
            "b" | "both" => {
                if let Some(model) = pick_model(&base_url, api_key_for_fetch.as_deref()).await? {
                    set_provider_model_field("ollama", "standard", &model)?;
                    println!(
                        "{}",
                        format!(
                            "Ollama standard model set to '{}' in ~/.reincore-io/config.toml.",
                            model
                        )
                        .green()
                        .bold()
                    );
                } else {
                    println!("{}", "Standard model not updated.".dimmed());
                }
                print!(
                    "{} ",
                    "Enter Ollama Cloud API key (leave empty for local mode):".dimmed()
                );
                io::stdout().flush()?;
                let k = read_line()?;
                if !k.is_empty() {
                    credentials::store_api_key("ollama", &k)?;
                    println!(
                        "{}",
                        "Ollama Cloud API key saved to ~/.reincore-io/credentials.json."
                            .green()
                            .bold()
                    );
                }
                break;
            }
            "q" | "quit" | "" => break,
            _ => println!("{}", "Invalid choice. Enter s, k, b or q.".yellow().bold()),
        }
    }

    Ok(())
}

/// `reincore provider remove <label>` - remove a provider's stored credentials.
fn provider_remove(provider: String) -> Result<()> {
    let label = provider.to_lowercase();
    let removed = credentials::remove_credential(&label)?;
    if removed {
        println!(
            "{}",
            format!(
                "Credentials for '{}' removed from ~/.reincore-io/credentials.json.",
                provider_display_name(&label)
            )
            .green()
            .bold()
        );
    } else {
        println!(
            "{}",
            format!(
                "No stored credentials found for '{}'.",
                provider_display_name(&label)
            )
            .yellow()
        );
    }
    Ok(())
}

async fn provider_list() -> Result<()> {
    let supported = supported_providers();
    let registry = build_provider_registry();

    println!("Supported providers:");
    for label in &supported {
        let env_configured = registry.get_by_label(label).is_some();
        let saved_cred = credentials::get_credential(label);
        let env_var = api_key_env_var(label);

        let (status_text, source, configured) = if env_configured && label != "ollama" {
            ("configured", format!("via {}", env_var), true)
        } else if label == "ollama" {
            // Ollama has two modes: local (no auth, needs binary + server)
            // and cloud (needs API key). Check both and report the status.
            let ollama_api_key = std::env::var("OLLAMA_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty());
            let cloud_configured = ollama_api_key.is_some() || saved_cred.is_some();

            let installed = crate::launch::is_ollama_installed_pub();
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| crate::launch::default_ollama_base_url().to_string());
            let server_running = if installed {
                crate::launch::is_ollama_server_running_pub(&base_url).await
            } else {
                false
            };

            match (cloud_configured, installed, server_running) {
                (true, true, true) => (
                    "configured (cloud + local)",
                    format!("API key set, server running at {}", base_url),
                    true,
                ),
                (true, true, false) => (
                    "configured (cloud)",
                    "API key set, local server not running".to_string(),
                    true,
                ),
                (true, false, _) => (
                    "configured (cloud)",
                    "API key set, ollama not installed".to_string(),
                    true,
                ),
                (false, true, true) => (
                    "available (local)",
                    format!("server running at {}, no auth needed", base_url),
                    true,
                ),
                (false, true, false) => (
                    "not running (local)",
                    "ollama installed, server not running (start with `ollama serve`)".to_string(),
                    false,
                ),
                (false, false, _) => (
                    "not configured",
                    "no API key, ollama not installed".to_string(),
                    false,
                ),
            }
        } else if saved_cred.is_some() {
            ("configured", "via saved credential".to_string(), true)
        } else if label == "copilot" {
            ("not configured", format!("via {} (OAuth)", env_var), false)
        } else {
            ("not configured", format!("set {}", env_var), false)
        };

        let status_colored = if configured {
            status_text.green().bold()
        } else {
            status_text.red().bold()
        };
        let display_name = provider_display_name(label);
        let label_colored = if configured {
            display_name.white().bold()
        } else {
            display_name.dimmed()
        };

        println!(
            "  {} ({}) - {}",
            label_colored,
            status_colored,
            source.dimmed()
        );

        // Show configured model tiers (standard/lite/pro/ultra) from config.toml.
        let models = read_provider_models(label);
        let mut tiers = Vec::new();
        if let Some(m) = &models.standard {
            tiers.push(format!("standard={}", m));
        }
        if let Some(m) = &models.lite {
            tiers.push(format!("lite={}", m));
        }
        if let Some(m) = &models.pro {
            tiers.push(format!("pro={}", m));
        }
        if let Some(m) = &models.ultra {
            tiers.push(format!("ultra={}", m));
        }
        if !tiers.is_empty() {
            println!("    models: {}", tiers.join(", "));
        }
    }

    if registry.is_empty() && credentials::CredentialStore::load().credentials.is_empty() {
        println!("\nNo providers are currently configured. Set one of the API key env vars to enable a provider:");
        println!("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, AZURE_OPENAI_API_KEY,");
        println!("  GROQ_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY, META_API_KEY, COPILOT_CLIENT_ID");
    }

    Ok(())
}

pub fn supported_providers() -> Vec<String> {
    vec![
        "openai".to_string(),
        "anthropic".to_string(),
        "gemini".to_string(),
        "azure".to_string(),
        "groq".to_string(),
        "xai".to_string(),
        "deepseek".to_string(),
        "meta".to_string(),
        "copilot".to_string(),
        "ollama".to_string(),
        "custom".to_string(),
    ]
}

/// Map a provider key (e.g. "openai", "xai") to a human-friendly display
/// name (e.g. "OpenAI", "xAI"). Used everywhere a provider name is shown
/// to the user - CLI output, HTTP responses, and the desktop app UI.
pub fn provider_display_name(label: &str) -> String {
    match label.to_lowercase().as_str() {
        "openai" => "OpenAI".to_string(),
        "anthropic" => "Anthropic".to_string(),
        "gemini" => "Google Gemini".to_string(),
        "azure" => "Azure OpenAI".to_string(),
        "groq" => "Groq".to_string(),
        "xai" => "xAI".to_string(),
        "deepseek" => "DeepSeek".to_string(),
        "meta" => "Meta Muse".to_string(),
        "copilot" => "GitHub Copilot".to_string(),
        "ollama" => "Ollama".to_string(),
        "custom" => "Custom".to_string(),
        other => {
            other
                .chars()
                .next()
                .map(|c| c.to_uppercase().collect::<String>())
                .unwrap_or_default()
                + &other[1..]
        }
    }
}

async fn provider_test(provider_label: String, prompt: Option<String>) -> Result<()> {
    let mut registry = build_provider_registry();
    let provider = registry
        .take_by_label(&provider_label)
        .or_else(|| {
            if provider_label.to_lowercase() == "copilot" {
                return None; // Copilot uses built-in OAuth, no key prompt
            }
            let key = prompt_for_api_key(&provider_label).ok()?;
            rebuild_provider_with_key(&provider_label, &key).ok()
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider '{}' is not configured",
                provider_display_name(&provider_label)
            )
        })?;

    let config = provider.config();
    let provider = if config.api_key.is_empty() && provider_label.to_lowercase() != "copilot" {
        let key = prompt_for_api_key(&provider_label)?;
        rebuild_provider_with_key(&provider_label, &key)?
    } else {
        provider
    };

    let prompt = prompt.unwrap_or_else(|| "Say hello in one sentence.".to_string());
    let request = ChatCompletionRequest {
        model: provider_label.clone(),
        messages: vec![engine_core::ChatMessage {
            provider_state: None,
            role: "user".to_string(),
            content: Some(engine_core::MessageContent::text(prompt)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        tools: None,
        tool_choice: None,
        temperature: Some(0.7),
        max_tokens: Some(50),
        stream: None,
        extra: std::collections::HashMap::new(),
    };

    match provider.complete(&request).await {
        Ok(response) => {
            println!(
                "Provider '{}' responded successfully:",
                provider_display_name(&provider_label)
            );
            println!("  model: {}", response.model);
            println!(
                "  content: {}",
                response
                    .message
                    .content_text()
                    .unwrap_or_else(|| "(empty)".to_string())
            );
            if let Some(usage) = response.usage {
                println!(
                    "  usage: {} prompt / {} completion / {} total",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "provider '{}' test failed: {}",
                provider_display_name(&provider_label),
                e
            ));
        }
    }

    Ok(())
}

async fn provider_models(provider_label: Option<String>) -> Result<()> {
    let registry = build_provider_registry();
    if registry.is_empty() {
        println!("No providers are configured.");
        return Ok(());
    }

    let models = if let Some(label) = provider_label {
        let provider = registry.get_by_label(&label).ok_or_else(|| {
            anyhow::anyhow!(
                "provider '{}' is not configured",
                provider_display_name(&label)
            )
        })?;
        provider.list_models().await.map_err(|e| {
            anyhow::anyhow!(
                "failed to list models for provider '{}': {}",
                provider_display_name(&label),
                e
            )
        })?
    } else {
        registry.list_all_models().await.map_err(|e| {
            anyhow::anyhow!("failed to list models from configured providers: {}", e)
        })?
    };

    if models.is_empty() {
        println!("No models returned.");
    } else {
        // Group models by provider and print as a header + bullet list:
        //
        //   OpenAI:
        //     - GPT-5.6 Terra (gpt-5.6-terra)
        //     - GPT-5.6 Luna (gpt-5.6-luna)
        //
        //   Anthropic:
        //     - Claude Sonnet 5 (claude-sonnet-5)
        //
        // This is easier to scan than the old "provider (model_id): display_name"
        // one-line-per-model format.
        use std::collections::BTreeMap;
        let mut by_provider: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        for m in models {
            let display = m.display_name.unwrap_or_else(|| m.id.clone());
            by_provider
                .entry(m.provider.clone())
                .or_default()
                .push((display, m.id));
        }

        for (provider_key, provider_models) in &by_provider {
            println!("{}:", provider_display_name(provider_key).white().bold());
            for (name, id) in provider_models {
                if name.is_empty() {
                    println!("  - {} ({})", id.cyan(), id.dimmed());
                } else {
                    println!("  - {} ({})", name.cyan(), id.dimmed());
                }
            }
        }
    }

    Ok(())
}

pub async fn provider_start_proxy(
    provider_label: String,
    host: String,
    port: u16,
    db_path: String,
    verbose: bool,
    mode: &str,
    project_path: Option<String>,
) -> Result<()> {
    let mut registry = build_provider_registry();
    let provider = registry
        .take_by_label(&provider_label)
        .or_else(|| {
            if provider_label.to_lowercase() == "copilot" {
                return None; // Copilot uses built-in OAuth, no key prompt
            }
            let key = prompt_for_api_key(&provider_label).ok()?;
            rebuild_provider_with_key(&provider_label, &key).ok()
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider '{}' is not configured",
                provider_display_name(&provider_label)
            )
        })?;

    let config = provider.config();
    let provider = if config.api_key.is_empty() && provider_label.to_lowercase() != "copilot" {
        let key = prompt_for_api_key(&provider_label)?;
        rebuild_provider_with_key(&provider_label, &key)?
    } else {
        provider
    };

    // For the Copilot provider, eagerly exchange the token to detect the
    // subscription tier and prompt for a model family (if paid) before
    // starting the proxy.
    if provider_label.to_lowercase() == "copilot" {
        println!("Authenticating with GitHub Copilot...");
        let tier_result = provider.check_tier().await?;

        if let Some(is_free_tier) = tier_result {
            if is_free_tier {
                println!("GitHub Copilot free tier detected. Using limited model.");
            } else {
                // Paid tier - prompt for model family if not already set via env var.
                let env_family_set = std::env::var("COPILOT_MODEL_FAMILY").is_ok();
                if !env_family_set {
                    let family = prompt_copilot_model_family();
                    // Reconstruct the provider with the selected family.
                    let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
                    let client_secret =
                        engine_core::provider::CopilotProvider::resolve_client_secret();
                    let github_instance =
                        engine_core::provider::CopilotProvider::resolve_github_instance();
                    let config = provider.config().clone();
                    let new_provider = engine_core::provider::CopilotProvider::new(
                        client_id,
                        client_secret,
                        github_instance,
                        config,
                        family,
                    )?;
                    println!(
                        "GitHub Copilot paid tier detected. Using {} model family.",
                        family.as_str()
                    );
                    return start_single_provider_daemon(
                        host,
                        port,
                        db_path,
                        verbose,
                        std::sync::Arc::new(new_provider)
                            as std::sync::Arc<dyn engine_core::Provider>,
                        mode,
                        project_path.clone(),
                        None,
                        None,
                    )
                    .await;
                }
                println!(
                    "GitHub Copilot paid tier detected. Using {} model family.",
                    provider.model_family().unwrap_or("openai")
                );
            }
        }
    }

    start_single_provider_daemon(
        host,
        port,
        db_path,
        verbose,
        provider,
        mode,
        project_path,
        None,
        None,
    )
    .await
}

/// Resolve a provider by label into an `Arc\u003cdyn Provider\u003e`, handling
/// API-key prompting and Copilot OAuth/tier detection. Used by
/// `start --provider \u003clabel\u003e` to share the same resolution logic as
/// `provider start-proxy` and `launch`.
pub async fn resolve_provider_arc(
    provider_label: &str,
) -> Result<std::sync::Arc<dyn engine_core::Provider>> {
    let provider_label_lc = provider_label.to_lowercase();
    let mut registry = build_provider_registry();
    let provider = registry
        .take_by_label(provider_label)
        .or_else(|| {
            if provider_label_lc == "copilot" {
                return None; // Copilot uses built-in OAuth, no key prompt
            }
            let key = prompt_for_api_key(provider_label).ok()?;
            rebuild_provider_with_key(provider_label, &key).ok()
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider '{}' is not configured",
                provider_display_name(provider_label)
            )
        })?;

    let config = provider.config();
    let provider = if config.api_key.is_empty() && provider_label_lc != "copilot" {
        let key = prompt_for_api_key(provider_label)?;
        rebuild_provider_with_key(provider_label, &key)?
    } else {
        provider
    };

    // For the Copilot provider, eagerly exchange the token to detect the
    // subscription tier and prompt for a model family (if paid).
    if provider_label_lc == "copilot" {
        println!("Authenticating with GitHub Copilot...");
        let tier_result = provider.check_tier().await?;

        if let Some(is_free_tier) = tier_result {
            if is_free_tier {
                println!("GitHub Copilot free tier detected. Using limited model.");
            } else {
                let env_family_set = std::env::var("COPILOT_MODEL_FAMILY").is_ok();
                if !env_family_set {
                    let family = prompt_copilot_model_family();
                    let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
                    let client_secret =
                        engine_core::provider::CopilotProvider::resolve_client_secret();
                    let github_instance =
                        engine_core::provider::CopilotProvider::resolve_github_instance();
                    let config = provider.config().clone();
                    let new_provider = engine_core::provider::CopilotProvider::new(
                        client_id,
                        client_secret,
                        github_instance,
                        config,
                        family,
                    )?;
                    println!(
                        "GitHub Copilot paid tier detected. Using {} model family.",
                        family.as_str()
                    );
                    return Ok(std::sync::Arc::new(new_provider)
                        as std::sync::Arc<dyn engine_core::Provider>);
                }
                println!(
                    "GitHub Copilot paid tier detected. Using {} model family.",
                    provider.model_family().unwrap_or("openai")
                );
            }
        }
    }

    Ok(provider)
}

pub fn build_provider_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    let local_config = load_local_config();

    // A provider is only registered if it has credentials - either an env
    // var set at startup, or a previously saved entry in
    // ~/.reincore-io/credentials.json (written by `reincore provider add`).
    // This prevents the Copilot OAuth device flow (or any other interactive
    // prompt) from firing on daemon startup when the user hasn't configured
    // the provider yet.

    // OpenAI
    {
        let key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("openai").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let policy = configured_model_policy(&local_config, "openai");
            let config = ProviderConfig {
                label: "openai".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::OpenAiProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // Anthropic
    {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("anthropic").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
            let policy = configured_model_policy(&local_config, "anthropic");
            let config = ProviderConfig {
                label: "anthropic".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::AnthropicProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // Gemini
    {
        let key = std::env::var("GEMINI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("gemini").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("GEMINI_BASE_URL").unwrap_or_else(|_| {
                "https://generativelanguage.googleapis.com/v1beta/openai/".to_string()
            });
            let policy = configured_model_policy(&local_config, "gemini");
            let config = ProviderConfig {
                label: "gemini".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::GeminiProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // Azure OpenAI
    {
        let key = std::env::var("AZURE_OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("azure").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("AZURE_OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://your-resource.openai.azure.com".to_string());
            let policy = configured_model_policy(&local_config, "azure");
            let config = ProviderConfig {
                label: "azure".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::AzureOpenAiProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // Groq
    {
        let key = std::env::var("GROQ_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("groq").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("GROQ_BASE_URL")
                .unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string());
            let policy = configured_model_policy(&local_config, "groq");
            let config = ProviderConfig {
                label: "groq".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::GroqProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // xAI
    {
        let key = std::env::var("XAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("xai").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url =
                std::env::var("XAI_BASE_URL").unwrap_or_else(|_| "https://api.x.ai/v1".to_string());
            let policy = configured_model_policy(&local_config, "xai");
            let config = ProviderConfig {
                label: "xai".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::XaiProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // DeepSeek
    {
        let key = std::env::var("DEEPSEEK_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("deepseek").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
            let policy = configured_model_policy(&local_config, "deepseek");
            let config = ProviderConfig {
                label: "deepseek".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::DeepSeekProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // Meta Muse (Meta Model API, OpenAI-compatible)
    {
        let key = std::env::var("META_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("meta").map(|c| c.value));
        if let Some(api_key) = key {
            let base_url = std::env::var("META_BASE_URL")
                .unwrap_or_else(|_| "https://api.meta.ai/v1".to_string());
            let policy = configured_model_policy(&local_config, "meta");
            let config = ProviderConfig {
                label: "meta".to_string(),
                base_url,
                api_key,
                standard_model: policy.standard_model,
                models: policy.models,
                routing: policy.routing,
                ..Default::default()
            };
            if let Ok(provider) = engine_core::provider::MetaProvider::new(config) {
                registry = registry.register(std::sync::Arc::new(provider));
            }
        }
    }

    // GitHub Copilot - only register if the user has previously completed the
    // OAuth device flow via `reincore provider add copilot` (token saved in
    // credentials.json). Without a stored token, the provider is skipped so
    // the daemon doesn't trigger an interactive login prompt on startup.
    if let Some(cred) = credentials::get_credential("copilot") {
        let client_id = engine_core::provider::CopilotProvider::resolve_client_id();
        let client_secret = engine_core::provider::CopilotProvider::resolve_client_secret();
        let github_instance = engine_core::provider::CopilotProvider::resolve_github_instance();
        let policy = configured_model_policy(&local_config, "copilot");
        let config = ProviderConfig {
            label: "copilot".to_string(),
            base_url: "https://api.githubcopilot.com".to_string(),
            api_key: cred.value,
            standard_model: policy.standard_model,
            models: policy.models,
            routing: policy.routing,
            ..Default::default()
        };
        if let Ok(provider) = engine_core::provider::CopilotProvider::new(
            client_id,
            client_secret,
            github_instance,
            config,
            resolve_copilot_model_family(),
        ) {
            registry = registry.register(std::sync::Arc::new(provider));
        }
    }

    // Custom OpenAI-compatible provider (for example OpenRouter). It is only
    // registered once the key, base URL, and standard model are all present.
    {
        let key = std::env::var("CUSTOM_PROVIDER_API_KEY")
            .ok()
            .filter(|key| !key.trim().is_empty())
            .or_else(|| credentials::get_credential("custom").map(|credential| credential.value));
        let details = read_custom_provider_details();
        let policy = configured_model_policy(&local_config, "custom");
        if let (Some(api_key), Some(base_url), Some(standard_model)) =
            (key, details.base_url, policy.standard_model.clone())
        {
            if !base_url.trim().is_empty() && !standard_model.trim().is_empty() {
                let config = ProviderConfig {
                    label: "custom".to_string(),
                    base_url,
                    api_key,
                    standard_model: Some(standard_model),
                    models: policy.models,
                    routing: policy.routing,
                    ..Default::default()
                };
                if let Ok(provider) = engine_core::provider::OpenAiProvider::new(config) {
                    registry = registry.register(std::sync::Arc::new(provider));
                }
            }
        }
    }

    // Ollama - local mode needs no auth, but requires the server to be
    // running. Cloud mode needs an API key saved in credentials.json.
    {
        let ollama_key = std::env::var("OLLAMA_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| credentials::get_credential("ollama").map(|c| c.value));

        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let policy = configured_model_policy(&local_config, "ollama");
        let config = ProviderConfig {
            label: "ollama".to_string(),
            base_url,
            api_key: ollama_key.clone().unwrap_or_default(),
            standard_model: policy.standard_model,
            models: policy.models,
            routing: policy.routing,
            ..Default::default()
        };
        // Use cloud mode if a key is present, local mode otherwise.
        let mode = if ollama_key.is_some() {
            engine_core::provider::OllamaMode::Cloud
        } else {
            engine_core::provider::OllamaMode::Local
        };
        if let Ok(provider) = engine_core::provider::OllamaProvider::new(config, mode) {
            registry = registry.register(std::sync::Arc::new(provider));
        }
    }

    registry
}
