//! Handlers for the `reincore routing` and `reincore model` CLI commands.
//!
//! These read and update `~/.reincore-io/config.toml` (or the path in
//! `REINCORE_CONFIG_PATH`). The file is created on first run with the default
//! contents from `config.example.toml`.

use anyhow::Result;
use colored::Colorize;

use crate::local_config;

/// Print or update the global routing policy.
pub fn handle_routing(
    mode: Option<String>,
    provider_strategy: Option<String>,
    minimum_quality: Option<String>,
    allow_upgrade: Option<String>,
    allow_degradation: Option<String>,
) -> Result<()> {
    local_config::ensure_config_exists().ok();

    let mut updated = false;
    if let Some(value) = mode {
        validate_mode(&value)?;
        local_config::set_routing_field("mode", &value)?;
        updated = true;
    }
    if let Some(value) = provider_strategy {
        local_config::set_routing_field("provider_strategy", &value)?;
        updated = true;
    }
    if let Some(value) = minimum_quality {
        validate_quality(&value)?;
        local_config::set_routing_field("minimum_quality", &value)?;
        updated = true;
    }
    if let Some(value) = allow_upgrade {
        local_config::set_routing_field("allow_upgrade", &bool_string(&value)?)?;
        updated = true;
    }
    if let Some(value) = allow_degradation {
        local_config::set_routing_field("allow_degradation", &bool_string(&value)?)?;
        updated = true;
    }

    let policy = local_config::read_global_routing();
    let path = local_config::local_config_path();

    if updated {
        println!(
            "{}  {}",
            "Updated global routing policy in".green().bold(),
            path.display()
        );
    } else {
        println!(
            "{}  {}",
            "Global routing policy".cyan().bold(),
            path.display().to_string().dimmed()
        );
    }

    println!("  mode                = {}", policy.mode.as_str());
    println!(
        "  provider_strategy   = {}",
        match policy.provider_strategy {
            engine_core::ProviderStrategy::SameProvider => "same_provider",
        }
    );
    println!(
        "  minimum_quality     = {}",
        match policy.minimum_quality {
            engine_core::MinimumQuality::Economy => "economy",
            engine_core::MinimumQuality::Balanced => "balanced",
            engine_core::MinimumQuality::High => "high",
        }
    );
    println!("  allow_upgrade       = {}", policy.allow_upgrade);
    println!("  allow_degradation   = {}", policy.allow_degradation);
    Ok(())
}

/// Print or update a provider's model mappings and per-provider routing.
pub fn handle_model(
    provider: &str,
    standard: Option<String>,
    lite: Option<String>,
    pro: Option<String>,
    ultra: Option<String>,
    routing_mode: Option<String>,
    routing_minimum_quality: Option<String>,
) -> Result<()> {
    local_config::ensure_config_exists().ok();
    let provider_label = provider.to_lowercase();

    let mut updated = false;
    if let Some(value) = standard {
        local_config::set_provider_model_field(&provider_label, "standard_model", &value)?;
        updated = true;
    }
    if let Some(value) = lite {
        if value.is_empty() {
            local_config::clear_provider_model_field(&provider_label, "lite_model")?;
        } else {
            local_config::set_provider_model_field(&provider_label, "lite_model", &value)?;
        }
        updated = true;
    }
    if let Some(value) = pro {
        if value.is_empty() {
            local_config::clear_provider_model_field(&provider_label, "pro_model")?;
        } else {
            local_config::set_provider_model_field(&provider_label, "pro_model", &value)?;
        }
        updated = true;
    }
    if let Some(value) = ultra {
        if value.is_empty() {
            local_config::clear_provider_model_field(&provider_label, "ultra_model")?;
        } else {
            local_config::set_provider_model_field(&provider_label, "ultra_model", &value)?;
        }
        updated = true;
    }
    if let Some(value) = routing_mode {
        validate_mode(&value)?;
        local_config::set_provider_routing_field(&provider_label, "mode", &value)?;
        updated = true;
    }
    if let Some(value) = routing_minimum_quality {
        validate_quality(&value)?;
        local_config::set_provider_routing_field(&provider_label, "minimum_quality", &value)?;
        updated = true;
    }

    let models = local_config::read_provider_models(&provider_label);
    let routing = local_config::read_provider_routing(&provider_label);
    let path = local_config::local_config_path();

    if updated {
        println!(
            "{}  {}[providers.{}]",
            "Updated provider config in".green().bold(),
            path.display().to_string().dimmed(),
            provider_label
        );
    } else {
        println!(
            "{}  {}",
            format!("Provider '{}' configuration", provider_label)
                .cyan()
                .bold(),
            path.display().to_string().dimmed()
        );
    }

    println!(
        "  standard_model = {}",
        models.standard.as_deref().unwrap_or("(none)")
    );
    println!(
        "  lite_model     = {}",
        models.lite.as_deref().unwrap_or("(none, auto-derived)")
    );
    println!(
        "  pro_model      = {}",
        models.pro.as_deref().unwrap_or("(none)")
    );
    println!(
        "  ultra_model    = {}",
        models.ultra.as_deref().unwrap_or("(none)")
    );
    println!("  routing.mode              = {}", routing.mode.as_str());
    println!(
        "  routing.minimum_quality   = {}",
        match routing.minimum_quality {
            engine_core::MinimumQuality::Economy => "economy",
            engine_core::MinimumQuality::Balanced => "balanced",
            engine_core::MinimumQuality::High => "high",
        }
    );
    Ok(())
}

fn validate_mode(value: &str) -> Result<()> {
    match value.to_ascii_lowercase().as_str() {
        "auto" | "custom" | "locked" => Ok(()),
        other => Err(anyhow::anyhow!(
            "invalid routing mode '{}'. Use auto, custom, or locked.",
            other
        )),
    }
}

fn validate_quality(value: &str) -> Result<()> {
    match value.to_ascii_lowercase().as_str() {
        "economy" | "balanced" | "high" => Ok(()),
        other => Err(anyhow::anyhow!(
            "invalid minimum quality '{}'. Use economy, balanced, or high.",
            other
        )),
    }
}

fn bool_string(value: &str) -> Result<String> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok("true".to_string()),
        "false" | "0" | "no" | "off" => Ok("false".to_string()),
        other => Err(anyhow::anyhow!(
            "invalid boolean '{}'. Use true or false.",
            other
        )),
    }
}
