use super::*;

// Provider & config management endpoints
//
// These endpoints expose the CLI-equivalent provider/config management
// functionality over HTTP so the desktop app (and any other HTTP client) can
// list providers, add/remove credentials, and read/write config.toml routing
// and model mappings.

/// Response shape for a single provider in `GET /v1/providers`.
#[derive(Serialize)]
pub struct ProviderStatusItem {
    pub id: String,
    pub label: String,
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub configured: bool,
    pub source: String,
    pub env_var: String,
    pub models: Vec<ModelTierInfo>,
    pub routing: engine_core::RoutingPolicy,
}

/// Model tier mapping for a provider.
#[derive(Serialize)]
pub struct ModelTierInfo {
    pub tier: String,
    pub model: Option<String>,
}

/// Response for `GET /v1/providers`.
#[derive(Serialize)]
pub struct ListProvidersResponse {
    pub providers: Vec<ProviderStatusItem>,
}

/// Request body for `POST /v1/providers/:provider/credentials`.
#[derive(Deserialize)]
pub struct AddCredentialRequest {
    pub api_key: String,
}

/// Response for credential operations.
#[derive(Serialize)]
pub struct CredentialResponse {
    pub provider: String,
    pub success: bool,
    pub message: String,
}

/// Response for `GET /v1/config/routing`.
#[derive(Serialize)]
pub struct RoutingConfigResponse {
    pub routing: engine_core::RoutingPolicy,
}

/// Request body for `PUT /v1/config/routing`.
#[derive(Default, Deserialize)]
#[serde(default)]
pub struct UpdateRoutingRequest {
    pub mode: Option<String>,
    pub provider_strategy: Option<String>,
    pub minimum_quality: Option<String>,
    pub allow_upgrade: Option<String>,
    pub allow_degradation: Option<String>,
}

/// Response for `GET /v1/config/providers/:provider`.
#[derive(Serialize)]
pub struct ProviderConfigResponse {
    pub provider: String,
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub standard: Option<String>,
    pub lite: Option<String>,
    pub pro: Option<String>,
    pub ultra: Option<String>,
    pub routing: engine_core::RoutingPolicy,
}

/// Request body for `PUT /v1/config/providers/:provider`.
#[derive(Default, Deserialize)]
#[serde(default)]
pub struct UpdateProviderConfigRequest {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub standard: Option<String>,
    pub lite: Option<String>,
    pub pro: Option<String>,
    pub ultra: Option<String>,
    pub routing_mode: Option<String>,
    pub routing_minimum_quality: Option<String>,
    pub routing_allow_upgrade: Option<String>,
    pub routing_allow_degradation: Option<String>,
}

/// `GET /v1/providers` - list all supported providers with configured status,
/// credential source, model tier mappings, and routing policy.
pub async fn list_providers() -> Json<ListProvidersResponse> {
    let supported = crate::provider::supported_providers();
    let registry = crate::provider::build_provider_registry();

    let mut items = Vec::with_capacity(supported.len());
    for label in &supported {
        let env_configured = registry.get_by_label(label).is_some();
        let saved_cred = crate::credentials::get_credential(label);
        let env_var = crate::provider::api_key_env_var(label);

        let model_summary = crate::local_config::read_provider_models(label);

        let custom_details = if label == "custom" {
            Some(crate::local_config::read_custom_provider_details())
        } else {
            None
        };

        let (source, configured) = if label == "custom" {
            let details = custom_details.as_ref().expect("custom details");
            let has_name = details
                .name
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            let has_base = details
                .base_url
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            let has_model = model_summary
                .standard
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            if env_configured {
                ("via custom configuration".to_string(), true)
            } else {
                let mut missing = Vec::new();
                if saved_cred.is_none() {
                    missing.push("API key");
                }
                if !has_name {
                    missing.push("name");
                }
                if !has_base {
                    missing.push("base URL");
                }
                if !has_model {
                    missing.push("standard model");
                }
                (format!("missing {}", missing.join(", ")), false)
            }
        } else if env_configured && label != "ollama" {
            (format!("via {}", env_var), true)
        } else if saved_cred.is_some() {
            ("via saved credential".to_string(), true)
        } else if label == "ollama" {
            let ollama_key = std::env::var("OLLAMA_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty());
            let has_standard_model = model_summary
                .standard
                .as_deref()
                .map(|m| !m.is_empty())
                .unwrap_or(false);
            let key_source = if ollama_key.is_some() {
                "OLLAMA_API_KEY env var"
            } else if saved_cred.is_some() {
                "saved credential"
            } else {
                "none"
            };
            if !has_standard_model {
                (
                    format!("standard model required (key: {})", key_source),
                    false,
                )
            } else if ollama_key.is_some() || saved_cred.is_some() {
                ("via API key".to_string(), true)
            } else {
                ("local (no auth needed)".to_string(), true)
            }
        } else if label == "copilot" {
            (format!("via {} (OAuth)", env_var), false)
        } else {
            (format!("set {}", env_var), false)
        };

        let routing = crate::local_config::read_provider_routing(label);

        let models = vec![
            ModelTierInfo {
                tier: "standard".to_string(),
                model: model_summary.standard,
            },
            ModelTierInfo {
                tier: "lite".to_string(),
                model: model_summary.lite,
            },
            ModelTierInfo {
                tier: "pro".to_string(),
                model: model_summary.pro,
            },
            ModelTierInfo {
                tier: "ultra".to_string(),
                model: model_summary.ultra,
            },
        ];

        items.push(ProviderStatusItem {
            id: label.clone(),
            label: custom_details
                .as_ref()
                .and_then(|details| details.name.clone())
                .unwrap_or_else(|| crate::provider::provider_display_name(label)),
            name: custom_details
                .as_ref()
                .and_then(|details| details.name.clone()),
            base_url: custom_details
                .as_ref()
                .and_then(|details| details.base_url.clone()),
            configured,
            source,
            env_var,
            models,
            routing,
        });
    }

    Json(ListProvidersResponse { providers: items })
}

/// `POST /v1/providers/:provider/credentials` - store an API key for a provider.
pub async fn add_provider_credentials(
    Path(provider): Path<String>,
    axum::Json(req): axum::Json<AddCredentialRequest>,
) -> Result<Json<CredentialResponse>, (StatusCode, String)> {
    let label = provider.to_lowercase();
    let supported = crate::provider::supported_providers();
    if !supported.contains(&label) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Unknown provider '{}'. Supported: {}",
                label,
                supported.join(", ")
            ),
        ));
    }

    if label == "copilot" {
        return Err((
            StatusCode::BAD_REQUEST,
            "Copilot uses OAuth device flow. Use 'reincore provider add copilot' in the CLI."
                .to_string(),
        ));
    }

    // Validate the key by trying to build the provider with it.
    crate::provider::rebuild_provider_with_key(&label, &req.api_key)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid API key: {}", e)))?;

    crate::credentials::store_api_key(&label, &req.api_key)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CredentialResponse {
        provider: label,
        success: true,
        message: "Credentials saved. Restart the daemon to activate.".to_string(),
    }))
}

/// `DELETE /v1/providers/:provider/credentials` - remove stored credentials.
pub async fn remove_provider_credentials(
    Path(provider): Path<String>,
) -> Result<Json<CredentialResponse>, (StatusCode, String)> {
    let label = provider.to_lowercase();

    let removed = crate::credentials::remove_credential(&label)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CredentialResponse {
        provider: label,
        success: removed,
        message: if removed {
            "Credentials removed. Restart the daemon to take effect.".to_string()
        } else {
            "No stored credentials found for this provider.".to_string()
        },
    }))
}

/// `GET /v1/config/routing` - read the global routing policy from config.toml.
pub async fn get_routing_config() -> Json<RoutingConfigResponse> {
    let routing = crate::local_config::read_global_routing();
    Json(RoutingConfigResponse { routing })
}

/// `PUT /v1/config/routing` - update global routing fields in config.toml.
pub async fn update_routing_config(
    axum::Json(req): axum::Json<UpdateRoutingRequest>,
) -> Result<Json<RoutingConfigResponse>, (StatusCode, String)> {
    if let Some(mode) = &req.mode {
        crate::local_config::set_routing_field("mode", mode)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(strategy) = &req.provider_strategy {
        crate::local_config::set_routing_field("provider_strategy", strategy)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(quality) = &req.minimum_quality {
        crate::local_config::set_routing_field("minimum_quality", quality)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(allow) = &req.allow_upgrade {
        crate::local_config::set_routing_field("allow_upgrade", allow)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(allow) = &req.allow_degradation {
        crate::local_config::set_routing_field("allow_degradation", allow)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    let routing = crate::local_config::read_global_routing();
    Ok(Json(RoutingConfigResponse { routing }))
}

/// `GET /v1/config/providers/:provider` - read a provider's model config and routing.
pub async fn get_provider_config(Path(provider): Path<String>) -> Json<ProviderConfigResponse> {
    let models = crate::local_config::read_provider_models(&provider);
    let routing = crate::local_config::read_provider_routing(&provider);

    Json(ProviderConfigResponse {
        provider: provider.to_lowercase(),
        name: if provider.eq_ignore_ascii_case("custom") {
            crate::local_config::read_custom_provider_details().name
        } else {
            None
        },
        base_url: if provider.eq_ignore_ascii_case("custom") {
            crate::local_config::read_custom_provider_details().base_url
        } else {
            None
        },
        standard: models.standard,
        lite: models.lite,
        pro: models.pro,
        ultra: models.ultra,
        routing,
    })
}

/// `PUT /v1/config/providers/:provider` - update a provider's model tiers and/or routing.
pub async fn update_provider_config(
    Path(provider): Path<String>,
    axum::Json(req): axum::Json<UpdateProviderConfigRequest>,
) -> Result<Json<ProviderConfigResponse>, (StatusCode, String)> {
    let label = provider.to_lowercase();
    if label == "custom" {
        // Validate the whole custom metadata patch before writing any fields.
        if req
            .name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty() || name.trim().len() > 80)
            || req
                .base_url
                .as_ref()
                .is_some_and(|url| !valid_local_custom_base_url(url.trim()))
            || req
                .standard
                .as_ref()
                .is_some_and(|model| model.trim().is_empty())
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "Enter a provider name, a valid HTTP(S) base URL, and a standard model."
                    .to_string(),
            ));
        }
        if let Some(name) = &req.name {
            let name = name.trim();
            if name.is_empty() || name.len() > 80 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "Custom provider name is required and must be at most 80 characters."
                        .to_string(),
                ));
            }
            crate::local_config::set_provider_detail_field(&label, "name", name)
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
        }
        if let Some(base_url) = &req.base_url {
            let base_url = base_url.trim().trim_end_matches('/');
            if !valid_local_custom_base_url(base_url) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "Custom provider base URL must be an http(s) URL with a host.".to_string(),
                ));
            }
            crate::local_config::set_provider_detail_field(&label, "base_url", base_url)
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
        }
    }

    if let Some(standard) = &req.standard {
        crate::local_config::set_provider_model_field(&label, "standard", standard)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(lite) = &req.lite {
        if lite.is_empty() {
            crate::local_config::clear_provider_model_field(&label, "lite")
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        } else {
            crate::local_config::set_provider_model_field(&label, "lite", lite)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        }
    }
    if let Some(pro) = &req.pro {
        if pro.is_empty() {
            crate::local_config::clear_provider_model_field(&label, "pro")
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        } else {
            crate::local_config::set_provider_model_field(&label, "pro", pro)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        }
    }
    if let Some(ultra) = &req.ultra {
        if ultra.is_empty() {
            crate::local_config::clear_provider_model_field(&label, "ultra")
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        } else {
            crate::local_config::set_provider_model_field(&label, "ultra", ultra)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        }
    }
    if let Some(mode) = &req.routing_mode {
        crate::local_config::set_provider_routing_field(&label, "mode", mode)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(quality) = &req.routing_minimum_quality {
        crate::local_config::set_provider_routing_field(&label, "minimum_quality", quality)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(allow) = &req.routing_allow_upgrade {
        crate::local_config::set_provider_routing_field(&label, "allow_upgrade", allow)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    if let Some(allow) = &req.routing_allow_degradation {
        crate::local_config::set_provider_routing_field(&label, "allow_degradation", allow)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    let models = crate::local_config::read_provider_models(&label);
    let routing = crate::local_config::read_provider_routing(&label);

    Ok(Json(ProviderConfigResponse {
        provider: label.clone(),
        name: if label == "custom" {
            crate::local_config::read_custom_provider_details().name
        } else {
            None
        },
        base_url: if label == "custom" {
            crate::local_config::read_custom_provider_details().base_url
        } else {
            None
        },
        standard: models.standard,
        lite: models.lite,
        pro: models.pro,
        ultra: models.ultra,
        routing,
    }))
}

fn valid_local_custom_base_url(raw: &str) -> bool {
    reqwest::Url::parse(raw).is_ok_and(|url| {
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
    })
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct CustomModelsRequest {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Serialize)]
pub struct CustomModelItem {
    pub id: String,
    pub name: String,
}

#[derive(Serialize)]
pub struct CustomModelsResponse {
    pub models: Vec<CustomModelItem>,
}

/// Probe a custom OpenAI-compatible provider's {base_url}/models endpoint.
/// A valid response with no model entries is returned as an empty list so
/// the desktop form can fall back to exact manual model IDs.
pub async fn list_custom_models(
    axum::Json(req): axum::Json<CustomModelsRequest>,
) -> Result<Json<CustomModelsResponse>, (StatusCode, String)> {
    let details = crate::local_config::read_custom_provider_details();
    let base_url = req
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or(details.base_url.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "Enter the custom provider base URL first.".to_string(),
            )
        })?;
    if !valid_local_custom_base_url(&base_url) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Custom provider base URL must be an http(s) URL with a host.".to_string(),
        ));
    }
    let use_saved_key = details
        .base_url
        .as_deref()
        .is_some_and(|saved| saved.trim_end_matches('/') == base_url.trim_end_matches('/'));
    let api_key = req
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            use_saved_key
                .then(|| std::env::var("CUSTOM_PROVIDER_API_KEY").ok())
                .flatten()
        })
        .or_else(|| {
            use_saved_key
                .then(|| {
                    crate::credentials::get_credential("custom").map(|credential| credential.value)
                })
                .flatten()
        });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let mut request = client
        .get(format!("{}/models", base_url.trim_end_matches('/')))
        .header("Accept", "application/json");
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    let mut response = request.send().await.map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Custom provider model discovery failed: {error}"),
        )
    })?;
    let status = response.status();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "Could not read custom provider models.".to_string(),
        )
    })? {
        if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
            return Err((
                StatusCode::BAD_GATEWAY,
                "Custom provider model response is too large.".to_string(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Custom provider /models returned HTTP {status}."),
        ));
    }
    let models = parse_custom_models(&bytes).map_err(|error| (StatusCode::BAD_GATEWAY, error))?;
    Ok(Json(CustomModelsResponse { models }))
}

fn parse_custom_models(bytes: &[u8]) -> Result<Vec<CustomModelItem>, String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("Custom provider /models did not return valid JSON: {error}"))?;
    let entries = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.get("models").and_then(serde_json::Value::as_array))
        .or_else(|| value.as_array());
    let mut models = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in entries.into_iter().flatten() {
        let Some(id) = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            continue;
        };
        if !seen.insert(id.to_string()) {
            continue;
        }
        let name = entry
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(id);
        models.push(CustomModelItem {
            id: id.to_string(),
            name: name.to_string(),
        });
    }
    Ok(models)
}

#[cfg(test)]
mod custom_provider_tests {
    use super::*;

    #[test]
    fn model_discovery_preserves_namespaced_ids_and_falls_back() {
        let models = parse_custom_models(br#"{"data":[{"id":"vendor/model","name":"Model"},{"id":"vendor/model"},{"id":"second"},{"name":"missing-id"}]}"#).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "vendor/model");
        assert_eq!(models[0].name, "Model");
        assert_eq!(models[1].name, "second");
        for body in [br#"{"data":[]}"#.as_slice(), br#"{"error":"unsupported"}"#] {
            assert!(parse_custom_models(body).unwrap().is_empty());
        }
        assert!(parse_custom_models(b"not json").is_err());
        assert!(valid_local_custom_base_url("http://127.0.0.1:1234/v1"));
        assert!(!valid_local_custom_base_url(
            "https://user:key@example.com/v1"
        ));
    }

    #[tokio::test]
    async fn custom_discovery_calls_models_with_bearer_auth() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            loop {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let wire = String::from_utf8(request).unwrap().to_lowercase();
            assert!(wire.starts_with("get /v1/models "));
            assert!(wire.contains("authorization: bearer test-key\r\n"));
            let body = r#"{"data":[{"id":"vendor/model"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let Json(response) = list_custom_models(Json(CustomModelsRequest {
            base_url: Some(format!("http://{address}/v1/")),
            api_key: Some("test-key".into()),
        }))
        .await
        .unwrap();
        server.join().unwrap();
        assert_eq!(response.models[0].id, "vendor/model");
    }
}

// Model catalog endpoints

/// A simplified model record for the catalog response. This shape matches
/// what the frontend `ProviderModelField` dropdown expects.
#[derive(Serialize)]
pub struct CatalogModelItem {
    pub provider: String,
    pub model_name: String,
    pub model_id: String,
    pub input_price: f64,
    pub output_price: f64,
    pub reasoning: String,
    pub speed: String,
}

/// Response for `GET /v1/providers/:provider/models`.
#[derive(Serialize)]
pub struct ProviderCatalogResponse {
    pub provider: String,
    pub models: Vec<CatalogModelItem>,
}

/// `GET /v1/providers/:provider/models` - list models from the curated catalog
/// for a given provider. No credentials needed - reads from the static catalog
/// or the local SQLite `provider_models` table.
///
/// For Copilot, the `:provider` should be a model family (`openai` or
/// `anthropic`), not `copilot` itself, since Copilot's models are cataloged
/// under the underlying provider's family.
pub async fn list_provider_catalog(Path(provider): Path<String>) -> Json<ProviderCatalogResponse> {
    let label = provider.to_lowercase();

    // The curated catalog uses provider labels like "openai", "anthropic", etc.
    // Copilot's models are cataloged under the family label.
    let catalog = engine_core::model_library::model_catalog();

    let mut records: Vec<&engine_core::model_library::ProviderModelRecord> = catalog
        .iter()
        .filter(|m| m.provider.eq_ignore_ascii_case(&label))
        .collect();
    // Higher rank means newer/preferred within the provider.
    records.sort_by(|a, b| b.rank.cmp(&a.rank));

    let models: Vec<CatalogModelItem> = records
        .into_iter()
        .map(|m| CatalogModelItem {
            provider: m.provider.clone(),
            model_name: m.model_name.clone(),
            model_id: m.model_id.clone(),
            input_price: m.input_price,
            output_price: m.output_price,
            reasoning: m.reasoning.as_str().to_string(),
            speed: m.speed.as_str().to_string(),
        })
        .collect();

    Json(ProviderCatalogResponse {
        provider: label,
        models,
    })
}

/// Request body for `POST /v1/providers/ollama/models` - fetch live models from
/// a running Ollama server. Mirrors `reincore-frontend`'s `listOllamaModels`.
#[derive(Default, Deserialize)]
#[serde(default)]
pub struct OllamaModelsRequest {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

/// Response model for Ollama model list.
#[derive(Serialize)]
pub struct OllamaModelItem {
    pub id: String,
    pub name: String,
    pub parameter_size: Option<String>,
}

/// Response for `POST /v1/providers/ollama/models`.
#[derive(Serialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModelItem>,
}

/// `POST /v1/providers/ollama/models` - fetch the live model list from a
/// running Ollama server at the given `base_url`. The API key is optional
/// (for Ollama Cloud) and is not persisted by this call.
pub async fn list_ollama_models(
    axum::Json(req): axum::Json<OllamaModelsRequest>,
) -> Result<Json<OllamaModelsResponse>, (StatusCode, String)> {
    let base_url = req
        .base_url
        .unwrap_or_else(|| crate::launch::default_ollama_base_url().to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut request = client
        .get(format!("{}/api/tags", base_url))
        .header("Content-Type", "application/json");

    if let Some(key) = &req.api_key {
        if !key.trim().is_empty() {
            request = request.header("Authorization", format!("Bearer {}", key));
        }
    }

    let response = request.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Ollama request failed: {}", e),
        )
    })?;

    let status = response.status();
    let body: serde_json::Value = response.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Failed to parse Ollama response: {}", e),
        )
    })?;

    if !status.is_success() {
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            format!("Ollama returned {}: {}", status, body),
        ));
    }

    let models = body
        .get("models")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let id = v.get("name").and_then(|v| v.as_str())?.to_string();
                    let name = v
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&id)
                        .to_string();
                    let parameter_size = v
                        .get("details")
                        .and_then(|d| d.get("parameter_size"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    Some(OllamaModelItem {
                        id,
                        name,
                        parameter_size,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(Json(OllamaModelsResponse { models }))
}
