//! Detached title generation: no history, tools, or completion request extras.
use super::*;
use crate::local_config::TitleSummarizer;
use engine_core::title_generation::generate_title;
#[cfg(test)]
use engine_core::title_generation::title_request;
use std::sync::Arc;

// Parallel tool-loop requests can resolve the same new task before its first
// LlmCall is stored. Deduplicate their background title jobs.
static TITLE_JOBS: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
struct TitleJob(String);
impl Drop for TitleJob {
    fn drop(&mut self) {
        if let Some(jobs) = TITLE_JOBS.get() {
            jobs.lock().unwrap().remove(&self.0);
        }
    }
}

pub(super) fn spawn_titles(
    db: Arc<LocalDatabase>,
    provider: Arc<dyn engine_core::Provider>,
    session: Option<&Session>,
    task: &engine_core::TaskAssignment,
    request: &ChatCompletionRequest,
) {
    if !task.is_new || request.extra.contains_key(engine_core::TASK_NAME_EXTRA_KEY) {
        return;
    }
    let Some(session) = session.cloned() else {
        return;
    };
    let Some(input) = engine_core::summary::latest_title_input(request) else {
        return;
    };
    let task = task.clone();
    tokio::spawn(async move {
        // Config I/O and SQLite work stay off the completion path.
        let Ok(config) = tokio::task::spawn_blocking(crate::local_config::load_local_config).await
        else {
            return;
        };
        if config.titles.summarizer != TitleSummarizer::Llm {
            return;
        }
        if task.task_name == engine_core::FALLBACK_TASK_SUMMARY {
            return;
        }
        let key = format!("{}:{}", session.id, task.task_id);
        if !TITLE_JOBS
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .insert(key.clone())
        {
            return;
        }
        let _job = TitleJob(key);
        let title =
            generate_title(provider.as_ref(), &input, Some((db.as_ref(), session.id))).await;
        let Some(title) = title else { return };
        if let Err(error) = db.save_generated_title(session, task, title).await {
            warn!(%error, "could not persist generated title");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{ModelTier, Provider, ProviderConfig, ProviderModelMapping};

    struct FakeProvider {
        config: ProviderConfig,
        output: String,
        usage: Option<engine_core::TokenUsage>,
    }

    #[async_trait::async_trait]
    impl Provider for FakeProvider {
        fn label(&self) -> &str {
            &self.config.label
        }
        fn config(&self) -> &ProviderConfig {
            &self.config
        }
        async fn complete(
            &self,
            request: &ChatCompletionRequest,
        ) -> engine_core::EngineResult<engine_core::ChatCompletionResponse> {
            assert_eq!(request.messages.len(), 2);
            assert_eq!(
                request.messages[1].content_str(),
                Some("Fix navigation scroll position")
            );
            assert!(request.tools.is_none());
            if self.output == "pending" {
                return std::future::pending().await;
            }
            if self.output == "error" {
                return Err(engine_core::EngineError::Provider("offline".into()));
            }
            Ok(serde_json::from_value(json!({
                "id":"title", "model": request.model,
                "message":{"role":"assistant", "content":self.output},
                "usage": self.usage, "raw":{}
            }))
            .unwrap())
        }
        async fn list_models(&self) -> engine_core::EngineResult<Vec<engine_core::ProviderModel>> {
            Ok(vec![])
        }
    }

    fn provider(output: &str) -> FakeProvider {
        FakeProvider {
            config: ProviderConfig {
                standard_model: Some("standard-model".into()),
                models: vec![
                    ProviderModelMapping {
                        tier: ModelTier::Lite,
                        provider_model: "lite-model".into(),
                    },
                    ProviderModelMapping {
                        tier: ModelTier::Pro,
                        provider_model: "pro-override".into(),
                    },
                ],
                ..Default::default()
            },
            output: output.into(),
            usage: None,
        }
    }

    #[tokio::test]
    async fn invalid_title_still_records_billable_attempt_and_missing_usage() {
        use engine_core::EventStore;
        let db = LocalDatabase::open_in_memory().unwrap();
        let mut session = Session::new("title", 100).unwrap();
        session.context_tokens = 1234;
        db.insert_session(&session).unwrap();
        assert!(generate_title(
            &provider(""),
            "Fix navigation scroll position",
            Some((&db, session.id))
        )
        .await
        .is_none());
        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.unknown_cost_calls, 1);
        assert_eq!(saved.context_tokens, 1234);
        let events = db.read_events(session.id, None, None).await.unwrap();
        assert_eq!(
            events[0].event_type,
            engine_core::SessionEventType::AuxiliaryCall
        );
        assert_eq!(events[0].payload["purpose"], "title");
    }
    #[tokio::test]
    async fn generated_title_charges_exact_usage() {
        let db = LocalDatabase::open_in_memory().unwrap();
        let session = Session::new("title", 100).unwrap();
        db.insert_session(&session).unwrap();
        let mut p = provider("Generate a billable example title");
        p.config.label = "openai".into();
        p.config.models[0].provider_model = "gpt-5-nano-2025-08-07".into();
        p.usage = Some(engine_core::TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            ..Default::default()
        });
        assert!(generate_title(
            &p,
            "Fix navigation scroll position",
            Some((&db, session.id))
        )
        .await
        .is_some());
        let saved = db.get_session(session.id).unwrap().unwrap();
        assert_eq!(saved.spend_nano_usd, Some(9_000));
        assert_eq!(saved.unknown_cost_calls, 0);
        assert_eq!(saved.context_tokens, 0);
        assert_eq!(saved.total_input_tokens, 100);
    }
    #[test]
    fn title_config_defaults_and_validates_choices() {
        let parse = |text: &str| {
            config::Config::builder()
                .add_source(config::File::from_str(text, config::FileFormat::Toml))
                .build()
                .unwrap()
                .try_deserialize::<crate::local_config::LocalConfig>()
        };
        assert_eq!(
            parse("").unwrap().titles.summarizer,
            TitleSummarizer::Reincore
        );
        let config = parse("[titles]\nsummarizer = 'llm'").unwrap();
        assert_eq!(config.titles.summarizer, TitleSummarizer::Llm);
        assert!(parse("[titles]\nsummarizer = 'invalid'").is_err());
        // The example may explicitly opt into either supported summarizer.
        // The omitted-setting default is verified separately above.
        assert!(parse(include_str!("../../config.example.toml")).is_ok());
    }

    #[tokio::test]
    async fn titles_select_current_provider_models_and_handle_failure() {
        for (explicit_lite, expected, routed_tier) in [
            (true, "lite-model", ModelTier::Lite),
            (false, "standard-model", ModelTier::Pro),
        ] {
            let mut provider = provider("Fix navigation scroll position");
            if !explicit_lite {
                provider
                    .config
                    .models
                    .retain(|mapping| mapping.tier != ModelTier::Lite);
            }
            let request = title_request(&provider, "Fix navigation scroll position").unwrap();
            assert_eq!(request.model, expected);
            assert_eq!(
                request.extra["reincore_model_tier"],
                json!(routed_tier.as_str())
            );
            assert_eq!(
                request.extra["reincore_title_standard_model"],
                json!(!explicit_lite)
            );
            assert_eq!(
                provider
                    .config
                    .model_for_request(&request, routed_tier)
                    .unwrap(),
                expected
            );
            assert_eq!(
                generate_title(&provider, "Fix navigation scroll position", None)
                    .await
                    .as_deref(),
                Some("Fix navigation scroll position")
            );
        }
        for output in [
            "error",
            "too short",
            "one two three four five six seven eight nine ten eleven",
        ] {
            let fake = self::provider(output);
            assert!(
                generate_title(&fake, "Fix navigation scroll position", None)
                    .await
                    .is_none()
            );
        }
    }

    #[tokio::test]
    async fn slow_title_request_can_run_independently() {
        let handle = tokio::spawn(async {
            generate_title(&provider("pending"), "Fix navigation scroll position", None).await
        });
        tokio::task::yield_now().await;
        assert!(!handle.is_finished());
        assert_eq!(
            tokio::spawn(async { "chat completed" }).await.unwrap(),
            "chat completed"
        );
        handle.abort();
    }
}
