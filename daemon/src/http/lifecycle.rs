use super::request_group::PROVIDER_HEADER;
use super::*;

pub(super) fn checkpoint_resume_error_to_status(error: &engine_core::EngineError) -> StatusCode {
    match error {
        engine_core::EngineError::CheckpointAlreadyRedeemed => StatusCode::CONFLICT,
        engine_core::EngineError::CheckpointNotFound(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub(super) fn provider_override_from_headers(
    headers: &HeaderMap,
    providers: &engine_core::ProviderRegistry,
    model: &str,
) -> Result<Option<String>, StatusCode> {
    let provider = headers
        .get(PROVIDER_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_lowercase);

    if provider.is_none() {
        // No explicit header: `reincore-auto` is the documented automatic
        // routing handle, so it never requires a header - selection falls
        // through to the default provider below.
        if model.eq_ignore_ascii_case("reincore-auto") {
            return Ok(None);
        }
        // Otherwise default to the provider owning the requested model, so
        // clients that send model ids (e.g. Muse Code sending
        // `muse-spark-1.3`) route to Muse models without extra headers.
        if let Some(owned) = providers.provider_owning_model(model) {
            return Ok(Some(owned.label().to_string()));
        }
    }

    if providers.is_multi_provider() && provider.is_none() {
        warn!("multi-provider mode requires a non-empty X-ReinCore-Provider request header");
        return Err(StatusCode::BAD_REQUEST);
    }

    Ok(provider)
}

pub(super) fn extra_text(request: &ChatCompletionRequest, key: &str) -> Option<String> {
    request
        .extra
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(super) fn extra_bool(request: &ChatCompletionRequest, key: &str) -> Option<bool> {
    let value = request.extra.get(key)?;
    if let Some(raw) = value.as_bool() {
        return Some(raw);
    }
    value.as_str().and_then(|raw| {
        let lowered = raw.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "true" | "1" | "yes" | "y" => Some(true),
            "false" | "0" | "no" | "n" => Some(false),
            _ => None,
        }
    })
}

pub(super) fn delegation_profile(
    headers: Option<&HeaderMap>,
    request: &ChatCompletionRequest,
) -> Option<String> {
    headers
        .and_then(|headers| headers.get(engine_core::DELEGATION_PROFILE_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| extra_text(request, "reincore_delegation_profile"))
        .or_else(|| extra_text(request, "reincore_client_id"))
}

pub(super) fn request_session_mode(request: &ChatCompletionRequest) -> SessionMode {
    if let Some(mode) = request
        .extra
        .get("reincore_session_mode")
        .and_then(|value| value.as_str())
        .and_then(|raw| raw.parse::<SessionMode>().ok())
    {
        return mode;
    }
    if extra_bool(request, "reincore_batch_eligible") == Some(true) {
        return SessionMode::Background;
    }
    SessionMode::Interactive
}

/// Sessions that open on a greeting, a smoke test or an exploratory question
/// only get a provisional name. Once a later turn states the actual task, adopt
/// it for the whole rollover lineage.
pub(super) async fn rename_session_if_clearer(
    db: &crate::db::LocalDatabase,
    session: &mut Session,
    request: &ChatCompletionRequest,
) {
    if extra_bool(request, "reincore_is_subagent").unwrap_or(false)
        && !extra_bool(request, "reincore_track_subagent_sessions").unwrap_or(true)
    {
        return;
    }
    let Some(better) = improved_task_summary(&session.task_summary, request) else {
        return;
    };
    let root_id = session.root_session_id.unwrap_or(session.id);
    if db
        .update_session_summary_async(root_id, better.clone())
        .await
        .is_ok()
    {
        session.task_summary = better;
    }
}

pub(super) async fn resolve_local_project_context(
    db: &crate::db::LocalDatabase,
    requested_name: String,
    requested_path: String,
) -> Result<(Option<Uuid>, String, String), StatusCode> {
    let (normalized_path, _) =
        crate::db::normalize_project_path(&requested_path).map_err(|_| StatusCode::BAD_REQUEST)?;
    let project = db
        .get_local_project_by_path_async(normalized_path.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(match project {
        Some(project) => (Some(project.id), project.name, normalized_path),
        None => (None, requested_name, normalized_path),
    })
}

pub(super) async fn resolve_local_session(
    db: &crate::db::LocalDatabase,
    request: &ChatCompletionRequest,
    default_project_name: &str,
    default_project_path: &str,
) -> Result<Option<Session>, StatusCode> {
    if let Some(raw_id) = request.extra.get("reincore_session_id") {
        let raw_id = raw_id.as_str().ok_or(StatusCode::BAD_REQUEST)?;
        let id = Uuid::parse_str(raw_id).map_err(|_| StatusCode::BAD_REQUEST)?;
        let mut session = db
            .get_active_session_in_lineage_async(id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .ok_or(StatusCode::NOT_FOUND)?;
        rename_session_if_clearer(db, &mut session, request).await;
        return Ok(Some(session));
    }

    let client_id = extra_text(request, "reincore_client_id").unwrap_or_else(|| "local".into());
    let requested_project_name = extra_text(request, "reincore_project_name")
        .unwrap_or_else(|| default_project_name.to_string());
    let requested_project_path = extra_text(request, "reincore_project_path")
        .unwrap_or_else(|| default_project_path.to_string());
    let (project_id, project_name, project_path) =
        resolve_local_project_context(db, requested_project_name, requested_project_path).await?;

    if let Some(conversation_id) = engine_core::client_conversation_id(request, &client_id) {
        if conversation_id.len() > 2048 {
            return Err(StatusCode::BAD_REQUEST);
        }
        let scope_key = project_id
            .map(|id| format!("project:{id}"))
            .unwrap_or_else(|| format!("path:{project_path}"));
        let binding_key =
            engine_core::conversation_binding_key(&client_id, &scope_key, &conversation_id);
        let conversation_hash = engine_core::conversation_id_hash(&conversation_id);
        let parent_conversation =
            extra_text(request, engine_core::PARENT_CONVERSATION_ID_EXTRA_KEY);
        let parent_binding_key = parent_conversation
            .as_ref()
            .map(|parent| engine_core::conversation_binding_key(&client_id, &scope_key, parent));
        let parent_session = if let Some(parent_key) = parent_binding_key.as_ref() {
            match db
                .get_conversation_binding_session_id_async(parent_key.clone())
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            {
                Some(id) => db
                    .get_active_session_in_lineage_async(id)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                None => None,
            }
        } else {
            None
        };
        let is_subagent = parent_conversation.is_some()
            || extra_bool(request, "reincore_is_subagent").unwrap_or(false);
        let track_subagent_sessions =
            extra_bool(request, "reincore_track_subagent_sessions").unwrap_or(true);
        let spawn_tool_call_id = extra_text(request, "reincore_subagent_spawn_tool_call_id");

        // A declared child must never degrade into an unrelated top-level
        // session. The parent request creates its binding before any tool call
        // can be streamed, so a missing parent indicates stale/bad metadata.
        if parent_conversation.is_some() && parent_session.is_none() {
            return Err(StatusCode::CONFLICT);
        }

        if is_subagent && !track_subagent_sessions {
            if let Some(parent) = parent_session.as_ref() {
                let id = db
                    .bind_conversation_to_session_async(
                        binding_key,
                        client_id,
                        scope_key,
                        conversation_hash,
                        parent.id,
                        true,
                        parent_binding_key,
                        spawn_tool_call_id,
                    )
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                return db
                    .get_active_session_in_lineage_async(id)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                    .ok_or(StatusCode::INTERNAL_SERVER_ERROR)
                    .map(Some);
            }
        }

        let budget_cents = request
            .extra
            .get("reincore_session_budget_cents")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);
        let mut proposed = Session::new_tracked(
            derive_task_summary(request),
            budget_cents,
            Some(project_name),
            Some(project_path),
            Some(client_id.clone()),
            Some(binding_key.clone()),
        )
        .map_err(|_| StatusCode::BAD_REQUEST)?;
        proposed.project_id = project_id;
        proposed.mode = request_session_mode(request);
        proposed.is_subagent = is_subagent;
        proposed.subagent_parent_session_id = parent_session.as_ref().map(|parent| {
            parent
                .external_session_id
                .clone()
                .unwrap_or_else(|| parent.id.to_string())
        });
        proposed.subagent_spawn_tool_call_id = spawn_tool_call_id.clone();
        proposed.subagent_spawn_request_group_id =
            extra_text(request, "reincore_subagent_spawn_request_group_id");

        let id = db
            .resolve_or_insert_conversation_session_async(
                binding_key,
                client_id,
                scope_key,
                conversation_hash,
                proposed,
                is_subagent,
                parent_binding_key,
                spawn_tool_call_id,
            )
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        if let Some(mut session) = db
            .get_active_session_in_lineage_async(id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        {
            rename_session_if_clearer(db, &mut session, request).await;
            return Ok(Some(session));
        }
        if let Some(session) = db
            .get_session_async(id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        {
            if matches!(
                session.status,
                SessionStatus::Checkpointed | SessionStatus::Paused
            ) {
                return Err(StatusCode::CONFLICT);
            }
        }
        return Err(StatusCode::CONFLICT);
    }

    // Backward-compatible identity for clients that have not adopted durable
    // conversation metadata yet.
    let external_id = extra_text(request, "reincore_external_session_id")
        .or_else(|| {
            engine_core::conversation_head_fingerprint(request)
                .map(|fingerprint| format!("auto:{}", &fingerprint[..fingerprint.len().min(64)]))
        })
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    if external_id.len() > 512 {
        return Err(StatusCode::BAD_REQUEST);
    }

    if let Some(mut session) = db
        .get_active_session_by_external_id_async(
            client_id.clone(),
            project_id,
            project_path.clone(),
            external_id.clone(),
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    {
        rename_session_if_clearer(db, &mut session, request).await;
        return Ok(Some(session));
    }

    // A checkpoint is an intentional pause. Do not silently detach the same
    // editor conversation into a new, unrelated session just because active-
    // only lookup did not find it; require explicit checkpoint redemption.
    if let Some(session) = db
        .get_latest_session_by_external_id_async(
            client_id.clone(),
            project_id,
            project_path.clone(),
            external_id.clone(),
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    {
        if matches!(
            session.status,
            SessionStatus::Checkpointed | SessionStatus::Paused
        ) {
            return Err(StatusCode::CONFLICT);
        }
    }

    let budget_cents = request
        .extra
        .get("reincore_session_budget_cents")
        .and_then(|value| value.as_i64())
        .unwrap_or(0);

    let mut session = Session::new_tracked(
        derive_task_summary(request),
        budget_cents,
        Some(project_name),
        Some(project_path),
        Some(client_id),
        Some(external_id),
    )
    .map_err(|_| StatusCode::BAD_REQUEST)?;
    session.project_id = project_id;
    session.mode = request_session_mode(request);

    // Propagate subagent identity from the request body so the
    // session row itself records whether this is a subagent and which
    // parent session it belongs to. The extension sets
    // `reincore_is_subagent` and `reincore_subagent_parent_session_id` in
    // the chat completion request body.
    session.subagent_parent_session_id = extra_text(request, "reincore_subagent_parent_session_id");
    session.is_subagent = extra_bool(request, "reincore_is_subagent").unwrap_or(false)
        || session.subagent_parent_session_id.is_some();
    session.subagent_spawn_tool_call_id =
        extra_text(request, "reincore_subagent_spawn_tool_call_id");
    session.subagent_spawn_request_group_id =
        extra_text(request, "reincore_subagent_spawn_request_group_id");

    db.insert_session_async(session.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Some(session))
}

/// Complete `previous` and open its linked successor.
///
/// The lineage link is kept even when the successor starts on an unrelated
/// task: it is what preserves budget carry-over, provenance, and resolution of
/// the session id the client is still sending. Only the *context* is optional.
pub(super) async fn open_successor_session(
    db: &crate::db::LocalDatabase,
    previous: Session,
    reason: &str,
    context_tokens_after: u64,
    fresh_task_summary: Option<String>,
) -> Result<Session, StatusCode> {
    db.rollover_session(
        previous,
        reason.to_owned(),
        context_tokens_after,
        fresh_task_summary,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// A successor session only inherits the previous session's name while it is
/// continuing the same work; a self-contained turn names itself.
pub(super) fn successor_task_summary(request: &ChatCompletionRequest) -> Option<String> {
    is_self_contained(request).then(|| derive_task_summary(request))
}

pub(super) async fn rollover_local_session_if_needed(
    db: &crate::db::LocalDatabase,
    session: Session,
    reset: &engine_core::ContextWindowReset,
    request: &ChatCompletionRequest,
) -> Result<(Session, Option<Uuid>), StatusCode> {
    if !reset.requires_session_rollover(session.context_tokens) {
        return Ok((session, None));
    }

    let mut previous = session;
    previous.context_tokens = reset.input_tokens_before;
    let previous_id = previous.id;

    let next = open_successor_session(
        db,
        previous,
        if reset.input_tokens_before > reset.limit_tokens {
            "context_window_full"
        } else {
            "context_pressure"
        },
        reset.input_tokens_after,
        successor_task_summary(request),
    )
    .await?;

    Ok((next, Some(previous_id)))
}

pub(super) struct OverflowRecovery {
    pub(super) handoff: SessionHandoff,
    pub(super) session: Option<Session>,
    pub(super) rolled_over_from: Option<Uuid>,
}

/// The provider rejected the request for exceeding its context window even
/// though our pre-flight check passed. Compact the payload down to a bounded
/// handoff and continue in a linked successor session so the retry has room.
pub(super) async fn recover_from_provider_overflow(
    db: &crate::db::LocalDatabase,
    session_id: Option<Uuid>,
    request: &mut ChatCompletionRequest,
    context_limit: u64,
) -> Result<OverflowRecovery, StatusCode> {
    let handoff = engine_core::continuation::compact_provider_overflow(request, context_limit);
    let previous = match session_id {
        Some(id) => db
            .get_session_async(id)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        None => None,
    };
    let Some(previous) = previous else {
        return Ok(OverflowRecovery {
            handoff,
            session: None,
            rolled_over_from: None,
        });
    };

    let previous_id = previous.id;
    let fresh_task_summary = (!handoff.carried_context).then(|| derive_task_summary(request));
    let next = open_successor_session(
        db,
        previous,
        "provider_context_window_full",
        0,
        fresh_task_summary,
    )
    .await?;

    Ok(OverflowRecovery {
        handoff,
        session: Some(next),
        rolled_over_from: Some(previous_id),
    })
}

// Helper to serialize session status consistently.
#[allow(dead_code)]
pub(super) trait SessionStatusExt {
    fn as_str(&self) -> &'static str;
}

#[allow(dead_code)]
impl SessionStatusExt for SessionStatus {
    fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Paused => "paused",
            SessionStatus::Completed => "completed",
            SessionStatus::Checkpointed => "checkpointed",
        }
    }
}
