//! In-memory implementations of the storage traits for testing and local-only
//! quick-start scenarios. These are not suitable for production persistence.

use crate::budget::{
    BudgetHistoryEntry, BudgetPolicy, BudgetScope, BudgetStore, LedgerEvent, LedgerReason,
    LedgerStore, SettingsStore,
};
use crate::cache::{
    CacheEntry, CacheKey, MemoizationCache, SemanticCacheCandidate, SemanticCacheEntry,
    SemanticCacheHit, SemanticCacheLookup, SemanticCacheProvenance, SemanticCacheQuery,
    SemanticCacheScope, ToolDefinitionForCache,
};
use crate::compression::CompressionSettings;
use crate::error::{EngineError, EngineResult};
use crate::event::{EventStore, SessionEvent, SessionEventType, SummaryPayload};
use crate::models::{ChatMessage, FileContextAttachment, ToolCall, ToolResult};
use crate::store::{
    CacheStore, IndexedToolState, TeamScopedCacheStore, TeamSemanticToolCacheEntry,
    TeamSemanticToolCacheHit, ToolStateStore,
};
use async_trait::async_trait;
use chrono::{Datelike, NaiveDate, Utc};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use uuid::Uuid;

/// Lock a mutex, recovering from poisoning instead of panicking.
///
/// These stores are test/local-only doubles, often shared across many
/// concurrent test tasks. If any one critical section ever panics while
/// holding the lock, plain `.lock().unwrap()` would poison the mutex and
/// make every *other* unrelated call on the same instance panic too,
/// turning one failure into a confusing cascade. Recovering the inner
/// value is the right tradeoff here: these are in-memory test doubles with
/// no durability guarantees to protect, so continuing with
/// possibly-partially-mutated state is safer than losing the whole store.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryEventStore {
    events: Arc<Mutex<HashMap<Uuid, Vec<SessionEvent>>>>,
    file_history: Arc<Mutex<HashMap<(Uuid, String, String), crate::file_history::FileWriteRecord>>>,
    file_call_identities: Arc<Mutex<HashMap<(Uuid, String), (String, bool)>>>,
    file_roots: Arc<Mutex<HashMap<Uuid, crate::file_history::FileHistoryScope>>>,
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn bind_file_history_scope(
        &self,
        session_id: Uuid,
        scope: &crate::file_history::FileHistoryScope,
    ) -> EngineResult<()> {
        let mut roots = lock(&self.file_roots);
        let old =
            roots
                .get(&session_id)
                .cloned()
                .unwrap_or(crate::file_history::FileHistoryScope {
                    project_id: scope.project_id,
                    roots: vec![],
                });
        let merged = old.merge(scope)?;
        if merged == old {
            return Ok(());
        }
        for record in lock(&self.file_history)
            .values_mut()
            .filter(|record| record.session_id == session_id && record.file_id.is_none())
        {
            record.file_id = merged.identity(&record.path);
        }
        roots.insert(session_id, merged);
        Ok(())
    }

    async fn read_file_history_scope(
        &self,
        session_id: Uuid,
    ) -> EngineResult<Option<crate::file_history::FileHistoryScope>> {
        Ok(lock(&self.file_roots).get(&session_id).cloned())
    }

    async fn observe_file_history_call_identities(
        &self,
        session_id: Uuid,
        calls: &[crate::models::ToolCall],
        results: &[crate::models::ToolResult],
    ) -> EngineResult<()> {
        let mut history = lock(&self.file_history);
        let mut identities = lock(&self.file_call_identities);
        for call in calls.iter().take(512) {
            if let Some((hash, ambiguous)) = identities.get_mut(&(session_id, call.id.clone())) {
                if hash != &crate::file_history::call_identity(call) {
                    *ambiguous = true;
                }
            }
        }
        for result in results.iter().take(512) {
            let hash = crate::file_history::result_identity(result);
            if history.values().any(|r| {
                r.session_id == session_id
                    && r.tool_call_id == result.tool_call_id
                    && r.result_hash.as_ref().is_some_and(|old| old != &hash)
            }) {
                if let Some((_, ambiguous)) =
                    identities.get_mut(&(session_id, result.tool_call_id.clone()))
                {
                    *ambiguous = true;
                }
            }
        }
        for record in history
            .values_mut()
            .filter(|record| record.session_id == session_id)
        {
            if identities
                .get(&(session_id, record.tool_call_id.clone()))
                .is_some_and(|(_, ambiguous)| *ambiguous)
            {
                crate::file_history::invalidate(record);
            }
        }
        Ok(())
    }

    async fn read_latest_continuation(
        &self,
        session_id: Uuid,
        task_id: &str,
    ) -> EngineResult<Option<SessionEvent>> {
        Ok(lock(&self.events).get(&session_id).and_then(|events| {
            events
                .iter()
                .rev()
                .find(|event| {
                    event.event_type == SessionEventType::Continuation
                        && (task_id.is_empty()
                            || event.payload["task_id"].as_str() == Some(task_id))
                })
                .cloned()
        }))
    }

    async fn compare_and_append_continuation(
        &self,
        session_id: Uuid,
        task_id: &str,
        expected_seq: Option<i64>,
        payload: serde_json::Value,
    ) -> EngineResult<Option<SessionEvent>> {
        let mut all = lock(&self.events);
        let events = all.entry(session_id).or_default();
        let previous = events.iter().rev().find(|event| {
            event.event_type == SessionEventType::Continuation
                && event.payload["task_id"].as_str() == Some(task_id)
        });
        if previous.map(|event| event.seq) != expected_seq {
            return Ok(None);
        }
        if let Some(event) =
            previous.filter(|event| crate::continuation::same_snapshot(&event.payload, &payload))
        {
            return Ok(Some(event.clone()));
        }
        let event = SessionEvent {
            session_id,
            seq: events.last().map_or(1, |event| event.seq + 1),
            event_type: SessionEventType::Continuation,
            payload,
            cost_cents: None,
            created_at: Utc::now(),
        };
        events.push(event.clone());
        Ok(Some(event))
    }

    async fn append_event(
        &self,
        session_id: Uuid,
        event_type: SessionEventType,
        mut payload: serde_json::Value,
        cost_cents: Option<i64>,
    ) -> EngineResult<SessionEvent> {
        // Match the durable store: payloads are privacy-sanitized before
        // they (and the file-history records derived from them) persist.
        crate::privacy::sanitize_event_payload(event_type, &mut payload);
        let mut events = lock(&self.events);
        let seq = events
            .get(&session_id)
            .map(|v| v.last().map(|e| e.seq).unwrap_or(0) + 1)
            .unwrap_or(1);
        let event = SessionEvent {
            session_id,
            seq,
            event_type,
            payload,
            cost_cents,
            created_at: Utc::now(),
        };
        {
            let roots = lock(&self.file_roots);
            let scope = roots.get(&session_id);
            let mut history = lock(&self.file_history);
            let mut identities = lock(&self.file_call_identities);
            for link in crate::event::tool_call_links_from_event(&event) {
                let identity = crate::file_history::call_identity(&link.tool_call);
                let previous = identities
                    .entry((session_id, link.tool_call.id.clone()))
                    .or_insert((identity.clone(), false));
                previous.1 |= previous.0 != identity;
                let ambiguous = previous.1;
                for mut record in
                    crate::file_history::records_for_call(session_id, seq, &link.tool_call)
                {
                    record.file_id = scope.and_then(|scope| scope.identity(&record.path));
                    let key = (session_id, record.tool_call_id.clone(), record.path.clone());
                    if let Some(old) = history.get_mut(&key) {
                        crate::file_history::merge_call(old, &record, ambiguous);
                    } else {
                        if ambiguous {
                            crate::file_history::invalidate(&mut record);
                        }
                        history.insert(key, record);
                    }
                }
            }
            for record in history
                .values_mut()
                .filter(|record| record.session_id == session_id)
            {
                if identities
                    .get(&(session_id, record.tool_call_id.clone()))
                    .is_some_and(|(_, ambiguous)| *ambiguous)
                {
                    crate::file_history::invalidate(record);
                }
                crate::file_history::apply_result(record, &event);
            }
        }
        events.entry(session_id).or_default().push(event.clone());
        Ok(event)
    }

    async fn read_file_history(
        &self,
        session_id: Uuid,
        paths: &[String],
    ) -> EngineResult<Vec<crate::file_history::FileWriteRecord>> {
        let scope = self.read_file_history_scope(session_id).await?;
        let history = lock(&self.file_history);
        let paths = crate::file_history::lookup_paths(paths, scope.as_ref());
        let mut result = Vec::new();
        for path in paths {
            let mut entries: Vec<_> = history
                .values()
                .filter(|r| r.session_id == session_id)
                .filter_map(|r| {
                    crate::file_history::rebase(r, std::slice::from_ref(&path), scope.as_ref())
                })
                .collect();
            entries.sort_by(|a, b| {
                (b.result_seq.unwrap_or(b.call_seq), &b.tool_call_id)
                    .cmp(&(a.result_seq.unwrap_or(a.call_seq), &a.tool_call_id))
            });
            entries.truncate(crate::file_history::PER_FILE_LIMIT);
            result.extend(entries);
        }
        Ok(result)
    }

    async fn read_events(
        &self,
        session_id: Uuid,
        since_seq: Option<i64>,
        limit: Option<usize>,
    ) -> EngineResult<Vec<SessionEvent>> {
        let events = lock(&self.events);
        // `limit`, when given, keeps the first `limit` events *after*
        // filtering by `since_seq` -- i.e. this paginates forward from the
        // oldest matching event, not backward from the newest. A caller
        // wanting "the most recent N events" needs to compute an
        // appropriate `since_seq` cursor (e.g. via `latest_seq`) and/or
        // reverse the result themselves; this method does not do that.
        let mut result: Vec<SessionEvent> = events
            .get(&session_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|e| since_seq.map(|s| e.seq >= s).unwrap_or(true))
            .collect();
        if let Some(l) = limit {
            result.truncate(l);
        }
        Ok(result)
    }

    async fn latest_seq(&self, session_id: Uuid) -> EngineResult<Option<i64>> {
        let events = lock(&self.events);
        Ok(events
            .get(&session_id)
            .and_then(|v| v.last().map(|e| e.seq)))
    }

    async fn read_latest_summary(&self, session_id: Uuid) -> EngineResult<Option<SummaryPayload>> {
        let events = lock(&self.events);
        let latest = events
            .get(&session_id)
            .and_then(|v| {
                v.iter()
                    .rev()
                    .find(|e| e.event_type == SessionEventType::Summary)
            })
            .and_then(|e| serde_json::from_value::<SummaryPayload>(e.payload.clone()).ok());
        Ok(latest)
    }

    async fn read_latest_files(
        &self,
        session_id: Uuid,
    ) -> EngineResult<Vec<FileContextAttachment>> {
        let events = lock(&self.events);
        let mut latest_by_path: HashMap<String, FileContextAttachment> = HashMap::new();
        for event in events.get(&session_id).iter().flat_map(|v| v.iter()) {
            if event.event_type != SessionEventType::ToolResult {
                continue;
            }
            if let Some(file) = event
                .payload
                .get("file")
                .and_then(|v| serde_json::from_value::<FileContextAttachment>(v.clone()).ok())
            {
                match latest_by_path.get(&file.path) {
                    Some(existing) if existing.version >= file.version => {}
                    _ => {
                        latest_by_path.insert(file.path.clone(), file);
                    }
                }
            }
        }
        let mut files: Vec<FileContextAttachment> = latest_by_path.into_values().collect();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(files)
    }

    async fn redeem_checkpoint(
        &self,
        session_id: Uuid,
        token_fingerprint: &str,
        payload: serde_json::Value,
    ) -> EngineResult<SessionEvent> {
        let mut events = lock(&self.events);
        let session_events = events.entry(session_id).or_default();
        let already_redeemed = session_events.iter().any(|event| {
            event.event_type == SessionEventType::CheckpointRedeemed
                && event
                    .payload
                    .get("token_fingerprint")
                    .and_then(serde_json::Value::as_str)
                    == Some(token_fingerprint)
        });
        if already_redeemed {
            return Err(EngineError::CheckpointAlreadyRedeemed);
        }

        let event = SessionEvent {
            session_id,
            seq: session_events
                .last()
                .map(|event| event.seq + 1)
                .unwrap_or(1),
            event_type: SessionEventType::CheckpointRedeemed,
            payload,
            cost_cents: None,
            created_at: Utc::now(),
        };
        session_events.push(event.clone());
        Ok(event)
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryLedgerStore {
    events: Arc<Mutex<Vec<LedgerEvent>>>,
}

#[async_trait]
impl LedgerStore for InMemoryLedgerStore {
    async fn record_spend(
        &self,
        session_id: Uuid,
        amount_cents: i64,
        reason: LedgerReason,
    ) -> EngineResult<LedgerEvent> {
        let event = LedgerEvent {
            session_id,
            amount_cents,
            reason,
            created_at: Utc::now(),
        };
        lock(&self.events).push(event.clone());
        Ok(event)
    }

    async fn total_spend(&self, session_id: Uuid) -> EngineResult<i64> {
        let total = lock(&self.events)
            .iter()
            .filter(|e| e.session_id == session_id)
            .map(|e| e.amount_cents)
            .sum();
        Ok(total)
    }

    async fn total_spend_day(&self, date: NaiveDate) -> EngineResult<i64> {
        let total = lock(&self.events)
            .iter()
            .filter(|e| e.created_at.date_naive() == date)
            .map(|e| e.amount_cents)
            .sum();
        Ok(total)
    }

    async fn total_spend_month(&self, year: i32, month: u32) -> EngineResult<i64> {
        let total = lock(&self.events)
            .iter()
            .filter(|e| {
                let d = e.created_at.date_naive();
                d.year() == year && d.month() == month
            })
            .map(|e| e.amount_cents)
            .sum();
        Ok(total)
    }

    async fn total_spend_all(&self) -> EngineResult<i64> {
        let total = lock(&self.events).iter().map(|e| e.amount_cents).sum();
        Ok(total)
    }

    async fn total_spend_project(&self, _project: &str) -> EngineResult<i64> {
        // In-memory ledger does not store project names; return 0.
        //
        // Worth flagging: a silent 0 reads as a real, verified answer
        // ("this project spent $0.00") rather than "project tracking isn't
        // implemented," which could mislead a caller building a UI on top
        // of this. Fixing that properly means adding a `project` field to
        // `LedgerEvent` itself (in `crate::budget`), which is out of scope
        // for this file -- flagging here since it's the actual root cause.
        Ok(0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryBudgetStore {
    policies: Arc<Mutex<HashMap<(BudgetScope, String), BudgetPolicy>>>,
    history: Arc<Mutex<Vec<BudgetHistoryEntry>>>,
}

#[async_trait]
impl BudgetStore for InMemoryBudgetStore {
    async fn set_budget(
        &self,
        scope: BudgetScope,
        scope_key: String,
        budget_cents: i64,
    ) -> EngineResult<(BudgetPolicy, BudgetHistoryEntry)> {
        let mut policies = lock(&self.policies);
        let previous_budget_cents = policies
            .get(&(scope, scope_key.clone()))
            .map(|p| p.budget_cents)
            .unwrap_or(0);

        let now = Utc::now();
        let policy = BudgetPolicy {
            id: Uuid::new_v4(),
            scope,
            scope_key: scope_key.clone(),
            budget_cents,
            created_at: now,
            updated_at: now,
        };
        policies.insert((scope, scope_key.clone()), policy.clone());

        let history_entry = BudgetHistoryEntry {
            policy_id: policy.id,
            scope,
            scope_key: scope_key.clone(),
            previous_budget_cents,
            new_budget_cents: budget_cents,
            updated_at: now,
        };
        lock(&self.history).push(history_entry.clone());

        Ok((policy, history_entry))
    }

    async fn get_budget(
        &self,
        scope: BudgetScope,
        scope_key: &str,
    ) -> EngineResult<Option<BudgetPolicy>> {
        Ok(lock(&self.policies)
            .get(&(scope, scope_key.to_string()))
            .cloned())
    }

    async fn list_budget_history(
        &self,
        scope: BudgetScope,
        scope_key: Option<&str>,
        limit: Option<usize>,
    ) -> EngineResult<Vec<BudgetHistoryEntry>> {
        let mut result: Vec<BudgetHistoryEntry> = lock(&self.history)
            .iter()
            .filter(|h| h.scope == scope)
            .filter(|h| scope_key.map(|k| h.scope_key == k).unwrap_or(true))
            .cloned()
            .collect();
        result.sort_by_key(|item| std::cmp::Reverse(item.updated_at));
        if let Some(l) = limit {
            result.truncate(l);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemorySettingsStore {
    settings: Arc<Mutex<CompressionSettings>>,
    loop_settings: Arc<Mutex<crate::circuit::LoopSettings>>,
}

#[async_trait]
impl SettingsStore for InMemorySettingsStore {
    async fn get_compression_settings(&self) -> EngineResult<CompressionSettings> {
        Ok(*lock(&self.settings))
    }

    async fn set_compression_settings(&self, settings: CompressionSettings) -> EngineResult<()> {
        *lock(&self.settings) = settings;
        Ok(())
    }

    async fn get_loop_settings(&self) -> EngineResult<crate::circuit::LoopSettings> {
        Ok(lock(&self.loop_settings).clone())
    }

    async fn set_loop_settings(&self, settings: crate::circuit::LoopSettings) -> EngineResult<()> {
        *lock(&self.loop_settings) = settings;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryCacheStore {
    memoized: Arc<Mutex<HashMap<String, CacheEntry>>>,
    semantic: Arc<Mutex<Vec<SemanticCacheEntry>>>,
}

#[async_trait]
impl MemoizationCache for InMemoryCacheStore {
    async fn lookup(
        &self,
        _tool: &ToolDefinitionForCache,
        call: &ToolCall,
    ) -> EngineResult<Option<ToolResult>> {
        let args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
        let key = CacheKey::for_tool_call(&call.function.name, &args)?;
        let entry = self.get_memoized(&key).await?;
        Ok(entry.and_then(|e| serde_json::from_value::<ToolResult>(e.result_payload).ok()))
    }

    async fn store(
        &self,
        tool: &ToolDefinitionForCache,
        call: &ToolCall,
        result: &ToolResult,
    ) -> EngineResult<()> {
        if !tool.safe_to_memoize {
            return Ok(());
        }
        let args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
        let key = CacheKey::for_tool_call(&call.function.name, &args)?;
        let entry = CacheEntry::from_tool_result(&tool.name, result)?;
        self.put_memoized(key, entry).await
    }

    async fn invalidate_for_file_update(&self, file: &FileContextAttachment) -> EngineResult<u64> {
        let path = &file.path;
        let mut memoized = lock(&self.memoized);
        let before = memoized.len();
        memoized.retain(|_k, v| {
            // Search the *entire serialized* result payload for the path,
            // not just the case where the payload happens to be a bare
            // JSON string. Most tool results are JSON objects/arrays (e.g.
            // {"content": "...", "path": "..."}), so the previous
            // `.as_str()`-only check returned `None` for nearly every real
            // entry, making this invalidation silently do nothing in the
            // common case. This is a best-effort text search (not a
            // structural JSON walk), so it can still theoretically miss a
            // path that's been escaped/transformed, but it now actually
            // matches the overwhelming majority of real cache entries.
            !v.result_payload.to_string().contains(path.as_str())
        });
        Ok((before - memoized.len()) as u64)
    }
}

#[async_trait]
impl CacheStore for InMemoryCacheStore {
    async fn get_memoized(&self, key: &CacheKey) -> EngineResult<Option<CacheEntry>> {
        Ok(lock(&self.memoized).get(&key.0).cloned())
    }

    async fn put_memoized(&self, key: CacheKey, entry: CacheEntry) -> EngineResult<()> {
        lock(&self.memoized).insert(key.0, entry);
        Ok(())
    }

    async fn get_semantic(&self, query: &SemanticCacheQuery) -> EngineResult<SemanticCacheLookup> {
        // Linear scan over every stored embedding. Fine for a test/local
        // double, but not something to scale up -- a real semantic cache
        // needs an actual vector index.
        let mut store = lock(&self.semantic);
        store.retain(|entry| entry.expires_at > query.now);
        let mut matches = Vec::new();
        for (index, entry) in store.iter().enumerate() {
            if entry.scope != query.scope
                || entry.fingerprint.digest()? != query.fingerprint_hash
                || !entry.policy.safe_to_reuse
            {
                continue;
            }
            let similarity = cosine_similarity(&entry.embedding, &query.embedding);
            if similarity >= query.candidate_threshold {
                let provenance = SemanticCacheProvenance {
                    entry_id: index.to_string(),
                    prompt_hash: hex::encode(Sha256::digest(entry.prompt.as_bytes())),
                    scope: entry.scope.clone(),
                    fingerprint_hash: entry.fingerprint.digest()?,
                    similarity,
                    created_at: entry.created_at,
                    expires_at: entry.expires_at,
                    provider: entry.fingerprint.provider.clone(),
                    resolved_model: entry.fingerprint.resolved_model.clone(),
                    source_data_version: entry.fingerprint.source_data_version.clone(),
                    immutable_workflow: entry.fingerprint.immutable_workflow.clone(),
                };
                matches.push((similarity, entry.response.clone(), provenance));
            }
        }
        matches.sort_by(|left, right| right.0.total_cmp(&left.0));
        let reusable_answer = matches
            .iter()
            .find(|(similarity, _, _)| *similarity >= query.answer_threshold)
            .map(|(_, response, provenance)| SemanticCacheHit {
                response: response.clone(),
                provenance: provenance.clone(),
            });
        let candidates = matches
            .into_iter()
            .filter(|(similarity, _, _)| *similarity < query.answer_threshold)
            .take(query.candidate_limit)
            .map(|(_, response, provenance)| SemanticCacheCandidate {
                response,
                provenance,
            })
            .collect();
        Ok(SemanticCacheLookup {
            reusable_answer,
            candidates,
        })
    }

    async fn put_semantic(&self, entry: SemanticCacheEntry) -> EngineResult<()> {
        if entry.policy.safe_to_reuse && entry.expires_at > entry.created_at {
            lock(&self.semantic).push(entry);
        }
        Ok(())
    }

    async fn invalidate_semantic(
        &self,
        scope: &SemanticCacheScope,
        fingerprint_hash: Option<&str>,
    ) -> EngineResult<u64> {
        let mut semantic = lock(&self.semantic);
        let before = semantic.len();
        semantic.retain(|entry| {
            let matches_scope = &entry.scope == scope;
            let matches_fingerprint = fingerprint_hash
                .map(|hash| {
                    entry
                        .fingerprint
                        .digest()
                        .map(|digest| digest == hash)
                        .unwrap_or(false)
                })
                .unwrap_or(true);
            !(matches_scope && matches_fingerprint)
        });
        Ok((before - semantic.len()) as u64)
    }

    async fn invalidate(
        &self,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64> {
        let mut memoized = lock(&self.memoized);
        let before = memoized.len();
        memoized.retain(|k, v| {
            // Only invalidate entries that match *all* filters that were
            // actually supplied (intersection), not entries matching *any*
            // one of them. The previous version ANDed the two "keep"
            // conditions, which is equivalent to ORing the two "invalidate"
            // conditions -- so `invalidate(Some("read_file"), Some("main.rs"))`
            // would drop every `read_file` entry regardless of pattern,
            // *plus* every entry of any tool whose key happened to contain
            // "main.rs". That's much broader than what a caller supplying
            // both filters together would reasonably expect. Single-filter
            // calls behave exactly as before.
            let tool_matches = tool_name.map(|n| v.tool_name == n).unwrap_or(true);
            let pattern_matches = key_pattern
                .map(|p| k.contains(p) || v.result_payload.to_string().contains(p))
                .unwrap_or(true);
            !(tool_matches && pattern_matches)
        });
        Ok((before - memoized.len()) as u64)
    }
}

/// In-memory team-scoped cache store for testing the cloud
/// [`TeamScopedCacheStore`] contract without a Postgres dependency.
///
/// Partitions entries by `(workspace_id, cache_key)`. `None` workspace_id
/// is its own partition - it never collides with a real workspace, matching
/// the Postgres `IS NOT DISTINCT FROM` semantics.
#[derive(Clone, Debug, Default)]
pub struct InMemoryTeamScopedCacheStore {
    // (workspace_id, cache_key) -> entry
    scoped: Arc<Mutex<ScopedCacheEntries>>,
    semantic: Arc<Mutex<Vec<TeamSemanticToolCacheEntry>>>,
}

type ScopedCacheEntries = HashMap<(Option<Uuid>, String), CacheEntry>;

#[async_trait]
impl CacheStore for InMemoryTeamScopedCacheStore {
    async fn get_memoized(&self, key: &CacheKey) -> EngineResult<Option<CacheEntry>> {
        Ok(lock(&self.scoped).get(&(None, key.0.clone())).cloned())
    }

    async fn put_memoized(&self, key: CacheKey, entry: CacheEntry) -> EngineResult<()> {
        lock(&self.scoped).insert((None, key.0), entry);
        Ok(())
    }

    async fn get_semantic(&self, _query: &SemanticCacheQuery) -> EngineResult<SemanticCacheLookup> {
        Ok(SemanticCacheLookup::default())
    }

    async fn put_semantic(&self, _entry: SemanticCacheEntry) -> EngineResult<()> {
        Ok(())
    }

    async fn invalidate_semantic(
        &self,
        _scope: &SemanticCacheScope,
        _fingerprint_hash: Option<&str>,
    ) -> EngineResult<u64> {
        Ok(0)
    }

    async fn invalidate(
        &self,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64> {
        let mut scoped = lock(&self.scoped);
        let before = scoped.len();
        scoped.retain(|(_, _k), v| {
            let tool_matches = tool_name.map(|n| v.tool_name == n).unwrap_or(true);
            let pattern_matches = key_pattern
                .map(|p| v.result_payload.to_string().contains(p))
                .unwrap_or(true);
            !(tool_matches && pattern_matches)
        });
        Ok((before - scoped.len()) as u64)
    }
}

#[async_trait]
impl TeamScopedCacheStore for InMemoryTeamScopedCacheStore {
    async fn get_memoized_scoped(
        &self,
        key: &CacheKey,
        workspace_id: Option<Uuid>,
    ) -> EngineResult<Option<CacheEntry>> {
        Ok(lock(&self.scoped)
            .get(&(workspace_id, key.0.clone()))
            .cloned())
    }

    async fn put_memoized_scoped(
        &self,
        key: CacheKey,
        entry: CacheEntry,
        workspace_id: Option<Uuid>,
    ) -> EngineResult<()> {
        lock(&self.scoped).insert((workspace_id, key.0), entry);
        Ok(())
    }

    async fn get_semantic_tool_scoped(
        &self,
        workspace_id: Uuid,
        request_group_id: &str,
        scope_fingerprint: &str,
        current_session_id: Uuid,
        tool_name: &str,
        query_embedding: &[f32],
        threshold: f32,
    ) -> EngineResult<Option<TeamSemanticToolCacheHit>> {
        if scope_fingerprint.trim().is_empty()
            || query_embedding.len() != crate::EMBEDDING_DIMENSIONS
            || query_embedding.iter().any(|v| !v.is_finite())
            || !threshold.is_finite()
        {
            return Err(crate::EngineError::InvalidInput(
                "invalid scoped tool-cache query".into(),
            ));
        }
        let mut matches = lock(&self.semantic)
            .iter()
            .filter(|entry| {
                entry.workspace_id == workspace_id
                    && entry.request_group_id == request_group_id
                    && entry.scope_fingerprint == scope_fingerprint
                    && entry.source_session_id != current_session_id
                    && entry.cache_entry.tool_name == tool_name
                    && entry.cache_entry.created_at
                        >= Utc::now()
                            - chrono::Duration::seconds(
                                crate::cache::DEFAULT_MEMOIZATION_TTL_SECONDS,
                            )
            })
            .filter_map(|entry| {
                let similarity = cosine_similarity(&entry.request_embedding, query_embedding);
                (similarity >= threshold).then_some((entry, similarity))
            })
            .map(|(entry, similarity)| TeamSemanticToolCacheHit {
                cache_entry: entry.cache_entry.clone(),
                source_session_id: entry.source_session_id,
                similarity,
            })
            .collect::<Vec<_>>();
        matches.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        Ok(matches.into_iter().next())
    }

    async fn put_semantic_tool_scoped(
        &self,
        entry: TeamSemanticToolCacheEntry,
    ) -> EngineResult<()> {
        if entry.scope_fingerprint.trim().is_empty()
            || entry.request_embedding.len() != crate::EMBEDDING_DIMENSIONS
            || entry.request_embedding.iter().any(|v| !v.is_finite())
        {
            return Err(crate::EngineError::InvalidInput(
                "invalid scoped tool-cache entry".into(),
            ));
        }
        let mut semantic = lock(&self.semantic);
        if let Some(existing) = semantic.iter_mut().find(|existing| {
            existing.workspace_id == entry.workspace_id
                && existing.request_group_id == entry.request_group_id
                && existing.scope_fingerprint == entry.scope_fingerprint
                && existing.source_session_id == entry.source_session_id
                && existing.cache_key == entry.cache_key
        }) {
            *existing = entry;
        } else {
            semantic.push(entry);
        }
        Ok(())
    }

    async fn invalidate_scoped(
        &self,
        workspace_id: Option<Uuid>,
        tool_name: Option<&str>,
        key_pattern: Option<&str>,
    ) -> EngineResult<u64> {
        let mut scoped = lock(&self.scoped);
        let before = scoped.len();
        scoped.retain(|(ws, _k), v| {
            if *ws != workspace_id {
                return true; // different partition, keep
            }
            let tool_matches = tool_name.map(|n| v.tool_name == n).unwrap_or(true);
            let pattern_matches = key_pattern
                .map(|p| v.result_payload.to_string().contains(p))
                .unwrap_or(true);
            !(tool_matches && pattern_matches)
        });
        let exact_removed = before - scoped.len();
        drop(scoped);

        let mut semantic = lock(&self.semantic);
        let semantic_before = semantic.len();
        semantic.retain(|entry| {
            if Some(entry.workspace_id) != workspace_id {
                return true;
            }
            let tool_matches = tool_name
                .map(|name| entry.cache_entry.tool_name == name)
                .unwrap_or(true);
            let pattern_matches = key_pattern
                .map(|pattern| entry.cache_key.0.contains(pattern))
                .unwrap_or(true);
            !(tool_matches && pattern_matches)
        });
        Ok((exact_removed + semantic_before - semantic.len()) as u64)
    }

    async fn invalidate_for_file_update_scoped(
        &self,
        workspace_id: Option<Uuid>,
        file: &FileContextAttachment,
    ) -> EngineResult<u64> {
        let path = &file.path;
        let mut scoped = lock(&self.scoped);
        let before = scoped.len();
        scoped.retain(|(ws, _k), v| {
            if *ws != workspace_id {
                return true; // different partition, keep
            }
            // Keep entries that do NOT reference the updated file path.
            !v.result_payload.to_string().contains(path.as_str())
        });
        let exact_removed = before - scoped.len();
        drop(scoped);

        let mut semantic = lock(&self.semantic);
        let semantic_before = semantic.len();
        semantic.retain(|entry| {
            Some(entry.workspace_id) != workspace_id
                || !entry
                    .cache_entry
                    .result_payload
                    .to_string()
                    .contains(path.as_str())
        });
        Ok((exact_removed + semantic_before - semantic.len()) as u64)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// In-memory reference implementation of the RAG tool-state tier.
///
/// Indexes `ToolResult` events with the embedding of the turn that produced
/// them and retrieves the top-k by cosine similarity. This is the local
/// double of the `sqlite-vec`/`pgvector`-backed stores used in production;
/// it is a linear scan and not suitable for scale, but it exercises the
/// full [`ToolStateStore`] contract for tests and the local-only edition.
#[derive(Clone, Debug, Default)]
pub struct InMemoryToolStateStore {
    entries: Arc<Mutex<Vec<IndexedToolState>>>,
}

#[async_trait]
impl ToolStateStore for InMemoryToolStateStore {
    async fn index_tool_result(&self, entry: IndexedToolState) -> EngineResult<()> {
        let mut entries = lock(&self.entries);
        // Upsert by (session_id, seq) so repeated indexing of the same
        // event during assembly is idempotent.
        if let Some(existing) = entries
            .iter_mut()
            .find(|e| e.session_id == entry.session_id && e.seq == entry.seq)
        {
            *existing = entry;
        } else {
            entries.push(entry);
        }
        Ok(())
    }

    async fn retrieve_tool_states(
        &self,
        session_id: uuid::Uuid,
        query_embedding: &[f32],
        k: usize,
        min_similarity: f32,
    ) -> EngineResult<Vec<ChatMessage>> {
        let entries = lock(&self.entries);
        let mut scored: Vec<(f32, &IndexedToolState)> = entries
            .iter()
            .filter(|entry| entry.session_id == session_id)
            .filter_map(|e| {
                let sim = cosine_similarity(&e.embedding, query_embedding);
                if sim >= min_similarity {
                    Some((sim, e))
                } else {
                    None
                }
            })
            .collect();
        // Sort by descending similarity; ties broken by ascending seq so
        // earlier events win, keeping retrieval order deterministic across
        // runs (consistent with the attached-files ordering contract).
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.seq.cmp(&b.1.seq))
        });
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(_, e)| e.message.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // EventStore Tests

    #[tokio::test]
    async fn append_event_assigns_increasing_seq_per_session() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        let e1 = store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        let e2 = store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(e1.seq, 1);
        assert_eq!(e2.seq, 2);
    }

    #[tokio::test]
    async fn read_events_filters_by_since_seq_and_paginates_forward() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        for _ in 0..5 {
            store
                .append_event(
                    session_id,
                    SessionEventType::LlmCall,
                    serde_json::json!({}),
                    None,
                )
                .await
                .unwrap();
        }
        // since_seq=3 should return seq 3, 4, 5.
        let since = store.read_events(session_id, Some(3), None).await.unwrap();
        assert_eq!(
            since.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![3, 4, 5]
        );

        // limit=2 after since_seq=3 keeps the *first* two matches (3, 4),
        // i.e. forward pagination from the oldest match, not the newest.
        let limited = store
            .read_events(session_id, Some(3), Some(2))
            .await
            .unwrap();
        assert_eq!(
            limited.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![3, 4]
        );
    }

    #[tokio::test]
    async fn latest_seq_reflects_most_recent_append() {
        let store = InMemoryEventStore::default();
        let session_id = Uuid::new_v4();
        assert_eq!(store.latest_seq(session_id).await.unwrap(), None);
        store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        store
            .append_event(
                session_id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(store.latest_seq(session_id).await.unwrap(), Some(2));
    }

    // LedgerStore Tests

    #[tokio::test]
    async fn ledger_totals_are_scoped_to_session() {
        let store = InMemoryLedgerStore::default();
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        store
            .record_spend(session_a, 100, LedgerReason::Completion)
            .await
            .unwrap();
        store
            .record_spend(session_a, 50, LedgerReason::Completion)
            .await
            .unwrap();
        store
            .record_spend(session_b, 999, LedgerReason::Completion)
            .await
            .unwrap();

        assert_eq!(store.total_spend(session_a).await.unwrap(), 150);
        assert_eq!(store.total_spend(session_b).await.unwrap(), 999);
    }

    #[tokio::test]
    async fn ledger_total_spend_project_is_documented_as_unsupported() {
        // Regression/documentation test: this in-memory store cannot track
        // per-project spend since `LedgerEvent` has no project field, so it
        // always returns 0 rather than erroring. If `LedgerEvent` ever
        // gains a project field, this test should be updated to assert
        // real tracking instead.
        let store = InMemoryLedgerStore::default();
        let session_id = Uuid::new_v4();
        store
            .record_spend(session_id, 500, LedgerReason::Completion)
            .await
            .unwrap();
        assert_eq!(store.total_spend_project("any-project").await.unwrap(), 0);
    }

    // SettingsStore Tests

    #[tokio::test]
    async fn settings_store_roundtrips_and_defaults_sensibly() {
        let store = InMemorySettingsStore::default();
        let initial = store.get_compression_settings().await.unwrap();
        assert_eq!(initial, CompressionSettings::default());

        let custom = CompressionSettings::disabled();
        store.set_compression_settings(custom).await.unwrap();
        assert_eq!(store.get_compression_settings().await.unwrap(), custom);
    }

    // CacheStore Tests (primitives underlying MemoizationCache)

    fn sample_result(content: &str) -> ToolResult {
        ToolResult {
            tool_call_id: "call_1".to_string(),
            role: "tool".to_string(),
            name: "read_file".to_string(),
            content: content.to_string(),
            is_error: None,
        }
    }

    #[tokio::test]
    async fn memoized_get_put_roundtrip() {
        let store = InMemoryCacheStore::default();
        let key =
            CacheKey::for_tool_call("read_file", &serde_json::json!({"path": "a.rs"})).unwrap();
        let entry =
            CacheEntry::from_tool_result("read_file", &sample_result("fn main() {}")).unwrap();

        assert!(store.get_memoized(&key).await.unwrap().is_none());
        store.put_memoized(key.clone(), entry).await.unwrap();
        assert!(store.get_memoized(&key).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn invalidate_for_file_update_matches_nested_json_payloads() {
        // Regression test for the `.as_str()`-only bug: a realistic result
        // payload is a JSON *object* containing the path, not a bare
        // string, and invalidation needs to catch that.
        let store = InMemoryCacheStore::default();
        let key = CacheKey::for_tool_call("read_file", &serde_json::json!({"path": "src/main.rs"}))
            .unwrap();
        let result = ToolResult {
            tool_call_id: "call_1".to_string(),
            role: "tool".to_string(),
            name: "read_file".to_string(),
            content: serde_json::json!({"path": "src/main.rs", "body": "fn main() {}"}).to_string(),
            is_error: None,
        };
        let entry = CacheEntry::from_tool_result("read_file", &result).unwrap();
        store.put_memoized(key.clone(), entry).await.unwrap();

        assert!(store.get_memoized(&key).await.unwrap().is_some());

        let file = FileContextAttachment {
            path: "src/main.rs".to_string(),
            version: 2,
            content_hash: None,
            content: None,
        };
        let invalidated = store.invalidate_for_file_update(&file).await.unwrap();
        assert_eq!(
            invalidated, 1,
            "expected the nested-path entry to be invalidated"
        );
        assert!(store.get_memoized(&key).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn invalidate_with_both_filters_uses_intersection_not_union() {
        // Regression test for the AND/OR composition bug.
        let store = InMemoryCacheStore::default();

        // Entry 1: tool "read_file", result payload contains "main.rs" -> matches BOTH filters.
        let key1 =
            CacheKey::for_tool_call("read_file", &serde_json::json!({"path": "a.rs"})).unwrap();
        let mut result1 = sample_result("a");
        result1.content = "content of main.rs".to_string();
        let entry1 = CacheEntry::from_tool_result("read_file", &result1).unwrap();
        store.put_memoized(key1.clone(), entry1).await.unwrap();

        // Entry 2: tool "write_file" (different tool), result payload contains "main.rs" -> matches pattern ONLY.
        let key2 =
            CacheKey::for_tool_call("write_file", &serde_json::json!({"path": "b.rs"})).unwrap();
        let mut result2 = sample_result("b");
        result2.content = "content of main.rs".to_string();
        let entry2 = CacheEntry::from_tool_result("write_file", &result2).unwrap();
        store.put_memoized(key2.clone(), entry2).await.unwrap();

        // Entry 3: tool "read_file", result payload does NOT contain "main.rs" -> matches tool ONLY.
        let key3 =
            CacheKey::for_tool_call("read_file", &serde_json::json!({"path": "c.rs"})).unwrap();
        let mut result3 = sample_result("c");
        result3.content = "content of other.rs".to_string();
        let entry3 = CacheEntry::from_tool_result("read_file", &result3).unwrap();
        store.put_memoized(key3.clone(), entry3).await.unwrap();

        // Filtering on BOTH tool_name="read_file" AND pattern="main.rs"
        // should only remove entry 1 (matches both), leaving entries 2 and
        // 3 (each matches only one filter) untouched.
        let invalidated = store
            .invalidate(Some("read_file"), Some("main.rs"))
            .await
            .unwrap();
        assert_eq!(invalidated, 1, "expected only entry 1 to be invalidated");
        assert!(
            store.get_memoized(&key1).await.unwrap().is_none(),
            "entry 1 should be removed"
        );
        assert!(
            store.get_memoized(&key2).await.unwrap().is_some(),
            "entry 2 should remain"
        );
        assert!(
            store.get_memoized(&key3).await.unwrap().is_some(),
            "entry 3 should remain"
        );
    }

    #[tokio::test]
    async fn semantic_cache_matches_above_threshold_only() {
        let store = InMemoryCacheStore::default();
        let scope = SemanticCacheScope {
            workspace_id: "local".to_string(),
            project_id: "project-a".to_string(),
        };
        let fingerprint = crate::SemanticCacheFingerprint {
            provider: "test".to_string(),
            resolved_model: "model".to_string(),
            system_prefix_hash: "prefix".to_string(),
            tool_schema_hash: "tools".to_string(),
            generation_params_hash: "params".to_string(),
            application_version: "1".to_string(),
            config_version: "1".to_string(),
            source_data_version: "none".to_string(),
            immutable_workflow: None,
        };
        let now = Utc::now();
        store
            .put_semantic(SemanticCacheEntry {
                embedding: vec![1.0, 0.0, 0.0],
                prompt: "what is rust".to_string(),
                response: serde_json::json!({"answer": "a systems language"}),
                scope: scope.clone(),
                fingerprint: fingerprint.clone(),
                policy: crate::SemanticCachePolicy {
                    safe_to_reuse: true,
                    ttl_seconds: 600,
                    immutable_workflow: None,
                },
                created_at: now,
                expires_at: now + chrono::Duration::minutes(10),
            })
            .await
            .unwrap();

        // Identical embedding: similarity 1.0, should match a high threshold.
        let query = SemanticCacheQuery {
            embedding: vec![1.0, 0.0, 0.0],
            scope: scope.clone(),
            fingerprint_hash: fingerprint.digest().unwrap(),
            answer_threshold: 0.97,
            candidate_threshold: 0.80,
            candidate_limit: 3,
            now,
        };
        let hit = store.get_semantic(&query).await.unwrap();
        assert!(hit.reusable_answer.is_some());

        // Orthogonal embedding: similarity 0.0, should not match.
        let miss = store
            .get_semantic(&SemanticCacheQuery {
                embedding: vec![0.0, 1.0, 0.0],
                ..query
            })
            .await
            .unwrap();
        assert!(miss.reusable_answer.is_none());
    }

    #[tokio::test]
    async fn semantic_cache_isolates_scope_fingerprint_expiry_and_candidates() {
        let store = InMemoryCacheStore::default();
        let now = Utc::now();
        let scope = SemanticCacheScope {
            workspace_id: "workspace-a".to_string(),
            project_id: "project-a".to_string(),
        };
        let fingerprint = crate::SemanticCacheFingerprint {
            provider: "provider-a".to_string(),
            resolved_model: "model-a".to_string(),
            system_prefix_hash: "prefix-a".to_string(),
            tool_schema_hash: "tools-a".to_string(),
            generation_params_hash: "params-a".to_string(),
            application_version: "app-a".to_string(),
            config_version: "config-a".to_string(),
            source_data_version: "source-a".to_string(),
            immutable_workflow: None,
        };
        let entry = |embedding: Vec<f32>, expires_at| SemanticCacheEntry {
            embedding,
            prompt: "cached prompt".to_string(),
            response: serde_json::json!({"answer": "cached"}),
            scope: scope.clone(),
            fingerprint: fingerprint.clone(),
            policy: crate::SemanticCachePolicy {
                safe_to_reuse: true,
                ttl_seconds: 600,
                immutable_workflow: None,
            },
            created_at: now - chrono::Duration::seconds(1),
            expires_at,
        };
        store
            .put_semantic(entry(
                vec![0.90, (1.0_f32 - 0.90_f32.powi(2)).sqrt(), 0.0],
                now + chrono::Duration::minutes(10),
            ))
            .await
            .unwrap();
        store
            .put_semantic(entry(
                vec![1.0, 0.0, 0.0],
                now - chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap();

        let base_query = SemanticCacheQuery {
            embedding: vec![1.0, 0.0, 0.0],
            scope: scope.clone(),
            fingerprint_hash: fingerprint.digest().unwrap(),
            answer_threshold: 0.97,
            candidate_threshold: 0.80,
            candidate_limit: 3,
            now,
        };
        let lookup = store.get_semantic(&base_query).await.unwrap();
        assert!(lookup.reusable_answer.is_none());
        assert_eq!(lookup.candidates.len(), 1);
        assert!((lookup.candidates[0].provenance.similarity - 0.90).abs() < 0.001);

        let wrong_scope = store
            .get_semantic(&SemanticCacheQuery {
                scope: SemanticCacheScope {
                    workspace_id: "workspace-b".to_string(),
                    project_id: "project-a".to_string(),
                },
                ..base_query.clone()
            })
            .await
            .unwrap();
        assert!(wrong_scope.candidates.is_empty());

        let wrong_fingerprint = store
            .get_semantic(&SemanticCacheQuery {
                fingerprint_hash: "different".to_string(),
                ..base_query.clone()
            })
            .await
            .unwrap();
        assert!(wrong_fingerprint.candidates.is_empty());

        assert_eq!(store.invalidate_semantic(&scope, None).await.unwrap(), 1);
        assert!(store
            .get_semantic(&base_query)
            .await
            .unwrap()
            .candidates
            .is_empty());
    }

    // TeamScopedCacheStore Tests

    fn scoped_entry(tool_name: &str, payload: &str) -> CacheEntry {
        CacheEntry {
            tool_name: tool_name.to_string(),
            result_payload: serde_json::json!({ "content": payload }),
            content_hash: payload.to_string(),
            created_at: chrono::Utc::now(),
        }
    }

    fn semantic_tool_entry(
        workspace_id: Uuid,
        request_group_id: &str,
        source_session_id: Uuid,
        cache_key: &str,
        query: &str,
    ) -> TeamSemanticToolCacheEntry {
        TeamSemanticToolCacheEntry {
            scope_fingerprint: "test-scope".into(),
            workspace_id,
            request_group_id: request_group_id.to_string(),
            source_session_id,
            cache_key: CacheKey(cache_key.to_string()),
            request_embedding: crate::embed_tool_request(
                "search_code",
                &serde_json::json!({"query": query}),
            )
            .unwrap(),
            cache_entry: scoped_entry("search_code", "authentication middleware is in auth.rs"),
        }
    }

    #[tokio::test]
    async fn semantic_tool_lookup_reuses_another_agents_result_in_same_team_group() {
        let store = InMemoryTeamScopedCacheStore::default();
        let workspace_id = Uuid::new_v4();
        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();
        store
            .put_semantic_tool_scoped(semantic_tool_entry(
                workspace_id,
                "coordinated-request",
                agent_a,
                "agent-a-key",
                "find authentication middleware in the codebase",
            ))
            .await
            .unwrap();

        let query = crate::embed_tool_request(
            "search_code",
            &serde_json::json!({
                "query": "search the codebase for authentication middleware"
            }),
        )
        .unwrap();
        let hit = store
            .get_semantic_tool_scoped(
                workspace_id,
                "coordinated-request",
                "test-scope",
                agent_b,
                "search_code",
                &query,
                crate::TEAM_TOOL_SEMANTIC_SIMILARITY,
            )
            .await
            .unwrap()
            .expect("agent B should reuse agent A's result");

        assert_eq!(hit.source_session_id, agent_a);
        assert_eq!(
            hit.cache_entry.result_payload,
            serde_json::json!({"content": "authentication middleware is in auth.rs"})
        );
    }

    #[tokio::test]
    async fn semantic_tool_lookup_enforces_team_group_session_and_tool_boundaries() {
        let store = InMemoryTeamScopedCacheStore::default();
        let workspace_id = Uuid::new_v4();
        let source_session_id = Uuid::new_v4();
        store
            .put_semantic_tool_scoped(semantic_tool_entry(
                workspace_id,
                "group-a",
                source_session_id,
                "source-key",
                "find authentication middleware in the codebase",
            ))
            .await
            .unwrap();
        let query = crate::embed_tool_request(
            "search_code",
            &serde_json::json!({"query": "find authentication middleware in the codebase"}),
        )
        .unwrap();

        for (workspace, group, session, tool) in [
            (Uuid::new_v4(), "group-a", Uuid::new_v4(), "search_code"),
            (workspace_id, "group-b", Uuid::new_v4(), "search_code"),
            (workspace_id, "group-a", source_session_id, "search_code"),
            (workspace_id, "group-a", Uuid::new_v4(), "read_file"),
        ] {
            assert!(store
                .get_semantic_tool_scoped(
                    workspace,
                    group,
                    "test-scope",
                    session,
                    tool,
                    &query,
                    0.99
                )
                .await
                .unwrap()
                .is_none());
        }
    }

    #[tokio::test]
    async fn scoped_lookup_returns_entry_within_same_workspace() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws = Uuid::new_v4();
        let key = CacheKey("shared_key".to_string());
        let entry = scoped_entry("grep_search", "found results");

        store
            .put_memoized_scoped(key.clone(), entry.clone(), Some(ws))
            .await
            .unwrap();

        let hit = store.get_memoized_scoped(&key, Some(ws)).await.unwrap();
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().tool_name, "grep_search");
    }

    #[tokio::test]
    async fn scoped_lookup_does_not_cross_workspace_boundary() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws_a = Uuid::new_v4();
        let ws_b = Uuid::new_v4();
        let key = CacheKey("shared_key".to_string());

        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("grep_search", "workspace A results"),
                Some(ws_a),
            )
            .await
            .unwrap();

        // Workspace B must not see workspace A's cached result.
        let miss = store.get_memoized_scoped(&key, Some(ws_b)).await.unwrap();
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn scoped_lookup_with_none_workspace_is_its_own_partition() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws = Uuid::new_v4();
        let key = CacheKey("shared_key".to_string());

        // Store under a real workspace.
        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("grep_search", "workspace result"),
                Some(ws),
            )
            .await
            .unwrap();

        // A None lookup must not see it.
        let miss_none = store.get_memoized_scoped(&key, None).await.unwrap();
        assert!(miss_none.is_none());

        // Store under None.
        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("grep_search", "unscoped result"),
                None,
            )
            .await
            .unwrap();

        // The real workspace must not see the None entry.
        let miss_ws = store.get_memoized_scoped(&key, Some(ws)).await.unwrap();
        assert_eq!(
            miss_ws.unwrap().result_payload,
            serde_json::json!({ "content": "workspace result" })
        );

        // None sees its own entry.
        let hit_none = store.get_memoized_scoped(&key, None).await.unwrap();
        assert_eq!(
            hit_none.unwrap().result_payload,
            serde_json::json!({ "content": "unscoped result" })
        );
    }

    #[tokio::test]
    async fn scoped_upsert_replaces_existing_entry() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws = Uuid::new_v4();
        let key = CacheKey("shared_key".to_string());

        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("grep_search", "old result"),
                Some(ws),
            )
            .await
            .unwrap();
        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("grep_search", "new result"),
                Some(ws),
            )
            .await
            .unwrap();

        let hit = store.get_memoized_scoped(&key, Some(ws)).await.unwrap();
        assert_eq!(hit.unwrap().content_hash, "new result".to_string());
    }

    #[tokio::test]
    async fn scoped_invalidate_only_touches_matching_partition() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws_a = Uuid::new_v4();
        let ws_b = Uuid::new_v4();
        let key = CacheKey("shared_key".to_string());

        store
            .put_memoized_scoped(key.clone(), scoped_entry("grep_search", "A"), Some(ws_a))
            .await
            .unwrap();
        store
            .put_memoized_scoped(key.clone(), scoped_entry("grep_search", "B"), Some(ws_b))
            .await
            .unwrap();

        let removed = store
            .invalidate_scoped(Some(ws_a), None, None)
            .await
            .unwrap();
        assert_eq!(removed, 1);

        // Workspace A's entry is gone.
        assert!(store
            .get_memoized_scoped(&key, Some(ws_a))
            .await
            .unwrap()
            .is_none());
        // Workspace B's entry survives.
        assert!(store
            .get_memoized_scoped(&key, Some(ws_b))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn same_cache_key_can_exist_in_multiple_workspaces() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws_a = Uuid::new_v4();
        let ws_b = Uuid::new_v4();
        let key = CacheKey("identical_arguments".to_string());

        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("read_file", "workspace A file content"),
                Some(ws_a),
            )
            .await
            .unwrap();
        store
            .put_memoized_scoped(
                key.clone(),
                scoped_entry("read_file", "workspace B file content"),
                Some(ws_b),
            )
            .await
            .unwrap();

        let a = store.get_memoized_scoped(&key, Some(ws_a)).await.unwrap();
        let b = store.get_memoized_scoped(&key, Some(ws_b)).await.unwrap();
        assert_eq!(
            a.unwrap().result_payload,
            serde_json::json!({ "content": "workspace A file content" })
        );
        assert_eq!(
            b.unwrap().result_payload,
            serde_json::json!({ "content": "workspace B file content" })
        );
    }

    #[tokio::test]
    async fn scoped_file_invalidation_does_not_cross_workspace() {
        let store = InMemoryTeamScopedCacheStore::default();
        let ws_a = Uuid::new_v4();
        let ws_b = Uuid::new_v4();
        let key = CacheKey("read_file_key".to_string());

        // Both workspaces cache a result referencing "src/main.rs".
        let mut entry_a = scoped_entry("read_file", "contents of src/main.rs v1");
        let mut entry_b = scoped_entry("read_file", "contents of src/main.rs v1");
        entry_a.content_hash = "hash_a".to_string();
        entry_b.content_hash = "hash_b".to_string();

        store
            .put_memoized_scoped(key.clone(), entry_a, Some(ws_a))
            .await
            .unwrap();
        store
            .put_memoized_scoped(key.clone(), entry_b, Some(ws_b))
            .await
            .unwrap();

        // File "src/main.rs" is updated in workspace A only.
        let file = FileContextAttachment {
            path: "src/main.rs".to_string(),
            version: 2,
            content_hash: None,
            content: None,
        };
        let removed = store
            .invalidate_for_file_update_scoped(Some(ws_a), &file)
            .await
            .unwrap();
        assert_eq!(removed, 1);

        // Workspace A's entry is gone.
        assert!(store
            .get_memoized_scoped(&key, Some(ws_a))
            .await
            .unwrap()
            .is_none());
        // Workspace B's entry survives - different partition.
        assert!(store
            .get_memoized_scoped(&key, Some(ws_b))
            .await
            .unwrap()
            .is_some());
    }
}
