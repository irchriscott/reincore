//! Budgets persistence for the local engine.

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use engine_core::{
    budget::{BudgetHistoryEntry, BudgetPolicy, BudgetScope, BudgetStore},
    error::{EngineError, EngineResult},
};
use rusqlite::{params, OptionalExtension};
use uuid::Uuid;

use super::LocalDatabase;

#[async_trait]
impl BudgetStore for LocalDatabase {
    async fn set_budget(
        &self,
        scope: BudgetScope,
        scope_key: String,
        budget_cents: i64,
    ) -> EngineResult<(BudgetPolicy, BudgetHistoryEntry)> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let now = Utc::now();
            let policy_id = Uuid::new_v4();

            let previous_budget_cents: i64 = conn
                .query_row(
                    "SELECT budget_cents FROM budget_policies WHERE scope = ?1 AND scope_key = ?2",
                    params![scope.as_str(), &scope_key],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            conn.execute(
                "INSERT INTO budget_policies (
                    id, scope, scope_key, budget_cents, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(scope, scope_key) DO UPDATE SET
                   budget_cents = EXCLUDED.budget_cents,
                   updated_at = EXCLUDED.updated_at",
                params![
                    policy_id.to_string(),
                    scope.as_str(),
                    &scope_key,
                    budget_cents,
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;

            conn.execute(
                "INSERT INTO budget_history (
                    policy_id, scope, scope_key, previous_budget_cents, new_budget_cents, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    policy_id.to_string(),
                    scope.as_str(),
                    &scope_key,
                    previous_budget_cents,
                    budget_cents,
                    now.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;

            let policy = BudgetPolicy {
                id: policy_id,
                scope,
                scope_key: scope_key.clone(),
                budget_cents,
                created_at: now,
                updated_at: now,
            };
            let history_entry = BudgetHistoryEntry {
                policy_id,
                scope,
                scope_key,
                previous_budget_cents,
                new_budget_cents: budget_cents,
                updated_at: now,
            };

            Ok((policy, history_entry))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn get_budget(
        &self,
        scope: BudgetScope,
        scope_key: &str,
    ) -> EngineResult<Option<BudgetPolicy>> {
        let db = self.clone();
        let scope_key = scope_key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, scope, scope_key, budget_cents, created_at, updated_at
                     FROM budget_policies WHERE scope = ?1 AND scope_key = ?2",
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let row = stmt
                .query_row([scope.as_str(), &scope_key], |row| {
                    let id_str: String = row.get(0)?;
                    let scope_str: String = row.get(1)?;
                    let created_str: String = row.get(4)?;
                    let updated_str: String = row.get(5)?;
                    Ok(BudgetPolicy {
                        id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        scope: str_to_budget_scope(&scope_str),
                        scope_key: row.get(2)?,
                        budget_cents: row.get(3)?,
                        created_at: created_str.parse().unwrap_or_else(|_| Utc::now()),
                        updated_at: updated_str.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .optional()
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(row)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn list_budget_history(
        &self,
        scope: BudgetScope,
        scope_key: Option<&str>,
        limit: Option<usize>,
    ) -> EngineResult<Vec<BudgetHistoryEntry>> {
        let db = self.clone();
        let scope_key = scope_key.map(|value| value.to_string());
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let sql = format!(
                "SELECT policy_id, scope, scope_key, previous_budget_cents, new_budget_cents, updated_at
                 FROM budget_history
                 WHERE scope = ?1 {}
                 ORDER BY updated_at DESC {}",
                scope_key.as_ref().map(|_| "AND scope_key = ?2").unwrap_or(""),
                limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            let scope_str = scope.as_str();
            let params_vec: Vec<Box<dyn rusqlite::ToSql>> = if let Some(key) = scope_key {
                vec![Box::new(scope_str), Box::new(key)]
            } else {
                vec![Box::new(scope_str)]
            };
            let params_refs: Vec<&dyn rusqlite::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();
            let rows = stmt
                .query_map(params_refs.as_slice(), |row| {
                    let policy_id_str: String = row.get(0)?;
                    let scope_str: String = row.get(1)?;
                    let updated_str: String = row.get(5)?;
                    Ok(BudgetHistoryEntry {
                        policy_id: Uuid::parse_str(&policy_id_str).unwrap_or_default(),
                        scope: str_to_budget_scope(&scope_str),
                        scope_key: row.get(2)?,
                        previous_budget_cents: row.get(3)?,
                        new_budget_cents: row.get(4)?,
                        updated_at: updated_str.parse().unwrap_or_else(|_| Utc::now()),
                    })
                })
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| EngineError::Storage(e.to_string()))
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}

pub(super) fn str_to_budget_scope(s: &str) -> BudgetScope {
    match s {
        "project" => BudgetScope::Project,
        "day" => BudgetScope::Day,
        "month" => BudgetScope::Month,
        _ => BudgetScope::Session,
    }
}
