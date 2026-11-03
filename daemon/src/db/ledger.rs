//! Ledger persistence for the local engine.

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use engine_core::{
    budget::{LedgerEvent, LedgerReason, LedgerStore},
    error::{EngineError, EngineResult},
};
use rusqlite::params;
use uuid::Uuid;

use super::LocalDatabase;

#[async_trait]
impl LedgerStore for LocalDatabase {
    async fn record_spend(
        &self,
        session_id: Uuid,
        amount_cents: i64,
        reason: LedgerReason,
    ) -> EngineResult<LedgerEvent> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let created_at = Utc::now();
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO ledger_events (
                    session_id, amount_cents, reason, created_at
                ) VALUES (?1, ?2, ?3, ?4)",
                params![
                    session_id.to_string(),
                    amount_cents,
                    reason_to_str(reason),
                    created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(LedgerEvent {
                session_id,
                amount_cents,
                reason,
                created_at,
            })
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn total_spend(&self, session_id: Uuid) -> EngineResult<i64> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let total: i64 = conn
                .query_row(
                    "SELECT (COALESCE(SUM(COALESCE(amount_nano_usd, amount_cents * 10000000)), 0) / 10000000) FROM ledger_events WHERE session_id = ?1",
                    [session_id.to_string()],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(total)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn total_spend_day(&self, date: NaiveDate) -> EngineResult<i64> {
        let start: DateTime<Utc> =
            DateTime::from_naive_utc_and_offset(date.and_hms_opt(0, 0, 0).unwrap_or_default(), Utc);
        let end: DateTime<Utc> = DateTime::from_naive_utc_and_offset(
            date.and_hms_opt(23, 59, 59).unwrap_or_default(),
            Utc,
        );
        let db = self.clone();
        let start = start.to_rfc3339();
        let end = end.to_rfc3339();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let total: i64 = conn
                .query_row(
                    "SELECT (COALESCE(SUM(COALESCE(amount_nano_usd, amount_cents * 10000000)), 0) / 10000000) FROM ledger_events
                     WHERE created_at >= ?1 AND created_at <= ?2",
                    [start, end],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(total)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn total_spend_month(&self, year: i32, month: u32) -> EngineResult<i64> {
        let start: DateTime<Utc> = DateTime::from_naive_utc_and_offset(
            NaiveDate::from_ymd_opt(year, month, 1)
                .unwrap_or_default()
                .and_hms_opt(0, 0, 0)
                .unwrap_or_default(),
            Utc,
        );
        let end_day = NaiveDate::from_ymd_opt(year, month + 1, 1)
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap_or_default())
            .pred_opt()
            .unwrap_or_default();
        let end: DateTime<Utc> = DateTime::from_naive_utc_and_offset(
            end_day.and_hms_opt(23, 59, 59).unwrap_or_default(),
            Utc,
        );
        let db = self.clone();
        let start = start.to_rfc3339();
        let end = end.to_rfc3339();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let total: i64 = conn
                .query_row(
                    "SELECT (COALESCE(SUM(COALESCE(amount_nano_usd, amount_cents * 10000000)), 0) / 10000000) FROM ledger_events
                     WHERE created_at >= ?1 AND created_at <= ?2",
                    [start, end],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(total)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }

    async fn total_spend_project(&self, _project: &str) -> EngineResult<i64> {
        // SQLite ledger does not store project names; return 0.
        Ok(0)
    }

    async fn total_spend_all(&self) -> EngineResult<i64> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            let total: i64 = conn
                .query_row(
                    "SELECT (COALESCE(SUM(COALESCE(amount_nano_usd, amount_cents * 10000000)), 0) / 10000000) FROM ledger_events",
                    [],
                    |row| row.get(0),
                )
                .map_err(|e| EngineError::Storage(e.to_string()))?;
            Ok(total)
        })
        .await
        .map_err(|e| EngineError::Storage(format!("sqlite task join error: {e}")))?
    }
}

pub(super) fn reason_to_str(reason: LedgerReason) -> &'static str {
    match reason {
        LedgerReason::LlmCall => "llm_call",
        LedgerReason::ToolCall => "tool_call",
        LedgerReason::EstimateLock => "estimate_lock",
        LedgerReason::Completion => "completion",
    }
}
