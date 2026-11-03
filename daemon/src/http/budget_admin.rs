use super::*;

// Budget management endpoints.
//
// These endpoints expose the BudgetStore and LedgerStore traits over HTTP
// so the desktop app frontend can manage budget limits, view spend, and
// browse budget history - all without a direct SQLite dependency on the
// Tauri side.

use engine_core::budget::{BudgetScope, BudgetStore};

/// Response shape for `GET /v1/budget` and `POST /v1/budget`.
#[derive(Serialize)]
pub struct BudgetResponse {
    pub scope: String,
    pub key: Option<String>,
    pub limit_cents: i64,
    pub spent_cents: i64,
    pub remaining_cents: i64,
    pub tier: u8,
}

/// Request body for `POST /v1/budget`.
#[derive(Deserialize)]
pub struct SetBudgetRequest {
    pub scope: String,
    pub key: Option<String>,
    /// Limit in **dollars** (not cents) - matches the frontend input.
    pub limit: String,
}

/// A single entry in the budget history list.
#[derive(Serialize)]
pub struct BudgetHistoryItem {
    pub timestamp: String,
    pub event: String,
    pub scope: String,
    pub key: Option<String>,
    pub limit_cents: Option<i64>,
    pub spent_cents: Option<i64>,
}

/// Response for `GET /v1/budget/history`.
#[derive(Serialize)]
pub struct BudgetHistoryResponse {
    pub history: Vec<BudgetHistoryItem>,
}

/// Parse a budget scope string ("session", "project", "day", "month")
/// into the enum, defaulting to `Session` for unknown values.
fn parse_scope(s: &str) -> BudgetScope {
    match s.to_lowercase().as_str() {
        "project" => BudgetScope::Project,
        "day" => BudgetScope::Day,
        "month" => BudgetScope::Month,
        _ => BudgetScope::Session,
    }
}

/// Compute the current spend (in cents) for a given budget scope + key.
async fn compute_spend(db: &crate::db::LocalDatabase, scope: BudgetScope, key: &str) -> i64 {
    use engine_core::budget::LedgerStore;
    match scope {
        BudgetScope::Session => {
            if key.is_empty() {
                // General session budget: sum spend across ALL sessions.
                db.total_spend_all().await.unwrap_or(0)
            } else {
                // key is a session UUID; sum ledger events for that session.
                match Uuid::parse_str(key) {
                    Ok(id) => db.total_spend(id).await.unwrap_or(0),
                    Err(_) => 0,
                }
            }
        }
        BudgetScope::Day => {
            // key is a date string like "2026-08-13".
            match chrono::NaiveDate::parse_from_str(key, "%Y-%m-%d") {
                Ok(date) => db.total_spend_day(date).await.unwrap_or(0),
                Err(_) => 0,
            }
        }
        BudgetScope::Month => {
            // key is "YYYY-M" (e.g. "2026-8").
            let parts: Vec<&str> = key.split('-').collect();
            if parts.len() == 2 {
                let year: i32 = parts[0].parse().unwrap_or(0);
                let month: u32 = parts[1].parse().unwrap_or(1);
                db.total_spend_month(year, month).await.unwrap_or(0)
            } else {
                0
            }
        }
        BudgetScope::Project => db.total_spend_project(key).await.unwrap_or(0),
    }
}

/// Derive the current tier (1–4) from the spend ratio.
fn compute_tier(limit_cents: i64, spent_cents: i64) -> u8 {
    if limit_cents <= 0 {
        return 1;
    }
    let ratio = spent_cents as f64 / limit_cents as f64;
    if ratio < 0.70 {
        1
    } else if ratio < 0.90 {
        2
    } else if ratio < 1.00 {
        3
    } else {
        4
    }
}

/// `GET /v1/budget?scope=&key=` - read the current budget policy and spend.
pub async fn get_budget(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    get_budget_inner(&state.db, params).await
}

/// Single-provider variant of [`get_budget`].
pub async fn get_budget_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    get_budget_inner(&state.db, params).await
}

async fn get_budget_inner(
    db: &crate::db::LocalDatabase,
    params: HashMap<String, String>,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    let scope_str = params.get("scope").cloned().unwrap_or_default();
    let key = params.get("key").cloned().unwrap_or_default();
    let scope = parse_scope(&scope_str);

    let policy = db
        .get_budget(scope, &key)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let limit_cents = policy.as_ref().map(|p| p.budget_cents).unwrap_or(0);
    let spent_cents = compute_spend(db, scope, &key).await;

    Ok(Json(BudgetResponse {
        scope: scope_str,
        key: if key.is_empty() { None } else { Some(key) },
        limit_cents,
        spent_cents,
        remaining_cents: limit_cents - spent_cents,
        tier: compute_tier(limit_cents, spent_cents),
    }))
}

/// `POST /v1/budget` - set or update a budget limit.
pub async fn set_budget(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<SetBudgetRequest>,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    set_budget_inner(&state.db, req).await
}

/// Single-provider variant of [`set_budget`].
pub async fn set_budget_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::Json(req): axum::Json<SetBudgetRequest>,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    set_budget_inner(&state.db, req).await
}

async fn set_budget_inner(
    db: &crate::db::LocalDatabase,
    req: SetBudgetRequest,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    let scope = parse_scope(&req.scope);
    let key = req.key.unwrap_or_default();

    // Parse the limit string as dollars → cents.
    let limit_dollars: f64 = req
        .limit
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid limit: {}", e)))?;
    let limit_cents = (limit_dollars * 100.0).round() as i64;
    if limit_cents < 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Budget limit cannot be negative.".to_string(),
        ));
    }

    db.set_budget(scope, key.clone(), limit_cents)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let spent_cents = compute_spend(db, scope, &key).await;

    Ok(Json(BudgetResponse {
        scope: req.scope,
        key: if key.is_empty() { None } else { Some(key) },
        limit_cents,
        spent_cents,
        remaining_cents: limit_cents - spent_cents,
        tier: compute_tier(limit_cents, spent_cents),
    }))
}

/// `GET /v1/budget/history?scope=&key=` - list budget change history.
pub async fn get_budget_history(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<BudgetHistoryResponse>, (StatusCode, String)> {
    get_budget_history_inner(&state.db, params).await
}

/// Single-provider variant of [`get_budget_history`].
pub async fn get_budget_history_single_provider(
    State(state): State<SingleProviderAppState>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<BudgetHistoryResponse>, (StatusCode, String)> {
    get_budget_history_inner(&state.db, params).await
}

async fn get_budget_history_inner(
    db: &crate::db::LocalDatabase,
    params: HashMap<String, String>,
) -> Result<Json<BudgetHistoryResponse>, (StatusCode, String)> {
    let scope_str = params.get("scope").cloned().unwrap_or_default();
    let key = params.get("key").cloned();
    let scope = parse_scope(&scope_str);

    let entries = db
        .list_budget_history(scope, key.as_deref(), Some(50))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let history = entries
        .into_iter()
        .map(|e| BudgetHistoryItem {
            timestamp: e.updated_at.to_rfc3339(),
            event: format!(
                "Budget updated: ${:.2} → ${:.2}",
                e.previous_budget_cents as f64 / 100.0,
                e.new_budget_cents as f64 / 100.0
            ),
            scope: scope_str.clone(),
            key: if e.scope_key.is_empty() {
                None
            } else {
                Some(e.scope_key)
            },
            limit_cents: Some(e.new_budget_cents),
            spent_cents: None,
        })
        .collect();

    Ok(Json(BudgetHistoryResponse { history }))
}
