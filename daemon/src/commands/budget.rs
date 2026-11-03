use crate::db::LocalDatabase;
use anyhow::Result;
use chrono::{Datelike, Utc};
use engine_core::{BudgetScope, BudgetStore, LedgerStore};
use tracing::{info, warn};
use uuid::Uuid;

pub async fn handle_budget(
    scope: String,
    key: Option<String>,
    limit: Option<String>,
    history: bool,
    usage: bool,
    db_path: String,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let db = LocalDatabase::open(&db_path).unwrap_or_else(|e| {
        warn!(
            "Failed to open SQLite DB at {}: {}. Falling back to in-memory.",
            db_path, e
        );
        LocalDatabase::open_in_memory().expect("in-memory DB must open")
    });

    let scope = parse_budget_scope(&scope)?;
    let scope_key = key.unwrap_or_else(|| default_scope_key(scope));

    if history {
        let entries = db
            .list_budget_history(scope, Some(&scope_key), None)
            .await?;
        if entries.is_empty() {
            println!("No budget history for {}:{}", scope.as_str(), scope_key);
        } else {
            println!("Budget history for {}:{}:", scope.as_str(), scope_key);
            for entry in entries {
                println!(
                    "  {} - ${:.2} -> ${:.2}",
                    entry.updated_at.to_rfc3339(),
                    entry.previous_budget_cents as f64 / 100.0,
                    entry.new_budget_cents as f64 / 100.0
                );
            }
        }
        return Ok(());
    }

    if usage {
        return show_budget_usage(
            &db,
            scope,
            &scope_key,
            Utc::now().date_naive(),
            Utc::now().year(),
            Utc::now().month(),
        )
        .await;
    }

    if let Some(limit_str) = limit {
        let limit_cents = parse_dollars_to_cents(&limit_str)?;
        let (policy, _history) = db.set_budget(scope, scope_key.clone(), limit_cents).await?;
        info!(
            "Budget set for {}:{} to {} cents (${})",
            scope.as_str(),
            scope_key,
            limit_cents,
            limit_str
        );
        println!(
            "Budget set for {}:{} to ${} ({} cents)",
            scope.as_str(),
            scope_key,
            limit_str,
            limit_cents
        );
        println!("Policy ID: {}", policy.id);
    } else {
        let policy = db.get_budget(scope, &scope_key).await?;
        match policy {
            Some(p) => println!(
                "Budget for {}:{} is ${} ({} cents)",
                scope.as_str(),
                scope_key,
                p.budget_cents as f64 / 100.0,
                p.budget_cents
            ),
            None => println!(
                "No budget set for {}:{}. Use --limit to set one.",
                scope.as_str(),
                scope_key
            ),
        }
    }
    Ok(())
}

async fn show_budget_usage(
    db: &LocalDatabase,
    scope: BudgetScope,
    scope_key: &str,
    today: chrono::NaiveDate,
    year: i32,
    month: u32,
) -> Result<()> {
    let policy = db.get_budget(scope, scope_key).await?;
    let budget_cents = policy.as_ref().map(|p| p.budget_cents);

    let spent_cents: i64 = match scope {
        BudgetScope::Session => {
            let session_id = Uuid::parse_str(scope_key)
                .map_err(|_| anyhow::anyhow!("session scope requires a valid UUID session id"))?;
            db.total_spend(session_id).await?
        }
        BudgetScope::Project => db.total_spend_project(scope_key).await?,
        BudgetScope::Day => db.total_spend_day(today).await?,
        BudgetScope::Month => db.total_spend_month(year, month).await?,
    };

    println!("Budget usage for {}:{}:", scope.as_str(), scope_key);
    println!(
        "  Spent:   ${:.2} ({} cents)",
        spent_cents as f64 / 100.0,
        spent_cents
    );
    match budget_cents {
        Some(limit) => {
            let remaining = limit - spent_cents;
            println!("  Limit:   ${:.2} ({} cents)", limit as f64 / 100.0, limit);
            println!(
                "  Remaining: ${:.2} ({} cents)",
                remaining as f64 / 100.0,
                remaining
            );
        }
        None => {
            println!("  Limit:   not set (use --limit to set one)");
        }
    }
    Ok(())
}

fn parse_budget_scope(input: &str) -> Result<engine_core::BudgetScope> {
    match input.to_lowercase().as_str() {
        "session" => Ok(engine_core::BudgetScope::Session),
        "project" => Ok(engine_core::BudgetScope::Project),
        "day" => Ok(engine_core::BudgetScope::Day),
        "month" => Ok(engine_core::BudgetScope::Month),
        _ => Err(anyhow::anyhow!(
            "invalid budget scope '{}'. Use session, project, day, or month.",
            input
        )),
    }
}

fn default_scope_key(scope: engine_core::BudgetScope) -> String {
    match scope {
        engine_core::BudgetScope::Session => {
            // Session scope requires an explicit key.
            "default".to_string()
        }
        engine_core::BudgetScope::Project => "default".to_string(),
        engine_core::BudgetScope::Day => Utc::now().date_naive().to_string(),
        engine_core::BudgetScope::Month => format!("{}-{}", Utc::now().year(), Utc::now().month()),
    }
}

pub fn parse_dollars_to_cents(input: &str) -> Result<i64> {
    let cleaned: String = input
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let dollars: f64 = cleaned.parse()?;
    Ok((dollars * 100.0).round() as i64)
}
