//! Providers persistence for the local engine.

use chrono::Utc;
use engine_core::{
    error::{EngineError, EngineResult},
    model_library::model_catalog,
};
use rusqlite::params;

use super::LocalDatabase;

impl LocalDatabase {
    /// Seed provider_models from the curated catalog using upsert.
    pub fn seed_provider_models(&self) -> EngineResult<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        for model in model_catalog() {
            let input_json = serde_json::to_string(&model.supported_input_data)
                .map_err(EngineError::Serialization)?;
            let output_json = serde_json::to_string(&model.supported_output_data)
                .map_err(EngineError::Serialization)?;
            tx.execute(
                "INSERT INTO provider_models (
                    id, provider, model_name, model_id, input_price, output_price, rank,
                    max_input_tokens, max_output_tokens, reasoning, speed,
                    supported_input_data, supported_output_data, created_at, updated_at, pricing, aliases, auto_select, notes
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
                ON CONFLICT(provider, model_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    input_price = EXCLUDED.input_price,
                    output_price = EXCLUDED.output_price,
                    rank = EXCLUDED.rank,
                    max_input_tokens = EXCLUDED.max_input_tokens,
                    max_output_tokens = EXCLUDED.max_output_tokens,
                    reasoning = EXCLUDED.reasoning,
                    speed = EXCLUDED.speed,
                    supported_input_data = EXCLUDED.supported_input_data,
                    supported_output_data = EXCLUDED.supported_output_data,
                    updated_at = EXCLUDED.updated_at,
                    pricing = EXCLUDED.pricing,
                    aliases = EXCLUDED.aliases,
                    auto_select = EXCLUDED.auto_select,
                    notes = EXCLUDED.notes",
                params![
                    model.id.to_string(),
                    &model.provider,
                    &model.model_name,
                    &model.model_id,
                    model.current_text_prices().0,
                    model.current_text_prices().1,
                    model.rank,
                    model.max_input_tokens,
                    model.max_output_tokens,
                    model.reasoning.as_str(),
                    model.speed.as_str(),
                    input_json,
                    output_json,
                    now,
                    now,
                    model.pricing.as_ref().map(serde_json::to_string).transpose().map_err(EngineError::Serialization)?,
                    serde_json::to_string(&model.aliases).map_err(EngineError::Serialization)?,
                    model.auto_select,
                    model.notes,
                ],
            )
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| EngineError::Storage(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_seed_persists_pricing_aliases_and_updates_existing_rows() {
        let db = LocalDatabase::open_in_memory().unwrap();
        {
            let conn = db.conn.lock().unwrap();
            let (pricing, aliases): (String, String) = conn.query_row(
                "SELECT pricing, aliases FROM provider_models WHERE provider='openai' AND model_id='gpt-5.6-sol'",
                [], |row| Ok((row.get(0)?, row.get(1)?)),
            ).unwrap();
            let stored: engine_core::model_catalog::PricingSchedule =
                serde_json::from_str(&pricing).unwrap();
            let expected = engine_core::model_library::find_model("openai", "gpt-5.6").unwrap();
            assert_eq!(
                serde_json::to_value(stored).unwrap(),
                serde_json::to_value(expected.pricing).unwrap()
            );
            assert_eq!(
                serde_json::from_str::<Vec<String>>(&aliases).unwrap(),
                vec!["gpt-5.6"]
            );
            conn.execute("UPDATE provider_models SET pricing=NULL, aliases=NULL, auto_select=1 WHERE model_id='claude-mythos-5-1'", []).unwrap();
        }
        db.seed_provider_models().unwrap();
        let conn = db.conn.lock().unwrap();
        let (count, enabled): (i64, bool) = conn.query_row(
            "SELECT COUNT(*), auto_select FROM provider_models WHERE model_id='claude-mythos-5-1' AND pricing IS NOT NULL", [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).unwrap();
        assert_eq!(count, 1);
        assert!(!enabled);
    }
}
