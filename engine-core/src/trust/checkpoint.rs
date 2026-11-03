//! Signed, time-bound resumable checkpoints.
//!
//! When a session exhausts its budget (Tier 4) or is explicitly paused,
//! the engine emits a [`Checkpoint`] containing the session ID, the last
//! event sequence number, and an HMAC-signed [`ResumableToken`]. A later
//! resume call validates the token signature and TTL, replays the event
//! log up to `last_event_seq`, and marks the checkpoint as redeemed so it
//! cannot be replayed a second time.

use crate::error::{EngineError, EngineResult};
use crate::event::{CheckpointRedeemedPayload, EventStore};
use crate::session::{Session, SessionStatus};
use chrono::{DateTime, Duration, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// A signed, time-bound reference to a resumable checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResumableToken {
    pub token: String,
    pub session_id: Uuid,
    pub last_event_seq: i64,
    pub expires_at: DateTime<Utc>,
}

/// A checkpoint record stored by the engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub token: String,
    pub session_id: Uuid,
    pub last_event_seq: i64,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

/// Manages creation and validation of resumable checkpoints.
#[derive(Clone, Debug)]
pub struct CheckpointManager {
    secret: Vec<u8>,
    default_ttl: Duration,
}

impl CheckpointManager {
    pub fn new(secret: impl AsRef<str>, default_ttl_minutes: i64) -> Self {
        let secret: Vec<u8> = secret.as_ref().as_bytes().to_vec();
        Self {
            secret,
            default_ttl: Duration::minutes(default_ttl_minutes),
        }
    }

    pub fn create(
        &self,
        session: &mut Session,
        last_event_seq: i64,
        _reason: impl Into<String>,
    ) -> EngineResult<ResumableToken> {
        let expires_at = Utc::now() + self.default_ttl;
        let token = self.sign_token(session.id, last_event_seq, expires_at)?;

        session.status = SessionStatus::Checkpointed;
        session.updated_at = Utc::now();

        Ok(ResumableToken {
            token,
            session_id: session.id,
            last_event_seq,
            expires_at,
        })
    }

    /// Validate a token and return the checkpoint it references.
    pub fn validate(&self, token: &str) -> EngineResult<ResumableToken> {
        if self.secret.is_empty() {
            return Err(EngineError::Internal(
                "checkpoint signing secret cannot be empty".to_string(),
            ));
        }
        // Bound parsing work for untrusted resume tokens.
        if token.len() > 512 {
            return Err(EngineError::CheckpointNotFound(
                "malformed token".to_string(),
            ));
        }
        let parts: Vec<&str> = token.splitn(5, ':').collect();
        if parts.len() != 4 {
            return Err(EngineError::CheckpointNotFound(
                "malformed token".to_string(),
            ));
        }

        let session_id = Uuid::parse_str(parts[0])
            .map_err(|_| EngineError::CheckpointNotFound("invalid session id".to_string()))?;
        let last_event_seq: i64 = parts[1]
            .parse()
            .map_err(|_| EngineError::CheckpointNotFound("invalid seq".to_string()))?;
        let expires_at = DateTime::from_timestamp(
            parts[2]
                .parse()
                .map_err(|_| EngineError::CheckpointNotFound("invalid expiry".to_string()))?,
            0,
        )
        .ok_or_else(|| EngineError::CheckpointNotFound("invalid expiry".to_string()))?;

        let payload = format!(
            "{}:{}:{}",
            session_id,
            last_event_seq,
            expires_at.timestamp()
        );
        let signature = hex::decode(parts[3])
            .map_err(|_| EngineError::CheckpointNotFound("invalid signature".to_string()))?;
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.secret)
            .map_err(|e| EngineError::Internal(format!("invalid checkpoint secret: {e}")))?;
        mac.update(payload.as_bytes());
        if mac.verify_slice(&signature).is_err() {
            return Err(EngineError::CheckpointNotFound(
                "token signature mismatch".to_string(),
            ));
        }

        if Utc::now() > expires_at {
            return Err(EngineError::CheckpointNotFound("token expired".to_string()));
        }

        Ok(ResumableToken {
            token: token.to_string(),
            session_id,
            last_event_seq,
            expires_at,
        })
    }

    /// Resume a session from a validated token by rehydrating from the event log.
    pub async fn resume<E: EventStore>(
        &self,
        token: &ResumableToken,
        session: &mut Session,
        event_store: &E,
    ) -> EngineResult<()> {
        // `ResumableToken` is public and deserializable, so callers can
        // construct one without going through `validate`. Re-validate the
        // signed string here and trust only the authenticated fields.
        let validated = self.validate(&token.token)?;
        if validated.session_id != token.session_id
            || validated.last_event_seq != token.last_event_seq
            || validated.expires_at.timestamp() != token.expires_at.timestamp()
        {
            return Err(EngineError::CheckpointNotFound(
                "checkpoint fields do not match signed token".to_string(),
            ));
        }
        if session.id != validated.session_id {
            return Err(EngineError::Internal(
                "session id does not match token".to_string(),
            ));
        }

        let latest = event_store.latest_seq(session.id).await?;
        if let Some(seq) = latest {
            if seq < validated.last_event_seq {
                return Err(EngineError::CheckpointNotFound(
                    "event log does not reach checkpoint sequence".to_string(),
                ));
            }
        } else {
            return Err(EngineError::CheckpointNotFound(
                "no events for session".to_string(),
            ));
        }

        let token_fingerprint = checkpoint_token_fingerprint(&validated.token);
        let payload = serde_json::to_value(CheckpointRedeemedPayload {
            token_fingerprint: token_fingerprint.clone(),
            last_event_seq: validated.last_event_seq,
            expires_at: validated.expires_at,
        })?;
        event_store
            .redeem_checkpoint(session.id, &token_fingerprint, payload)
            .await?;

        session.status = SessionStatus::Active;
        session.updated_at = Utc::now();
        Ok(())
    }

    fn sign_token(
        &self,
        session_id: Uuid,
        last_event_seq: i64,
        expires_at: DateTime<Utc>,
    ) -> EngineResult<String> {
        if self.secret.is_empty() {
            return Err(EngineError::Internal(
                "checkpoint signing secret cannot be empty".to_string(),
            ));
        }
        let payload = format!(
            "{}:{}:{}",
            session_id,
            last_event_seq,
            expires_at.timestamp()
        );
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.secret)
            .map_err(|e| EngineError::Internal(format!("invalid checkpoint secret: {e}")))?;
        mac.update(payload.as_bytes());
        let sig = hex::encode(mac.finalize().into_bytes());
        Ok(format!("{}:{}", payload, sig))
    }
}

fn checkpoint_token_fingerprint(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EventStore, SessionEventType};
    use crate::in_memory_stores::InMemoryEventStore;
    use crate::models::ChatMessage;

    #[test]
    fn empty_checkpoint_secret_is_rejected() {
        let manager = CheckpointManager::new("", 60);
        let mut session = Session::new("test task", 1000).unwrap();
        assert!(manager.create(&mut session, 0, "test").is_err());
    }

    #[tokio::test]
    async fn resume_revalidates_caller_constructed_token_fields() {
        let manager = CheckpointManager::new("test-secret", 60);
        let store = InMemoryEventStore::default();
        let mut session = Session::new("test task", 1000).unwrap();
        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        let mut token = manager.create(&mut session, 1, "test").unwrap();
        token.last_event_seq = 0;

        let error = manager
            .resume(&token, &mut session, &store)
            .await
            .expect_err("unsigned token fields must not be trusted");
        assert!(matches!(error, EngineError::CheckpointNotFound(_)));
    }

    #[tokio::test]
    async fn checkpoint_create_validate_resume_round_trip() {
        let manager = CheckpointManager::new("test-secret", 60);
        let mut session = Session::new("test task", 1000).unwrap();
        let store = InMemoryEventStore::default();

        // Seed an event so latest_seq is non-zero.
        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({
                    "response_message": ChatMessage {
                        provider_state: None,
                        role: "assistant".to_string(),
                        content: Some(crate::MessageContent::text("hello".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    }
                }),
                Some(1),
            )
            .await
            .unwrap();

        let latest_seq = store.latest_seq(session.id).await.unwrap().unwrap_or(0);
        let token = manager
            .create(&mut session, latest_seq, "budget-exhausted")
            .unwrap();
        assert_eq!(session.status, SessionStatus::Checkpointed);

        let validated = manager.validate(&token.token).unwrap();
        assert_eq!(validated.session_id, session.id);
        assert_eq!(validated.last_event_seq, latest_seq);

        manager
            .resume(&validated, &mut session, &store)
            .await
            .unwrap();
        assert_eq!(session.status, SessionStatus::Active);

        let events = store.read_events(session.id, None, None).await.unwrap();
        let redeemed = events.last().unwrap();
        assert_eq!(redeemed.event_type, SessionEventType::CheckpointRedeemed);
        let payload: CheckpointRedeemedPayload =
            serde_json::from_value(redeemed.payload.clone()).unwrap();
        assert_eq!(
            payload.token_fingerprint,
            checkpoint_token_fingerprint(&token.token)
        );
        assert!(!redeemed.payload.to_string().contains(&token.token));
    }

    #[tokio::test]
    async fn checkpoint_rejects_token_reuse() {
        let manager = CheckpointManager::new("test-secret", 60);
        let mut session = Session::new("test task", 1000).unwrap();
        let store = InMemoryEventStore::default();
        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        let token = manager.create(&mut session, 1, "test").unwrap();

        manager.resume(&token, &mut session, &store).await.unwrap();
        let error = manager
            .resume(&token, &mut session, &store)
            .await
            .unwrap_err();

        assert!(matches!(error, EngineError::CheckpointAlreadyRedeemed));
    }

    #[tokio::test]
    async fn concurrent_checkpoint_resume_redeems_exactly_once() {
        let manager = CheckpointManager::new("test-secret", 60);
        let mut session = Session::new("test task", 1000).unwrap();
        let store = InMemoryEventStore::default();
        store
            .append_event(
                session.id,
                SessionEventType::LlmCall,
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        let token = manager.create(&mut session, 1, "test").unwrap();
        let session_id = session.id;
        let mut first_session = session.clone();
        let mut second_session = session;

        let (first, second) = tokio::join!(
            manager.resume(&token, &mut first_session, &store),
            manager.resume(&token, &mut second_session, &store),
        );

        assert_eq!(usize::from(first.is_ok()) + usize::from(second.is_ok()), 1);
        assert!(matches!(
            first.err().or_else(|| second.err()),
            Some(EngineError::CheckpointAlreadyRedeemed)
        ));
        let redemptions = store
            .read_events(session_id, None, None)
            .await
            .unwrap()
            .into_iter()
            .filter(|event| event.event_type == SessionEventType::CheckpointRedeemed)
            .count();
        assert_eq!(redemptions, 1);
    }

    #[test]
    fn checkpoint_rejects_tampered_token() {
        let manager = CheckpointManager::new("test-secret", 60);
        let mut session = Session::new("test task", 1000).unwrap();
        let token = manager.create(&mut session, 5, "budget-exhausted").unwrap();

        let tampered = format!("{}:tampered", token.token);
        assert!(manager.validate(&tampered).is_err());
    }

    #[test]
    fn checkpoint_rejects_expired_token() {
        let manager = CheckpointManager::new("test-secret", -1);
        let mut session = Session::new("test task", 1000).unwrap();
        let token = manager.create(&mut session, 5, "budget-exhausted").unwrap();

        assert!(manager.validate(&token.token).is_err());
    }
}
