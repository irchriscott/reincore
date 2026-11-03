//! Live delivery for the durable session event log.
//!
//! The event database remains the source of truth. This module provides an
//! in-process notification channel plus a replaying stream that repairs
//! missed or lagged notifications from [`EventStore`]. Cloud deployments can
//! replace the notification side with NATS while keeping the same replay and
//! browser-facing SSE contract.

use crate::{EngineError, EngineResult, EventStore, SessionEvent};
use futures::Stream;
use std::time::Duration;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Default number of committed notifications retained by the in-process bus.
/// Lag is harmless: [`live_session_events`] repairs it from the event store.
const DEFAULT_CHANNEL_CAPACITY: usize = 2_048;
const REPLAY_PAGE_SIZE: usize = 500;
/// Core NATS is intentionally ephemeral. This periodic source-of-truth check
/// repairs a final event missed during a broker reconnect even when no later
/// notification arrives to reveal the sequence gap.
#[cfg(not(test))]
const DURABLE_REPAIR_INTERVAL: Duration = Duration::from_secs(5);
#[cfg(test)]
const DURABLE_REPAIR_INTERVAL: Duration = Duration::from_millis(20);

/// Lightweight post-commit notification bus for session events.
#[derive(Clone, Debug)]
pub struct SessionEventHub {
    sender: broadcast::Sender<SessionEvent>,
}

impl Default for SessionEventHub {
    fn default() -> Self {
        Self::new(DEFAULT_CHANNEL_CAPACITY)
    }
}

impl SessionEventHub {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self { sender }
    }

    /// Publish an event after its storage transaction has committed.
    ///
    /// Having no active subscribers is expected and is not an error.
    pub fn publish(&self, event: SessionEvent) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.sender.subscribe()
    }
}

/// Return a replaying live stream for one session.
///
/// The receiver is created before the stream begins its database catch-up,
/// closing the subscribe/replay race. Sequence gaps and broadcast lag switch
/// the stream back into catch-up mode; duplicate notifications are discarded.
pub fn live_session_events<S>(
    store: std::sync::Arc<S>,
    hub: SessionEventHub,
    session_id: Uuid,
    after_seq: i64,
) -> impl Stream<Item = EngineResult<SessionEvent>>
where
    S: EventStore + 'static,
{
    let mut receiver = hub.subscribe();

    async_stream::stream! {
        let mut cursor = after_seq.max(0);
        let mut catch_up = true;
        let first_repair = tokio::time::Instant::now() + DURABLE_REPAIR_INTERVAL;
        let mut repair_tick = tokio::time::interval_at(first_repair, DURABLE_REPAIR_INTERVAL);

        loop {
            if catch_up {
                let page = match store
                    .read_event_page(
                        session_id,
                        Some(cursor.saturating_add(1)),
                        None,
                        REPLAY_PAGE_SIZE,
                    )
                    .await
                {
                    Ok(page) => page,
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                };

                let page_was_full = page.len() == REPLAY_PAGE_SIZE;
                for event in page {
                    if event.seq > cursor {
                        cursor = event.seq;
                        yield Ok(event);
                    }
                }

                if page_was_full {
                    continue;
                }
                catch_up = false;
            }

            tokio::select! {
                received = receiver.recv() => match received {
                    Ok(event) if event.session_id != session_id || event.seq <= cursor => {}
                    Ok(event) if event.seq == cursor.saturating_add(1) => {
                        cursor = event.seq;
                        yield Ok(crate::compact_session_event(event));
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {
                        // The durable log contains the missing notification(s).
                        catch_up = true;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        yield Err(EngineError::Storage(
                            "session event notification channel closed".to_string(),
                        ));
                        return;
                    }
                },
                _ = repair_tick.tick() => {
                    catch_up = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InMemoryEventStore, SessionEventType};
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn stream_replays_then_delivers_live_events() {
        let store = std::sync::Arc::new(InMemoryEventStore::default());
        let hub = SessionEventHub::new(8);
        let session_id = Uuid::new_v4();
        let first = store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                json!({"n": 1}),
                None,
            )
            .await
            .unwrap();

        let stream = live_session_events(store.clone(), hub.clone(), session_id, 0);
        tokio::pin!(stream);
        assert_eq!(stream.next().await.unwrap().unwrap().seq, first.seq);

        let second = store
            .append_event(
                session_id,
                SessionEventType::ToolResult,
                json!({"n": 2}),
                None,
            )
            .await
            .unwrap();
        hub.publish(second.clone());

        assert_eq!(stream.next().await.unwrap().unwrap().seq, second.seq);
    }

    #[tokio::test]
    async fn sequence_gap_is_repaired_from_the_store() {
        let store = std::sync::Arc::new(InMemoryEventStore::default());
        let hub = SessionEventHub::new(8);
        let session_id = Uuid::new_v4();
        let first = store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                json!({"n": 1}),
                None,
            )
            .await
            .unwrap();

        let stream = live_session_events(store.clone(), hub.clone(), session_id, first.seq);
        tokio::pin!(stream);

        let second = store
            .append_event(
                session_id,
                SessionEventType::ToolResult,
                json!({"n": 2}),
                None,
            )
            .await
            .unwrap();
        let third = store
            .append_event(session_id, SessionEventType::LlmCall, json!({"n": 3}), None)
            .await
            .unwrap();
        hub.publish(third.clone());

        assert_eq!(stream.next().await.unwrap().unwrap().seq, second.seq);
        assert_eq!(stream.next().await.unwrap().unwrap().seq, third.seq);
    }

    #[tokio::test]
    async fn periodic_repair_recovers_a_missed_tail_notification() {
        let store = std::sync::Arc::new(InMemoryEventStore::default());
        let hub = SessionEventHub::new(8);
        let session_id = Uuid::new_v4();
        let first = store
            .append_event(
                session_id,
                SessionEventType::ToolCall,
                json!({"n": 1}),
                None,
            )
            .await
            .unwrap();

        let stream = live_session_events(store.clone(), hub.clone(), session_id, first.seq);
        tokio::pin!(stream);

        let second = store
            .append_event(
                session_id,
                SessionEventType::ToolResult,
                json!({"n": 2}),
                None,
            )
            .await
            .unwrap();
        hub.publish(second.clone());
        assert_eq!(stream.next().await.unwrap().unwrap().seq, second.seq);

        let third = store
            .append_event(session_id, SessionEventType::LlmCall, json!({"n": 3}), None)
            .await
            .unwrap();
        let repaired = tokio::time::timeout(Duration::from_millis(200), stream.next())
            .await
            .expect("periodic durable replay should wake the stream")
            .unwrap()
            .unwrap();

        assert_eq!(repaired.seq, third.seq);
    }
}
