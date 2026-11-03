//! Fail-fast admission; permits cover response bodies, including SSE.
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct Admission {
    permits: Arc<Semaphore>,
}

impl Admission {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            permits: Arc::new(Semaphore::new(capacity)),
        }
    }

    pub fn from_env(default: usize) -> anyhow::Result<Self> {
        let capacity = match std::env::var("REINCORE_MAX_IN_FLIGHT") {
            Ok(value) => value.parse::<usize>().map_err(|_| {
                anyhow::anyhow!("REINCORE_MAX_IN_FLIGHT must be an integer from 1 to 4096")
            })?,
            Err(std::env::VarError::NotPresent) => default,
            Err(_) => anyhow::bail!("invalid REINCORE_MAX_IN_FLIGHT"),
        };
        anyhow::ensure!(
            (1..=4096).contains(&capacity),
            "REINCORE_MAX_IN_FLIGHT must be from 1 to 4096"
        );
        Ok(Self::new(capacity))
    }
}

pub async fn admit(State(state): State<Admission>, request: Request, next: Next) -> Response {
    if matches!(request.uri().path(), "/health" | "/livez" | "/readyz") {
        return next.run(request).await;
    }

    let Ok(permit) = state.permits.clone().try_acquire_owned() else {
        crate::operational::increment(crate::operational::Counter::RequestsShed, 1);
        return (StatusCode::SERVICE_UNAVAILABLE, [("retry-after", "1"), ("cache-control", "no-store")], Json(serde_json::json!({"error":{"code":"overloaded","message":"Request capacity exhausted; retry with backoff"}}))).into_response();
    };

    let response = next.run(request).await;
    let (parts, body) = response.into_parts();
    if body.is_end_stream() {
        drop(permit);
        return Response::from_parts(parts, body);
    }
    Response::from_parts(
        parts,
        Body::new(PermitBody {
            body,
            permit: Some(permit),
        }),
    )
}

struct PermitBody {
    body: Body,
    permit: Option<OwnedSemaphorePermit>,
}

impl HttpBody for PermitBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let result = Pin::new(&mut self.body).poll_frame(cx);
        if matches!(result, Poll::Ready(None) | Poll::Ready(Some(Err(_))))
            || self.body.is_end_stream()
        {
            self.permit.take();
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

pub async fn live() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status":"alive"}))
}

pub fn readiness(database: bool, relay: Option<bool>) -> Response {
    let ready = database && relay.unwrap_or(true);
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        [("cache-control", "no-store")],
        Json(serde_json::json!({
            "status": if ready { "ready" } else { "not_ready" },
            "database": database,
            "relay": relay
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stream_permit_survives_headers_and_releases_on_disconnect() {
        let permits = Arc::new(Semaphore::new(1));
        let body = PermitBody {
            body: Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>()),
            permit: Some(permits.clone().acquire_owned().await.unwrap()),
        };
        assert!(permits.clone().try_acquire_owned().is_err());
        drop(body);
        assert!(permits.clone().try_acquire_owned().is_ok());
    }

    #[tokio::test]
    async fn completed_body_releases_capacity_and_outages_are_not_ready() {
        use futures::future::poll_fn;
        let permits = Arc::new(Semaphore::new(1));
        let mut body = PermitBody {
            body: Body::empty(),
            permit: Some(permits.clone().acquire_owned().await.unwrap()),
        };
        assert!(poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .is_none());
        assert_eq!(permits.available_permits(), 1);
        assert_eq!(
            readiness(false, None).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            readiness(true, Some(false)).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(readiness(true, None).status(), StatusCode::OK);
    }
}

#[cfg(test)]
mod route_tests {
    use super::*;
    use axum::{routing::get, Router};
    use tower::ServiceExt;

    #[tokio::test]
    async fn empty_response_releases_capacity_without_polling() {
        let app = Router::new()
            .route("/empty", get(|| async { StatusCode::NO_CONTENT }))
            .layer(axum::middleware::from_fn_with_state(
                Admission::new(1),
                admit,
            ));
        let request = || {
            Request::builder()
                .uri("/empty")
                .body(Body::empty())
                .unwrap()
        };
        let held = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(held.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            app.oneshot(request()).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );
    }

    #[tokio::test]
    async fn overload_never_queues_and_probes_survive_saturated_streams() {
        let app = Router::new()
            .route(
                "/stream",
                get(|| async {
                    Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>())
                }),
            )
            .route("/livez", get(live))
            .layer(axum::middleware::from_fn_with_state(
                Admission::new(1),
                admit,
            ));
        let request = |path| Request::builder().uri(path).body(Body::empty()).unwrap();
        let held = app.clone().oneshot(request("/stream")).await.unwrap();
        let denied = app.clone().oneshot(request("/stream")).await.unwrap();
        assert_eq!(denied.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(denied.headers()["retry-after"], "1");
        assert_eq!(
            app.clone()
                .oneshot(request("/livez"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        drop(held);
        assert_eq!(
            app.oneshot(request("/stream")).await.unwrap().status(),
            StatusCode::OK
        );
    }
}
