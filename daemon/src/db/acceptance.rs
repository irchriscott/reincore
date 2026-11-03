//! Opt-in developer-machine acceptance evidence. Never uses the user's daemon or DB.
use super::*;
use axum::{routing::get, Router};
use engine_core::{Session, SessionStatus};
use std::time::{Duration, Instant};

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let position = (values.len() - 1) as f64 * p;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * (position - lo as f64)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "manual isolated load/outage/restore run; writes JSON to REINCORE_ACCEPTANCE_REPORT"]
async fn local_acceptance_run() {
    let report_path = std::env::var("REINCORE_ACCEPTANCE_REPORT").expect("report path required");
    let mut report_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)
        .unwrap();
    let directory = std::env::temp_dir().join(format!("reincore-acceptance-{}", Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("source.db");
    let snapshot = directory.join("snapshot.db");
    let restored = directory.join("restored.db");
    let db = LocalDatabase::open(&source).unwrap();

    let seed_started = Instant::now();
    {
        let mut conn = db.conn.lock().unwrap();
        let tx = conn.transaction().unwrap();
        for i in 1..=10_000 {
            let mut session = Session::new(format!("acceptance {i}"), 100).unwrap();
            session.id = Uuid::from_u128(i);
            session.status = SessionStatus::Completed;
            session.updated_at = "2020-01-01T00:00:00Z".parse().unwrap();
            LocalDatabase::insert_session_on(&tx, &session).unwrap();
            tx.execute(
                "WITH RECURSIVE n(x) AS (
                    SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100
                )
                INSERT INTO session_events(session_id,seq,event_type,payload,created_at)
                SELECT ?1,x,'llm_call','{}','2020-01-01T00:00:00+00:00' FROM n",
                [session.id.to_string()],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }

    let seed_ms = ms(seed_started);
    let state = crate::app::AppState::new_with_project_path(db.clone(), None);
    let router = Router::new()
        .route("/v1/sessions", get(crate::http::list_sessions))
        .route(
            "/v1/sessions/:id/events/stream",
            get(crate::http::stream_session_events),
        )
        .route("/readyz", get(crate::http::ready))
        .route("/livez", get(engine_core::resilience::live))
        .route(
            "/held",
            get(|| async {
                axum::body::Body::from_stream(futures::stream::pending::<
                    Result<axum::body::Bytes, std::io::Error>,
                >())
            }),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            engine_core::resilience::Admission::new(32),
            engine_core::resilience::admit,
        ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
            .unwrap()
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let mut profiles = Vec::new();
    for concurrency in [1, 16, 64] {
        for _ in 0..20 {
            client
                .get(format!("{base}/v1/sessions?limit=50"))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .bytes()
                .await
                .unwrap();
        }
        let started = Instant::now();
        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..concurrency {
            let client = client.clone();
            let base = base.clone();
            tasks.spawn(async move {
                let mut samples = Vec::new();
                let mut rejected = 0;
                for _ in 0..20 {
                    let now = Instant::now();
                    let response = client
                        .get(format!("{base}/v1/sessions?limit=50"))
                        .send()
                        .await
                        .unwrap();
                    if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                        assert_eq!(response.headers()["retry-after"], "1");
                        rejected += 1;
                        response.bytes().await.unwrap();
                    } else {
                        let value: serde_json::Value =
                            response.error_for_status().unwrap().json().await.unwrap();
                        assert_eq!(value["sessions"].as_array().unwrap().len(), 50);
                        samples.push(ms(now));
                    }
                }
                (samples, rejected)
            });
        }
        let mut samples = Vec::new();
        let mut rejected = 0;
        while let Some(task) = tasks.join_next().await {
            let (part, shed) = task.unwrap();
            samples.extend(part);
            rejected += shed;
        }
        samples.sort_by(f64::total_cmp);
        assert!(!samples.is_empty());
        profiles.push(serde_json::json!({
            "concurrency": concurrency,
            "requests": concurrency * 20,
            "successes": samples.len(),
            "shed": rejected,
            "elapsed_ms": ms(started),
            "p95_ms": percentile(&samples, 0.95),
            "p99_ms": percentile(&samples, 0.99),
        }));
    }
    // Reconnect through the actual SSE handler using the durable event cursor.
    let mut replay = client
        .get(format!(
            "{base}/v1/sessions/{}/events/stream",
            Uuid::from_u128(1)
        ))
        .header("Last-Event-ID", "90")
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();

    let mut pending = String::new();
    let mut replayed = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while replayed.len() < 10 {
            let chunk = replay.chunk().await.unwrap().expect("replay ended early");
            pending.push_str(std::str::from_utf8(&chunk).unwrap());
            assert!(pending.len() < 65536);
            while let Some(end) = pending.find("\n\n") {
                let frame: String = pending.drain(..end + 2).collect();
                for line in frame.lines() {
                    if let Some(id) = line.strip_prefix("id:") {
                        replayed.push(id.trim().parse::<i64>().unwrap());
                    }
                }
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(replayed, (91..=100).collect::<Vec<_>>());
    drop(replay);
    // Force saturation with response bodies held open, not merely simultaneous headers.
    let mut held = Vec::new();
    for _ in 0..32 {
        held.push(
            client
                .get(format!("{base}/held"))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap(),
        );
    }
    let shed_start = Instant::now();
    assert_eq!(
        client
            .get(format!("{base}/v1/sessions?limit=50"))
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
    let shed_ms = ms(shed_start);
    assert_eq!(
        client
            .get(format!("{base}/livez"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .get(format!("{base}/readyz"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client
            .get(format!("{base}/v1/sessions?limit=50"))
            .send()
            .await
            .unwrap()
            .status()
            == 200
        {
            break;
        }
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    db.conn
        .lock()
        .unwrap()
        .execute_batch("ALTER TABLE sessions RENAME TO unavailable_sessions")
        .unwrap();
    let outage_start = Instant::now();
    assert_eq!(
        client
            .get(format!("{base}/readyz"))
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
    let outage_ms = ms(outage_start);
    assert_eq!(
        client
            .get(format!("{base}/livez"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    db.conn
        .lock()
        .unwrap()
        .execute_batch("ALTER TABLE unavailable_sessions RENAME TO sessions")
        .unwrap();
    let recovery_start = Instant::now();
    assert_eq!(
        client
            .get(format!("{base}/readyz"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );

    let readiness_recovery_ms = ms(recovery_start);
    stop.send(()).unwrap();
    server.await.unwrap();

    let backup_start = Instant::now();

    super::data_lifecycle::backup(&source, &snapshot).unwrap();
    let backup_ms = ms(backup_start);
    let restore_start = Instant::now();

    super::data_lifecycle::restore(&snapshot, &restored).unwrap();
    let restored_db = LocalDatabase::open(&restored).unwrap();
    let restore_ms = ms(restore_start);

    for database in [&db, &restored_db] {
        let conn = database.conn.lock().unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            10_000
        );
        let aggregate: (i64, i64) = conn
            .query_row("SELECT COUNT(*),SUM(seq) FROM session_events", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(aggregate, (1_000_000, 50_500_000));
    }
    assert!(restored_db.ready().await);
    let bytes = std::fs::metadata(&snapshot).unwrap().len();
    let report = serde_json::json!({
        "profile": "developer-local-uniform",
        "reference_acceptance": false,
        "sessions": 10000,
        "events": 1000000,
        "payload": "empty JSON; no skew, attachments or provider traffic",
        "seed_ms": seed_ms,
        "snapshot_bytes": bytes,
        "load": profiles,
        "forced_shed_ms": shed_ms,
        "schema_outage_detection_ms": outage_ms,
        "readiness_recovery_ms": readiness_recovery_ms,
        "backup_ms": backup_ms,
        "restore_open_ms": restore_ms,
        "restored_counts_and_sequence_sum_match": true,
        "sse_replayed_ids": replayed,
    });

    use std::io::Write;
    report_file
        .write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
        .unwrap();
    report_file.sync_all().unwrap();
    println!("{report}");
    drop(restored_db);
    drop(db);
    drop(client);
    std::fs::remove_dir_all(&directory).unwrap();
}
