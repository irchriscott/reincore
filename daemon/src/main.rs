// ReinCore daemon + CLI binary entry point.
//
// All logic lives in the `reincore` library crate so the desktop app can
// link against the same code. The binary just parses CLI args and dispatches
// to the library's handlers - behavior is identical to before the lib+bin
// split.

use anyhow::Result;
use clap::Parser;

use reincore::budget::handle_budget;
use reincore::cli::{Cli, Commands, DaemonConfig, LaunchArgs};
use reincore::daemon::{show_status, start_daemon, stop_daemon};
use reincore::launch::handle_launch;
use reincore::provider::handle_provider_command;
use reincore::session::{
    handle_compress, handle_config, handle_export_timeline, handle_pause, handle_resume,
    handle_sessions,
};
use reincore::telemetry_cmd::handle_telemetry_command;

#[tokio::main]
async fn main() -> Result<()> {
    reincore::local_config::install_privacy_policy();
    let cli = Cli::parse();

    match cli.command {
        Commands::Data { command } => {
            use reincore::{cli::DataCommand, db::data_lifecycle as data};
            match command {
                DataCommand::Backup { source, output } => data::backup(&source, &output)?,
                DataCommand::Restore { source, output } => data::restore(&source, &output)?,
                DataCommand::Verify { source } => data::verify(&source)?,
                DataCommand::Retain {
                    source,
                    before,
                    retention_days,
                    limit,
                    apply,
                    offline: _,
                    backup,
                } => {
                    anyhow::ensure!((1..=100).contains(&limit), "limit must be 1 through 100");
                    let mut privacy = engine_core::privacy::current_policy();
                    if let Some(days) = retention_days {
                        privacy.retention_days = days;
                    }
                    let before =
                        before.unwrap_or_else(|| privacy.retention_cutoff(chrono::Utc::now()));
                    anyhow::ensure!(
                        before < chrono::Utc::now(),
                        "retention cutoff must be in the past"
                    );
                    if apply {
                        data::backup(&source, backup.as_deref().expect("clap requires backup"))?;
                    }
                    let ids = data::retain(&source, before, limit, apply)?;
                    println!(
                        "{}",
                        serde_json::json!({"applied": apply, "session_ids": ids, "ledger_preserved": true})
                    );
                }
            }
            Ok(())
        }

        Commands::Mcp { command } => reincore::mcp_config::command(&command).await,
        Commands::Start {
            provider,
            mode,
            host,
            port,
            db_path,
            path,
            verbose,
            cloud,
            model,
        } => {
            start_daemon(
                host, port, db_path, verbose, provider, mode, path, cloud, model, None, None,
            )
            .await
        }
        Commands::Budget {
            scope,
            key,
            limit,
            history,
            usage,
            db_path,
        } => handle_budget(scope, key, limit, history, usage, db_path).await,
        Commands::Status => {
            let config = DaemonConfig::load();
            show_status(config.host, config.port).await
        }
        Commands::Stop => {
            let config = DaemonConfig::load();
            stop_daemon(config.host, config.port).await
        }
        Commands::Compact {
            session,
            mode,
            json,
            db_path,
        } => handle_compress(session, mode, json, db_path).await,
        Commands::Config {
            compaction,
            compaction_mode,
            db_path,
        } => handle_config(compaction, compaction_mode, db_path).await,
        Commands::Pause { session, reason } => {
            let config = DaemonConfig::load();
            handle_pause(session, reason, config.host, config.port).await
        }
        Commands::Resume { token } => {
            let config = DaemonConfig::load();
            handle_resume(token, config.host, config.port).await
        }
        Commands::Sessions { path, checkpoints } => {
            let config = DaemonConfig::load();
            handle_sessions(path, checkpoints, config.host, config.port).await
        }
        Commands::Export { session, output } => {
            let config = DaemonConfig::load();
            handle_export_timeline(session, output, config.host, config.port).await
        }
        Commands::Provider { command } => handle_provider_command(command).await,
        Commands::Launch {
            app,
            provider,
            mode,
            host,
            port,
            db_path,
            path,
            verbose,
            restore,
            cloud,
            model,
        } => {
            handle_launch(LaunchArgs {
                app,
                provider,
                mode,
                host,
                port,
                db_path,
                path,
                verbose,
                restore,
                cloud,
                model,
            })
            .await
        }
        Commands::Setup {
            app,
            provider,
            mode,
            host,
            port,
        } => {
            reincore::launch::handle_setup(reincore::launch::SetupArgs {
                app,
                provider,
                mode,
                host,
                port,
            })
            .await
        }
        Commands::Migrate { db_path } => {
            let db = reincore::db::LocalDatabase::open(&db_path)?;
            let summary = db.run_migrations()?;
            println!("{}", summary);
            Ok(())
        }
        Commands::Restore { app } => match reincore::launch::restore_profile_for_app(&app)? {
            reincore::launch::ProfileStatus::Handled => {
                println!("Restored '{}' to its original configuration.", app);
                Ok(())
            }
            reincore::launch::ProfileStatus::Unsupported => {
                println!(
                    "'{}' has no persistent reincore profile to restore (nothing to do).",
                    app
                );
                Ok(())
            }
        },
        Commands::Logout { provider } => reincore::launch::handle_logout(&provider),
        Commands::Routing {
            mode,
            provider_strategy,
            minimum_quality,
            allow_upgrade,
            allow_degradation,
        } => reincore::config_cmd::handle_routing(
            mode,
            provider_strategy,
            minimum_quality,
            allow_upgrade,
            allow_degradation,
        ),
        Commands::Model {
            provider,
            standard,
            lite,
            pro,
            ultra,
            routing_mode,
            routing_minimum_quality,
        } => reincore::config_cmd::handle_model(
            &provider,
            standard,
            lite,
            pro,
            ultra,
            routing_mode,
            routing_minimum_quality,
        ),
        Commands::Telemetry { command } => handle_telemetry_command(command).await,
    }
}
