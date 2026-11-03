//! Projects persistence for the local engine.

use anyhow::Result;
use chrono::Utc;
use engine_core::error::{EngineError, EngineResult};
use rusqlite::{params, OptionalExtension};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use super::{LocalDatabase, LocalProject};

impl LocalDatabase {
    /// Import the path-based project information recorded by older releases
    /// into the registry. This is deliberately idempotent and runs after the
    /// schema migrations, so existing users do not have to recreate projects.
    pub(super) fn backfill_local_projects(&self) -> Result<()> {
        let legacy_projects = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT DISTINCT project_name, project_path FROM sessions
                 WHERE project_path IS NOT NULL AND TRIM(project_path) <> ''",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, Option<String>>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        for (name, path) in legacy_projects {
            if self.get_local_project_by_path(&path)?.is_some() {
                continue;
            }
            let fallback_name = Path::new(&path)
                .file_name()
                .and_then(|part| part.to_str())
                .filter(|part| !part.is_empty())
                .unwrap_or("Project");
            self.create_local_project_internal(name.as_deref().unwrap_or(fallback_name), &path)?;
        }
        Ok(())
    }

    /// Register a new logical project and attach its first working directory.
    pub fn create_local_project(
        &self,
        name: &str,
        project_path: &str,
    ) -> EngineResult<LocalProject> {
        let (display_path, _) = normalize_project_path(project_path)?;
        if !Path::new(&display_path).is_dir() {
            return Err(EngineError::Storage(format!(
                "project path is not an existing directory: {display_path}"
            )));
        }
        self.create_local_project_internal(name, &display_path)
    }

    pub(super) fn create_local_project_internal(
        &self,
        name: &str,
        project_path: &str,
    ) -> EngineResult<LocalProject> {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 128 {
            return Err(EngineError::Storage(
                "project name must contain between 1 and 128 characters".to_string(),
            ));
        }
        let (display_path, normalized_path) = normalize_project_path(project_path)?;
        if self.get_local_project_by_path(&display_path)?.is_some() {
            return Err(EngineError::Storage(
                "working directory is already assigned to a project".to_string(),
            ));
        }

        let project_id = Uuid::new_v4();
        let now = Utc::now();
        {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn
                .transaction()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "INSERT INTO local_projects (id, name, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?3)",
                params![project_id.to_string(), name, now.to_rfc3339()],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "INSERT INTO local_project_directories
                    (project_id, project_path, normalized_path, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    project_id.to_string(),
                    display_path,
                    normalized_path,
                    now.to_rfc3339()
                ],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.commit()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
        }
        self.attribute_sessions_for_directory(project_id, name, &display_path)?;
        self.get_local_project(project_id)?.ok_or_else(|| {
            EngineError::Storage("newly created local project was not found".to_string())
        })
    }

    /// Attach another working directory to an existing logical project.
    pub fn assign_local_project_directory(
        &self,
        project_id: Uuid,
        project_path: &str,
    ) -> EngineResult<LocalProject> {
        let (display_path, normalized_path) = normalize_project_path(project_path)?;
        if !Path::new(&display_path).is_dir() {
            return Err(EngineError::Storage(format!(
                "project path is not an existing directory: {display_path}"
            )));
        }

        if let Some(existing) = self.get_local_project_by_path(&display_path)? {
            if existing.id == project_id {
                return Ok(existing);
            }
            return Err(EngineError::Storage(format!(
                "working directory is already assigned to project '{}'",
                existing.name
            )));
        }

        let project = self
            .get_local_project(project_id)?
            .ok_or_else(|| EngineError::Storage("local project was not found".to_string()))?;
        let now = Utc::now();
        {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn
                .transaction()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "INSERT INTO local_project_directories
                    (project_id, project_path, normalized_path, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    project_id.to_string(),
                    display_path,
                    normalized_path,
                    now.to_rfc3339()
                ],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "UPDATE local_projects SET updated_at = ?2 WHERE id = ?1",
                params![project_id.to_string(), now.to_rfc3339()],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.commit()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
        }
        self.attribute_sessions_for_directory(project_id, &project.name, &display_path)?;
        self.get_local_project(project_id)?
            .ok_or_else(|| EngineError::Storage("updated local project was not found".to_string()))
    }

    pub(super) fn attribute_sessions_for_directory(
        &self,
        project_id: Uuid,
        project_name: &str,
        project_path: &str,
    ) -> EngineResult<()> {
        let (_, target_key) = normalize_project_path(project_path)?;
        let recorded_paths = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT project_path FROM sessions
                     WHERE project_path IS NOT NULL AND TRIM(project_path) <> ''",
                )
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|error| EngineError::Storage(error.to_string()))?
        };

        let matching_paths: Vec<String> = recorded_paths
            .into_iter()
            .filter(|path| {
                normalize_project_path(path)
                    .map(|(_, key)| key == target_key)
                    .unwrap_or(false)
            })
            .collect();
        let conn = self.conn.lock().unwrap();
        for path in matching_paths {
            conn.execute(
                "UPDATE sessions SET project_id = ?1, project_name = ?2
                 WHERE project_path = ?3",
                params![project_id.to_string(), project_name, path],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
        }
        Ok(())
    }

    pub fn list_local_projects(&self) -> EngineResult<Vec<LocalProject>> {
        let project_rows = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, name, created_at, updated_at
                     FROM local_projects ORDER BY LOWER(name), created_at",
                )
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|error| EngineError::Storage(error.to_string()))?
        };

        project_rows
            .into_iter()
            .map(|(id, name, created_at, updated_at)| {
                let id = Uuid::parse_str(&id)
                    .map_err(|error| EngineError::Storage(error.to_string()))?;
                let directories = {
                    let conn = self.conn.lock().unwrap();
                    let mut stmt = conn
                        .prepare(
                            "SELECT project_path FROM local_project_directories
                             WHERE project_id = ?1 ORDER BY project_path",
                        )
                        .map_err(|error| EngineError::Storage(error.to_string()))?;
                    let rows = stmt
                        .query_map([id.to_string()], |row| row.get::<_, String>(0))
                        .map_err(|error| EngineError::Storage(error.to_string()))?;
                    rows.collect::<rusqlite::Result<Vec<_>>>()
                        .map_err(|error| EngineError::Storage(error.to_string()))?
                };
                Ok(LocalProject {
                    id,
                    name,
                    directories,
                    created_at: created_at.parse().unwrap_or_else(|_| Utc::now()),
                    updated_at: updated_at.parse().unwrap_or_else(|_| Utc::now()),
                })
            })
            .collect()
    }

    pub fn get_local_project(&self, id: Uuid) -> EngineResult<Option<LocalProject>> {
        Ok(self
            .list_local_projects()?
            .into_iter()
            .find(|project| project.id == id))
    }

    pub fn get_local_project_by_path(
        &self,
        project_path: &str,
    ) -> EngineResult<Option<LocalProject>> {
        let (_, normalized_path) = normalize_project_path(project_path)?;
        let project_id = {
            let conn = self.read_conn();
            conn.query_row(
                "SELECT project_id FROM local_project_directories
                 WHERE normalized_path = ?1",
                [normalized_path],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| EngineError::Storage(error.to_string()))?
        };
        match project_id {
            Some(id) => {
                let id = Uuid::parse_str(&id)
                    .map_err(|error| EngineError::Storage(error.to_string()))?;
                self.get_local_project(id)
            }
            None => Ok(None),
        }
    }

    pub async fn list_local_projects_async(&self) -> EngineResult<Vec<LocalProject>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.list_local_projects())
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    pub async fn get_local_project_by_path_async(
        &self,
        project_path: String,
    ) -> EngineResult<Option<LocalProject>> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.get_local_project_by_path(&project_path))
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    pub async fn create_local_project_async(
        &self,
        name: String,
        project_path: String,
    ) -> EngineResult<LocalProject> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.create_local_project(&name, &project_path))
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    pub async fn assign_local_project_directory_async(
        &self,
        project_id: Uuid,
        project_path: String,
    ) -> EngineResult<LocalProject> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || {
            db.assign_local_project_directory(project_id, &project_path)
        })
        .await
        .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }

    /// Rename a local project. Updates `updated_at` and rewrites `project_name`
    /// on any sessions that reference the project so the UI stays consistent.
    pub fn rename_local_project(&self, project_id: Uuid, name: &str) -> EngineResult<LocalProject> {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 128 {
            return Err(EngineError::Storage(
                "project name must contain between 1 and 128 characters".to_string(),
            ));
        }

        let project = self
            .get_local_project(project_id)?
            .ok_or_else(|| EngineError::Storage("local project was not found".to_string()))?;
        if project.name == name {
            return Ok(project);
        }

        let now = Utc::now();
        {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn
                .transaction()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "UPDATE local_projects SET name = ?2, updated_at = ?3 WHERE id = ?1",
                params![project_id.to_string(), name, now.to_rfc3339()],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.execute(
                "UPDATE sessions SET project_name = ?2 WHERE project_id = ?1",
                params![project_id.to_string(), name],
            )
            .map_err(|error| EngineError::Storage(error.to_string()))?;
            tx.commit()
                .map_err(|error| EngineError::Storage(error.to_string()))?;
        }

        self.get_local_project(project_id)?
            .ok_or_else(|| EngineError::Storage("renamed local project was not found".to_string()))
    }

    pub async fn rename_local_project_async(
        &self,
        project_id: Uuid,
        name: String,
    ) -> EngineResult<LocalProject> {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.rename_local_project(project_id, &name))
            .await
            .map_err(|error| EngineError::Storage(format!("sqlite task join error: {error}")))?
    }
}

/// Produce a stable, absolute display path and a comparison key. Existing
/// directories are canonicalized (resolving `.`/`..` and symlinks); missing
/// historical paths still receive a deterministic lexical key for migration.
pub(crate) fn normalize_project_path(raw: &str) -> EngineResult<(String, String)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(EngineError::Storage(
            "project path must not be empty".to_string(),
        ));
    }
    let input = PathBuf::from(raw);
    let absolute = if input.is_absolute() {
        input
    } else {
        std::env::current_dir()
            .map_err(|error| EngineError::Storage(error.to_string()))?
            .join(input)
    };
    let resolved = std::fs::canonicalize(&absolute).unwrap_or(absolute);
    let mut display = resolved.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        if let Some(without_prefix) = display.strip_prefix(r"\\?\UNC\") {
            display = format!(r"\\{}", without_prefix);
        } else if let Some(without_prefix) = display.strip_prefix(r"\\?\") {
            display = without_prefix.to_string();
        }
    }
    while display.len() > 1 && (display.ends_with('/') || display.ends_with('\\')) {
        if display.len() == 3 && display.as_bytes().get(1) == Some(&b':') {
            break;
        }
        display.pop();
    }
    let mut key = display.replace('\\', "/");
    #[cfg(windows)]
    {
        key = key.to_lowercase();
    }
    Ok((display, key))
}
