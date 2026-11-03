//! Versioned schema migrations shared between SQLite and Postgres backends.
//!
//! This module provides a tiny, dependency-free migration framework.  Migrations
//! are declared as a numbered sequence of `Migration` structs.  Each migration
//! contains a list of `SchemaChange` operations (`create_table`, `add_field`,
//! etc.).  A dialect-specific runner translates those operations into SQL and
//! applies them inside a transaction, recording progress in a
//! `schema_migrations` table.
//!
//! The DSL is intentionally simple so engineers can add a new migration by
//! editing a single Rust file rather than maintaining separate SQL scripts for
//! each database.

use std::collections::HashMap;

/// Supported column / field data types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    /// SQLite `TEXT` / Postgres `TEXT`.
    Text,
    /// SQLite `INTEGER` / Postgres `BIGINT`.
    Integer,
    /// SQLite `REAL` / Postgres `DOUBLE PRECISION`.
    Float,
    /// SQLite `INTEGER` (0/1) / Postgres `BOOLEAN`.
    Boolean,
    /// SQLite `TEXT` (RFC3339) / Postgres `TIMESTAMPTZ`.
    Timestamp,
    /// SQLite `BLOB` / Postgres `BYTEA`.
    Blob,
    /// SQLite `TEXT` / Postgres `JSONB`.
    Json,
    /// SQLite `TEXT` / Postgres `UUID`.
    Uuid,
    /// SQLite `TEXT` / Postgres `TEXT` with a max-length constraint.
    VarChar { max_chars: usize },
    /// SQLite `BLOB` / Postgres pgvector `vector(N)`.
    Vector { dimensions: usize },
}

/// A column definition used when creating a table or adding a field.
#[derive(Clone, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub auto_increment: bool,
    pub unique: bool,
    pub default: Option<String>,
    pub foreign_key: Option<ForeignKey>,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: false,
            primary_key: false,
            auto_increment: false,
            unique: false,
            default: None,
            foreign_key: None,
        }
    }

    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    pub fn primary_key(mut self) -> Self {
        self.primary_key = true;
        self
    }

    pub fn auto_increment(mut self) -> Self {
        self.auto_increment = true;
        self
    }

    pub fn unique(mut self) -> Self {
        self.unique = true;
        self
    }

    pub fn default(mut self, value: impl Into<String>) -> Self {
        self.default = Some(value.into());
        self
    }

    pub fn foreign_key(mut self, table: impl Into<String>, column: impl Into<String>) -> Self {
        self.foreign_key = Some(ForeignKey {
            table: table.into(),
            column: column.into(),
        });
        self
    }
}

/// Reference to another table/column for a foreign-key constraint.
#[derive(Clone, Debug)]
pub struct ForeignKey {
    pub table: String,
    pub column: String,
}

/// A high-level schema change.
#[derive(Clone, Debug)]
pub enum SchemaChange {
    /// Create a new table with the given columns.
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
        if_not_exists: bool,
    },
    /// Add a column to an existing table.
    AddField {
        table: String,
        column: ColumnDef,
        if_not_exists: bool,
    },
    /// Create an index on an existing table.
    CreateIndex {
        name: String,
        table: String,
        columns: Vec<String>,
        unique: bool,
        if_not_exists: bool,
        predicate: Option<String>,
    },
    /// Create a pgvector IVFFlat cosine-distance index. SQLite has no
    /// equivalent and renders this change as a no-op.
    CreateIvfFlatCosineIndex {
        name: String,
        table: String,
        column: String,
        if_not_exists: bool,
    },
    /// Run raw SQL for a specific dialect.
    Raw { sqlite: String, postgres: String },
}

/// Convenience constructors for `SchemaChange`.
impl SchemaChange {
    pub fn create_table(name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self::CreateTable {
            name: name.into(),
            columns,
            if_not_exists: false,
        }
    }

    pub fn create_table_if_not_exists(name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self::CreateTable {
            name: name.into(),
            columns,
            if_not_exists: true,
        }
    }

    pub fn add_field(table: impl Into<String>, column: ColumnDef) -> Self {
        Self::AddField {
            table: table.into(),
            column,
            if_not_exists: false,
        }
    }

    pub fn add_field_if_not_exists(table: impl Into<String>, column: ColumnDef) -> Self {
        Self::AddField {
            table: table.into(),
            column,
            if_not_exists: true,
        }
    }

    pub fn create_index(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<impl Into<String>>,
    ) -> Self {
        Self::CreateIndex {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            unique: false,
            if_not_exists: false,
            predicate: None,
        }
    }

    pub fn create_unique_index(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<impl Into<String>>,
    ) -> Self {
        Self::CreateIndex {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            unique: true,
            if_not_exists: false,
            predicate: None,
        }
    }

    /// Create a non-unique index with `IF NOT EXISTS`, matching the behaviour
    /// of `CREATE INDEX IF NOT EXISTS` in both SQLite and Postgres.
    pub fn create_index_if_not_exists(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<impl Into<String>>,
    ) -> Self {
        Self::CreateIndex {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            unique: false,
            if_not_exists: true,
            predicate: None,
        }
    }

    /// Create a unique index with `IF NOT EXISTS`.
    pub fn create_unique_index_if_not_exists(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<impl Into<String>>,
    ) -> Self {
        Self::CreateIndex {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            unique: true,
            if_not_exists: true,
            predicate: None,
        }
    }

    /// Create a non-unique partial index with `IF NOT EXISTS`.
    ///
    /// The predicate is a SQL boolean expression, for example
    /// `published_at IS NULL`. Both supported dialects implement partial
    /// indexes with the same `WHERE` syntax.
    pub fn create_partial_index_if_not_exists(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<impl Into<String>>,
        predicate: impl Into<String>,
    ) -> Self {
        Self::CreateIndex {
            name: name.into(),
            table: table.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            unique: false,
            if_not_exists: true,
            predicate: Some(predicate.into()),
        }
    }

    /// Create a Postgres pgvector IVFFlat index using cosine distance.
    pub fn create_ivfflat_cosine_index_if_not_exists(
        name: impl Into<String>,
        table: impl Into<String>,
        column: impl Into<String>,
    ) -> Self {
        Self::CreateIvfFlatCosineIndex {
            name: name.into(),
            table: table.into(),
            column: column.into(),
            if_not_exists: true,
        }
    }
}

/// A single numbered migration.
#[derive(Clone, Debug)]
pub struct Migration {
    pub version: i64,
    pub description: String,
    pub changes: Vec<SchemaChange>,
}

impl Migration {
    pub fn new(version: i64, description: impl Into<String>, changes: Vec<SchemaChange>) -> Self {
        Self {
            version,
            description: description.into(),
            changes,
        }
    }
}

/// SQL dialect being targeted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    SQLite,
    Postgres,
}

/// Render a `DataType` to SQL for the given dialect.
pub fn render_data_type(dialect: Dialect, dt: &DataType) -> String {
    match (dialect, dt) {
        (Dialect::SQLite, DataType::Text) => "TEXT".to_string(),
        (Dialect::Postgres, DataType::Text) => "TEXT".to_string(),
        (Dialect::SQLite, DataType::Integer) => "INTEGER".to_string(),
        (Dialect::Postgres, DataType::Integer) => "BIGINT".to_string(),
        (Dialect::SQLite, DataType::Float) => "REAL".to_string(),
        (Dialect::Postgres, DataType::Float) => "DOUBLE PRECISION".to_string(),
        (Dialect::SQLite, DataType::Boolean) => "INTEGER".to_string(),
        (Dialect::Postgres, DataType::Boolean) => "BOOLEAN".to_string(),
        (Dialect::SQLite, DataType::Timestamp) => "TEXT".to_string(),
        (Dialect::Postgres, DataType::Timestamp) => "TIMESTAMPTZ".to_string(),
        (Dialect::SQLite, DataType::Blob) => "BLOB".to_string(),
        (Dialect::Postgres, DataType::Blob) => "BYTEA".to_string(),
        (Dialect::SQLite, DataType::Json) => "TEXT".to_string(),
        (Dialect::Postgres, DataType::Json) => "JSONB".to_string(),
        (Dialect::SQLite, DataType::Uuid) => "TEXT".to_string(),
        (Dialect::Postgres, DataType::Uuid) => "UUID".to_string(),
        (Dialect::SQLite, DataType::VarChar { .. }) => "TEXT".to_string(),
        (Dialect::Postgres, DataType::VarChar { max_chars }) => format!("VARCHAR({})", max_chars),
        (Dialect::SQLite, DataType::Vector { .. }) => "BLOB".to_string(),
        (Dialect::Postgres, DataType::Vector { dimensions }) => {
            format!("vector({dimensions})")
        }
    }
}

/// Render a column definition to SQL for the given dialect.
pub fn render_column(dialect: Dialect, col: &ColumnDef) -> String {
    let mut parts = vec![format!(
        "\"{}\" {}",
        col.name,
        render_data_type(dialect, &col.data_type)
    )];

    if col.primary_key {
        parts.push("PRIMARY KEY".to_string());
    }

    if col.auto_increment {
        match dialect {
            Dialect::SQLite => parts.push("AUTOINCREMENT".to_string()),
            Dialect::Postgres => {
                // Postgres uses BIGSERIAL / SERIAL; we assume the caller set the
                // data type to Integer and we append the serial keyword.
                parts.push("GENERATED ALWAYS AS IDENTITY".to_string());
            }
        }
    }

    if col.unique && !col.primary_key {
        parts.push("UNIQUE".to_string());
    }

    if !col.nullable || col.primary_key {
        parts.push("NOT NULL".to_string());
    }

    if let Some(default) = &col.default {
        parts.push(format!(
            "DEFAULT {}",
            render_default(&col.data_type, default)
        ));
    }

    parts.join(" ")
}

/// Render a column DEFAULT value, quoting string-typed literals so they are not
/// mistaken for column references (e.g. Postgres rejects `DEFAULT none` with
/// "cannot use column reference in DEFAULT expression").
fn render_default(data_type: &DataType, value: &str) -> String {
    let is_string_type = matches!(
        data_type,
        DataType::Text
            | DataType::VarChar { .. }
            | DataType::Uuid
            | DataType::Json
            | DataType::Timestamp
            | DataType::Blob
    );

    if !is_string_type {
        return value.to_string();
    }

    let trimmed = value.trim();
    // Leave already-quoted values, SQL expressions/function calls, and keywords
    // (NULL, CURRENT_TIMESTAMP, etc.) untouched.
    let looks_like_expr = trimmed.starts_with('\'')
        || trimmed.contains('(')
        || trimmed.eq_ignore_ascii_case("null")
        || trimmed.eq_ignore_ascii_case("current_timestamp");

    if looks_like_expr {
        value.to_string()
    } else {
        format!("'{}'", trimmed.replace('\'', "''"))
    }
}

/// Render a `SchemaChange` to one or more SQL statements for the given dialect.
pub fn render_change(dialect: Dialect, change: &SchemaChange) -> Vec<String> {
    match change {
        SchemaChange::CreateTable {
            name,
            columns,
            if_not_exists,
        } => {
            let if_clause = if *if_not_exists { "IF NOT EXISTS " } else { "" };
            let mut col_sql: Vec<String> =
                columns.iter().map(|c| render_column(dialect, c)).collect();

            // Add foreign-key constraints at the table level.
            for col in columns {
                if let Some(fk) = &col.foreign_key {
                    col_sql.push(format!(
                        "FOREIGN KEY (\"{}\") REFERENCES \"{}\"(\"{}\")",
                        col.name, fk.table, fk.column
                    ));
                }
            }

            vec![format!(
                "CREATE TABLE {if_clause}\"{}\" (\n  {}\n);",
                name,
                col_sql.join(",\n  ")
            )]
        }
        SchemaChange::AddField {
            table,
            column,
            if_not_exists,
        } => {
            let rendered = render_column(dialect, column);
            match dialect {
                Dialect::SQLite => {
                    // SQLite supports ALTER TABLE ADD COLUMN directly.
                    let if_clause = if *if_not_exists { "IF NOT EXISTS " } else { "" };
                    vec![format!(
                        "ALTER TABLE \"{}\" ADD COLUMN {if_clause}{};",
                        table, rendered
                    )]
                }
                Dialect::Postgres => {
                    // Postgres does not support IF NOT EXISTS on ADD COLUMN.
                    let stmt = format!("ALTER TABLE \"{}\" ADD COLUMN {};", table, rendered);
                    if *if_not_exists {
                        vec![format!(
                            "DO $$ BEGIN {} EXCEPTION WHEN duplicate_column THEN NULL; END $$;",
                            stmt
                        )]
                    } else {
                        vec![stmt]
                    }
                }
            }
        }
        SchemaChange::CreateIndex {
            name,
            table,
            columns,
            unique,
            if_not_exists,
            predicate,
        } => {
            let unique_kw = if *unique { "UNIQUE " } else { "" };
            let if_clause = if *if_not_exists { "IF NOT EXISTS " } else { "" };
            let cols = columns
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", ");
            let where_clause = predicate
                .as_ref()
                .map(|predicate| format!(" WHERE {predicate}"))
                .unwrap_or_default();
            vec![format!(
                "CREATE {unique_kw}INDEX {if_clause}\"{}\" ON \"{}\" ({}){};",
                name, table, cols, where_clause
            )]
        }
        SchemaChange::CreateIvfFlatCosineIndex {
            name,
            table,
            column,
            if_not_exists,
        } => match dialect {
            Dialect::SQLite => vec![],
            Dialect::Postgres => {
                let if_clause = if *if_not_exists { "IF NOT EXISTS " } else { "" };
                vec![format!(
                    "CREATE INDEX {if_clause}\"{}\" ON \"{}\" USING ivfflat (\"{}\" vector_cosine_ops);",
                    name, table, column
                )]
            }
        },
        SchemaChange::Raw { sqlite, postgres } => match dialect {
            Dialect::SQLite => vec![sqlite.clone()],
            Dialect::Postgres => vec![postgres.clone()],
        },
    }
}

/// A migration runner that applies pending migrations and records them.
///
/// Dialect-specific implementations provide the actual execution and
/// transaction handling.
pub struct MigrationRunner<'a> {
    pub dialect: Dialect,
    pub migrations: &'a [Migration],
}

impl<'a> MigrationRunner<'a> {
    pub fn new(dialect: Dialect, migrations: &'a [Migration]) -> Self {
        Self {
            dialect,
            migrations,
        }
    }

    /// Render all pending migrations (those with a version greater than
    /// `current_version`) to SQL in order.
    pub fn pending_sql(&self, current_version: i64) -> Vec<(i64, String, Vec<String>)> {
        let mut pending = Vec::new();
        for migration in self.migrations {
            if migration.version > current_version {
                let stmts: Vec<String> = migration
                    .changes
                    .iter()
                    .flat_map(|c| render_change(self.dialect, c))
                    .filter(|s| !s.trim().is_empty())
                    .collect();
                pending.push((migration.version, migration.description.clone(), stmts));
            }
        }
        pending
    }

    /// SQL to create the migration tracking table.
    pub fn migration_table_sql(&self) -> String {
        match self.dialect {
            Dialect::SQLite => "CREATE TABLE IF NOT EXISTS schema_migrations (\n  version INTEGER PRIMARY KEY,\n  description TEXT NOT NULL,\n  applied_at TEXT NOT NULL\n);".to_string(),
            Dialect::Postgres => "CREATE TABLE IF NOT EXISTS schema_migrations (\n  version BIGINT PRIMARY KEY,\n  description TEXT NOT NULL,\n  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);".to_string(),
        }
    }

    /// SQL to insert a migration record.
    pub fn record_migration_sql(&self, version: i64, description: &str) -> String {
        match self.dialect {
            Dialect::SQLite => format!(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES ({}, '{}', datetime('now'));",
                version,
                escape_sql_string(description)
            ),
            Dialect::Postgres => format!(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES ({}, '{}', now());",
                version,
                escape_sql_string(description)
            ),
        }
    }
}

/// Very small helper to escape single quotes in migration descriptions.
fn escape_sql_string(s: &str) -> String {
    s.replace('\'', "''")
}

/// Validate that migrations are ordered and have unique versions.
pub fn validate_migrations(migrations: &[Migration]) -> Result<(), String> {
    let mut versions = HashMap::new();
    let mut last_version = -1i64;
    for m in migrations {
        if m.version <= last_version {
            return Err(format!(
                "migration versions must be strictly increasing: {} follows {}",
                m.version, last_version
            ));
        }
        if let Some(prev) = versions.insert(m.version, m.description.clone()) {
            return Err(format!(
                "duplicate migration version {} (previously {})",
                m.version, prev
            ));
        }
        last_version = m.version;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_sqlite_create_table() {
        let sql = render_change(
            Dialect::SQLite,
            &SchemaChange::create_table(
                "users",
                vec![
                    ColumnDef::new("id", DataType::Integer)
                        .primary_key()
                        .auto_increment(),
                    ColumnDef::new("name", DataType::VarChar { max_chars: 255 }),
                    ColumnDef::new("email", DataType::Text).unique().nullable(),
                ],
            ),
        );
        assert_eq!(sql.len(), 1);
        assert!(sql[0].contains("\"id\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL"));
        assert!(sql[0].contains("\"name\" TEXT NOT NULL"));
        assert!(sql[0].contains("\"email\" TEXT UNIQUE"));
    }

    #[test]
    fn render_postgres_create_table() {
        let sql = render_change(
            Dialect::Postgres,
            &SchemaChange::create_table(
                "users",
                vec![
                    ColumnDef::new("id", DataType::Integer)
                        .primary_key()
                        .auto_increment(),
                    ColumnDef::new("name", DataType::VarChar { max_chars: 255 }),
                    ColumnDef::new("email", DataType::Text).unique().nullable(),
                ],
            ),
        );
        assert_eq!(sql.len(), 1);
        assert!(sql[0].contains("\"id\" BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY NOT NULL"));
        assert!(sql[0].contains("\"name\" VARCHAR(255) NOT NULL"));
        assert!(sql[0].contains("\"email\" TEXT UNIQUE"));
    }

    #[test]
    fn render_add_field_if_not_exists_postgres_wraps_do_block() {
        let sql = render_change(
            Dialect::Postgres,
            &SchemaChange::add_field_if_not_exists(
                "sessions",
                ColumnDef::new("metadata", DataType::Json).nullable(),
            ),
        );
        assert_eq!(sql.len(), 1);
        assert!(sql[0].starts_with("DO $$"));
    }

    #[test]
    fn render_postgres_vector_column_and_ivfflat_cosine_index() {
        assert_eq!(
            render_data_type(Dialect::Postgres, &DataType::Vector { dimensions: 384 }),
            "vector(384)"
        );
        let sql = render_change(
            Dialect::Postgres,
            &SchemaChange::create_ivfflat_cosine_index_if_not_exists(
                "idx_embeddings",
                "semantic_cache",
                "embedding",
            ),
        );
        assert_eq!(
            sql,
            vec!["CREATE INDEX IF NOT EXISTS \"idx_embeddings\" ON \"semantic_cache\" USING ivfflat (\"embedding\" vector_cosine_ops);".to_string()]
        );
    }

    #[test]
    fn render_partial_index() {
        let sql = render_change(
            Dialect::Postgres,
            &SchemaChange::create_partial_index_if_not_exists(
                "idx_jobs_pending",
                "jobs",
                vec!["available_at", "id"],
                "published_at IS NULL",
            ),
        );
        assert_eq!(
            sql,
            vec!["CREATE INDEX IF NOT EXISTS \"idx_jobs_pending\" ON \"jobs\" (\"available_at\", \"id\") WHERE published_at IS NULL;".to_string()]
        );
    }

    #[test]
    fn sqlite_vector_index_is_a_no_op() {
        let sql = render_change(
            Dialect::SQLite,
            &SchemaChange::create_ivfflat_cosine_index_if_not_exists(
                "idx_embeddings",
                "semantic_cache",
                "embedding",
            ),
        );
        assert!(sql.is_empty());
    }

    #[test]
    fn validate_migrations_rejects_unordered() {
        let migrations = vec![
            Migration::new(0, "a", vec![]),
            Migration::new(0, "b", vec![]),
        ];
        assert!(validate_migrations(&migrations).is_err());
    }
}
