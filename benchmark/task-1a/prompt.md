# TASK-1A benchmark — agent prompt (run verbatim)

You are a backend engineer. Execute TASK-1A from `./Task.md` using the
specification in `./ARCHITECTURE.md`. Read BOTH files fully before acting.

Scope (from Task.md acceptance criteria):
- Repository structure: `services/` with all 6 microservices
  (portfolio, market-feed, alert-engine, dashboard, notification, nav-batch),
  each with its own `requirements.txt` (FastAPI deps) and `main.py` entry
  point. Follow section 12.1 of ARCHITECTURE.md exactly.
- Root files: `.env.example`, `README.md`, `.gitignore`, pre-commit config
  (black, flake8).
- `scripts/` with `setup.sh` and `run-tests.sh`; `k8s/` directory;
  `logs/` with `.gitkeep`.
- PostgreSQL: database `stockvisor_main` with the section 3.2.1 schema
  (7 tables, 9 named indexes, FK CASCADE deletes, UNIQUE(portfolio_id,
  symbol) on holdings, gen_random_uuid() defaults). A PostgreSQL server is
  listening on 127.0.0.1:5432 - discover working credentials using standard
  local defaults (do NOT brute-force; try the obvious defaults and stop if
  none work, reporting that).
- Alembic initialized in `services/portfolio/migrations/` with an initial
  migration covering the complete schema.
- Python 3.11+; include pytest, pytest-asyncio, pytest-cov in test deps;
  use asyncpg; UTC timestamps; JSONB columns.

Verification you must perform yourself (not just write files):
- Create a project-local venv (`./.venv`, never the system interpreter),
  install dependencies, run `alembic upgrade head` against
  `stockvisor_main`, confirm all 7 tables + 9 indexes exist.
- Run `alembic downgrade -1` then `alembic upgrade head` again to prove
  rollback works.
- Run each service's `main.py --help` (or import check) to prove the
  scaffolding imports.

Tool naming (infrastructure requirement — follow strictly):
- Call tools using their BARE contract names with NO namespace prefix:
  `read_file`, `write_file`, `execute_command`, never `muse.read_file`,
  `muse.write_file`, etc. (The gateway now resolves `muse.*` aliases, but
  bare names are still required.)

Tool-use discipline (infrastructure requirement — follow strictly):
- Create files strictly sequentially: at most 2 file-writing tool calls per
  message, then stop and confirm in the next message that the batch executed
  (check tool results) before proposing the next batch.
- If any batch is reported as rejected / failed validation, do NOT re-propose
  the same batch: retry the calls strictly one at a time, checking each
  result, and adjust fields to match the reported tool schema.
- Never abandon the task after a rejected batch; work through remaining files
  one at a time.

Constraints:
- Only create/modify files inside the current working directory.
- Only touch the `stockvisor_main` database. Never drop or modify any other
  database. Never delete data outside this task.
- Keep secrets out of files: real credentials go in `.env` (gitignored,
  never committed); only placeholders in `.env.example`.

Final summary (print at the end, always):
1. Files/directories created (list).
2. Database connection used: host, port, user, database name, auth method
   (no passwords).
3. Alembic revision id(s).
4. Commands run with results (upgrade, downgrade, import checks).
5. Anything from the acceptance criteria NOT completed, and why.
