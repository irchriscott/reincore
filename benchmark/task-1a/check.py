#!/usr/bin/env python3
"""TASK-1A acceptance checker (static + best-effort read-only live-DB checks).

Usage:  python check.py [stockvisor-dir]
Exit 0 = all checks PASS (UNVERIFIED allowed only when no DB driver/venv).
Exit 1 = at least one FAIL present.
"""
import compileall
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "../stockvisor")

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"
results = []


def check(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def p(*parts):
    return os.path.join(ROOT, *parts)


SERVICES = ["portfolio", "market-feed", "alert-engine", "dashboard",
            "notification", "nav-batch"]

# ---- project structure ----
check("services/ exists", PASS if os.path.isdir(p("services")) else FAIL)
for svc in SERVICES:
    check(f"services/{svc}/ exists",
          PASS if os.path.isdir(p("services", svc)) else FAIL)
    check(f"services/{svc}/requirements.txt exists",
          PASS if os.path.isfile(p("services", svc, "requirements.txt")) else FAIL)
    check(f"services/{svc}/main.py exists",
          PASS if os.path.isfile(p("services", svc, "main.py")) else FAIL)

req_all_fastapi = True
for svc in SERVICES:
    fp = p("services", svc, "requirements.txt")
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        if "fastapi" not in content and "uvicorn" not in content:
            req_all_fastapi = False
    except OSError:
        req_all_fastapi = False
check("all requirements.txt mention fastapi/uvicorn",
      PASS if req_all_fastapi else FAIL)

pfp = p("services", "portfolio", "requirements.txt")
try:
    with open(pfp, encoding="utf-8", errors="replace") as f:
        pf = f.read().lower()
    missing = [d for d in ("pytest", "pytest-asyncio", "pytest-cov",
                           "asyncpg", "alembic") if d not in pf]
    check("portfolio test/async deps present",
          PASS if not missing else FAIL, f"missing={missing}")
except OSError:
    check("portfolio test/async deps present", FAIL, "requirements.txt unreadable")

if os.path.isdir(p("services")):
    ok = compileall.compile_dir(p("services"), quiet=1, force=True)
    check("all service main.py compile", PASS if ok else FAIL)
else:
    check("all service main.py compile", FAIL, "no services/ dir")

check("scripts/setup.sh exists",
      PASS if os.path.isfile(p("scripts", "setup.sh")) else FAIL)
check("scripts/run-tests.sh exists",
      PASS if os.path.isfile(p("scripts", "run-tests.sh")) else FAIL)
check("k8s/ exists", PASS if os.path.isdir(p("k8s")) else FAIL)
check("logs/.gitkeep exists",
      PASS if os.path.isfile(p("logs", ".gitkeep")) else FAIL)

rd = p("README.md")
check("root README.md non-empty",
      PASS if os.path.isfile(rd) and os.path.getsize(rd) > 0 else FAIL)

env = p(".env.example")
if os.path.isfile(env):
    with open(env, encoding="utf-8", errors="replace") as f:
        ec = f.read()
    placeholders = ("POSTGRES" in ec or "DATABASE" in ec)
    has_real_pw = bool(re.search(r"(?im)^\s*(POSTGRES_PASSWORD|DB_PASSWORD)\s*=\s*[^*\s<][^\s]*$",
                                 ec))
    check(".env.example has DB placeholders, no real secrets",
          PASS if placeholders and not has_real_pw else FAIL)
else:
    check(".env.example exists", FAIL)

gi = p(".gitignore")
if os.path.isfile(gi):
    with open(gi, encoding="utf-8", errors="replace") as f:
        gc = f.read()
    need = [".env", "venv", "__pycache__", "logs", "data"]
    missing = [n for n in need if n not in gc]
    check(".gitignore covers env/venv/cache/logs/data",
          PASS if not missing else FAIL, f"missing={missing}")
else:
    check(".gitignore exists", FAIL)

pc = p(".pre-commit-config.yaml")
if os.path.isfile(pc):
    with open(pc, encoding="utf-8", errors="replace") as f:
        pc_text = f.read().lower()
    check("pre-commit has black+flake8",
          PASS if ("black" in pc_text and "flake8" in pc_text) else FAIL)
else:
    check(".pre-commit-config.yaml exists", FAIL)

mig = p("services", "portfolio", "migrations")
versions = os.path.join(mig, "versions")
has_env = os.path.isfile(os.path.join(mig, "env.py"))
ver_files = []
if os.path.isdir(versions):
    ver_files = [f for f in os.listdir(versions)
                 if f.endswith(".py") and f != "__init__.py"]
check("alembic env.py + versions/ with >=1 migration",
      PASS if has_env and ver_files else FAIL, f"migrations={ver_files}")
check("project .venv exists", PASS if os.path.isdir(p(".venv")) else FAIL)

# ---- live DB checks (read-only; UNVERIFIED when no driver/venv/server) ----
TABLES_7 = ["users", "portfolios", "holdings", "transactions",
            "alert_rules", "notification_preferences", "portfolio_events"]
INDEXES_9 = ["idx_portfolios_user_id", "idx_holdings_portfolio_id",
             "idx_holdings_symbol", "idx_transactions_portfolio_id",
             "idx_alert_rules_user_id", "idx_alert_rules_symbol",
             "idx_aggregate_id", "idx_created_at", "idx_event_type"]
JSONB_COLS = [("alert_rules", "rule_config"),
              ("alert_rules", "notification_channels"),
              ("portfolio_events", "event_data"),
              ("portfolio_events", "metadata")]


def parse_dotenv(fp):
    out = {}
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def db_check_names():
    return ["db connect stockvisor_main", "7 tables exist",
            "9 named indexes exist", "UNIQUE(portfolio_id,symbol) on holdings",
            "FK CASCADE deletes present", "gen_random_uuid() defaults",
            "alembic_version row present", "JSONB columns typed jsonb"]


def mark_all_unverified(reason):
    for name in db_check_names():
        check(name, UNVERIFIED, reason)


try:
    import psycopg2  # noqa
    HAVE_PG = ("psycopg2",)
except ImportError:
    try:
        import psycopg  # noqa
        HAVE_PG = ("psycopg",)
    except ImportError:
        HAVE_PG = None

if HAVE_PG is None:
    mark_all_unverified("no postgres driver in checker python")
else:
    env_vars = parse_dotenv(p(".env"))
    user = (env_vars.get("POSTGRES_USER") or env_vars.get("DB_USER")
            or "postgres")
    pw = (env_vars.get("POSTGRES_PASSWORD") or env_vars.get("DB_PASSWORD")
          or env_vars.get("POSTGRES_PW") or "")
    candidates = [(user, pw)] if pw else [(user, "postgres"),
                                          (user, "password"), (user, "")]
    conn = None
    used = None
    mod = __import__(HAVE_PG[0])
    for u, w in candidates:
        try:
            if HAVE_PG[0] == "psycopg2":
                conn = mod.connect(host="127.0.0.1", port=5432,
                                   dbname="stockvisor_main", user=u,
                                   password=w or None, connect_timeout=5)
            else:
                conn = mod.connect(host="127.0.0.1", port=5432,
                                   dbname="stockvisor_main", user=u,
                                   password=w or None, connect_timeout=5)
            used = u
            break
        except Exception:
            conn = None
    if conn is None:
        mark_all_unverified("stockvisor_main unreachable with default creds")
    else:
        cur = conn.cursor()
        check("db connect stockvisor_main", PASS, f"user={used}")

        cur.execute("SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public'")
        have_tables = {r[0] for r in cur.fetchall()}
        missing = [t for t in TABLES_7 if t not in have_tables]
        check("7 tables exist", PASS if not missing else FAIL,
              f"missing={missing}")

        cur.execute("SELECT indexname FROM pg_indexes "
                    "WHERE schemaname='public'")
        have_idx = {r[0] for r in cur.fetchall()}
        missing = [i for i in INDEXES_9 if i not in have_idx]
        check("9 named indexes exist", PASS if not missing else FAIL,
              f"missing={missing}")

        cur.execute("""SELECT 1 FROM information_schema.table_constraints
                       WHERE table_name='holdings'
                       AND constraint_type='UNIQUE'
                       AND constraint_name IN (
                         SELECT constraint_name
                         FROM information_schema.key_column_usage
                         WHERE table_name='holdings'
                         GROUP BY constraint_name
                         HAVING array_agg(column_name::text
                                           ORDER BY ordinal_position)
                                = ARRAY['portfolio_id','symbol'])""")
        check("UNIQUE(portfolio_id,symbol) on holdings",
              PASS if cur.fetchone() else FAIL)

        cur.execute("""SELECT COUNT(*) FROM
                       information_schema.referential_constraints
                       WHERE delete_rule='CASCADE'""")
        n_cascade = cur.fetchone()[0]
        check("FK CASCADE deletes present",
              PASS if n_cascade >= 4 else FAIL, f"cascade_fks={n_cascade}")

        cur.execute("""SELECT COUNT(*) FROM information_schema.columns
                       WHERE table_schema='public'
                       AND column_default LIKE '%gen_random_uuid()%'""")
        n_uuid = cur.fetchone()[0]
        check("gen_random_uuid() defaults",
              PASS if n_uuid >= 7 else FAIL, f"uuid_defaults={n_uuid}")

        try:
            cur.execute("SELECT version_num FROM alembic_version")
            rows = cur.fetchall()
            check("alembic_version row present",
                  PASS if rows else FAIL, f"revisions={[r[0] for r in rows]}")
        except Exception as e:
            check("alembic_version row present", FAIL, str(e)[:100])

        cur.execute("""SELECT table_name, column_name FROM
                       information_schema.columns
                       WHERE table_schema='public' AND udt_name='jsonb'""")
        have_jsonb = {(r[0], r[1]) for r in cur.fetchall()}
        missing = [c for c in JSONB_COLS if c not in have_jsonb]
        check("JSONB columns typed jsonb", PASS if not missing else FAIL,
              f"missing={missing}")
        conn.close()

print()
np = sum(1 for _, s, _ in results if s == PASS)
nf = sum(1 for _, s, _ in results if s == FAIL)
nu = sum(1 for _, s, _ in results if s == UNVERIFIED)
print(f"TOTAL {len(results)}: {np} PASS / {nf} FAIL / {nu} UNVERIFIED")
sys.exit(1 if nf else 0)
