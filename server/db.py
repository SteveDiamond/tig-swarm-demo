import aiosqlite
from contextlib import asynccontextmanager
from pathlib import Path

import os
# Use /data for Railway persistent volume, fallback to local for dev
_data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))
DB_PATH = _data_dir / "swarm.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    registered_at TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL,
    status TEXT DEFAULT 'idle',
    experiments_completed INTEGER DEFAULT 0,
    best_score REAL,
    runs_since_improvement INTEGER DEFAULT 0,
    improvements INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    strategy_tag TEXT NOT NULL,
    status TEXT DEFAULT 'failed',
    fingerprint TEXT NOT NULL,
    parent_hypothesis_id TEXT,
    created_at TEXT NOT NULL,
    target_best_experiment_id TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS agent_bests (
    agent_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    algorithm_code TEXT NOT NULL,
    score REAL NOT NULL,
    feasible INTEGER NOT NULL DEFAULT 1,
    num_vehicles INTEGER DEFAULT 0,
    total_distance REAL DEFAULT 0.0,
    route_data TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    hypothesis_id TEXT,
    algorithm_code TEXT DEFAULT '',
    score REAL NOT NULL,
    feasible INTEGER DEFAULT 1,
    num_vehicles INTEGER DEFAULT 0,
    total_distance REAL DEFAULT 0.0,
    runtime_seconds REAL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    route_data TEXT,
    delta_vs_best_pct REAL,
    delta_vs_own_best_pct REAL,
    beats_own_best INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    agent_name TEXT NOT NULL,
    content TEXT NOT NULL,
    msg_type TEXT DEFAULT 'agent',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS best_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    agent_id TEXT,
    agent_name TEXT NOT NULL,
    score REAL NOT NULL,
    route_data TEXT,
    created_at TEXT NOT NULL
);
"""

# Indexes are split out from the main schema so they can be applied after
# ALTER TABLE migrations in init_db, which keeps both fresh and upgraded
# databases working.
SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_exp_feasible_score ON experiments(feasible, score);
CREATE INDEX IF NOT EXISTS idx_exp_agent ON experiments(agent_id);
CREATE INDEX IF NOT EXISTS idx_hyp_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hyp_fingerprint ON hypotheses(fingerprint);
CREATE INDEX IF NOT EXISTS idx_agent_bests_score ON agent_bests(feasible, score);
CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_hyp_agent_target ON hypotheses(agent_id, target_best_experiment_id);
"""

DEFAULT_CONFIG = {
    "benchmark_instances": '["R1_4_1","R1_4_2","R1_4_3","R1_4_4","R1_4_5","R2_4_1","R2_4_2","R2_4_3","R2_4_4","R2_4_5","RC1_4_1","RC1_4_2","RC1_4_3","RC1_4_4","RC1_4_5","RC2_4_1","RC2_4_2","RC2_4_3","RC2_4_4","RC2_4_5","C1_4_1","C1_4_2","C2_4_1","C2_4_2"]',
    "admin_key": "ads-2026",
}


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        # 1) Tables first. All table DDL is IF NOT EXISTS so fresh and
        #    upgraded databases both work.
        await db.executescript(SCHEMA)
        # 2) Column migrations. ADD COLUMN fails if the column exists;
        #    that's expected on every subsequent run.
        try:
            await db.execute("ALTER TABLE experiments RENAME COLUMN algorithm_diff TO algorithm_code")
            await db.commit()
        except Exception:
            pass
        for stmt in (
            "ALTER TABLE agents ADD COLUMN runs_since_improvement INTEGER DEFAULT 0",
            "ALTER TABLE agents ADD COLUMN improvements INTEGER DEFAULT 0",
            "ALTER TABLE hypotheses ADD COLUMN target_best_experiment_id TEXT",
            "ALTER TABLE best_history ADD COLUMN agent_id TEXT",
            "ALTER TABLE experiments ADD COLUMN delta_vs_best_pct REAL",
            "ALTER TABLE experiments ADD COLUMN delta_vs_own_best_pct REAL",
            "ALTER TABLE experiments ADD COLUMN beats_own_best INTEGER DEFAULT 0",
        ):
            try:
                await db.execute(stmt)
                await db.commit()
            except Exception:
                pass
        # The current model has no "active" hypotheses: every attempt is
        # recorded as succeeded/failed once evaluated. Legacy statuses are
        # normalized to failed so old rows don't appear in a third state.
        await db.execute(
            "UPDATE hypotheses SET status = 'failed' "
            "WHERE status IN ('proposed', 'claimed', 'testing')"
        )
        await db.commit()
        # 3) Indexes last, *after* the migrations above — some of them
        #    reference columns that only exist post-migration.
        await db.executescript(SCHEMA_INDEXES)
        # 4) Backfill agent_bests from the existing experiments table on
        #    first upgrade. Without this, an existing deployment would see
        #    an empty agent_bests, collapse to cold start, and serve every
        #    agent the Solomon seed until someone republishes. ON CONFLICT
        #    DO NOTHING makes this a no-op on subsequent boots.
        await db.execute(
            """INSERT INTO agent_bests
               (agent_id, experiment_id, algorithm_code, score, feasible,
                num_vehicles, total_distance, route_data, updated_at)
               SELECT agent_id, id, algorithm_code, score, 1,
                      num_vehicles, total_distance, route_data, created_at
               FROM (
                   SELECT e.*,
                          ROW_NUMBER() OVER (
                              PARTITION BY e.agent_id
                              ORDER BY e.score ASC, e.created_at ASC
                          ) AS rn
                   FROM experiments e
                   WHERE e.feasible = 1
               )
               WHERE rn = 1
               ON CONFLICT(agent_id) DO NOTHING"""
        )
        await db.commit()
        for key, value in DEFAULT_CONFIG.items():
            await db.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                (key, value),
            )
        await db.commit()


@asynccontextmanager
async def connect():
    """Context manager for DB connections — ensures cleanup on error."""
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()


async def get_config(conn: aiosqlite.Connection) -> dict:
    cursor = await conn.execute("SELECT key, value FROM config")
    rows = await cursor.fetchall()
    return {row["key"]: row["value"] for row in rows}


async def get_global_best(conn: aiosqlite.Connection) -> dict | None:
    # Global best is the best-scoring `agent_bests` row — i.e. whichever
    # agent's branch currently holds the lowest feasible score. `id` is
    # aliased to experiment_id so callers that expect the old experiments
    # shape (best["id"] meaning the experiment row) keep working.
    cursor = await conn.execute(
        "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
        "       score, feasible, num_vehicles, total_distance, route_data, updated_at "
        "FROM agent_bests WHERE feasible = 1 "
        "ORDER BY score ASC LIMIT 1"
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_agent_best(
    conn: aiosqlite.Connection, agent_id: str
) -> dict | None:
    cursor = await conn.execute(
        "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
        "       score, feasible, num_vehicles, total_distance, route_data, updated_at "
        "FROM agent_bests WHERE agent_id = ?",
        (agent_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def upsert_agent_best(
    conn: aiosqlite.Connection,
    agent_id: str,
    experiment_id: str,
    algorithm_code: str,
    score: float,
    feasible: bool,
    num_vehicles: int,
    total_distance: float,
    route_data: str | None,
    updated_at: str,
) -> None:
    await conn.execute(
        """INSERT INTO agent_bests
           (agent_id, experiment_id, algorithm_code, score, feasible,
            num_vehicles, total_distance, route_data, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(agent_id) DO UPDATE SET
             experiment_id = excluded.experiment_id,
             algorithm_code = excluded.algorithm_code,
             score = excluded.score,
             feasible = excluded.feasible,
             num_vehicles = excluded.num_vehicles,
             total_distance = excluded.total_distance,
             route_data = excluded.route_data,
             updated_at = excluded.updated_at""",
        (agent_id, experiment_id, algorithm_code, score,
         1 if feasible else 0, num_vehicles, total_distance,
         route_data, updated_at),
    )


async def list_agent_bests(
    conn: aiosqlite.Connection,
    exclude_agent_ids: list[str] | None = None,
) -> list[dict]:
    # All feasible agent-bests, optionally excluding specific agent ids.
    # Returned shape matches `get_global_best` so callers can treat the
    # rows interchangeably.
    exclude = exclude_agent_ids or []
    if exclude:
        placeholders = ",".join("?" for _ in exclude)
        query = (
            "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
            "       score, feasible, num_vehicles, total_distance, route_data, updated_at "
            f"FROM agent_bests WHERE feasible = 1 AND agent_id NOT IN ({placeholders}) "
            "ORDER BY score ASC"
        )
        cursor = await conn.execute(query, exclude)
    else:
        cursor = await conn.execute(
            "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
            "       score, feasible, num_vehicles, total_distance, route_data, updated_at "
            "FROM agent_bests WHERE feasible = 1 ORDER BY score ASC"
        )
    return [dict(row) for row in await cursor.fetchall()]


async def get_agent_count(
    conn: aiosqlite.Connection,
    active_only: bool = False,
    inactive_cutoff: str | None = None,
) -> int:
    if active_only:
        if inactive_cutoff is None:
            raise ValueError("inactive_cutoff is required when active_only=True")
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM agents WHERE last_heartbeat >= ?",
            (inactive_cutoff,),
        )
    else:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM agents")
    return (await cursor.fetchone())["c"]


async def get_all_agent_names(conn: aiosqlite.Connection) -> set[str]:
    cursor = await conn.execute("SELECT name FROM agents")
    return {row["name"] for row in await cursor.fetchall()}


async def compute_leaderboard(
    conn: aiosqlite.Connection,
    inactive_cutoff: str | None = None,
) -> list[dict]:
    # All counters are stored directly on the agents table and updated
    # atomically by POST /api/iterations.  best_score comes from agent_bests.
    cursor = await conn.execute(
        """
        SELECT
            a.id   as agent_id,
            a.name as agent_name,
            a.experiments_completed as runs,
            a.improvements as improvements,
            a.runs_since_improvement as runs_since_improvement,
            a.last_heartbeat as last_heartbeat,
            ab.score as best_score
        FROM agents a
        LEFT JOIN agent_bests ab ON ab.agent_id = a.id AND ab.feasible = 1
        ORDER BY best_score IS NULL, best_score ASC, a.name ASC
        """
    )
    rows = await cursor.fetchall()
    return [
        {
            "rank": i + 1,
            "agent_id": row["agent_id"],
            "agent_name": row["agent_name"],
            "runs": row["runs"],
            "improvements": row["improvements"],
            "runs_since_improvement": row["runs_since_improvement"],
            "best_score": row["best_score"],
            "active": row["last_heartbeat"] >= inactive_cutoff if inactive_cutoff and row["last_heartbeat"] else False,
        }
        for i, row in enumerate(rows)
    ]
