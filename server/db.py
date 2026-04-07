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
    best_score REAL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    strategy_tag TEXT NOT NULL,
    status TEXT DEFAULT 'proposed',
    fingerprint TEXT NOT NULL,
    parent_hypothesis_id TEXT,
    claimed_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    hypothesis_id TEXT,
    algorithm_diff TEXT DEFAULT '',
    score REAL NOT NULL,
    feasible INTEGER DEFAULT 1,
    num_vehicles INTEGER DEFAULT 0,
    total_distance REAL DEFAULT 0.0,
    runtime_seconds REAL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    route_data TEXT,
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

CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    content TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    updated_by TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_exp_feasible_score ON experiments(feasible, score);
CREATE INDEX IF NOT EXISTS idx_exp_agent ON experiments(agent_id);
CREATE INDEX IF NOT EXISTS idx_hyp_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hyp_fingerprint ON hypotheses(fingerprint);
CREATE TABLE IF NOT EXISTS best_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    score REAL NOT NULL,
    route_data TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);
"""

DEFAULT_CONFIG = {
    "baseline_score": "1850.5",
    "benchmark_instances": '["C101","C201","R101","RC101"]',
    "admin_key": "ads-2026",
}


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
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
    cursor = await conn.execute(
        "SELECT * FROM experiments WHERE feasible = 1 ORDER BY score ASC LIMIT 1"
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_agent_count(conn: aiosqlite.Connection, active_only: bool = False) -> int:
    if active_only:
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM agents WHERE status != 'offline'"
        )
    else:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM agents")
    return (await cursor.fetchone())["c"]


async def get_all_agent_names(conn: aiosqlite.Connection) -> set[str]:
    cursor = await conn.execute("SELECT name FROM agents")
    return {row["name"] for row in await cursor.fetchall()}


async def compute_leaderboard(conn: aiosqlite.Connection, baseline_score: float) -> list[dict]:
    cursor = await conn.execute("""
        SELECT
            a.id as agent_id, a.name as agent_name, a.experiments_completed,
            MIN(e.score) as best_score, e.id as best_experiment_id
        FROM agents a
        JOIN experiments e ON e.agent_id = a.id AND e.feasible = 1
        GROUP BY a.id
        ORDER BY best_score ASC
    """)
    rows = await cursor.fetchall()
    return [
        {
            "rank": i + 1,
            "agent_id": row["agent_id"],
            "agent_name": row["agent_name"],
            "best_score": row["best_score"],
            "best_experiment_id": row["best_experiment_id"],
            "experiments_completed": row["experiments_completed"],
            "improvement_pct": round(
                ((baseline_score - row["best_score"]) / baseline_score) * 100, 2
            ) if baseline_score > 0 else 0,
        }
        for i, row in enumerate(rows)
    ]
