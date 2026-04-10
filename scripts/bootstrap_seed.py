#!/usr/bin/env python3
"""Reset the swarm server and publish a real Solomon benchmark as row #1.

Steps:
  1. POST /api/admin/reset (wipes experiments, hypotheses, agents, messages).
  2. Swap src/vehicle_routing/algorithm/mod.rs with server/seed_algorithm.rs.
  3. Run scripts/benchmark.py to get a real Solomon score + route_data.
  4. Register a bootstrap agent and create a "construction" hypothesis.
  5. POST the benchmark to /api/experiments so row #1 is a real Solomon run.
  6. Restore the original mod.rs (always, even on failure).
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SERVER = "https://swarm-coordination-production.up.railway.app"
ROOT = Path(__file__).parent.parent
SEED = ROOT / "server/seed_algorithm.rs"
ALGO = ROOT / "src/vehicle_routing/algorithm/mod.rs"
ADMIN_KEY = "ads-2026"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    if not SEED.exists():
        log(f"seed not found: {SEED}")
        return 1

    log("1/6 resetting coordination server...")
    log(f"    {post('/api/admin/reset', {'admin_key': ADMIN_KEY})}")

    backup = ALGO.read_text() if ALGO.exists() else None
    try:
        log("2/6 writing Solomon seed into mod.rs...")
        ALGO.write_text(SEED.read_text())

        log("3/6 running benchmark (build + 8 instances)...")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/benchmark.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            log(proc.stderr)
            return 2
        bench = json.loads(proc.stdout)
        log(
            f"    score={bench['score']} feasible={bench['feasible']} "
            f"vehicles={bench['num_vehicles']} "
            f"({bench['instances_feasible']}/{bench['instances_solved']} feasible)"
        )

        log("4/6 registering bootstrap agent...")
        agent = post("/api/agents/register", {"client_version": "1.0"})
        agent_id = agent["agent_id"]
        log(f"    agent_id={agent_id} name={agent['agent_name']}")

        log("5/6 creating bootstrap hypothesis...")
        hyp = post(
            "/api/hypotheses",
            {
                "agent_id": agent_id,
                "title": "Bootstrap: Solomon I1 insertion (seed)",
                "description": (
                    "Establish the initial DB row by benchmarking the unmodified "
                    "Solomon I1 sequential-insertion seed. This is the reference "
                    "score every other hypothesis builds on."
                ),
                "strategy_tag": "construction",
                "auto_claim": True,
            },
        )
        hyp_id = hyp["hypothesis_id"]
        log(f"    hypothesis_id={hyp_id} status={hyp.get('status')}")

        log("6/6 publishing experiment...")
        payload = {
            "agent_id": agent_id,
            "hypothesis_id": hyp_id,
            "algorithm_code": SEED.read_text(),
            "score": bench["score"],
            "feasible": bench["feasible"],
            "num_vehicles": bench["num_vehicles"],
            "total_distance": bench.get("total_distance", bench["score"]),
            "notes": "Initial Solomon I1 insertion benchmark (bootstrap seed row).",
            "route_data": bench.get("route_data"),
        }
        result = post("/api/experiments", payload)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if backup is not None:
            ALGO.write_text(backup)
            log("restored original mod.rs")


if __name__ == "__main__":
    sys.exit(main())
