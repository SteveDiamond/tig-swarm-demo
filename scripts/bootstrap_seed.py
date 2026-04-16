#!/usr/bin/env python3
"""Reset the swarm server and publish a real Solomon benchmark as row #1.

Steps:
  1. POST /api/admin/reset (wipes experiments, hypotheses, agents, messages).
  2. Overwrite src/vehicle_routing/algorithm/mod.rs with server/seed_algorithm.rs.
  3. Run scripts/benchmark.py to get a real Solomon score + route_data.
  4. Register a bootstrap agent and create a "construction" hypothesis.
  5. POST the benchmark to /api/experiments so row #1 is a real Solomon run.

mod.rs is intentionally left at the Solomon seed after the script exits —
any leftover algorithm code from a previous experiment would otherwise be
the starting point for the next agent that inspects the local tree, even
though the server's best_algorithm_code is already the seed.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "https://demo.discoveryatscale.com"
ROOT = Path(__file__).parent.parent
SEED = ROOT / "server/seed_algorithm.rs"
ALGO = ROOT / "src/vehicle_routing/algorithm/mod.rs"
ADMIN_KEY = "ads-2026"


def post(path: str, payload: dict, retries: int = 4) -> dict:
    # The benchmark step blocks for ~30s, which is exactly the window in
    # which a Railway redeploy (triggered by a fresh push) can flip the
    # upstream and cause a transient 502/503. Retry a few times with
    # backoff so a rolling deploy doesn't kill the whole bootstrap run.
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    if not SEED.exists():
        log(f"seed not found: {SEED}")
        return 1

    log("1/6 resetting coordination server...")
    log(f"    {post('/api/admin/reset', {'admin_key': ADMIN_KEY})}")

    log("2/6 overwriting mod.rs with the Solomon seed...")
    ALGO.write_text(SEED.read_text())

    log("3/6 running benchmark (build + 24 instances)...")
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
    log("mod.rs left at the Solomon seed — ready for the next agent run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
