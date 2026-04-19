#!/usr/bin/env python3
"""Publish DFlash training results to the swarm coordination server.

Usage:
    python3 scripts/dflash_benchmark.py 2>/dev/null \
      | python3 scripts/dflash_publish.py AGENT_ID "title" "description" strategy_tag "notes"
"""

import json
import sys
import urllib.request
from pathlib import Path

SERVER = "https://demo.discoveryatscale.com"
TRAIN_PATH = Path(__file__).parent.parent / "dflash/train.py"


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python3 scripts/dflash_publish.py <agent_id> <title> <description> <strategy_tag> [notes]",
            file=sys.stderr,
        )
        sys.exit(1)

    agent_id = sys.argv[1]
    title = sys.argv[2]
    description = sys.argv[3]
    strategy_tag = sys.argv[4]
    notes = sys.argv[5] if len(sys.argv) > 5 else ""

    bench = json.load(sys.stdin)
    code = TRAIN_PATH.read_text()

    payload = {
        "agent_id": agent_id,
        "title": title,
        "description": description,
        "strategy_tag": strategy_tag,
        "algorithm_code": code,
        "score": bench["score"],
        "feasible": bench.get("feasible", True),
        "notes": notes,
        "training_metrics": bench.get("training_metrics"),
        "hyperparameters": bench.get("hyperparameters"),
        "per_position_accuracy": bench.get("per_position_accuracy"),
    }

    req = urllib.request.Request(
        f"{SERVER}/api/iterations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
