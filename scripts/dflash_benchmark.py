#!/usr/bin/env python3
"""Run DFlash training + evaluation and output JSON results.

Usage:
    python3 scripts/dflash_benchmark.py [--eval-tier 1|2] [--skip-train]

Output: JSON to stdout with score, metrics, training script.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DFLASH_DIR = ROOT_DIR / "dflash"
CACHE_DIR = Path(os.environ.get("DFLASH_CACHE", os.path.expanduser("~/.cache/dflash_swarm")))
TRAIN_SCRIPT = DFLASH_DIR / "train.py"
EVAL_SCRIPT = DFLASH_DIR / "evaluate.py"

EVAL_TIER = 1
SKIP_TRAIN = False

for arg in sys.argv[1:]:
    if arg.startswith("--eval-tier"):
        EVAL_TIER = int(sys.argv[sys.argv.index(arg) + 1])
    if arg == "--skip-train":
        SKIP_TRAIN = True


def run_training():
    """Run train.py and capture output."""
    print("=== Running training ===", file=sys.stderr)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(DFLASH_DIR),
        capture_output=True,
        text=True,
        timeout=7200,
    )

    training_time = time.time() - t0
    print(result.stdout, file=sys.stderr)
    if result.returncode != 0:
        print(f"Training FAILED (exit code {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None, training_time

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("=") and not line.startswith("-"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val

    return metrics, training_time


def run_evaluation(tier: int):
    """Run evaluate.py and capture results."""
    print(f"=== Running Tier {tier} evaluation ===", file=sys.stderr)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--tier", str(tier)],
        cwd=str(DFLASH_DIR),
        capture_output=True,
        text=True,
        timeout=3600,
    )

    eval_time = time.time() - t0
    print(result.stdout, file=sys.stderr)
    if result.returncode != 0:
        print(f"Evaluation FAILED (exit code {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None, eval_time

    eval_results_path = CACHE_DIR / "eval_results.json"
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            return json.load(f), eval_time

    return None, eval_time


def main():
    training_metrics = None
    training_time = 0

    if not SKIP_TRAIN:
        training_metrics, training_time = run_training()
        if training_metrics is None:
            output = {
                "score": 0.0,
                "feasible": False,
                "error": "Training failed",
                "training_time": training_time,
            }
            print(json.dumps(output))
            sys.exit(1)

    eval_results, eval_time = run_evaluation(EVAL_TIER)
    if eval_results is None:
        output = {
            "score": 0.0,
            "feasible": False,
            "error": "Evaluation failed",
            "training_time": training_time,
            "eval_time": eval_time,
        }
        print(json.dumps(output))
        sys.exit(1)

    score = eval_results.get("score", eval_results.get("mean_accepted_length",
                              eval_results.get("estimated_acceptance", 0)))

    hyperparams = {}
    try:
        with open(TRAIN_SCRIPT) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#") and not line.startswith("def "):
                    if any(line.startswith(kw) for kw in [
                        "NUM_DRAFT_LAYERS", "NUM_TARGET_FEATURES", "BLOCK_SIZE",
                        "LR", "OPTIMIZER", "BETAS", "WEIGHT_DECAY", "WARMUP_STEPS",
                        "LR_SCHEDULE", "GRAD_CLIP", "BATCH_SIZE", "GAMMA",
                        "LABEL_SMOOTHING", "NUM_STEPS", "USE_EMA", "EMA_DECAY",
                    ]):
                        key, _, val = line.partition("=")
                        hyperparams[key.strip()] = val.split("#")[0].strip()
    except Exception:
        pass

    output = {
        "score": score,
        "feasible": True,
        "training_time": training_time,
        "eval_time": eval_time,
        "eval_tier": EVAL_TIER,
        "training_metrics": training_metrics or {},
        "eval_results": eval_results,
        "hyperparameters": hyperparams,
        "per_position_accuracy": eval_results.get("per_position_accuracy", []),
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
