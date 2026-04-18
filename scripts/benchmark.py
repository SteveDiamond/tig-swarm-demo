#!/usr/bin/env python3
"""Run VRPTW benchmark and output JSON results. Robust to solver failures."""

import subprocess
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "datasets/vehicle_routing/HG")
INSTANCES = [
    "R1_4_1.txt",
    "R1_4_2.txt",
    "R1_4_3.txt",
    "R1_4_4.txt",
    "R1_4_5.txt",
    "R2_4_1.txt",
    "R2_4_2.txt",
    "R2_4_3.txt",
    "R2_4_4.txt",
    "R2_4_5.txt",
    "RC1_4_1.txt",
    "RC1_4_2.txt",
    "RC1_4_3.txt",
    "RC1_4_4.txt",
    "RC1_4_5.txt",
    "RC2_4_1.txt",
    "RC2_4_2.txt",
    "RC2_4_3.txt",
    "RC2_4_4.txt",
    "RC2_4_5.txt",
    "C1_4_1.txt",
    "C1_4_2.txt",
    "C2_4_1.txt",
    "C2_4_2.txt",
]
SOLVER_TIMEOUT = 30

def build():
    subprocess.run(
        ["cargo", "build", "-r", "--bin", "tig_solver", "--features", "solver,vehicle_routing"],
        cwd=ROOT_DIR, check=True, capture_output=True,
    )
    subprocess.run(
        ["cargo", "build", "-r", "--bin", "tig_evaluator", "--features", "evaluator,vehicle_routing"],
        cwd=ROOT_DIR, check=True, capture_output=True,
    )

def parse_instance_positions(inst_path: str) -> dict:
    """Parse node positions from Solomon-format instance file."""
    positions = {}
    in_customer = False
    with open(inst_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("CUST NO"):
                in_customer = True
                continue
            if in_customer and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        positions[int(parts[0])] = (int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
    return positions

def parse_solution_routes(sol_path: str) -> list:
    """Parse routes from solution file."""
    routes = []
    with open(sol_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Route"):
                parts = line.split(":")
                if len(parts) == 2:
                    nodes = [int(x) for x in parts[1].split() if x.strip()]
                    routes.append(nodes)
    return routes

def make_route_data(inst_path: str, sol_path: str) -> dict | None:
    """Build route_data JSON for dashboard visualization."""
    positions = parse_instance_positions(inst_path)
    routes = parse_solution_routes(sol_path)
    if not positions or not routes:
        return None
    depot = positions.get(0, (500, 500))
    route_data = {
        "depot": {"x": depot[0], "y": depot[1]},
        "routes": [],
    }
    for i, route_nodes in enumerate(routes):
        path = []
        for node in route_nodes:
            if node in positions:
                path.append({"x": positions[node][0], "y": positions[node][1], "customer_id": node})
        route_data["routes"].append({"vehicle_id": i, "path": path})
    return route_data


def _first_nonempty_line(*chunks: str) -> str:
    for chunk in chunks:
        if not chunk:
            continue
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                return line
    return ""


def parse_evaluator_distance(eval_result: subprocess.CompletedProcess) -> tuple[int | None, str | None]:
    """Parse strict evaluator JSON output.

    Returns `(distance, None)` on success, `(None, error)` on failure.
    """
    if eval_result.returncode != 0:
        msg = _first_nonempty_line(eval_result.stderr, eval_result.stdout)
        return None, msg or f"evaluator failed (exit {eval_result.returncode})"

    stdout = (eval_result.stdout or "").strip()
    if not stdout:
        return None, "evaluator produced no output"

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        sample = _first_nonempty_line(stdout)
        return None, f"invalid evaluator JSON: {sample or 'empty'}"

    distance = payload.get("distance")
    if not isinstance(distance, (int, float)):
        return None, "evaluator JSON missing numeric distance"
    if distance < 0:
        return None, "evaluator distance must be non-negative"

    return int(round(distance)), None

def run_instance(instance_name, dataset_path, solver, evaluator):
    """Run solver + evaluator for a single instance. Returns a result dict."""
    inst = dataset_path / instance_name
    with tempfile.NamedTemporaryFile(suffix=".solution", delete=False) as tmp:
        sol_path = tmp.name
    try:
        result = subprocess.run(
            [solver, "vehicle_routing", str(inst), sol_path],
            capture_output=True, text=True, timeout=SOLVER_TIMEOUT,
        )
        if result.returncode != 0 or not os.path.exists(sol_path):
            return {"instance": instance_name, "error": "solver failed", "feasible": False}

        routes = parse_solution_routes(sol_path)
        eval_result = subprocess.run(
            [evaluator, "vehicle_routing", str(inst), sol_path],
            capture_output=True, text=True, timeout=SOLVER_TIMEOUT,
        )
        dist, err = parse_evaluator_distance(eval_result)
        if err:
            return {"instance": instance_name, "error": err, "num_vehicles": len(routes), "feasible": False}
        rd = make_route_data(str(inst), sol_path)
        return {"instance": instance_name, "dist": dist, "num_vehicles": len(routes), "feasible": True, "route_data": rd}

    except subprocess.TimeoutExpired:
        # Solver timed out, but save_solution() may have written a partial solution
        if os.path.exists(sol_path) and os.path.getsize(sol_path) > 0:
            try:
                routes = parse_solution_routes(sol_path)
                eval_result = subprocess.run(
                    [evaluator, "vehicle_routing", str(inst), sol_path],
                    capture_output=True, text=True, timeout=SOLVER_TIMEOUT,
                )
                dist, err = parse_evaluator_distance(eval_result)
                if err:
                    return {"instance": instance_name, "error": err, "num_vehicles": len(routes), "feasible": False}
                rd = make_route_data(str(inst), sol_path)
                return {"instance": instance_name, "dist": dist, "num_vehicles": len(routes), "feasible": True, "route_data": rd}
            except Exception:
                return {"instance": instance_name, "error": "timeout (evaluation failed)", "feasible": False}
        return {"instance": instance_name, "error": "timeout (no solution saved)", "feasible": False}
    finally:
        if os.path.exists(sol_path):
            os.unlink(sol_path)


def main():
    print("Building solver...", file=sys.stderr)
    build()
    print(f"Running benchmark on {DATASET}...", file=sys.stderr)

    solver = str(ROOT_DIR / "target/release/tig_solver")
    evaluator = str(ROOT_DIR / "target/release/tig_evaluator")

    total_dist = 0
    total_vehicles = 0
    solved = 0
    feasible_count = 0
    infeasible_count = 0
    errors = []
    all_route_data = {}

    dataset_path = Path(DATASET)
    workers = min(len(INSTANCES), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_instance, name, dataset_path, solver, evaluator): name
            for name in INSTANCES
        }
        for future in as_completed(futures):
            r = future.result()
            if "error" in r:
                errors.append(f"{r['instance']}: {r['error']}")
                infeasible_count += 1
                solved += 1
            else:
                solved += 1
                feasible_count += 1
                total_dist += r["dist"]
                total_vehicles += r["num_vehicles"]
                if r.get("route_data"):
                    all_route_data[r["instance"]] = r["route_data"]

    all_feasible = infeasible_count == 0 and feasible_count > 0
    PENALTY_PER_INFEASIBLE = 1_000_000
    num_instances = len(INSTANCES)
    score = (total_dist + infeasible_count * PENALTY_PER_INFEASIBLE) / max(num_instances, 1)

    result = {
        "score": score,
        "total_distance": total_dist,
        "num_vehicles": total_vehicles,
        "feasible": all_feasible,
        "instances_solved": solved,
        "instances_feasible": feasible_count,
        "instances_infeasible": infeasible_count,
        "route_data": all_route_data if all_route_data else None,
        "errors": errors if errors else None,
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
