#!/usr/bin/env python3
"""Run VRPTW benchmark and output JSON results. Robust to solver failures."""

import subprocess
import json
import os
import sys
import tempfile
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "datasets/vehicle_routing/demo")

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
    best_route_data = None
    best_instance_dist = float("inf")

    dataset_path = Path(DATASET)
    for track_dir in sorted(dataset_path.glob("n_nodes=*")):
        for inst in sorted(track_dir.glob("*.txt")):
            if ".solution" in inst.name:
                continue
            instance_name = f"{track_dir.name}/{inst.name}"

            with tempfile.NamedTemporaryFile(suffix=".solution", delete=False) as tmp:
                sol_path = tmp.name

            try:
                # Run solver
                result = subprocess.run(
                    [solver, "vehicle_routing", str(inst), sol_path],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0 or not os.path.exists(sol_path):
                    errors.append(f"{instance_name}: solver failed")
                    continue

                solved += 1
                routes = parse_solution_routes(sol_path)
                total_vehicles += len(routes)

                # Evaluate
                eval_result = subprocess.run(
                    [evaluator, "vehicle_routing", str(inst), sol_path],
                    capture_output=True, text=True, timeout=10,
                )
                output = (eval_result.stdout + eval_result.stderr).strip()

                if "Error" in output:
                    infeasible_count += 1
                    errors.append(f"{instance_name}: {output.split(chr(10))[0]}")
                else:
                    feasible_count += 1
                    # Extract distance from evaluator output
                    nums = re.findall(r"\d+", output)
                    if nums:
                        dist = int(nums[0])
                        total_dist += dist
                        # Keep route data from a feasible instance
                        if dist < best_instance_dist:
                            best_instance_dist = dist
                            best_route_data = make_route_data(str(inst), sol_path)

            except subprocess.TimeoutExpired:
                errors.append(f"{instance_name}: timeout")
            finally:
                if os.path.exists(sol_path):
                    os.unlink(sol_path)

    all_feasible = infeasible_count == 0 and feasible_count > 0
    # Score: total distance if all feasible, otherwise huge penalty per infeasible instance
    PENALTY_PER_INFEASIBLE = 1_000_000
    score = total_dist + infeasible_count * PENALTY_PER_INFEASIBLE if feasible_count > 0 else float("inf")

    result = {
        "score": score,
        "total_distance": total_dist,
        "num_vehicles": total_vehicles,
        "feasible": all_feasible,
        "instances_solved": solved,
        "instances_feasible": feasible_count,
        "instances_infeasible": infeasible_count,
        "route_data": best_route_data,
        "errors": errors if errors else None,
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
