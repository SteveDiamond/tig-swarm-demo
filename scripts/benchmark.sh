#!/bin/bash
# Run the vehicle routing benchmark and output JSON results.
# Robust to solver failures — reports what it can.
# Usage: ./scripts/benchmark.sh [dataset_dir]

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-datasets/vehicle_routing/demo}"
RESULTS_DIR="/tmp/tig_benchmark_$$"
mkdir -p "$RESULTS_DIR"

# Build solver
echo "Building solver..." >&2
cargo build -r --bin tig_solver --features "solver,vehicle_routing" 2>&1 >&2
cargo build -r --bin tig_evaluator --features "evaluator,vehicle_routing" 2>&1 >&2

echo "Running benchmark on $DATASET..." >&2

TOTAL_DIST=0
TOTAL_VEHICLES=0
SOLVED=0
FEASIBLE=0
INFEASIBLE=0
ERRORS=""
ROUTE_JSON="["

for dir in "$DATASET"/n_nodes=*/; do
  [ -d "$dir" ] || continue
  for inst in "$dir"*.txt; do
    [[ "$inst" == *.solution ]] && continue
    [ -f "$inst" ] || continue

    instance_name="$(basename "$(dirname "$inst")")/$(basename "$inst")"
    sol="$RESULTS_DIR/$(echo "$instance_name" | tr '/' '_').solution"

    # Run solver (timeout 30s per instance)
    if timeout 30 ./target/release/tig_solver vehicle_routing "$inst" "$sol" 2>/dev/null; then
      if [ -f "$sol" ]; then
        SOLVED=$((SOLVED + 1))
        NV=$(grep -c "^Route" "$sol" 2>/dev/null || echo "0")
        TOTAL_VEHICLES=$((TOTAL_VEHICLES + NV))

        # Evaluate
        eval_result=$(./target/release/tig_evaluator vehicle_routing "$inst" "$sol" 2>&1)
        if echo "$eval_result" | grep -q "Error"; then
          INFEASIBLE=$((INFEASIBLE + 1))
          err_msg=$(echo "$eval_result" | head -1)
          ERRORS="${ERRORS}${instance_name}: ${err_msg}\n"
        else
          FEASIBLE=$((FEASIBLE + 1))
          dist=$(echo "$eval_result" | grep -oE '[0-9]+' | head -1)
          if [ -n "$dist" ]; then
            TOTAL_DIST=$((TOTAL_DIST + dist))
          fi
        fi
      fi
    else
      ERRORS="${ERRORS}${instance_name}: solver timeout or crash\n"
    fi
  done
done

# Build route data JSON from the last feasible solution (for visualization)
# Parse node positions from instance + routes from solution
ROUTE_DATA=""
for dir in "$DATASET"/n_nodes=*/; do
  [ -d "$dir" ] || continue
  for inst in "$dir"*.txt; do
    [[ "$inst" == *.solution ]] && continue
    [ -f "$inst" ] || continue
    instance_name="$(basename "$(dirname "$inst")")/$(basename "$inst")"
    sol="$RESULTS_DIR/$(echo "$instance_name" | tr '/' '_').solution"
    [ -f "$sol" ] || continue

    # Only use if evaluator passes
    eval_result=$(./target/release/tig_evaluator vehicle_routing "$inst" "$sol" 2>&1)
    echo "$eval_result" | grep -q "Error" && continue

    # Extract route data
    ROUTE_DATA=$(python3 -c "
import sys

# Parse instance for node positions
positions = {}
in_customer = False
with open('$inst') as f:
    for line in f:
        line = line.strip()
        if line.startswith('CUST NO'):
            in_customer = True
            continue
        if in_customer and line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = int(parts[1])
                    y = int(parts[2])
                    positions[node_id] = (x, y)
                except ValueError:
                    pass

# Parse solution for routes
import json
routes = []
with open('$sol') as f:
    for line in f:
        line = line.strip()
        if line.startswith('Route'):
            parts = line.split(':')
            if len(parts) == 2:
                nodes = [int(x) for x in parts[1].split() if x.strip()]
                path = []
                for n in nodes:
                    if n in positions:
                        path.append({'x': positions[n][0], 'y': positions[n][1], 'customer_id': n})
                routes.append({'vehicle_id': len(routes), 'path': path})

depot = positions.get(0, (500, 500))
print(json.dumps({'depot': {'x': depot[0], 'y': depot[1]}, 'routes': routes}))
" 2>/dev/null)
    [ -n "$ROUTE_DATA" ] && break 2
  done
done

# Output JSON
python3 -c "
import json
result = {
    'score': $TOTAL_DIST if $FEASIBLE > 0 else 0,
    'total_distance': $TOTAL_DIST,
    'num_vehicles': $TOTAL_VEHICLES,
    'feasible': $INFEASIBLE == 0 and $FEASIBLE > 0,
    'instances_solved': $SOLVED,
    'instances_feasible': $FEASIBLE,
    'instances_infeasible': $INFEASIBLE,
    'avg_distance': round($TOTAL_DIST / max($FEASIBLE, 1), 1),
    'runtime_seconds': 0,
    'route_data': json.loads('''${ROUTE_DATA:-null}''') if '''${ROUTE_DATA}''' != '' else None,
    'errors': '''$(echo -e "$ERRORS")'''.strip() if '''$(echo -e "$ERRORS")'''.strip() else None,
}
print(json.dumps(result, indent=2))
"

# Cleanup
rm -rf "$RESULTS_DIR"
