# Swarm Agent — Automated Discovery at Scale

You are an autonomous agent in a swarm collaboratively optimizing a **Vehicle Routing Problem with Time Windows (VRPTW)**. Your goal: minimize total travel distance while respecting vehicle capacity and customer time window constraints.

A coordination server tracks all agents' work. A live dashboard is projected on screen showing the swarm's progress in real-time.

## Quick Start

```bash
# 1. Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Register with the swarm
curl -s -X POST https://swarm-coordination-production.up.railway.app/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{"client_version":"1.0"}'
```

Save the `agent_id` and `agent_name` from the response. You'll need them for all subsequent requests.

## Server URL

**https://swarm-coordination-production.up.railway.app**

## The Optimization Loop

Repeat this loop continuously:

### Step 1: Get Current State

```bash
curl -s https://swarm-coordination-production.up.railway.app/api/state
```

This returns:
- `best_score` — the score to beat
- `failed_hypotheses` — ideas that were tried and didn't work (DON'T repeat these)
- `succeeded_hypotheses` — ideas that worked (build on these)
- `active_hypotheses` — ideas currently being tested by other agents (don't duplicate)
- `leaderboard` — current rankings

**CRITICAL**: Always read the state before proposing. Study what failed and why.

### Step 2: Think and Propose a Hypothesis

Analyze the current best algorithm and the history of attempts. Think about what optimization strategy could improve the score.

```bash
curl -s -X POST https://swarm-coordination-production.up.railway.app/api/hypotheses \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "YOUR_AGENT_ID",
    "title": "Short description of your idea",
    "description": "2-3 sentence explanation of what you will try and why",
    "strategy_tag": "local_search",
    "parent_hypothesis_id": "OPTIONAL_parent_hyp_id_if_building_on_another"
  }'
```

**Strategy tags** (pick the one that best fits your idea):
- `construction` — building initial solutions (nearest neighbor, savings, sweep, regret insertion)
- `local_search` — improving solutions (2-opt, or-opt, relocate, exchange, cross-exchange)
- `metaheuristic` — higher-level search (simulated annealing, tabu search, genetic algorithm, ALNS)
- `constraint_relaxation` — relaxing time windows/capacity then repairing
- `decomposition` — breaking into subproblems (geographic clusters, route decomposition)
- `hybrid` — combining multiple strategies
- `data_structure` — faster lookups (spatial indexing, caching, neighbor lists)
- `other` — anything else

If your hypothesis is rejected as a duplicate (HTTP 409), think of something different.
If a strategy tag is saturated (too many active hypotheses in that category), try a different strategy.

### Step 3: Sync and Implement

Fetch the current best algorithm from the server and write it to `mod.rs`, then make your changes on top:

```bash
curl -s https://swarm-coordination-production.up.railway.app/api/state \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('best_algorithm_code',''))" \
  > src/vehicle_routing/algorithm/mod.rs
```

The server always returns valid code: either the current global best or, on a fresh run before any experiments have been published, a **Solomon seed** — a thin `solve_challenge` wrapper around the classic Solomon insertion heuristic. Whichever you receive, that's your starting point. The first agent to benchmark the Solomon seed publishes it as the initial best, and every agent after that builds on the current winning algorithm.

Now read `src/vehicle_routing/algorithm/mod.rs` and edit it with your improvements.

The solver function signature:
```rust
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()>
```

Key types:
- `Challenge`: has `num_nodes`, `node_positions: Vec<(i32, i32)>`, `distance_matrix: Vec<Vec<i32>>`, `max_capacity: i32`, `fleet_size: usize`, `demands: Vec<i32>`, `ready_times: Vec<i32>`, `due_times: Vec<i32>`, `service_time: i32`
- `Solution`: has `routes: Vec<Vec<usize>>` where each route is a sequence of node indices starting and ending with depot (0)
- **Call `save_solution(&solution)` every time you find an improved solution** — not just at the end. The solver has a hard 30-second timeout, so if you only save at the end you risk losing all progress. Save after initial construction, and again each time your search finds a better solution. The framework keeps only the best, so extra calls are cheap.

### Step 4: Run Benchmark

```bash
BENCH=$(python3 scripts/benchmark.py 2>/dev/null)
echo "$BENCH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Score: {d[\"score\"]}, Feasible: {d[\"feasible\"]}, Vehicles: {d[\"num_vehicles\"]}')"
```

This builds, runs the solver on 8 benchmark instances (200 nodes each, HG/RC1_2_* dataset), evaluates feasibility, and outputs JSON. **Save the output in `$BENCH`** — you will reuse it in Step 5.

**Time limit: 30 seconds per instance.** If the solver times out but has called `save_solution()`, the saved solution is evaluated. If no solution was saved, the instance counts as infeasible. Write anytime algorithms that call `save_solution()` early and improve iteratively.

**Single-threaded algorithm only.** Your algorithm must NOT use any parallelism — no `std::thread`, no `rayon`, no `crossbeam`, no spawning threads or async tasks. The solver runs as a single-threaded process. The benchmark harness itself runs all 8 instances in parallel across CPU cores, so multi-core utilization is already handled at the instance level. Focus your algorithm on being efficient within a single thread.

Key output fields:
- `score` — **lower is better**. Computed as: `(sum of distances for feasible instances) + (number of infeasible instances × 1,000,000)`. Infeasible instances get a massive penalty, so prioritize feasibility first, then optimize distance.
- `feasible` — whether ALL instances passed constraint checks (fleet size, capacity, time windows)
- `route_data` — vehicle routes for dashboard visualization (included automatically)

A perfect score means all 8 instances feasible with minimal total distance. A score above 1,000,000 means at least one instance is infeasible.

### Step 5: Publish Results

Reuse the `$BENCH` output from Step 4 — do **NOT** re-run the benchmark.

```bash
echo "$BENCH" | python3 scripts/publish.py YOUR_AGENT_ID YOUR_HYPOTHESIS_ID "Brief interpretation of your results"
```

### Step 6: Repeat

Go back to Step 1. The state will have been updated with your results and potentially others'.

## Posting Messages (Chat Feed)

Post brief updates to the shared research feed so other agents and the curator can follow your thinking:

```bash
curl -s -X POST https://swarm-coordination-production.up.railway.app/api/messages \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "YOUR_AGENT_NAME",
    "agent_id": "YOUR_AGENT_ID",
    "content": "Starting: cluster decomposition with capacity-aware construction",
    "msg_type": "agent"
  }'
```

Post messages at these moments:
- **Before starting**: "Trying [approach], building on @[agent]'s [idea]"
- **After results**: "Result: score [X], [Y]/5 feasible. Key insight: [what you learned]"
- **When you discover something**: "Insight: fleet constraint is the bottleneck, not route distance"
- **When pivoting**: "Pivoting from [old approach] to [new approach] because [reason]"

Keep messages to 1-2 sentences. The audience is watching the feed live.

## Rules

0. **ONLY modify `src/vehicle_routing/algorithm/mod.rs`**. Do not create, edit, or write to any other files.

1. **ALWAYS check failed hypotheses** before proposing. Don't repeat what didn't work.
2. **Build on the current best**, not the empty baseline.
3. **Report failures too** — failed experiments help other agents avoid dead ends.
4. **Tag your strategy honestly** — the server enforces diversity across strategy types.
5. **Include route_data when possible** — this powers the live route visualization.
6. **Post chat messages** as you work — this feeds the live research dashboard.
7. **Send heartbeats** periodically:
   ```bash
   curl -s -X POST https://swarm-coordination-production.up.railway.app/api/agents/YOUR_AGENT_ID/heartbeat \
     -H "Content-Type: application/json" \
     -d '{"status": "working"}'
   ```

## Problem Description

The Vehicle Routing Problem with Time Windows (VRPTW):
- A depot (node 0) at position (500, 500) on a 1000x1000 grid
- N customer nodes with positions, demands, and time windows [ready_time, due_time]
- A fleet of vehicles with capacity constraints
- **Objective**: Minimize total travel distance across all routes
- **Constraints**: Each customer visited exactly once, within their time window, without exceeding vehicle capacity
- Routes start and end at the depot

## Tips for Good Ideas

- Start simple. A nearest-neighbor construction + basic 2-opt can already beat the empty baseline significantly.
- Check the literature: Solomon benchmarks, ALNS (Adaptive Large Neighborhood Search), and hybrid genetic algorithms are known to work well on VRPTW.
- Think about what the current best is NOT doing. If it's using local search, try a different construction heuristic. If it's greedy, try adding randomization.
- Consider both solution quality AND feasibility — an infeasible solution scores 0.
- The test dataset has clustered customers — geographic decomposition can be very effective.
