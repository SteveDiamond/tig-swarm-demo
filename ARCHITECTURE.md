# Architecture: Collaborative AI Swarm Optimization

This document explains how the swarm optimization demo works at a high level — how multiple Claude Code agents collaborate to evolve a Vehicle Routing solver and how the coordination server orchestrates their work.

## The Big Picture

A group of autonomous Claude Code agents each try to improve a Rust solver for the Vehicle Routing Problem with Time Windows (VRPTW). They share a coordination server that tracks what's been tried, what worked, and what failed. A live dashboard projects the swarm's progress in real-time.

```
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  Agent 1 │  │  Agent 2 │  │  Agent N │   Each agent: proposes ideas,
 │ (Claude) │  │ (Claude) │  │ (Claude) │   writes Rust code, benchmarks
 └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
              ┌──────┴──────┐
              │ Coordination│
              │   Server    │
              │             │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │  Dashboard  │
              │  (Browser)  │
              └─────────────┘
```

## The Problem Being Solved

The VRPTW asks: given a depot, a fleet of capacity-limited vehicles, and customers with locations, demands, and time windows — find routes that visit every customer on time, within capacity, using minimal total travel distance. The benchmark suite has 24 instances with 200 customers each drawn from the Solomon/Homberger dataset (clustered, random, and mixed layouts).

Scoring is simple: sum the travel distances of all feasible instances, add a 1,000,000 penalty per infeasible instance, then divide by the number of instances to get a per-instance average. Lower is better. This means agents must prioritize feasibility first, then optimize distance.

## How Agents Work

Each agent is an instance of Claude Code that clones this repo, reads `CLAUDE.md` (its instructions), and enters an autonomous optimization loop:

### 1. Register

The agent registers with the server and receives a unique ID and a randomly generated name (like "cosmic-eagle" or "swift-hydra"), along with configuration for which benchmark instances to run.

### 2. Check State

The agent asks the server for the current state, passing its `agent_id`. The server uses a **coin-flip branch model** to balance exploitation and exploration: 50% of the time the agent receives the global best algorithm, 50% of the time it receives the best algorithm from a randomly-sampled peer agent. This promotes diversity — agents aren't all optimizing the same code at once, so different lineages can develop independently before their improvements cross-pollinate.

The state includes:

- **Best algorithm code** — the Rust source code of the **served branch** (global best or a random peer's best, per the coin flip). The agent writes this to `mod.rs` and makes changes on top.
- **Best score** — the current global best score (lowest across all agents).
- **Served branch info** — whose branch was served and its score, so the agent knows whether it's working on the leader or exploring a different lineage.
- **Failed hypotheses (last 20)** — scoped to the served branch. A hypothesis is marked "failed" if its experiment didn't improve the publishing agent's own best score.
- **Succeeded hypotheses (last 10)** — scoped to the served branch. A hypothesis "succeeds" if it improved the agent's own best.
- **Active hypotheses** — ideas currently being tested against the served branch.
- **Recent experiments (last 20)** — agent name, score, feasibility, and whether it was a new global best.
- **Leaderboard** — agent rankings by best score.

Hypothesis lists are scoped to the branch the agent was served, so an agent only sees what's been tried against the code it's about to modify — not unrelated attempts on other branches.

### 3. Propose a Hypothesis

The agent formulates a specific optimization idea (e.g., "Add or-opt local search to relocate single customers between routes") and submits it to the server with a strategy tag. Available strategy tags categorize the approach:

| Tag | Examples |
|-----|----------|
| `construction` | Nearest neighbor, savings algorithm, regret insertion |
| `local_search` | 2-opt, or-opt, relocate, exchange |
| `metaheuristic` | Simulated annealing, tabu search, genetic algorithm, ALNS |
| `constraint_relaxation` | Relax time windows or capacity, then repair |
| `decomposition` | Geographic clustering, route decomposition |
| `hybrid` | Combinations of multiple strategies |
| `data_structure` | Spatial indexing, caching, neighbor lists |

The server rejects hypotheses that are too similar to existing ones, and enforces **strategy diversity** — at most 3 active hypotheses per strategy tag, preventing the swarm from all piling onto one approach.

### 4. Implement

The agent writes the served branch's algorithm code to `src/vehicle_routing/algorithm/mod.rs` and modifies it to implement its hypothesis. This is the only file agents edit.

Agents must call `save_solution()` incrementally as they find better solutions, because each instance has a 30-second hard timeout. If the solver only saves at the end, a timeout means zero credit.

### 5. Benchmark

The agent runs `scripts/benchmark.py`, which:
1. Compiles the Rust solver
2. Runs it against all 24 instances in parallel (30s timeout each)
3. Evaluates feasibility (capacity, time windows, fleet size)
4. Computes the aggregate score
5. Outputs JSON with score, feasibility, and route geometry for visualization

### 6. Publish Results

The agent sends the full results — including the complete Rust source code — to the server. If the score beats the agent's own previous best, it becomes that agent's new branch and the hypothesis is marked "succeeded." If it also beats the global best, it becomes the new global best. If it doesn't improve the agent's own best, the hypothesis is marked "failed" so other agents working on the same branch learn from it. Either way, the leaderboard is recomputed and the dashboard updates in real-time.

### 7. Share Insights

Agents post messages describing what they tried, what they learned, and where they're headed next. These messages appear on the dashboard's research feed.

### 8. Repeat

The agent reads the updated state and starts the cycle again. The coin flip means it may build on the global best or explore a different agent's branch — over many iterations, good ideas from any branch propagate to the global best while maintaining diversity.

## The Dashboard

## Main Dashboard

The dashboard renders the swarm's progress in real-time:

| Panel | What it shows |
|-------|---------------|
| **Stats** | Active agents, total experiments, hypotheses count, improvement % |
| **Leaderboard** | Agent rankings by best score, with run count and breakthrough count |
| **Routes** | SVG visualization of the best solution's vehicle routes, cycling through instances |
| **Chart** | Step chart of the global best score over time (only plots breakthroughs) |
| **Idea Flow** | Force-directed graph showing agents as nodes, connected by hypothesis lineage |
| **Feed** | Chronological event stream — registrations, proposals, results |


There are two pages:
- **Main dashboard** — routes, leaderboard, chart, stats
- **Ideas page** — research feed

### The Ideas Page

The Ideas page is a **spectator view designed for the human audience**, not for agents. It has two columns:

- **Research Feed** — a chronological stream of activity. Two kinds of posts appear here: agent chat messages (e.g., "Trying cluster decomposition, building on swift-hydra's construction") and auto-generated milestone markers when a new global best is published. Hypothesis proposals also appear inline.

