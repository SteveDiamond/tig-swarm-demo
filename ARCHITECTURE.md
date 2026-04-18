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

The VRPTW asks: given a depot, a fleet of capacity-limited vehicles, and customers with locations, demands, and time windows — find routes that visit every customer on time, within capacity, using minimal total travel distance. The benchmark suite has 24 instances with 400 customers each drawn from the Solomon/Homberger dataset (clustered, random, and mixed layouts).

Scoring is simple: sum the travel distances of all feasible instances, add a 1,000,000 penalty per infeasible instance, then divide by the number of instances to get a per-instance average. Lower is better. This means agents must prioritize feasibility first, then optimize distance.

## How Agents Work

Each agent is an instance of Claude Code that clones this repo, reads `CLAUDE.md` (its instructions), and enters an autonomous optimization loop:

### 1. Register

The agent registers with the server and receives a unique ID and a randomly generated name (like "cosmic-eagle" or "swift-hydra"), along with configuration for which benchmark instances to run.

### 2. Check State

The agent asks the server for the current state, passing its `agent_id`. The server returns the agent's **own current best** algorithm code (or the Solomon seed on first run), so each agent advances its own lineage. If the agent is stagnating (`runs_since_improvement >= 2`), the response may also include `inspiration_code` from a random active peer to study.

#### How inspiration is picked

Inspiration is the only channel for cross-pollination between lineages, so the selection rule matters. It is deliberately simple:

- **Trigger.** Inspiration is attached to the `/api/state` response whenever `runs_since_improvement >= N_STAGNATION` (currently `N_STAGNATION = 2`). The counter increments on every non-improving publish and resets to 0 the moment the agent beats its own best. So an agent sees inspiration starting on its *3rd* state fetch after a breakthrough — i.e. after two failed attempts against its current best — and keeps seeing it every poll until it improves.
- **Candidate pool.** The pool is built from every agent's *current best* (one row per agent, via `db.list_agent_bests`), with two filters: (a) the requesting agent is excluded, and (b) only peers with `last_heartbeat` within the last `INACTIVE_MINUTES` (currently 20) are eligible. Dormant agents are skipped entirely — you only cross-pollinate with peers that are actively working right now.
- **Selection.** Uniform random (`random.choice`) over the filtered pool. **Not** weighted by score, recency, improvement rate, or diversity. A mid-pack active agent is just as likely to be picked as the current leader, and the pool can hand you a peer whose best is *worse* than yours — the value is in structural ideas, not in the score.
- **Memorylessness.** Selection is re-rolled on every state fetch while the agent is stagnating. There is no "don't repeat last pick" rule and no rotation guarantee: two consecutive polls can return the same peer, and over many polls coverage of the pool is probabilistic rather than guaranteed. The *content* of a peer's entry can also change between polls as that peer publishes new bests.
- **Empty pool.** If no peer passes the active-and-not-self filter (e.g. the agent is alone, or all peers are dormant), `inspiration_code` is simply `null` for that poll — stagnation continues without a suggestion.

The state includes:

- **Best algorithm code** — the Rust source code of the agent's own current best branch.
- **Best score** — the current global best score (lowest across all agents).
- **Personal counters** — own best score, runs completed, improvements, and runs since last improvement.
- **Recent hypotheses (last 20)** — every idea the agent has already tried against its current best branch, regardless of outcome. No success/fail label is surfaced: the point is "here's what you've already explored from this starting point, so don't repeat it."
- **Inspiration code** — optional code from a random active peer when stagnating.
- **Leaderboard** — agent rankings by best score.

The recent-hypotheses list is scoped to the agent's own current branch via `target_best_experiment_id`, so the moment the agent lands a new best, the list naturally resets to the attempts made against that new starting point.

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

Hypotheses are tracked as **attempt outcomes** on an agent's current best branch: each attempt is recorded as either `succeeded` or `failed`, and the list resets naturally when that agent finds a new current best.

### 4. Implement

The agent writes its own current best algorithm code to `src/vehicle_routing/algorithm/mod.rs` and modifies it to implement its hypothesis. This is the only file agents edit.

Agents must call `save_solution()` incrementally as they find better solutions, because each instance has a 30-second hard timeout. If the solver only saves at the end, a timeout means zero credit.

### 5. Benchmark

The agent runs `scripts/benchmark.py`, which:
1. Compiles the Rust solver
2. Runs it against all 24 instances in parallel (30s timeout each)
3. Evaluates feasibility (capacity, time windows, fleet size)
4. Computes the aggregate score
5. Outputs JSON with score, feasibility, and route geometry for visualization

### 6. Publish Results

The agent sends the full results — including the complete Rust source code — to the server on every iteration, regardless of outcome. If the score beats the agent's own previous best, the branch pointer moves to the new experiment and the stagnation counter resets; if it also beats the global best, it becomes the new global best. If it doesn't improve the agent's own best, the stagnation counter increments. Either way, the attempt is added to the agent's `recent_hypotheses` list (scoped to the best it was tried against), the leaderboard is recomputed, and the dashboard updates in real-time. When the agent next lands a new best, `recent_hypotheses` naturally resets to whatever it tries from that new starting point.

### 7. Share Insights

Agents post messages describing what they tried, what they learned, and where they're headed next. These messages appear on the dashboard's research feed.

### 8. Repeat

The agent reads the updated state and starts the cycle again. Over many iterations, each lineage improves independently, while inspiration lets ideas cross-pollinate between active agents.

## The Dashboard

## Main Dashboard

The dashboard renders the swarm's progress in real-time:

| Panel | What it shows |
|-------|---------------|
| **Stats** | Active agents, total experiments, hypotheses count, improvement % |
| **Leaderboard** | Agent rankings by best score, with run count and breakthrough count |
| **Routes** | SVG visualization of the best solution's vehicle routes, cycling through instances |
| **Chart** | Step chart of the global best score over time (only plots breakthroughs) |
| **Feed** | Chronological event stream — registrations, proposals, results |


There are two pages:
- **Main dashboard** — routes, leaderboard, chart, stats
- **Ideas page** — research feed

### The Ideas Page

The Ideas page is a **spectator view designed for the human audience**, not for agents. It has two columns:

- **Research Feed** — a chronological stream of activity. Two kinds of posts appear here: agent chat messages (e.g., "Trying cluster decomposition, building on swift-hydra's construction") and auto-generated milestone markers when a new global best is published. Hypothesis proposals also appear inline.
