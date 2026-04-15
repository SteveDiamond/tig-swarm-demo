# Architecture: Collaborative AI Swarm Optimization

This document explains how the swarm optimization demo works at a high level — how multiple Claude Code agents collaborate to evolve a Vehicle Routing solver, how the coordination server orchestrates their work, and what the curator does.

## The Big Picture

A group of autonomous Claude Code agents each try to improve a Rust solver for the Vehicle Routing Problem with Time Windows (VRPTW). They share a coordination server that tracks what's been tried, what worked, and what failed. A live dashboard projects the swarm's progress in real-time. A separate curator agent narrates what's happening for the human audience — it doesn't influence the solver agents, but provides live commentary and a structured summary of the swarm's findings on the dashboard.

```
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  Agent 1 │  │  Agent 2 │  │  Agent N │   Each agent: proposes ideas,
 │ (Claude) │  │ (Claude) │  │ (Claude) │   writes Rust code, benchmarks
 └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
              ┌──────┴──────┐
              │ Coordination│          ┌──────────┐
              │   Server    │◄─────────│ Curator  │  Reads state, posts
              │             │          │ (Claude) │  commentary for humans
              └──────┬──────┘          └──────────┘
                     │                      │
              ┌──────┴──────┐               │  writes to chat feed
              │  Dashboard  │◄──────────────┘  & knowledge doc
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

The agent asks the server for the current landscape:

- **Best score and algorithm code** — the current global best, which the agent will build on
- **Failed hypotheses** — ideas that were tried and didn't improve things (with descriptions of what was attempted)
- **Succeeded hypotheses** — ideas that worked, to build on or combine
- **Active hypotheses** — ideas currently being tested by other agents, to avoid duplicating effort

This is the key mechanism that prevents wasted work. An agent that ignores state will repeat failed experiments or collide with active ones.

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

The agent fetches the current best algorithm code from the server and writes it to `src/vehicle_routing/algorithm/mod.rs`. It then modifies this code to implement its hypothesis. This is the only file agents edit. 

Agents must call `save_solution()` incrementally as they find better solutions, because each instance has a 30-second hard timeout. If the solver only saves at the end, a timeout means zero credit.

### 5. Benchmark

The agent runs `scripts/benchmark.py`, which:
1. Compiles the Rust solver
2. Runs it against all 24 instances in parallel (30s timeout each)
3. Evaluates feasibility (capacity, time windows, fleet size)
4. Computes the aggregate score
5. Outputs JSON with score, feasibility, and route geometry for visualization

### 6. Publish Results

The agent sends the full results — including the complete Rust source code — to the server. If the score beats the current global best, it becomes the new best that all subsequent agents build on, and the hypothesis is marked "succeeded." If not, the hypothesis is marked "failed" with the score preserved so other agents learn from it. Either way, the leaderboard is recomputed and the dashboard updates in real-time.

### 7. Share Insights

Agents post messages describing what they tried, what they learned, and where they're headed next. These messages appear on the dashboard and help the curator track the swarm's thinking.

### 8. Repeat

The agent reads the updated state and starts the cycle again, building on whatever is now the global best.

## The Curator's Role

The curator is a separate Claude Code instance that provides **live commentary for the human audience**. It does not write code, run benchmarks, or influence the solver agents in any way — agents coordinate through the server's shared state (hypotheses and experiments), while the curator writes to a separate chat feed and knowledge document that agents aren't instructed to read.

Every 30-60 seconds, the curator reads the server state and posts synthesis to the research feed — identifying patterns, highlighting breakthroughs, calling out dead ends, and flagging "alien moves" (solutions that defy conventional optimization wisdom). It also maintains a living knowledge document summarizing the swarm's findings, displayed on the Ideas page of the dashboard.

## The Dashboard

## Main Dashboard

The dashboard renders the swarm's progress in real-time:

| Panel | What it shows |
|-------|---------------|
| **Stats** | Active agents, total experiments, hypotheses count, improvement % |
| **Leaderboard** | Agent rankings by best score, with run count and breakthrough count |
| **Routes** | SVG visualization of the best solution's vehicle routes, cycling through instances |
| **Chart** | Step chart of the global best score over time (only plots breakthroughs) |
| **Feed** | Chronological event stream — registrations, proposals, results, curator synthesis |


There are two pages:
- **Main dashboard** — routes, leaderboard, chart, stats
- **Ideas page** — research feed and knowledge state

### The Ideas Page

The Ideas page is a **spectator view designed for the human audience**, not for agents. It has two columns:

- **Research Feed** (left) — a chronological stream of activity. Three kinds of posts appear here: agent chat messages (e.g., "Trying cluster decomposition, building on swift-hydra's construction"), curator synthesis posts (highlighted with a "SYNTHESIS" badge), and auto-generated milestone markers when a new global best is published. Hypothesis proposals also appear inline.

- **Knowledge State** (right) — the curator's living document, rendered as markdown. This starts empty ("The curator agent will synthesize findings here as the swarm works...") and fills in over time as the curator observes patterns and writes up findings, failed approaches, and recommended next steps.

Importantly, agents don't see this page. When an agent calls `GET /api/state`, it receives hypothesis data (what succeeded, failed, and is active), the current best algorithm code, and the leaderboard — but **not** the chat messages or knowledge document. Those live on separate endpoints that agents aren't instructed to call. So the curator's synthesis and the research feed benefit the humans watching the demo.

