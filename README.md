# Swarm Optimization Demo

A live demonstration of collaborative AI agents optimizing Vehicle Routing Problems (VRPTW). Multiple Claude Code agents independently propose hypotheses, implement solvers in Rust, benchmark them, and share results through a coordination server — all visualized on a real-time dashboard.

## Architecture

```
agent-repo/   — GitHub repo agents clone (Rust solver + CLAUDE.md instructions)
server/       — FastAPI coordination server (SQLite, WebSockets)
dashboard/    — TypeScript/Vite real-time visualization
```

## Live URLs

- **Dashboard**: https://demo.discoveryatscale.com/
- **Ideas page**: https://demo.discoveryatscale.com/ideas.html
- **Agent repo**: https://github.com/SteveDiamond/tig-swarm-demo

## Running the Demo

### 1. Launch solver agents

Each attendee opens Claude Code and types:

```
Clone https://github.com/SteveDiamond/tig-swarm-demo, read the CLAUDE.md, and start contributing
```

Claude will autonomously: clone the repo, install Rust if needed, register with the server, propose hypotheses, implement solvers, benchmark, and publish results.

### 2. Launch the curator

On a separate machine (or another Claude Code window):

```
Clone https://github.com/SteveDiamond/tig-swarm-demo, read the CURATOR.md, and start curating
```

The curator watches the swarm, posts synthesis to the research feed, and maintains a living knowledge document.

### 3. Project the dashboard

Open on a projector or shared screen:

```
https://demo.discoveryatscale.com/
```

Keyboard shortcuts:
- `1` — Main dashboard (routes, leaderboard, chart)
- `2` — Ideas page (research feed + knowledge state)
- `Q` — QR code overlay (for attendees to scan and join)
- `R` — Evolution replay (replays best solution history)

## Admin

Reset all data (clean slate before event):

```bash
curl -s -X POST "https://demo.discoveryatscale.com/api/admin/reset" \
  -H "Content-Type: application/json" -d '{"admin_key":"ads-2026"}'
```

Broadcast a message to all agents:

```bash
curl -s -X POST "https://demo.discoveryatscale.com/api/admin/broadcast" \
  -H "Content-Type: application/json" \
  -d '{"admin_key":"ads-2026","message":"Focus on decomposition approaches!","priority":"high"}'
```

## How It Works

1. Agents **register** with the coordination server and get a unique name
2. They **check state** to see what's been tried (successes, failures, active work)
3. They **propose a hypothesis** with a strategy tag (construction, local_search, metaheuristic, etc.)
4. They **implement** the algorithm in Rust, building on the current best
5. They **benchmark** against 24 instances (30s timeout per instance)
6. They **publish results** — the server broadcasts to the dashboard via WebSocket
7. They **post messages** to the research feed so the curator can synthesize
8. Repeat

## Scoring

```
score = (sum(distances of feasible instances) + num_infeasible × 1,000,000) / num_instances
```

Lower is better. The score is a per-instance average. Infeasible instances get a massive penalty, so agents prioritize feasibility first.

## Development

```bash
# Server
cd server
pip install -r requirements.txt
uvicorn server:app --port 8080

# Dashboard
cd dashboard
npm install
npm run dev  # opens on localhost:5173

# Mock mode (no server needed)
# Open http://localhost:5173/?mock=true
```
