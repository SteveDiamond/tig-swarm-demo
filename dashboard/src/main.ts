import "./style.css";
import { initParticles } from "./lib/particles";
import { SwarmWebSocket } from "./lib/websocket";
import { MockDataGenerator } from "./mock";
import { viewportFlash } from "./lib/animate";
import {
  soundAgentJoined, soundHypothesisProposed, soundExperimentPublished,
  soundNewGlobalBest, startHeartbeat,
} from "./lib/sounds";
import { initWelcome, toggleWelcome } from "./lib/welcome";
import { startReplay } from "./lib/replay";

import { StatsPanel } from "./panels/stats";
import { RoutesPanel } from "./panels/routes";
import { ChartPanel } from "./panels/chart";
import { IdeaFlowPanel } from "./panels/ideaflow";
import { FeedPanel } from "./panels/feed";
import { LeaderboardPanel } from "./panels/leaderboard";

import type { WSMessage, Panel } from "./types";

// ── Config ──
const params = new URLSearchParams(window.location.search);
const isMock = params.has("mock");
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = params.get("ws") || `${wsProtocol}//${window.location.host}/ws/dashboard`;

// Derive REST API URL from WS URL
function getApiUrl(): string {
  const explicit = params.get("api");
  if (explicit) return explicit;
  // Convert ws(s)://host/ws/dashboard -> http(s)://host
  return wsUrl
    .replace("ws://", "http://")
    .replace("wss://", "https://")
    .replace("/ws/dashboard", "");
}

// ── Background particles ──
const canvas = document.getElementById("particleCanvas") as HTMLCanvasElement;
initParticles(canvas);

// ── Initialize panels ──
const panels: Panel[] = [];

function initPanel(PanelClass: new () => Panel, containerId: string) {
  const panel = new PanelClass();
  const container = document.getElementById(containerId)!;
  panel.init(container);
  panels.push(panel);
  return panel;
}

initPanel(StatsPanel, "panel-stats");
initPanel(RoutesPanel, "panel-routes");
const chartPanel = initPanel(ChartPanel, "panel-chart") as ChartPanel;
initPanel(IdeaFlowPanel, "panel-ideaflow");
initPanel(FeedPanel, "panel-feed");
initPanel(LeaderboardPanel, "panel-leaderboard");

// ── Message dispatch ──
let soundEnabled = false; // disabled during initial state hydration

function handleMessage(msg: WSMessage) {
  if (soundEnabled) {
    if (msg.type === "agent_joined") soundAgentJoined();
    if (msg.type === "hypothesis_proposed") soundHypothesisProposed(msg.strategy_tag);
    if (msg.type === "experiment_published") soundExperimentPublished();
    if (msg.type === "new_global_best") soundNewGlobalBest();
    if (msg.type === "stats_update") startHeartbeat(msg.active_agents);
  }

  if (msg.type === "new_global_best") {
    viewportFlash("rgba(0, 229, 255, 0.03)", 150);
  }

  panels.forEach((panel) => panel.handleMessage(msg));
}

// ── Fetch initial state from REST API ──
async function loadInitialState(apiUrl: string) {
  try {
    // /api/state gives current snapshot; /api/replay gives the full
    // best-so-far trajectory. We need both: the snapshot drives stats /
    // leaderboard / feed, and the trajectory seeds the chart and lets us
    // compute the incremental delta for the routes panel.
    const [stateRes, replayRes] = await Promise.all([
      fetch(`${apiUrl}/api/state`),
      fetch(`${apiUrl}/api/replay`),
    ]);
    if (!stateRes.ok) return;
    const state = await stateRes.json();
    const replay: Array<{
      experiment_id: string;
      agent_name: string;
      score: number;
      created_at: string;
    }> = replayRes.ok ? await replayRes.json() : [];

    // Seed the chart with the entire trajectory, not just recent_experiments.
    chartPanel.seedHistory(replay);

    // Incremental % improvement of the current best vs the prior best.
    // Null only when there has been exactly one (or zero) global bests.
    const incrementalPct =
      replay.length >= 2
        ? ((replay[replay.length - 2].score - replay[replay.length - 1].score) /
            replay[replay.length - 2].score) *
          100
        : null;

    // Emit stats
    handleMessage({
      type: "stats_update",
      active_agents: state.active_agents,
      total_experiments: state.recent_experiments?.length || 0,
      hypotheses_count: state.active_hypotheses?.length || 0,
      best_score: state.best_score,
      baseline_score: state.baseline_score,
      num_instances: state.num_instances || 1,
      improvement_pct: state.improvement_pct || 0,
      timestamp: new Date().toISOString(),
    });

    // Emit route data if available
    if (state.best_route_data && state.best_score != null) {
      handleMessage({
        type: "new_global_best",
        experiment_id: state.best_experiment_id || "",
        agent_name: replay[replay.length - 1]?.agent_name || "swarm",
        agent_id: "",
        score: state.best_score,
        improvement_pct: state.improvement_pct || 0,
        // Derived from /api/replay above. Null only when the current best
        // is the very first global best of the run.
        incremental_improvement_pct: incrementalPct,
        num_instances: state.num_instances || 1,
        route_data: state.best_route_data,
        timestamp: new Date().toISOString(),
      });
    }

    // Emit leaderboard
    if (state.leaderboard?.length) {
      handleMessage({
        type: "leaderboard_update",
        entries: state.leaderboard,
        timestamp: new Date().toISOString(),
      });
    }

    // Replay recent experiments as feed items (most recent last so they stack correctly)
    const recent = (state.recent_experiments || []).slice().reverse();
    for (const exp of recent) {
      handleMessage({
        type: "experiment_published",
        experiment_id: exp.id || "",
        agent_name: exp.agent_name,
        agent_id: "",
        score: exp.score,
        feasible: exp.feasible !== false,
        improvement_pct: exp.improvement_pct || 0,
        // We don't know the historical prev-best from /api/state, so leave
        // the delta null for the replayed feed items. The feed just hides
        // the % when null.
        delta_vs_best_pct: null,
        num_instances: state.num_instances || 1,
        is_new_best: exp.is_new_best === true,
        hypothesis_id: null,
        notes: exp.notes || "",
        timestamp: exp.created_at || new Date().toISOString(),
      });
    }

    // Replay active hypotheses as feed items
    for (const hyp of state.active_hypotheses || []) {
      handleMessage({
        type: "hypothesis_proposed",
        hypothesis_id: hyp.id || "",
        agent_name: hyp.agent_name,
        agent_id: "",
        title: hyp.title,
        description: hyp.description || "",
        strategy_tag: hyp.strategy_tag,
        parent_hypothesis_id: hyp.parent_hypothesis_id || null,
        timestamp: new Date().toISOString(),
      });
    }

    soundEnabled = true;
    console.log("[Dashboard] Loaded initial state:", {
      agents: state.active_agents,
      experiments: state.recent_experiments?.length,
      bestScore: state.best_score,
    });
  } catch (e) {
    console.warn("[Dashboard] Failed to load initial state:", e);
  }
}

// ── Welcome overlay ──
initWelcome();

// ── Keyboard navigation ──
document.addEventListener("keydown", (e) => {
  if (e.key === "2") window.location.href = "/ideas.html";
  if (e.key === "j" || e.key === "J") toggleWelcome();
  if (e.key === "r" || e.key === "R") startReplay(getApiUrl(), handleMessage);
});

// ── Connect ──
if (isMock) {
  console.log("[Dashboard] Running in MOCK mode");
  soundEnabled = true;
  const mock = new MockDataGenerator();
  mock.onMessage(handleMessage);
  mock.start();

  const wsEl = document.getElementById("ws-status");
  if (wsEl) {
    wsEl.textContent = "MOCK";
    wsEl.className = "ws-status connected";
  }
} else {
  const apiUrl = getApiUrl();
  console.log(`[Dashboard] Connecting to ${wsUrl}, API: ${apiUrl}`);

  setTimeout(() => loadInitialState(apiUrl), 500);

  const ws = new SwarmWebSocket(wsUrl);
  ws.onMessage(handleMessage);
  ws.connect();
}
