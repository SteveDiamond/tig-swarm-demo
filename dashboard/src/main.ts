import "./style.css";
import { initParticles } from "./lib/particles";
import { SwarmWebSocket } from "./lib/websocket";
import { MockDataGenerator } from "./mock";
import { viewportFlash } from "./lib/animate";
import {
  soundAgentJoined, soundHypothesisProposed, soundExperimentPublished,
  soundNewGlobalBest, startHeartbeat,
} from "./lib/sounds";
import { initQROverlay, toggleQR } from "./lib/qrcode";
import { startReplay } from "./lib/replay";

import { StatsPanel } from "./panels/stats";
import { RoutesPanel } from "./panels/routes";
import { ChartPanel } from "./panels/chart";
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
initPanel(ChartPanel, "panel-chart");
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
    const res = await fetch(`${apiUrl}/api/state`);
    if (!res.ok) return;
    const state = await res.json();

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
        agent_name: "swarm",
        agent_id: "",
        score: state.best_score,
        improvement_pct: state.improvement_pct || 0,
        // No prior best to compare against on initial load
        incremental_improvement_pct: null,
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

// ── QR overlay ──
initQROverlay();

// ── Keyboard navigation ──
document.addEventListener("keydown", (e) => {
  if (e.key === "2") window.location.href = "/ideas.html";
  if (e.key === "q" || e.key === "Q") toggleQR();
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
