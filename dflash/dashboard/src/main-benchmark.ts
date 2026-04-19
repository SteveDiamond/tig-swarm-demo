import "./style.css";
import { initParticles } from "./lib/particles";
import { SwarmWebSocket } from "./lib/websocket";
import { MockDataGenerator } from "./mock";
import { ChartPanel } from "./panels/chart";
import type { WSMessage } from "./types";

// ── Config ──
const params = new URLSearchParams(window.location.search);
const isMock = params.has("mock");
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = params.get("ws") || `${wsProtocol}//${window.location.host}/ws/dashboard`;

function getApiUrl(): string {
  const explicit = params.get("api");
  if (explicit) return explicit;
  return wsUrl
    .replace("ws://", "http://")
    .replace("wss://", "https://")
    .replace("/ws/dashboard", "");
}

// ── Background particles ──
const canvas = document.getElementById("particleCanvas") as HTMLCanvasElement;
initParticles(canvas);

// ── Initialize single panel ──
const chartPanel = new ChartPanel();
chartPanel.init(document.getElementById("panel-chart")!);

function handleMessage(msg: WSMessage) {
  chartPanel.handleMessage(msg);
}

// ── Hydrate from /api/state + /api/replay ──
async function loadInitialState(apiUrl: string) {
  try {
    const [stateRes, replayRes] = await Promise.all([
      fetch(`${apiUrl}/api/state`),
      fetch(`${apiUrl}/api/replay`),
    ]);
    if (!stateRes.ok) return;
    const state = await stateRes.json();
    const replay: Array<{
      experiment_id: string;
      agent_name: string;
      agent_id?: string;
      score: number;
      created_at: string;
    }> = replayRes.ok ? await replayRes.json() : [];

    chartPanel.seedHistory(replay);

    if (state.leaderboard?.length) {
      handleMessage({
        type: "leaderboard_update",
        entries: state.leaderboard,
        timestamp: new Date().toISOString(),
      });
    }

    console.log(`[Benchmark] Loaded ${replay.length} best-history points`);
  } catch (e) {
    console.warn("[Benchmark] Failed to load initial state:", e);
  }
}

// ── Keyboard navigation ──
document.addEventListener("keydown", (e) => {
  if (e.key === "1") window.location.href = "/";
  if (e.key === "2") window.location.href = "/ideas.html";
  if (e.key === "3") window.location.href = "/diversity.html";
});

// ── Connect ──
if (isMock) {
  console.log("[Benchmark] Running in MOCK mode");
  const mock = new MockDataGenerator();
  mock.onMessage(handleMessage);
  mock.start();
} else {
  const apiUrl = getApiUrl();
  console.log(`[Benchmark] Connecting to ${wsUrl}, API: ${apiUrl}`);
  setTimeout(() => loadInitialState(apiUrl), 300);
  const ws = new SwarmWebSocket(wsUrl);
  ws.onMessage(handleMessage);
  ws.connect();
}
