import "./style.css";
import { initParticles } from "./lib/particles";
import { SwarmWebSocket } from "./lib/websocket";
import { MockDataGenerator } from "./mock";
import { IdeasTree } from "./panels/ideas-tree";
import { StrategyLeaderboardPanel } from "./panels/strategy-leaderboard";
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

// ── Initialize ideas tree ──
const root = document.getElementById("ideas-root")!;
const ideasTree = new IdeasTree();
ideasTree.init(root);

const strategyLb = new StrategyLeaderboardPanel();
const strategyMount = document.getElementById("strategy-lb-mount");
if (strategyMount) strategyLb.init(strategyMount);

function handleMessage(msg: WSMessage) {
  ideasTree.handleMessage(msg);
  strategyLb.handleMessage(msg);
}

// ── Keyboard navigation ──
document.addEventListener("keydown", (e) => {
  if (e.key === "1") window.location.href = "/";
});

// ── Fetch initial state ──
async function loadInitialState(apiUrl: string) {
  try {
    const res = await fetch(`${apiUrl}/api/state`);
    if (!res.ok) return;
    const state = await res.json();

    // Replay all hypotheses (active, failed, succeeded)
    const allHyps = [
      ...(state.active_hypotheses || []),
      ...(state.failed_hypotheses || []),
      ...(state.succeeded_hypotheses || []),
    ];

    for (const h of allHyps) {
      handleMessage({
        type: "hypothesis_proposed",
        hypothesis_id: h.id,
        agent_name: h.agent_name,
        agent_id: h.agent_id || "",
        title: h.title,
        description: h.description || "",
        strategy_tag: h.strategy_tag,
        parent_hypothesis_id: h.parent_hypothesis_id || null,
        timestamp: new Date().toISOString(),
      });
      // Apply status if not active
      if (h.status === "succeeded" || h.status === "failed") {
        handleMessage({
          type: "hypothesis_status_changed",
          hypothesis_id: h.id,
          new_status: h.status,
          agent_name: h.agent_name,
          timestamp: new Date().toISOString(),
        });
      }
    }

    console.log(`[Ideas] Loaded ${allHyps.length} hypotheses`);

    const msgRes = await fetch(`${apiUrl}/api/messages?limit=50`);

    if (msgRes.ok) {
      const messages = await msgRes.json();
      for (const m of messages.reverse()) {
        handleMessage({
          type: "chat_message",
          message_id: m.id,
          agent_name: m.agent_name,
          agent_id: m.agent_id,
          content: m.content,
          msg_type: m.msg_type,
          timestamp: m.created_at,
        });
      }
    }
  } catch (e) {
    console.warn("[Ideas] Failed to load initial state:", e);
  }
}

// ── Connect ──
if (isMock) {
  console.log("[Ideas] Running in MOCK mode");
  const mock = new MockDataGenerator();
  mock.onMessage(handleMessage);
  mock.start();
} else {
  const apiUrl = getApiUrl();
  console.log(`[Ideas] Connecting to ${wsUrl}, API: ${apiUrl}`);
  setTimeout(() => loadInitialState(apiUrl), 300);
  const ws = new SwarmWebSocket(wsUrl);
  ws.onMessage(handleMessage);
  ws.connect();
}
