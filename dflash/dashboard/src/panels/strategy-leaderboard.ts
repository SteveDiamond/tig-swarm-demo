import type { Panel, WSMessage } from "../types";
import { getAgentColor } from "../lib/colors";

interface TopEntry {
  experiment_id: string;
  score: number;
  agent_id: string;
  agent_name: string | null;
  strategy_tag: string | null;
  title: string | null;
  created_at?: string;
}

const MAX_ROWS = 20;

// Colored pills for the eight canonical strategy tags. Unknown / null tags
// fall back to the "other" styling.
const TAG_COLORS: Record<string, string> = {
  construction:          "#b388ff",
  local_search:          "#00e5ff",
  metaheuristic:         "#ffab00",
  constraint_relaxation: "#ff80ab",
  decomposition:         "#00e676",
  hybrid:                "#40c4ff",
  data_structure:        "#00bfa5",
  other:                 "#7a869a",
};

export class StrategyLeaderboardPanel implements Panel {
  private listEl!: HTMLElement;
  private apiUrl = "";
  // Keyed by experiment_id so the same event delivered twice (initial /state
  // + live WS catch-up after reconnect) doesn't produce duplicate rows.
  private entries = new Map<string, TopEntry>();

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="strategy-lb">
        <div class="ideas-col-label">TOP 20 SCORES · BY STRATEGY</div>
        <div class="strategy-lb-header">
          <span class="sl-rank">#</span>
          <span class="sl-score">Score</span>
          <span class="sl-tag-col">Strategy</span>
          <span class="sl-title">Hypothesis</span>
          <span class="sl-agent">Agent</span>
        </div>
        <div class="strategy-lb-list" id="strategy-lb-list"></div>
      </div>
    `;
    this.listEl = document.getElementById("strategy-lb-list")!;

    const params = new URLSearchParams(window.location.search);
    const explicit = params.get("api");
    if (explicit) this.apiUrl = explicit;
    else {
      const ws = params.get("ws") || "";
      if (ws) {
        this.apiUrl = ws
          .replace("ws://", "http://")
          .replace("wss://", "https://")
          .replace("/ws/dashboard", "");
      } else {
        this.apiUrl = `${window.location.protocol}//${window.location.host}`;
      }
    }

    this.loadInitial();
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.entries.clear();
      this.render();
      this.loadInitial();
      return;
    }

    if (msg.type !== "experiment_published") return;
    if (!msg.feasible) return;
    if (this.entries.has(msg.experiment_id)) return; // already recorded

    const worst = this.worstScore();
    if (this.entries.size >= MAX_ROWS && msg.score >= worst) return;

    this.entries.set(msg.experiment_id, {
      experiment_id: msg.experiment_id,
      score: msg.score,
      agent_id: msg.agent_id,
      agent_name: msg.agent_name,
      strategy_tag: (msg as any).strategy_tag ?? null,
      title: (msg as any).title ?? null,
    });
    this.trim();
    this.render();
  }

  private async loadInitial() {
    try {
      const res = await fetch(`${this.apiUrl}/api/top_scores?limit=${MAX_ROWS}`);
      if (!res.ok) return;
      const data: { entries: TopEntry[] } = await res.json();
      for (const e of data.entries) {
        if (!this.entries.has(e.experiment_id)) {
          this.entries.set(e.experiment_id, e);
        }
      }
      this.trim();
      this.render();
    } catch {
      // noop — WS events will backfill
    }
  }

  private worstScore(): number {
    let worst = -Infinity;
    for (const e of this.entries.values()) {
      if (e.score > worst) worst = e.score;
    }
    return worst;
  }

  private trim() {
    if (this.entries.size <= MAX_ROWS) return;
    const sorted = [...this.entries.values()].sort((a, b) => a.score - b.score);
    this.entries.clear();
    for (const e of sorted.slice(0, MAX_ROWS)) {
      this.entries.set(e.experiment_id, e);
    }
  }

  private render() {
    const sorted = [...this.entries.values()].sort((a, b) => a.score - b.score);

    if (sorted.length === 0) {
      this.listEl.innerHTML = `
        <div class="strategy-lb-empty">no feasible iterations yet</div>
      `;
      return;
    }

    this.listEl.innerHTML = "";
    sorted.forEach((entry, i) => {
      const row = document.createElement("div");
      row.className = "strategy-lb-row";

      const tag = entry.strategy_tag || "—";
      const tagColor = entry.strategy_tag
        ? (TAG_COLORS[entry.strategy_tag] || TAG_COLORS.other)
        : "var(--text-dim)";
      const title = entry.title || "—";
      const agentName = entry.agent_name || "—";
      const agentColor = entry.agent_id ? getAgentColor(entry.agent_id) : "var(--text-dim)";

      row.innerHTML = `
        <span class="sl-rank">${i + 1}</span>
        <span class="sl-score">${entry.score.toFixed(1)}</span>
        <span class="sl-tag-col">
          <span class="sl-tag" style="color:${tagColor};border-color:${tagColor}">${this.escape(tag)}</span>
        </span>
        <span class="sl-title" title="${this.escape(title)}">${this.escape(title)}</span>
        <span class="sl-agent" style="color:${agentColor}">${this.escape(agentName)}</span>
      `;
      this.listEl.appendChild(row);
    });
  }

  private escape(s: string): string {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}
