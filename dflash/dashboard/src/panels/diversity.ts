import { getAgentColor } from "../lib/colors";
import type { Panel, WSMessage } from "../types";

interface DiversityData {
  agents: { agent_id: string; agent_name: string }[];
  matrix: number[][];
}

export class DiversityPanel implements Panel {
  private container!: HTMLElement;
  private inner!: HTMLElement;
  private apiUrl = "";
  private throttleTimer: ReturnType<typeof setTimeout> | null = null;
  private lastFetch = 0;
  private static THROTTLE_MS = 30_000;

  init(container: HTMLElement) {
    this.container = container;
    container.innerHTML = `
      <div class="panel-inner diversity-panel">
        <div class="panel-label">CODE DIVERSITY</div>
        <div class="diversity-grid" id="diversity-grid"></div>
      </div>
    `;
    this.inner = document.getElementById("diversity-grid")!;

    const wsEl = document.querySelector(".ws-status");
    if (wsEl) {
      const proto = window.location.protocol;
      this.apiUrl = `${proto}//${window.location.host}`;
    }
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
      }
    }

    this.fetchAndRender();
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.inner.innerHTML = "";
      return;
    }
    if (msg.type !== "leaderboard_update") return;

    const elapsed = Date.now() - this.lastFetch;
    if (elapsed >= DiversityPanel.THROTTLE_MS) {
      this.fetchAndRender();
    } else if (!this.throttleTimer) {
      this.throttleTimer = setTimeout(() => {
        this.throttleTimer = null;
        this.fetchAndRender();
      }, DiversityPanel.THROTTLE_MS - elapsed);
    }
  }

  private async fetchAndRender() {
    this.lastFetch = Date.now();
    try {
      const res = await fetch(`${this.apiUrl}/api/diversity`);
      if (!res.ok) return;
      const data: DiversityData = await res.json();
      this.render(data);
    } catch {
      // silently retry on next update
    }
  }

  private render(data: DiversityData) {
    const { agents, matrix } = data;
    if (!agents.length) {
      this.inner.innerHTML = `<span style="color:var(--text-dim);font-size:11px">No agents yet</span>`;
      return;
    }

    const n = agents.length;
    const grid = document.createElement("div");
    grid.className = "dv-grid";
    grid.style.gridTemplateColumns = `56px repeat(${n}, 1fr)`;
    grid.style.gridTemplateRows = `20px repeat(${n}, 1fr)`;

    // Column headers
    grid.appendChild(this.corner());
    for (let j = 0; j < n; j++) {
      const hdr = document.createElement("div");
      hdr.className = "dv-col-hdr";
      hdr.style.color = getAgentColor(agents[j].agent_id);
      hdr.textContent = this.shortName(agents[j].agent_name);
      hdr.title = agents[j].agent_name;
      grid.appendChild(hdr);
    }

    // Rows
    for (let i = 0; i < n; i++) {
      // Row header
      const rh = document.createElement("div");
      rh.className = "dv-row-hdr";
      rh.style.color = getAgentColor(agents[i].agent_id);
      rh.textContent = this.shortName(agents[i].agent_name);
      rh.title = agents[i].agent_name;
      grid.appendChild(rh);

      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        const cell = document.createElement("div");
        cell.className = i === j ? "dv-cell dv-diag" : "dv-cell";
        cell.style.background = i === j
          ? this.diagColor(val)
          : this.cellColor(val);
        cell.textContent = (val * 100).toFixed(0);
        cell.title = i === j
          ? `${agents[i].agent_name}: ${(val * 100).toFixed(1)}% unique lines`
          : `${(val * 100).toFixed(1)}% of ${agents[i].agent_name}'s lines found in ${agents[j].agent_name}`;
        grid.appendChild(cell);
      }
    }

    this.inner.innerHTML = "";
    this.inner.appendChild(grid);
  }

  private corner(): HTMLElement {
    const el = document.createElement("div");
    el.className = "dv-corner";
    return el;
  }

  private shortName(name: string): string {
    if (name.length <= 8) return name;
    return name.slice(0, 7) + "…";
  }

  private cellColor(val: number): string {
    // 0 = dark, 1 = bright cyan
    const a = Math.max(0.05, val * 0.7);
    return `rgba(0, 229, 255, ${a})`;
  }

  private diagColor(val: number): string {
    // 0 = dark, 1 = bright amber (uniqueness)
    const a = Math.max(0.05, val * 0.8);
    return `rgba(255, 170, 0, ${a})`;
  }
}
