import { getAgentColor } from "../lib/colors";
import type { Panel, WSMessage } from "../types";

interface PerPositionEntry {
  agent_name: string;
  agent_id: string;
  accuracies: number[];
  score: number;
}

export class RoutesPanel implements Panel {
  private container!: HTMLElement;
  private gridEl!: HTMLElement;
  private scoreEl!: HTMLElement;
  private scoreDeltaEl!: HTMLElement;
  private entries: PerPositionEntry[] = [];
  private apiUrl = "";

  init(container: HTMLElement) {
    this.container = container;
    container.innerHTML = `
      <div class="panel-inner routes-panel">
        <div class="panel-label">BLOCK ACCURACY</div>
        <div class="routes-agent-name" id="routes-agent-name" style="font-size:10px;color:var(--text-dim)">Per-position prediction accuracy across block</div>
        <div class="routes-svg-wrap" id="routes-svg-wrap" style="overflow:auto">
          <div id="block-accuracy-grid" style="padding:8px"></div>
        </div>
        <div class="routes-score">
          <div class="routes-score-label">BEST ACCEPTANCE LENGTH</div>
          <div class="routes-score-value" id="routes-score">---</div>
          <div class="routes-score-delta" id="routes-score-delta"></div>
        </div>
      </div>
    `;

    this.gridEl = document.getElementById("block-accuracy-grid")!;
    this.scoreEl = document.getElementById("routes-score")!;
    this.scoreDeltaEl = document.getElementById("routes-score-delta")!;

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
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.entries = [];
      this.gridEl.innerHTML = "";
      this.scoreEl.textContent = "---";
      this.scoreDeltaEl.textContent = "";
      return;
    }

    if (msg.type === "stats_update") {
      if (msg.best_score != null) {
        this.scoreEl.textContent = msg.best_score.toFixed(4);
      }
    }

    if (msg.type === "new_global_best") {
      this.scoreEl.textContent = msg.score.toFixed(4);
      if (msg.incremental_improvement_pct != null) {
        const pct = msg.incremental_improvement_pct;
        const sign = pct >= 0 ? "+" : "";
        this.scoreDeltaEl.textContent = `${sign}${pct.toFixed(4)}% vs prev best`;
        this.scoreDeltaEl.style.color = pct > 0 ? "var(--green)" : "var(--red)";
      } else {
        this.scoreDeltaEl.textContent = "first global best";
        this.scoreDeltaEl.style.color = "var(--text-dim)";
      }
    }

    if (msg.type === "experiment_published") {
      const routeData = (msg as any).route_data;
      if (routeData && typeof routeData === "object") {
        const perPos = routeData.per_position_accuracy;
        if (Array.isArray(perPos) && perPos.length > 0) {
          this.updateEntry({
            agent_name: msg.agent_name,
            agent_id: msg.agent_id,
            accuracies: perPos,
            score: msg.score,
          });
          this.render();
        }
      }
    }
  }

  private updateEntry(entry: PerPositionEntry) {
    const idx = this.entries.findIndex((e) => e.agent_id === entry.agent_id);
    if (idx >= 0) {
      this.entries[idx] = entry;
    } else {
      this.entries.push(entry);
    }
    this.entries.sort((a, b) => b.score - a.score);
  }

  private render() {
    if (!this.entries.length) {
      this.gridEl.innerHTML = `<span style="color:var(--text-dim);font-size:11px">Waiting for per-position accuracy data...</span>`;
      return;
    }

    const maxPositions = Math.max(...this.entries.map((e) => e.accuracies.length));

    const grid = document.createElement("div");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = `80px repeat(${maxPositions}, 1fr)`;
    grid.style.gap = "1px";
    grid.style.fontSize = "9px";
    grid.style.fontFamily = "var(--mono)";

    // Header row
    const corner = document.createElement("div");
    corner.style.padding = "2px 4px";
    corner.style.color = "var(--text-dim)";
    corner.textContent = "Agent";
    grid.appendChild(corner);

    for (let p = 0; p < maxPositions; p++) {
      const hdr = document.createElement("div");
      hdr.style.padding = "2px";
      hdr.style.textAlign = "center";
      hdr.style.color = "var(--text-dim)";
      hdr.textContent = `${p + 1}`;
      grid.appendChild(hdr);
    }

    // Agent rows
    for (const entry of this.entries) {
      const nameCell = document.createElement("div");
      nameCell.style.padding = "2px 4px";
      nameCell.style.color = getAgentColor(entry.agent_id);
      nameCell.style.overflow = "hidden";
      nameCell.style.textOverflow = "ellipsis";
      nameCell.style.whiteSpace = "nowrap";
      nameCell.textContent = entry.agent_name;
      nameCell.title = `${entry.agent_name} (score: ${entry.score.toFixed(4)})`;
      grid.appendChild(nameCell);

      for (let p = 0; p < maxPositions; p++) {
        const acc = p < entry.accuracies.length ? entry.accuracies[p] : 0;
        const cell = document.createElement("div");
        cell.style.padding = "2px";
        cell.style.textAlign = "center";
        cell.style.borderRadius = "2px";
        cell.style.background = this.accuracyColor(acc);
        cell.style.color = acc > 0.5 ? "#000" : "#fff";
        cell.textContent = (acc * 100).toFixed(0);
        cell.title = `Position ${p + 1}: ${(acc * 100).toFixed(1)}%`;
        grid.appendChild(cell);
      }
    }

    this.gridEl.innerHTML = "";
    this.gridEl.appendChild(grid);
  }

  private accuracyColor(acc: number): string {
    // 0 = dark red/black, 0.5 = dim, 1.0 = bright green
    if (acc < 0.3) {
      const t = acc / 0.3;
      return `rgba(255, 82, 82, ${0.1 + t * 0.4})`;
    } else if (acc < 0.6) {
      const t = (acc - 0.3) / 0.3;
      return `rgba(255, 234, 0, ${0.15 + t * 0.4})`;
    } else {
      const t = (acc - 0.6) / 0.4;
      return `rgba(0, 230, 118, ${0.3 + t * 0.6})`;
    }
  }
}
