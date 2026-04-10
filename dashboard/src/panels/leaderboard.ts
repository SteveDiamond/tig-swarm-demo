import type { Panel, WSMessage, LeaderboardEntry } from "../types";
import { getAgentColor } from "../lib/colors";

type SortKey = "avg_score" | "runs" | "improvements";
type SortDir = "asc" | "desc";

// Default direction when a column is first clicked: lower-is-better for score,
// higher-is-better for activity counts.
const DEFAULT_DIR: Record<SortKey, SortDir> = {
  avg_score: "asc",
  runs: "desc",
  improvements: "desc",
};

export class LeaderboardPanel implements Panel {
  private list!: HTMLElement;
  private currentEntries: LeaderboardEntry[] = [];
  private sortKey: SortKey = "avg_score";
  private sortDir: SortDir = "asc";

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner">
        <div class="panel-label">LEADERBOARD</div>
        <div class="leaderboard-header">
          <span class="lb-rank">#</span>
          <span class="lb-name">Agent</span>
          <button type="button" class="lb-runs lb-sortable" data-sort="runs">Runs<span class="lb-arrow"></span></button>
          <button type="button" class="lb-imp lb-sortable" data-sort="improvements">Imp.<span class="lb-arrow"></span></button>
          <button type="button" class="lb-score lb-sortable" data-sort="avg_score">Score<span class="lb-arrow"></span></button>
        </div>
        <div class="leaderboard-list" id="leaderboard-list"></div>
      </div>
    `;
    this.list = document.getElementById("leaderboard-list")!;

    container.querySelectorAll<HTMLButtonElement>(".lb-sortable").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.sort as SortKey;
        if (this.sortKey === key) {
          this.sortDir = this.sortDir === "asc" ? "desc" : "asc";
        } else {
          this.sortKey = key;
          this.sortDir = DEFAULT_DIR[key];
        }
        this.render();
      });
    });

    this.updateHeaderIndicators();
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.currentEntries = [];
      this.list.innerHTML = "";
      return;
    }

    if (msg.type !== "leaderboard_update") return;
    this.currentEntries = msg.entries.slice();
    this.render();
  }

  private updateHeaderIndicators() {
    document.querySelectorAll<HTMLButtonElement>(".lb-sortable").forEach((btn) => {
      const isActive = btn.dataset.sort === this.sortKey;
      btn.classList.toggle("lb-sortable--active", isActive);
      const arrow = btn.querySelector<HTMLElement>(".lb-arrow")!;
      arrow.textContent = isActive ? (this.sortDir === "asc" ? " ↑" : " ↓") : "";
    });
  }

  private sortEntries(entries: LeaderboardEntry[]): LeaderboardEntry[] {
    const sorted = entries.slice();
    const dir = this.sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      const av = a[this.sortKey];
      const bv = b[this.sortKey];
      // Nulls (no runs yet) always sink to the bottom regardless of direction.
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return ((av as number) - (bv as number)) * dir;
    });
    return sorted;
  }

  private render() {
    this.updateHeaderIndicators();

    // Record first positions for FLIP animation
    const firstRects = new Map<string, DOMRect>();
    Array.from(this.list.children).forEach((child) => {
      const el = child as HTMLElement;
      firstRects.set(el.dataset.agentId || "", el.getBoundingClientRect());
    });

    // Track previous displayed score per agent so we can highlight improvements
    // (improvement = the value in the *currently sorted column* moved in the
    // "better" direction, which is whatever DEFAULT_DIR considers good).
    const prevValues = new Map<string, number | null>();
    this.list.childNodes.forEach((node) => {
      const el = node as HTMLElement;
      const id = el.dataset.agentId || "";
      const v = el.dataset.sortValue;
      prevValues.set(id, v === "" || v === undefined ? null : Number(v));
    });

    const sorted = this.sortEntries(this.currentEntries).slice(0, 10);

    this.list.innerHTML = "";
    sorted.forEach((entry, i) => {
      const rank = i + 1;
      const row = document.createElement("div");
      row.className = "leaderboard-row";
      row.dataset.agentId = entry.agent_id;
      const sortVal = entry[this.sortKey];
      row.dataset.sortValue = sortVal === null ? "" : String(sortVal);

      const rankClass =
        rank === 1 ? "rank-gold" :
        rank === 2 ? "rank-cyan" :
        rank === 3 ? "rank-teal" : "";

      const color = getAgentColor(entry.agent_id);

      const prev = prevValues.get(entry.agent_id);
      const goodDir = DEFAULT_DIR[this.sortKey];
      const improved =
        prev !== undefined && prev !== null && sortVal !== null &&
        ((goodDir === "asc" && (sortVal as number) < prev) ||
         (goodDir === "desc" && (sortVal as number) > prev));

      const scoreText = entry.avg_score === null ? "—" : entry.avg_score.toFixed(1);

      row.innerHTML = `
        <span class="lb-rank ${rankClass}">${rank}</span>
        <span class="lb-name">
          <span class="lb-dot" style="background:${color}"></span>
          ${entry.agent_name}
        </span>
        <span class="lb-runs">${entry.runs}</span>
        <span class="lb-imp">${entry.improvements}</span>
        <span class="lb-score ${improved ? "lb-score--improved" : ""}">${scoreText}</span>
      `;

      if (rank <= 3) {
        const accentColor = rank === 1 ? "var(--amber)" : rank === 2 ? "var(--cyan)" : "var(--teal)";
        row.style.borderLeft = `2px solid ${accentColor}`;
        row.style.boxShadow = `inset 4px 0 12px -4px ${accentColor}44`;
      }

      this.list.appendChild(row);
    });

    // FLIP animation for reordered rows
    if (firstRects.size > 0) {
      Array.from(this.list.children).forEach((child) => {
        const el = child as HTMLElement;
        const agentId = el.dataset.agentId || "";
        const first = firstRects.get(agentId);
        if (!first) {
          el.style.opacity = "0";
          el.style.transform = "translateX(20px)";
          requestAnimationFrame(() => {
            el.style.transition = "opacity 0.4s ease, transform 0.4s ease";
            el.style.opacity = "1";
            el.style.transform = "translateX(0)";
            setTimeout(() => { el.style.transition = ""; }, 400);
          });
          return;
        }

        const last = el.getBoundingClientRect();
        const deltaY = first.top - last.top;
        if (Math.abs(deltaY) < 1) return;

        el.style.transform = `translateY(${deltaY}px)`;
        el.style.transition = "none";

        requestAnimationFrame(() => {
          el.style.transition = "transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)";
          el.style.transform = "";
          setTimeout(() => { el.style.transition = ""; }, 500);
        });
      });
    }
  }
}
