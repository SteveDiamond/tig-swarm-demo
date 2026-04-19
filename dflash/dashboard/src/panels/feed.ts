import type { Panel, WSMessage } from "../types";
import { formatTime } from "../lib/animate";
import { getAgentColor } from "../lib/colors";

const MAX_ITEMS = 200;

const EVENT_CONFIG: Record<string, { dot: string; icon: string }> = {
  agent_joined: { dot: "var(--cyan)", icon: "+" },
  hypothesis_proposed: { dot: "var(--purple)", icon: "?" },
  experiment_success: { dot: "var(--green)", icon: "\u2713" },
  experiment_fail: { dot: "var(--red)", icon: "\u2717" },
  new_global_best: { dot: "var(--amber)", icon: "\u2605" },
};

export class FeedPanel implements Panel {
  private list!: HTMLElement;
  private items: HTMLElement[] = [];

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner">
        <div class="panel-label">LIVE FEED</div>
        <div class="feed-list" id="feed-list"></div>
      </div>
    `;
    this.list = document.getElementById("feed-list")!;
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.items.forEach((el) => el.remove());
      this.items = [];
      this.list.innerHTML = "";
      return;
    }

    let text = "";
    let eventType = "";

    switch (msg.type) {
      case "agent_joined":
        text = `<b>${msg.agent_name}</b> joined the swarm`;
        eventType = "agent_joined";
        break;
      case "hypothesis_proposed":
        text = `<b>${msg.agent_name}</b> proposed: "${msg.title}"`;
        eventType = "hypothesis_proposed";
        break;
      case "experiment_published": {
        // Three outcomes:
        //   1. beats own best AND new global best → show both %s
        //   2. beats own best only → show own-best %
        //   3. no improvement → just the score
        // Server deltas are improvement-positive (positive = score rose =
        // improvement). Display directly with matching sign.
        const fmtDelta = (d: number | null | undefined): string => {
          if (d == null) return "";
          const sign = d >= 0 ? "+" : "";
          const color = d > 0 ? "var(--green)" : d < 0 ? "var(--red)" : "var(--text-dim)";
          return `<span style="color:${color}">${sign}${d.toFixed(4)}%</span>`;
        };

        const ownDelta = msg.delta_vs_own_best_pct;
        const globalDelta = msg.delta_vs_best_pct;
        const beatsOwn = msg.beats_own_best === true;

        if (msg.is_new_best) {
          // Beat own best AND global best.
          const ownStr = ownDelta != null ? ` (${fmtDelta(ownDelta)} own)` : "";
          const globalStr = globalDelta != null ? ` ${fmtDelta(globalDelta)} vs global` : "";
          text = `<b>${msg.agent_name}</b> improved &mdash; ${msg.score.toFixed(1)}${ownStr} · NEW GLOBAL BEST${globalStr}`;
          eventType = "new_global_best";
        } else if (beatsOwn) {
          const ownStr = ownDelta != null ? ` (${fmtDelta(ownDelta)})` : "";
          text = `<b>${msg.agent_name}</b> improvement &mdash; ${msg.score.toFixed(1)}${ownStr}`;
          eventType = "experiment_success";
        } else {
          // Show the regression vs own best when available so the magnitude
          // of "no improvement" is visible (e.g. +0.42% = slightly worse).
          const ownStr = ownDelta != null ? ` (${fmtDelta(ownDelta)} vs own)` : "";
          text = `<b>${msg.agent_name}</b> no improvement &mdash; ${msg.score.toFixed(1)}${ownStr}`;
          eventType = "experiment_fail";
        }
        break;
      }
      case "admin_broadcast":
        text = `<b>ADMIN</b>: ${msg.message}`;
        eventType = "new_global_best";
        break;
      default:
        return;
    }

    const config = EVENT_CONFIG[eventType] || EVENT_CONFIG.agent_joined;
    const agentId = "agent_id" in msg ? (msg as any).agent_id : "";
    const agentColor = agentId ? getAgentColor(agentId) : config.dot;
    const timestamp = "timestamp" in msg ? formatTime(msg.timestamp as string) : "";

    const item = document.createElement("div");
    item.className = `feed-item ${eventType === "new_global_best" ? "feed-item--best" : ""}`;
    item.innerHTML = `
      <span class="feed-time">${timestamp}</span>
      <span class="feed-dot" style="background:${agentColor}"></span>
      <span class="feed-icon">${config.icon}</span>
      <span class="feed-text">${text}</span>
    `;

    // Animate in
    item.style.transform = "translateY(-28px)";
    item.style.opacity = "0";
    this.list.prepend(item);

    requestAnimationFrame(() => {
      item.style.transition = "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease";
      item.style.transform = "translateY(0)";
      item.style.opacity = "1";
    });

    this.items.unshift(item);

    // Older items stay fully visible — user can scroll to see them

    // Remove excess
    while (this.items.length > MAX_ITEMS) {
      const old = this.items.pop()!;
      old.remove();
    }
  }
}
