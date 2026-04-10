import type { Panel, WSMessage } from "../types";
import { formatTime } from "../lib/animate";
import { getAgentColor } from "../lib/colors";

const MAX_ITEMS = 15;

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
        // Score shown in the feed is the avg per-instance score, so it matches
        // the leaderboard / routes panel. The % delta is vs the previous
        // global best: positive when this run beat it (improved), negative
        // when it was worse.
        const divisor = Math.max(msg.num_instances || 1, 1);
        const avgScore = msg.score / divisor;
        const delta = msg.delta_vs_best_pct;
        const deltaStr =
          delta == null
            ? ""
            : ` (${delta >= 0 ? "+" : ""}${delta.toFixed(6)}%)`;
        if (msg.is_new_best) {
          text = `<b>${msg.agent_name}</b> improved &mdash; ${avgScore.toFixed(1)}${deltaStr}`;
          eventType = "new_global_best";
        } else {
          text = `<b>${msg.agent_name}</b> no improvement &mdash; ${avgScore.toFixed(1)}${deltaStr}`;
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

    // Fade older items
    this.items.forEach((el, i) => {
      if (i >= 8) el.style.opacity = `${Math.max(0.15, 1 - (i - 7) * 0.12)}`;
    });

    // Remove excess
    while (this.items.length > MAX_ITEMS) {
      const old = this.items.pop()!;
      old.remove();
    }
  }
}
