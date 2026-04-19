import type { WSMessage } from "../types";
import { getAgentColor } from "../lib/colors";
import { formatTime } from "../lib/animate";

interface FeedItem {
  id: string;
  agentName: string;
  agentId: string;
  content: string;
  msgType: "agent" | "milestone";
  timestamp: string;
}

const MAX_FEED_ITEMS = 40;

export class IdeasTree {
  private feedEl!: HTMLElement;
  private feedItems: HTMLElement[] = [];
  private statsEl!: HTMLElement;
  private hypothesisCount = 0;
  private succeededCount = 0;
  private failedCount = 0;
  private messageCount = 0;

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="ideas-page">
        <div class="ideas-header">
          <div class="ideas-title">
            <span class="stats-diamond">&#9670;</span>
            <span class="ideas-title-text">Collective Intelligence</span>
          </div>
          <div class="ideas-nav">
            <a href="/" class="ideas-nav-link">Dashboard</a>
            <span class="ideas-nav-active">Ideas</span>
          </div>
        </div>

        <div class="ideas-body">
          <div class="ideas-feed-col">
            <div class="ideas-col-label">RESEARCH FEED</div>
            <div class="ideas-feed" id="ideas-feed"></div>
          </div>
          <div class="ideas-right-col" id="strategy-lb-mount"></div>
        </div>

        <div class="ideas-stats" id="ideas-stats"></div>
      </div>
    `;

    this.feedEl = document.getElementById("ideas-feed")!;
    this.statsEl = document.getElementById("ideas-stats")!;
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.feedItems.forEach((el) => el.remove());
      this.feedItems = [];
      this.feedEl.innerHTML = "";
      this.hypothesisCount = 0;
      this.succeededCount = 0;
      this.failedCount = 0;
      this.messageCount = 0;
      this.updateStats();
      return;
    }

    switch (msg.type) {
      case "chat_message":
        this.addFeedItem({
          id: msg.message_id,
          agentName: msg.agent_name,
          agentId: msg.agent_id || "",
          content: msg.content,
          msgType: msg.msg_type,
          timestamp: msg.timestamp,
        });
        this.messageCount++;
        break;

      case "hypothesis_proposed":
        this.hypothesisCount++;
        this.addFeedItem({
          id: msg.hypothesis_id,
          agentName: msg.agent_name,
          agentId: msg.agent_id,
          content: `Proposed: "${msg.title}"`,
          msgType: "agent",
          timestamp: msg.timestamp,
        });
        break;

      case "hypothesis_status_changed":
        if (msg.new_status === "succeeded") this.succeededCount++;
        if (msg.new_status === "failed") this.failedCount++;
        break;

      case "experiment_published": {
        // Three outcomes: new global best → milestone with own + global %s;
        // beats own best but not global → lightweight "improvement" post;
        // no improvement → skip (research feed stays narrative-focused).
        const fmtPct = (p: number | null | undefined): string => {
          if (p == null) return "";
          const sign = p >= 0 ? "-" : "+";
          return `${sign}${Math.abs(p).toFixed(2)}%`;
        };

        if (msg.is_new_best) {
          const ownPart = msg.delta_vs_own_best_pct != null
            ? ` (${fmtPct(msg.delta_vs_own_best_pct)} own)`
            : "";
          const globalPart = msg.delta_vs_best_pct != null
            ? ` and NEW GLOBAL BEST (${fmtPct(msg.delta_vs_best_pct)} vs global)`
            : " and NEW GLOBAL BEST";
          this.addFeedItem({
            id: msg.experiment_id,
            agentName: msg.agent_name,
            agentId: msg.agent_id,
            content: `Improvement — Score ${msg.score.toFixed(1)}${ownPart}${globalPart}`,
            msgType: "milestone",
            timestamp: msg.timestamp,
          });
        } else if (msg.beats_own_best === true) {
          const ownPart = msg.delta_vs_own_best_pct != null
            ? ` (${fmtPct(msg.delta_vs_own_best_pct)})`
            : "";
          this.addFeedItem({
            id: msg.experiment_id,
            agentName: msg.agent_name,
            agentId: msg.agent_id,
            content: `Improvement — Score ${msg.score.toFixed(1)}${ownPart}`,
            msgType: "agent",
            timestamp: msg.timestamp,
          });
        }
        break;
      }
    }

    this.updateStats();
  }

  private addFeedItem(item: FeedItem) {
    const el = document.createElement("div");
    el.className = `feed-post feed-post--${item.msgType}`;

    const agentColor = getAgentColor(item.agentId || item.agentName);
    const time = formatTime(item.timestamp);

    if (item.msgType === "milestone") {
      el.innerHTML = `
        <div class="feed-post-header">
          <span class="feed-post-badge milestone-badge">&#9733; MILESTONE</span>
          <span class="feed-post-time">${time}</span>
        </div>
        <div class="feed-post-content milestone-content">${item.content}</div>
        <div class="feed-post-author">
          <span class="feed-post-dot" style="background:${agentColor}"></span>
          ${item.agentName}
        </div>
      `;
    } else {
      el.innerHTML = `
        <div class="feed-post-agent">
          <span class="feed-post-dot" style="background:${agentColor}"></span>
          <span class="feed-post-name">${item.agentName}</span>
          <span class="feed-post-time">${time}</span>
        </div>
        <div class="feed-post-content">${item.content}</div>
      `;
    }

    el.style.opacity = "0";
    el.style.transform = "translateY(-16px)";
    this.feedEl.prepend(el);
    requestAnimationFrame(() => {
      el.style.transition = "opacity 0.35s ease, transform 0.35s ease";
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";
    });

    this.feedItems.unshift(el);

    while (this.feedItems.length > MAX_FEED_ITEMS) {
      const old = this.feedItems.pop()!;
      old.remove();
    }
  }

  private updateStats() {
    const active = this.hypothesisCount - this.succeededCount - this.failedCount;
    this.statsEl.innerHTML = `
      <span class="ideas-stat">HYPOTHESES <b>${this.hypothesisCount}</b></span>
      <span class="ideas-stat">SUCCEEDED <b style="color:var(--green)">${this.succeededCount}</b></span>
      <span class="ideas-stat">FAILED <b style="color:var(--red)">${this.failedCount}</b></span>
      <span class="ideas-stat">ACTIVE <b style="color:var(--cyan)">${Math.max(0, active)}</b></span>
      <span class="ideas-stat">MESSAGES <b>${this.messageCount}</b></span>
    `;
  }
}
