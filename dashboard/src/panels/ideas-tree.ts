import type { WSMessage } from "../types";
import { getAgentColor } from "../lib/colors";
import { formatTime } from "../lib/animate";

interface FeedItem {
  id: string;
  agentName: string;
  agentId: string;
  content: string;
  msgType: "agent" | "synthesis" | "milestone";
  timestamp: string;
}

const MAX_FEED_ITEMS = 40;

export class IdeasTree {
  private feedEl!: HTMLElement;
  private knowledgeEl!: HTMLElement;
  private knowledgeEmptyEl!: HTMLElement;
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
          <div class="ideas-knowledge-col">
            <div class="ideas-col-label">KNOWLEDGE STATE</div>
            <div class="ideas-knowledge" id="ideas-knowledge">
              <div class="ideas-knowledge-empty" id="ideas-knowledge-empty">
                <div class="knowledge-empty-icon">&#9671;</div>
                <div class="knowledge-empty-text">The curator agent will synthesize findings here as the swarm works...</div>
              </div>
            </div>
          </div>
        </div>

        <div class="ideas-stats" id="ideas-stats"></div>
      </div>
    `;

    this.feedEl = document.getElementById("ideas-feed")!;
    this.knowledgeEl = document.getElementById("ideas-knowledge")!;
    this.knowledgeEmptyEl = document.getElementById("ideas-knowledge-empty")!;
    this.statsEl = document.getElementById("ideas-stats")!;
  }

  handleMessage(msg: WSMessage) {
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

      case "experiment_published":
        if (msg.is_new_best) {
          this.addFeedItem({
            id: msg.experiment_id,
            agentName: msg.agent_name,
            agentId: msg.agent_id,
            content: `NEW BEST: Score ${msg.score.toFixed(0)} (${msg.improvement_pct > 0 ? "+" : ""}${msg.improvement_pct.toFixed(1)}% improvement)`,
            msgType: "milestone",
            timestamp: msg.timestamp,
          });
        }
        break;

      case "knowledge_updated":
        this.renderKnowledge(msg.content, msg.updated_by);
        break;
    }

    this.updateStats();
  }

  private addFeedItem(item: FeedItem) {
    const el = document.createElement("div");
    el.className = `feed-post feed-post--${item.msgType}`;

    const agentColor = getAgentColor(item.agentId || item.agentName);
    const time = formatTime(item.timestamp);

    if (item.msgType === "synthesis") {
      el.innerHTML = `
        <div class="feed-post-header">
          <span class="feed-post-badge synthesis-badge">SYNTHESIS</span>
          <span class="feed-post-time">${time}</span>
        </div>
        <div class="feed-post-content synthesis-content">${this.renderMarkdown(item.content)}</div>
        <div class="feed-post-author">— ${item.agentName}</div>
      `;
    } else if (item.msgType === "milestone") {
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

    // Animate in
    el.style.opacity = "0";
    el.style.transform = "translateY(-16px)";
    this.feedEl.prepend(el);
    requestAnimationFrame(() => {
      el.style.transition = "opacity 0.35s ease, transform 0.35s ease";
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";
    });

    this.feedItems.unshift(el);

    // Cleanup old items
    while (this.feedItems.length > MAX_FEED_ITEMS) {
      const old = this.feedItems.pop()!;
      old.remove();
    }
  }

  private renderKnowledge(content: string, updatedBy: string) {
    this.knowledgeEmptyEl.style.display = "none";
    // Render markdown-ish content (## headers, bullet lists, bold)
    const html = this.renderMarkdown(content);
    this.knowledgeEl.innerHTML = `
      <div class="knowledge-doc">${html}</div>
      <div class="knowledge-meta">Updated by ${updatedBy} at ${formatTime(new Date().toISOString())}</div>
    `;

    // Brief glow effect on update
    this.knowledgeEl.style.boxShadow = "inset 0 0 30px rgba(0, 229, 255, 0.05)";
    setTimeout(() => { this.knowledgeEl.style.boxShadow = ""; }, 2000);
  }

  private renderMarkdown(text: string): string {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      // ## Headers
      .replace(/^## (.+)$/gm, '<h3 class="knowledge-h2">$1</h3>')
      .replace(/^### (.+)$/gm, '<h4 class="knowledge-h3">$1</h4>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      // Bullet lists
      .replace(/^- (.+)$/gm, '<div class="knowledge-bullet">$1</div>')
      // Line breaks
      .replace(/\n\n/g, '<div class="knowledge-gap"></div>')
      .replace(/\n/g, "<br>");
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
