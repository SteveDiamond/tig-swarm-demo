import type { Panel, WSMessage } from "../types";
import { counterTween, pulseGlow } from "../lib/animate";
import { toggleMute, isMuted } from "../lib/sounds";

export class StatsPanel implements Panel {
  private agentsEl!: HTMLElement;
  private experimentsEl!: HTMLElement;
  private hypothesesEl!: HTMLElement;
  private improvementEl!: HTMLElement;
  private heroEl!: HTMLElement;

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="stats-bar">
        <div class="stats-logo">
          <span class="stats-diamond">&#9670;</span>
          <span class="stats-title">Automated Discovery</span>
          <span id="ws-status" class="ws-status connected">LIVE</span>
          <a href="/ideas.html" class="stats-nav-link">Ideas &rarr;</a>
          <button class="stats-mute-btn" id="mute-btn" title="Toggle sound">&#9835;</button>
        </div>
        <div class="stats-chips">
          <div class="stat-chip" id="stat-agents">
            <span class="stat-label">AGENTS</span>
            <span class="stat-value" id="stat-agents-val">0</span>
          </div>
          <div class="stat-chip" id="stat-experiments">
            <span class="stat-label">EXPERIMENTS</span>
            <span class="stat-value" id="stat-experiments-val">0</span>
          </div>
          <div class="stat-chip" id="stat-hypotheses">
            <span class="stat-label">HYPOTHESES</span>
            <span class="stat-value" id="stat-hypotheses-val">0</span>
          </div>
          <div class="stat-chip" id="stat-improvement">
            <span class="stat-label">IMPROVEMENT</span>
            <span class="stat-value" id="stat-improvement-val">0%</span>
          </div>
          <div class="stat-hero" id="stat-hero"></div>
        </div>
      </div>
    `;

    this.agentsEl = document.getElementById("stat-agents-val")!;
    this.experimentsEl = document.getElementById("stat-experiments-val")!;
    this.hypothesesEl = document.getElementById("stat-hypotheses-val")!;
    this.improvementEl = document.getElementById("stat-improvement-val")!;
    this.heroEl = document.getElementById("stat-hero")!;

    document.getElementById("mute-btn")!.addEventListener("click", () => {
      const m = toggleMute();
      document.getElementById("mute-btn")!.classList.toggle("muted", m);
    });
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "stats_update") {
      counterTween(this.agentsEl, msg.active_agents);
      counterTween(this.experimentsEl, msg.total_experiments);
      counterTween(this.hypothesesEl, msg.hypotheses_count);

      const impEl = this.improvementEl;
      const target = msg.improvement_pct;
      impEl.textContent = `${target > 0 ? "+" : ""}${target.toFixed(1)}%`;
      if (target > 0) impEl.style.color = "var(--green)";
    }

    if (msg.type === "agent_joined") {
      pulseGlow(document.getElementById("stat-agents")!);
    }

    if (msg.type === "experiment_published") {
      pulseGlow(document.getElementById("stat-experiments")!);
    }

    if (msg.type === "new_global_best") {
      this.heroEl.textContent = msg.agent_name;
      this.heroEl.style.opacity = "1";
      pulseGlow(document.getElementById("stat-improvement")!, "#ffab00");
      setTimeout(() => {
        this.heroEl.style.opacity = "0";
      }, 5000);
    }
  }
}
