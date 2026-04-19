import type { Panel, WSMessage } from "../types";
import { counterTween, pulseGlow } from "../lib/animate";


export class StatsPanel implements Panel {
  private agentsEl!: HTMLElement;
  private experimentsEl!: HTMLElement;
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
          <a href="/diversity.html" class="stats-nav-link">Diversity &rarr;</a>
          <a href="/benchmark.html" class="stats-nav-link">Benchmark &rarr;</a>
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
    this.improvementEl = document.getElementById("stat-improvement-val")!;
    this.heroEl = document.getElementById("stat-hero")!;

  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.agentsEl.textContent = "0";
      this.experimentsEl.textContent = "0";
      this.improvementEl.textContent = "0%";
      this.improvementEl.style.color = "";
      this.heroEl.textContent = "";
      this.heroEl.style.opacity = "0";
      return;
    }

    if (msg.type === "stats_update") {
      counterTween(this.agentsEl, msg.total_agents ?? msg.active_agents);
      counterTween(this.experimentsEl, msg.total_experiments);

      // improvement_pct is positive when score rose (= improvement for
      // higher-is-better). Display directly: "+5.0%" green for improvement.
      const impEl = this.improvementEl;
      const target = msg.improvement_pct;
      const sign = target >= 0 ? "+" : "";
      impEl.textContent = `${sign}${target.toFixed(1)}%`;
      impEl.style.color = target > 0 ? "var(--green)" : target < 0 ? "var(--red)" : "";
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
