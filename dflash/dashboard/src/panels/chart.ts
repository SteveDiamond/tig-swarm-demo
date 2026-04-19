import * as d3 from "d3";
import { getAgentColor } from "../lib/colors";
import type { Panel, WSMessage } from "../types";

interface DataPoint {
  time: number; // ms since start
  score: number;
  agentName?: string;
  agentId?: string;
  isBreakthrough?: boolean;
}

type Tab =
  | { type: "global" }
  | { type: "agent"; agentId: string; agentName: string };

interface AgentProgress {
  registeredAt: number; // epoch ms
  experiments: { time: number; score: number; feasible: boolean; experimentId?: string }[]; // time = ms since registeredAt
  experimentIds: Set<string>;
  loaded: boolean;
  lastEventTime: number; // epoch ms of most recent appended experiment
}

export class ChartPanel implements Panel {
  private svg!: any;
  private g!: any;
  private globalData: DataPoint[] = [];
  private globalStartTime = 0;
  private width = 0;
  private height = 0;
  private margin = { top: 28, right: 16, bottom: 28, left: 52 };

  private apiUrl = "";

  private tabs: Tab[] = [{ type: "global" }];
  private currentTabIndex = 0;

  private agentProgress = new Map<string, AgentProgress>();
  // Live experiment events that arrive before /api/agent_experiments has
  // finished loading for a given agent.
  private pendingAgentExperiments = new Map<string, any[]>();

  private tabLabelEl!: HTMLElement;
  private tabPrevEl!: HTMLElement;
  private tabNextEl!: HTMLElement;

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner chart-panel">
        <div class="panel-label">BENCHMARK PROGRESS</div>
        <div class="chart-tabs" id="chart-tabs">
          <button class="chart-tab-btn" id="chart-tab-prev" type="button">&lsaquo;</button>
          <span class="chart-tab-label" id="chart-tab-label">GLOBAL</span>
          <button class="chart-tab-btn" id="chart-tab-next" type="button">&rsaquo;</button>
        </div>
        <svg id="chart-svg"></svg>
      </div>
    `;

    this.tabLabelEl = document.getElementById("chart-tab-label")!;
    this.tabPrevEl = document.getElementById("chart-tab-prev")!;
    this.tabNextEl = document.getElementById("chart-tab-next")!;

    this.tabPrevEl.addEventListener("click", () => this.cycleTab(-1));
    this.tabNextEl.addEventListener("click", () => this.cycleTab(1));

    const svgEl = document.getElementById("chart-svg")!;
    const rect = svgEl.parentElement!.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height - 48; // label + tab row

    this.svg = d3.select("#chart-svg")
      .attr("width", this.width)
      .attr("height", this.height);

    this.g = this.svg.append("g");

    // Resolve API base URL the same way other panels do.
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

    const observer = new ResizeObserver(() => {
      const newRect = svgEl.parentElement!.getBoundingClientRect();
      this.width = newRect.width;
      this.height = newRect.height - 48;
      this.svg.attr("width", this.width).attr("height", this.height);
      this.redraw();
    });
    observer.observe(svgEl.parentElement!);

    this.renderTabLabel();
  }

  // Seed the chart with the full best-so-far trajectory in one batch.
  // `entries` must be in chronological order. Called on initial load so the
  // chart reflects the entire run, not just the recent-20 window returned by
  // /api/state.
  //
  // We apply a running-maximum filter: server-side best_history can contain
  // non-improving rows, but the chart is a best-so-far trajectory, so only
  // strictly-improving points belong on it. Higher score = better.
  seedHistory(entries: { score: number; agent_name: string; agent_id?: string; created_at: string }[]) {
    if (!entries.length) return;
    const first = new Date(entries[0].created_at).getTime();
    this.globalStartTime = first;
    const filtered: DataPoint[] = [];
    let runningBest = -Infinity;
    for (const e of entries) {
      if (e.score <= runningBest) continue;
      runningBest = e.score;
      filtered.push({
        time: Math.max(0, new Date(e.created_at).getTime() - first),
        score: e.score,
        agentName: e.agent_name,
        agentId: e.agent_id,
        isBreakthrough: true,
      });
    }
    this.globalData = filtered;
    if (this.currentTab().type === "global") this.redraw();
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.globalData = [];
      this.globalStartTime = 0;
      this.agentProgress.clear();
      this.pendingAgentExperiments.clear();
      this.tabs = [{ type: "global" }];
      this.currentTabIndex = 0;
      this.renderTabLabel();
      this.g.selectAll("*").remove();
      return;
    }

    if (msg.type === "leaderboard_update") {
      this.syncTabsFromLeaderboard(msg.entries);
    }

    if (msg.type === "experiment_published") {
      this.updateGlobalFromMessage(msg);
      this.appendAgentExperiment(msg);
    }
  }

  // ── Tab navigation ──

  private currentTab(): Tab {
    return this.tabs[this.currentTabIndex];
  }

  private cycleTab(delta: number) {
    if (this.tabs.length === 0) return;
    this.currentTabIndex = (this.currentTabIndex + delta + this.tabs.length) % this.tabs.length;
    this.renderTabLabel();
    const tab = this.currentTab();
    if (tab.type === "agent") {
      this.ensureAgentLoaded(tab.agentId).then(() => {
        if (this.currentTab().type === "agent"
            && (this.currentTab() as any).agentId === tab.agentId) {
          this.redraw();
        }
      });
    } else {
      this.redraw();
    }
  }

  private renderTabLabel() {
    const tab = this.currentTab();
    if (tab.type === "global") {
      this.tabLabelEl.textContent = "GLOBAL";
      this.tabLabelEl.style.color = "";
    } else {
      this.tabLabelEl.textContent = tab.agentName;
      this.tabLabelEl.style.color = getAgentColor(tab.agentId);
    }
  }

  private syncTabsFromLeaderboard(entries: { agent_id: string; agent_name: string }[]) {
    const currentTab = this.currentTab();
    const activeAgentId = currentTab.type === "agent" ? currentTab.agentId : null;

    // Keep GLOBAL first, then agents in leaderboard order.
    const newTabs: Tab[] = [{ type: "global" }];
    for (const entry of entries) {
      if (!entry.agent_id) continue;
      newTabs.push({
        type: "agent",
        agentId: entry.agent_id,
        agentName: entry.agent_name,
      });
    }
    this.tabs = newTabs;

    // Preserve the user's current selection across reorderings.
    if (activeAgentId) {
      const idx = this.tabs.findIndex(
        (t) => t.type === "agent" && t.agentId === activeAgentId
      );
      this.currentTabIndex = idx >= 0 ? idx : 0;
    } else {
      this.currentTabIndex = Math.min(this.currentTabIndex, this.tabs.length - 1);
    }
    this.renderTabLabel();
  }

  // ── Global chart data (existing behavior) ──

  private updateGlobalFromMessage(msg: any) {
    if (!msg.feasible) return;
    const msgTime = msg.timestamp ? new Date(msg.timestamp).getTime() : Date.now();
    if (this.globalStartTime === 0) this.globalStartTime = msgTime;
    const time = msgTime - this.globalStartTime;

    const tryAppend = () => {
      this.globalData.push({
        time: Math.max(0, time),
        score: msg.score,
        agentName: msg.agent_name,
        agentId: msg.agent_id,
        isBreakthrough: msg.is_new_best,
      });
      if (this.currentTab().type === "global") this.redraw();
    };

    if (this.globalData.length === 0) {
      tryAppend();
    } else {
      const currentBest = this.globalData[this.globalData.length - 1].score;
      if (msg.score > currentBest) tryAppend();
    }
  }

  // ── Per-agent chart data ──

  private async ensureAgentLoaded(agentId: string): Promise<void> {
    const existing = this.agentProgress.get(agentId);
    if (existing?.loaded) return;

    try {
      const res = await fetch(`${this.apiUrl}/api/agent_experiments?agent_id=${encodeURIComponent(agentId)}`);
      if (!res.ok) return;
      const data: {
        agent_id: string;
        agent_name: string | null;
        registered_at: string | null;
        experiments: { id?: string; score: number; feasible: boolean; created_at: string }[];
      } = await res.json();

      const registeredAt = data.registered_at
        ? new Date(data.registered_at).getTime()
        : Date.now();

      const experiments = data.experiments.map((e) => ({
        time: Math.max(0, new Date(e.created_at).getTime() - registeredAt),
        score: e.score,
        feasible: e.feasible,
        experimentId: e.id,
      }));

      const experimentIds = new Set(
        experiments
          .map((e) => e.experimentId)
          .filter((id): id is string => Boolean(id))
      );

      const lastEventTime = data.experiments.length
        ? new Date(data.experiments[data.experiments.length - 1].created_at).getTime()
        : 0;

      const progress: AgentProgress = {
        registeredAt,
        experiments,
        experimentIds,
        loaded: true,
        lastEventTime,
      };

      // Merge any live events that landed while the history request was in-flight.
      const pending = this.pendingAgentExperiments.get(agentId) || [];
      for (const msg of pending) {
        this.appendToAgentProgress(progress, msg);
      }
      this.pendingAgentExperiments.delete(agentId);

      this.agentProgress.set(agentId, progress);
    } catch {
      // leave unloaded; next tab visit will retry
    }
  }

  private appendToAgentProgress(progress: AgentProgress, msg: any): boolean {
    const msgTime = msg.timestamp ? new Date(msg.timestamp).getTime() : Date.now();
    const experimentId = typeof msg.experiment_id === "string" ? msg.experiment_id : null;

    if (experimentId && progress.experimentIds.has(experimentId)) {
      return false;
    }

    const time = Math.max(0, msgTime - progress.registeredAt);
    const feasible = msg.feasible !== false;

    // Legacy safety net when an event lacks experiment_id.
    if (!experimentId) {
      const duplicate = progress.experiments.some(
        (e) => !e.experimentId && e.time === time && e.score === msg.score && e.feasible === feasible
      );
      if (duplicate) return false;
    }

    progress.experiments.push({
      time,
      score: msg.score,
      feasible,
      experimentId: experimentId || undefined,
    });
    if (experimentId) progress.experimentIds.add(experimentId);
    progress.lastEventTime = Math.max(progress.lastEventTime, msgTime);
    return true;
  }

  private appendAgentExperiment(msg: any) {
    if (!msg.agent_id) return;
    const progress = this.agentProgress.get(msg.agent_id);
    if (!progress || !progress.loaded) {
      const pending = this.pendingAgentExperiments.get(msg.agent_id) || [];
      pending.push(msg);
      this.pendingAgentExperiments.set(msg.agent_id, pending);
      return;
    }
    const added = this.appendToAgentProgress(progress, msg);
    if (!added) return;

    const tab = this.currentTab();
    if (tab.type === "agent" && tab.agentId === msg.agent_id) {
      this.redraw();
    }
  }

  // ── Rendering ──

  private redraw() {
    const tab = this.currentTab();
    if (tab.type === "global") {
      this.redrawGlobal();
    } else {
      this.redrawAgent(tab.agentId, tab.agentName);
    }
  }

  private redrawGlobal() {
    this.g.selectAll("*").remove();
    if (this.globalData.length < 1) return;

    const m = this.margin;
    const w = this.width - m.left - m.right;
    const h = this.height - m.top - m.bottom;

    const latestData = d3.max(this.globalData, (d) => d.time)!;
    const xPad = Math.max(latestData * 0.15, 5000);
    const xScale = d3.scaleLinear()
      .domain([0, latestData + xPad])
      .range([0, w]);

    const yDomain = this.getGlobalYDomain();
    if (!yDomain) return;

    const yScale = d3.scaleLinear()
      .domain(yDomain)
      .range([h, 0]);

    const chartG = this.g.append("g")
      .attr("transform", `translate(${m.left},${m.top})`);

    const yTicks = yScale.ticks(6);
    yTicks.forEach((tick) => {
      chartG.append("line")
        .attr("x1", 0).attr("x2", w)
        .attr("y1", yScale(tick)).attr("y2", yScale(tick))
        .attr("stroke", "#141c2a")
        .attr("stroke-width", 0.5);
    });

    const trailTime = latestData + xPad;
    for (let i = 0; i < this.globalData.length; i++) {
      const d = this.globalData[i];
      const nextX = i < this.globalData.length - 1 ? xScale(this.globalData[i + 1].time) : xScale(trailTime);
      const x0 = xScale(d.time);
      const y0 = yScale(d.score);
      const color = getAgentColor(d.agentId || d.agentName || "unknown");

      chartG.append("rect")
        .attr("x", x0)
        .attr("y", y0)
        .attr("width", Math.max(0, nextX - x0))
        .attr("height", Math.max(0, h - y0))
        .attr("fill", color)
        .attr("opacity", 0.08);

      chartG.append("line")
        .attr("x1", x0).attr("x2", nextX)
        .attr("y1", y0).attr("y2", y0)
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.9);

      if (i < this.globalData.length - 1) {
        const nextY = yScale(this.globalData[i + 1].score);
        const nextColor = getAgentColor(this.globalData[i + 1].agentId || this.globalData[i + 1].agentName || "unknown");
        chartG.append("line")
          .attr("x1", nextX).attr("x2", nextX)
          .attr("y1", y0).attr("y2", nextY)
          .attr("stroke", nextColor)
          .attr("stroke-width", 2)
          .attr("stroke-opacity", 0.9);
      }
    }

    this.globalData.filter((d) => d.isBreakthrough).forEach((d) => {
      const x = xScale(d.time);
      const y = yScale(d.score);
      const color = getAgentColor(d.agentId || d.agentName || "unknown");

      chartG.append("line")
        .attr("x1", x).attr("x2", x)
        .attr("y1", 0).attr("y2", h)
        .attr("stroke", color)
        .attr("stroke-width", 0.5)
        .attr("stroke-dasharray", "3 3")
        .attr("stroke-opacity", 0.5);

      chartG.append("path")
        .attr("d", d3.symbol(d3.symbolDiamond, 24)())
        .attr("transform", `translate(${x},${y})`)
        .attr("fill", color)
        .attr("opacity", 0.9);

      if (d.agentName) {
        chartG.append("text")
          .attr("x", x + 6)
          .attr("y", y - 8)
          .attr("fill", color)
          .attr("font-size", "9px")
          .attr("font-family", "var(--mono)")
          .attr("opacity", 0.8)
          .text(d.agentName);
      }
    });

    yTicks.forEach((tick) => {
      chartG.append("text")
        .attr("x", -8)
        .attr("y", yScale(tick) + 3)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "9px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "end")
        .text(tick.toFixed(1));
    });

    const xTicks = xScale.ticks(6);
    xTicks.forEach((tick) => {
      chartG.append("text")
        .attr("x", xScale(tick))
        .attr("y", h + 16)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "9px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "middle")
        .text(formatElapsed(tick));
    });
  }

  private redrawAgent(agentId: string, agentName: string) {
    this.g.selectAll("*").remove();

    const progress = this.agentProgress.get(agentId);
    const m = this.margin;
    const w = this.width - m.left - m.right;
    const h = this.height - m.top - m.bottom;

    const chartG = this.g.append("g")
      .attr("transform", `translate(${m.left},${m.top})`);

    if (!progress || progress.experiments.length === 0) {
      chartG.append("text")
        .attr("x", w / 2)
        .attr("y", h / 2)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "11px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "middle")
        .text(progress ? `no attempts yet from ${agentName}` : "loading…");
      return;
    }

    const color = getAgentColor(agentId);
    const exps = progress.experiments;

    // X: from registration (0) to last attempt (frozen — no live trailing).
    const latestTime = exps[exps.length - 1].time;
    const xDomainEnd = Math.max(latestTime, 1000);
    const xScale = d3.scaleLinear()
      .domain([0, xDomainEnd])
      .range([0, w]);

    // Y: match the GLOBAL chart domain/limits exactly when available.
    // This intentionally allows agent points outside the domain to render
    // off-chart so all tabs share the same visual scale.
    const globalYDomain = this.getGlobalYDomain();
    const minScore = d3.min(exps, (d) => d.score)!;
    const maxScore = d3.max(exps, (d) => d.score)!;
    const fallbackMin = Math.max(0, minScore * 0.9);
    const fallbackMax = Math.max(fallbackMin + 0.5, maxScore * 1.1);
    const yScale = d3.scaleLinear()
      .domain(globalYDomain ?? [fallbackMin, fallbackMax])
      .range([h, 0]);

    const yTicks = yScale.ticks(5);
    yTicks.forEach((tick) => {
      chartG.append("line")
        .attr("x1", 0).attr("x2", w)
        .attr("y1", yScale(tick)).attr("y2", yScale(tick))
        .attr("stroke", "#141c2a")
        .attr("stroke-width", 0.5);
    });

    // Step plot: each attempt's score is held until the next attempt.
    // The final attempt terminates at its own time (frozen x-axis).
    for (let i = 0; i < exps.length; i++) {
      const d = exps[i];
      const x0 = xScale(d.time);
      const y0 = yScale(d.score);
      const next = exps[i + 1];
      const xEnd = next ? xScale(next.time) : x0;

      if (xEnd > x0) {
        chartG.append("line")
          .attr("x1", x0).attr("x2", xEnd)
          .attr("y1", y0).attr("y2", y0)
          .attr("stroke", color)
          .attr("stroke-width", 2)
          .attr("stroke-opacity", 0.9);
      }

      if (next) {
        const yNext = yScale(next.score);
        chartG.append("line")
          .attr("x1", xEnd).attr("x2", xEnd)
          .attr("y1", y0).attr("y2", yNext)
          .attr("stroke", color)
          .attr("stroke-width", 2)
          .attr("stroke-opacity", 0.9);
      }

      // Attempt marker — dimmer for infeasible so they're distinguishable.
      chartG.append("circle")
        .attr("cx", x0)
        .attr("cy", y0)
        .attr("r", 2.5)
        .attr("fill", color)
        .attr("opacity", d.feasible ? 0.9 : 0.4);
    }

    yTicks.forEach((tick) => {
      chartG.append("text")
        .attr("x", -8)
        .attr("y", yScale(tick) + 3)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "9px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "end")
        .text(tick.toFixed(1));
    });

    const xTicks = xScale.ticks(6);
    xTicks.forEach((tick) => {
      chartG.append("text")
        .attr("x", xScale(tick))
        .attr("y", h + 16)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "9px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "middle")
        .text(formatElapsed(tick));
    });
  }

  private getGlobalYDomain(): [number, number] | null {
    if (this.globalData.length < 1) return null;
    const scoreMax = d3.max(this.globalData, (d) => d.score);
    const seedScore = this.globalData[0]?.score;
    if (scoreMax == null || seedScore == null) return null;

    const yMin = 0;
    const yMax = Math.max(scoreMax * 1.15, seedScore + 1, 2);
    return [yMin, yMax];
  }
}

function formatElapsed(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}
