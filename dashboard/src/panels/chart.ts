import * as d3 from "d3";
import type { Panel, WSMessage } from "../types";

interface DataPoint {
  time: number; // ms since start
  score: number;
  agentName?: string;
  isBreakthrough?: boolean;
}

export class ChartPanel implements Panel {
  private svg!: any;
  private g!: any;
  private data: DataPoint[] = [];
  private startTime = 0; // set from first data point
  private baseline = 1850.5;
  private width = 0;
  private height = 0;
  private margin = { top: 28, right: 16, bottom: 28, left: 52 };
  private tickHandle: ReturnType<typeof setInterval> | null = null;

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner chart-panel">
        <div class="panel-label">BENCHMARK PROGRESS</div>
        <svg id="chart-svg"></svg>
      </div>
    `;

    const svgEl = document.getElementById("chart-svg")!;
    const rect = svgEl.parentElement!.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height - 24; // account for label

    this.svg = d3.select("#chart-svg")
      .attr("width", this.width)
      .attr("height", this.height);

    // Gradient for area fill
    const defs = this.svg.append("defs");
    const grad = defs.append("linearGradient")
      .attr("id", "area-gradient")
      .attr("x1", "0").attr("y1", "0")
      .attr("x2", "0").attr("y2", "1");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#00e5ff").attr("stop-opacity", 0.2);
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#00e5ff").attr("stop-opacity", 0.0);

    this.g = this.svg.append("g");

    // Handle resize
    const observer = new ResizeObserver(() => {
      const newRect = svgEl.parentElement!.getBoundingClientRect();
      this.width = newRect.width;
      this.height = newRect.height - 24;
      this.svg.attr("width", this.width).attr("height", this.height);
      this.redraw();
    });
    observer.observe(svgEl.parentElement!);

    // Tick the chart so the x-axis keeps growing with elapsed wall time even
    // when no new improvements are landing.
    this.tickHandle = setInterval(() => {
      if (this.startTime > 0 && this.data.length > 0) this.redraw();
    }, 1000);
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "stats_update" && msg.baseline_score) {
      this.baseline = msg.baseline_score;
    }

    if (msg.type === "experiment_published" && msg.feasible) {
      // Use server timestamp if available, otherwise wall clock
      const msgTime = msg.timestamp ? new Date(msg.timestamp).getTime() : Date.now();
      if (this.startTime === 0) this.startTime = msgTime;
      const time = msgTime - this.startTime;

      // Only track the running best
      const currentBest = this.data.length > 0 ? this.data[this.data.length - 1].score : this.baseline;
      const newBest = Math.min(currentBest, msg.score);
      if (newBest < currentBest || this.data.length === 0) {
        this.data.push({
          time: Math.max(0, time),
          score: newBest,
          agentName: msg.agent_name,
          isBreakthrough: msg.is_new_best,
        });
        this.redraw();
      }
    }
  }

  private redraw() {
    if (this.data.length < 1) return;

    this.g.selectAll("*").remove();
    const m = this.margin;
    const w = this.width - m.left - m.right;
    const h = this.height - m.top - m.bottom;

    // X-axis grows with the actual elapsed time since the first data point —
    // this lets the chart keep "scrolling" even when no new bests arrive.
    const elapsed = Math.max(0, Date.now() - this.startTime);
    const latestData = d3.max(this.data, (d) => d.time)!;
    const xScale = d3.scaleLinear()
      .domain([0, Math.max(elapsed, latestData, 1)])
      .range([0, w]);

    const scoreMin = d3.min(this.data, (d) => d.score)! * 0.98;
    // Soft cap at 8k: normally the curve tops out at 8000 regardless of the
    // baseline, but if a data point exceeds that (e.g. an unusually poor
    // feasible run), the scale expands to fit it so the curve never escapes
    // the plot region.
    const dataMax = d3.max(this.data, (d) => d.score)!;
    const scoreMax = Math.max(8000, dataMax);

    // Standard Y axis: high values at the top, low at the bottom. The curve
    // descends as the score improves.
    const yScale = d3.scaleLinear()
      .domain([scoreMin, scoreMax])
      .range([h, 0]);

    const chartG = this.g.append("g")
      .attr("transform", `translate(${m.left},${m.top})`);

    // Grid lines
    const yTicks = yScale.ticks(5);
    yTicks.forEach((tick) => {
      chartG.append("line")
        .attr("x1", 0).attr("x2", w)
        .attr("y1", yScale(tick)).attr("y2", yScale(tick))
        .attr("stroke", "#141c2a")
        .attr("stroke-width", 0.5);
    });

    // Area
    const area = d3.area<DataPoint>()
      .x((d) => xScale(d.time))
      .y0(h)
      .y1((d) => yScale(d.score))
      .curve(d3.curveMonotoneX);

    chartG.append("path")
      .datum(this.data)
      .attr("d", area)
      .attr("fill", "url(#area-gradient)");

    // Line
    const line = d3.line<DataPoint>()
      .x((d) => xScale(d.time))
      .y((d) => yScale(d.score))
      .curve(d3.curveMonotoneX);

    chartG.append("path")
      .datum(this.data)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#00e5ff")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.9);

    // Breakthrough markers
    this.data.filter((d) => d.isBreakthrough).forEach((d) => {
      const x = xScale(d.time);
      const y = yScale(d.score);

      // Vertical dashed line
      chartG.append("line")
        .attr("x1", x).attr("x2", x)
        .attr("y1", 0).attr("y2", h)
        .attr("stroke", "#ffab00")
        .attr("stroke-width", 0.5)
        .attr("stroke-dasharray", "3 3")
        .attr("stroke-opacity", 0.5);

      // Diamond marker
      chartG.append("path")
        .attr("d", d3.symbol(d3.symbolDiamond, 24)())
        .attr("transform", `translate(${x},${y})`)
        .attr("fill", "#ffab00")
        .attr("opacity", 0.9);

      // Label
      if (d.agentName) {
        chartG.append("text")
          .attr("x", x + 6)
          .attr("y", y - 8)
          .attr("fill", "#ffab00")
          .attr("font-size", "9px")
          .attr("font-family", "var(--mono)")
          .attr("opacity", 0.8)
          .text(d.agentName);
      }
    });

    // Y axis labels
    yTicks.forEach((tick) => {
      chartG.append("text")
        .attr("x", -8)
        .attr("y", yScale(tick) + 3)
        .attr("fill", "#3d4a5c")
        .attr("font-size", "9px")
        .attr("font-family", "var(--mono)")
        .attr("text-anchor", "end")
        .text(tick.toFixed(0));
    });

    // X axis labels (mm:ss elapsed)
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
}

function formatElapsed(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}
