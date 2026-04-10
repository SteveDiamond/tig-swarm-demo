import * as d3 from "d3";
import type { Panel, WSMessage, RouteData, AllRouteData, RoutePoint } from "../types";
import { getRouteColor } from "../lib/colors";

// Drawing sizes as fractions of the viewBox side length. Everything else in
// this file should reference these constants — never hardcode pixel/unit
// values, because the viewBox is fit tightly to the data and its absolute
// scale varies per dataset. Tweak these to resize elements.
const STYLE = {
  customerRadius: 0.006,          // customer dot radius
  depotSize:      0.020,          // depot diamond side length (before rotate)
  routeStroke:    0.004,          // main route line thickness
  glowStroke:     0.012,          // blurred glow halo behind each route
  routeDashOn:    0.018,          // dash length for the flowing stroke
  routeDashOff:   0.007,          // gap length for the flowing stroke
} as const;

const routeLine = d3.line<RoutePoint>()
  .x((d) => d.x)
  .y((d) => d.y)
  .curve(d3.curveCatmullRom.alpha(0.5));

function fullPath(data: RouteData, route: { path: RoutePoint[] }): RoutePoint[] {
  const depot = { x: data.depot.x, y: data.depot.y, customer_id: -1 };
  return [depot, ...route.path, depot];
}

// Sum of Euclidean distances over every leg of every vehicle's route, with the
// depot stitched onto each end. Matches how the solver computes route length.
function computeRouteDistance(data: RouteData): number {
  let total = 0;
  for (const route of data.routes) {
    const path = fullPath(data, route);
    for (let i = 0; i < path.length - 1; i++) {
      const dx = path[i + 1].x - path[i].x;
      const dy = path[i + 1].y - path[i].y;
      total += Math.sqrt(dx * dx + dy * dy);
    }
  }
  return total;
}

export class RoutesPanel implements Panel {
  private svg!: any;
  private routeGroup!: any;
  private customerGroup!: any;
  private depotGroup!: any;
  private scoreEl!: HTMLElement;
  private scoreDeltaEl!: HTMLElement;
  private routeDistanceEl!: HTMLElement;
  private instanceLabelEl!: HTMLElement;
  private navEl!: HTMLElement;

  private allInstances: AllRouteData = {};
  private currentIndex = 0;
  private currentRouteData: RouteData | null = null;
  private numInstances = 1;
  // Side length of the current viewBox in SVG user units. All draw sizes
  // are computed as STYLE.* × viewSide so they stay visually consistent
  // regardless of how spread out the underlying data is.
  private viewSide = 1000;
  // Raw experiment score (sum across all instances). The displayed SCORE is
  // this divided by numInstances so it matches the leaderboard's avg metric.
  private rawScore: number | null = null;

  private get instanceKeys(): string[] {
    return Object.keys(this.allInstances).sort();
  }

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner routes-panel">
        <div class="panel-label">ROUTES</div>
        <div class="routes-nav" id="routes-nav" style="display:none">
          <button class="routes-nav-btn" id="routes-prev">&lsaquo;</button>
          <span class="routes-instance-label" id="routes-instance-label"></span>
          <button class="routes-nav-btn" id="routes-next">&rsaquo;</button>
        </div>
        <div class="routes-svg-wrap" id="routes-svg-wrap">
          <svg id="routes-svg"></svg>
        </div>
        <div class="routes-route-distance">
          <div class="routes-sub-label">ROUTE DISTANCE</div>
          <div class="routes-sub-value" id="routes-route-distance">---</div>
        </div>
        <div class="routes-score">
          <div class="routes-score-label">SCORE</div>
          <div class="routes-score-value" id="routes-score">---</div>
          <div class="routes-score-delta" id="routes-score-delta"></div>
        </div>
      </div>
    `;

    this.scoreEl = document.getElementById("routes-score")!;
    this.scoreDeltaEl = document.getElementById("routes-score-delta")!;
    this.routeDistanceEl = document.getElementById("routes-route-distance")!;
    this.instanceLabelEl = document.getElementById("routes-instance-label")!;
    this.navEl = document.getElementById("routes-nav")!;

    document.getElementById("routes-prev")!.addEventListener("click", () => this.navigate(-1));
    document.getElementById("routes-next")!.addEventListener("click", () => this.navigate(1));

    this.svg = d3.select("#routes-svg");
    this.svg
      .attr("viewBox", "0 0 1000 1000")
      .attr("preserveAspectRatio", "xMidYMid meet");

    const defs = this.svg.append("defs");
    const filter = defs.append("filter").attr("id", "route-glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "1.5").attr("result", "blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    this.routeGroup = this.svg.append("g").attr("class", "routes");
    this.customerGroup = this.svg.append("g").attr("class", "customers");
    this.depotGroup = this.svg.append("g").attr("class", "depot");

    // Make the SVG element a square sized to the largest square that fits
    // inside the wrap. Without this the SVG fills the wrap rectangle but the
    // 1:1 viewBox letterboxes a square inside it, leaving large empty side
    // margins on a wide panel.
    const wrap = document.getElementById("routes-svg-wrap")!;
    const resize = () => {
      const size = Math.max(0, Math.min(wrap.clientWidth, wrap.clientHeight));
      this.svg.attr("width", size).attr("height", size);
    };
    new ResizeObserver(resize).observe(wrap);
    resize();

    setInterval(() => {
      if (this.instanceKeys.length > 1) {
        this.navigate(1);
      }
    }, 8000);
  }

  // Compute a square viewBox that tightly bounds *all* instances' data with a
  // small padding margin. Using all instances (rather than per-instance) keeps
  // the zoom stable as you click through them.
  private updateViewBox() {
    const all = Object.values(this.allInstances);
    if (all.length === 0) {
      this.viewSide = 1000;
      this.svg.attr("viewBox", "0 0 1000 1000");
      return;
    }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const inst of all) {
      const consider = (x: number, y: number) => {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      };
      consider(inst.depot.x, inst.depot.y);
      for (const route of inst.routes) {
        for (const p of route.path) consider(p.x, p.y);
      }
    }
    if (!isFinite(minX)) {
      this.viewSide = 1000;
      this.svg.attr("viewBox", "0 0 1000 1000");
      return;
    }
    const w = maxX - minX;
    const h = maxY - minY;
    const side = Math.max(w, h, 1);
    const padding = side * 0.06;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const finalSide = side + padding * 2;
    const x = cx - finalSide / 2;
    const y = cy - finalSide / 2;
    this.viewSide = finalSide;
    this.svg.attr("viewBox", `${x} ${y} ${finalSide} ${finalSide}`);
  }

  private navigate(delta: number) {
    const keys = this.instanceKeys;
    if (keys.length === 0) return;
    this.currentIndex = (this.currentIndex + delta + keys.length) % keys.length;
    this.updateInstanceLabel();
    this.showInstance(this.allInstances[keys[this.currentIndex]]);
  }

  private updateInstanceLabel() {
    const keys = this.instanceKeys;
    if (keys.length <= 1) {
      this.navEl.style.display = "none";
      return;
    }
    this.navEl.style.display = "flex";
    const key = keys[this.currentIndex];
    const label = key.replace(/\.txt$/, "");
    this.instanceLabelEl.textContent = `${label}  (${this.currentIndex + 1}/${keys.length})`;
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.allInstances = {};
      this.currentRouteData = null;
      this.currentIndex = 0;
      this.rawScore = null;
      this.viewSide = 1000;
      this.routeGroup.selectAll("*").remove();
      this.customerGroup.selectAll("*").remove();
      this.depotGroup.selectAll("*").remove();
      this.svg.attr("viewBox", "0 0 1000 1000");
      this.scoreEl.textContent = "---";
      this.scoreDeltaEl.textContent = "";
      this.routeDistanceEl.textContent = "---";
      this.navEl.style.display = "none";
      this.instanceLabelEl.textContent = "";
      return;
    }

    if (msg.type === "stats_update") {
      if (msg.num_instances) this.numInstances = msg.num_instances;
      // Show score even before any route data has arrived. Once route data
      // exists, new_global_best is the source of truth.
      if (msg.best_score != null && !this.currentRouteData) {
        this.rawScore = msg.best_score;
        this.scoreEl.textContent = (msg.best_score / Math.max(this.numInstances, 1)).toFixed(1);
      }
    }

    if (msg.type === "new_global_best" && msg.route_data) {
      if (msg.num_instances) this.numInstances = msg.num_instances;
      this.rawScore = msg.score;
      this.allInstances = msg.route_data;
      this.updateViewBox();

      const keys = this.instanceKeys;
      if (this.currentIndex >= keys.length) this.currentIndex = 0;

      this.updateInstanceLabel();
      this.showInstance(this.allInstances[keys[this.currentIndex]]);

      // SCORE = avg per-instance score for the global best algorithm
      this.scoreEl.textContent = (msg.score / Math.max(this.numInstances, 1)).toFixed(1);

      // % improvement vs the previous global best. By construction this fires
      // only on a new best, so the value is positive (lower score = better).
      if (msg.incremental_improvement_pct != null) {
        const v = msg.incremental_improvement_pct;
        this.scoreDeltaEl.textContent = `+${v.toFixed(5)}% vs prev best`;
        this.scoreDeltaEl.style.color = "var(--green)";
      } else {
        this.scoreDeltaEl.textContent = "first global best";
        this.scoreDeltaEl.style.color = "var(--text-dim)";
      }
    }
  }

  // Immediate, non-animated draw of one instance's route data.
  private showInstance(data: RouteData) {
    this.currentRouteData = data;

    this.routeGroup.selectAll("*").remove();
    this.customerGroup.selectAll("*").remove();
    this.depotGroup.selectAll("*").remove();

    const s = this.viewSide;
    const customerR = STYLE.customerRadius * s;
    const routeW = STYLE.routeStroke * s;
    const glowW = STYLE.glowStroke * s;
    const dashOn = STYLE.routeDashOn * s;
    const dashOff = STYLE.routeDashOff * s;

    data.routes.forEach((route, i) => {
      const path = fullPath(data, route);
      const color = getRouteColor(i);

      // Glow halo
      this.routeGroup.append("path")
        .datum(path)
        .attr("d", routeLine as any)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", glowW)
        .attr("stroke-opacity", 0.1)
        .attr("filter", "url(#route-glow)");

      // Main path
      this.routeGroup.append("path")
        .datum(path)
        .attr("d", routeLine as any)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", routeW)
        .attr("stroke-opacity", 0.9)
        .attr("stroke-dasharray", `${dashOn} ${dashOff}`)
        .attr("class", "route-flowing");

      // Customers
      route.path.forEach((pt) => {
        this.customerGroup.append("circle")
          .attr("cx", pt.x)
          .attr("cy", pt.y)
          .attr("r", customerR)
          .attr("fill", color)
          .attr("opacity", 0.75);
      });
    });

    // Depot
    const depotSize = STYLE.depotSize * s;
    this.depotGroup.append("rect")
      .attr("x", data.depot.x - depotSize / 2)
      .attr("y", data.depot.y - depotSize / 2)
      .attr("width", depotSize)
      .attr("height", depotSize)
      .attr("fill", "#fff")
      .attr("opacity", 0.9)
      .attr("transform", `rotate(45, ${data.depot.x}, ${data.depot.y})`)
      .attr("class", "depot-pulse");

    // ROUTE DISTANCE = total Euclidean distance for the currently shown instance
    this.routeDistanceEl.textContent = computeRouteDistance(data).toFixed(1);
  }
}
