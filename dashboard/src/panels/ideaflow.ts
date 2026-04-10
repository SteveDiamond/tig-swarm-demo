import * as d3 from "d3";
import type { Panel, WSMessage } from "../types";
import { getAgentColor } from "../lib/colors";

interface AgentNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  contributions: number;
  color: string;
}

interface IdeaLink extends d3.SimulationLinkDatum<AgentNode> {
  sourceId: string;
  targetId: string;
}

export class IdeaFlowPanel implements Panel {
  private svg!: any;
  private linkGroup!: any;
  private nodeGroup!: any;
  private labelGroup!: any;
  private simulation!: d3.Simulation<AgentNode, IdeaLink>;
  private nodes: AgentNode[] = [];
  private links: IdeaLink[] = [];
  private width = 0;
  private height = 0;

  // Track hypothesis ownership for building links
  private hypothesisOwner = new Map<string, string>(); // hypothesis_id -> agent_id

  init(container: HTMLElement) {
    container.innerHTML = `
      <div class="panel-inner ideaflow-panel">
        <div class="panel-label">IDEA FLOW</div>
        <svg id="ideaflow-svg"></svg>
      </div>
    `;

    const svgEl = document.getElementById("ideaflow-svg")!;
    const rect = svgEl.parentElement!.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height - 24;

    this.svg = d3.select("#ideaflow-svg")
      .attr("width", this.width)
      .attr("height", this.height);

    // Arrow marker
    const defs = this.svg.append("defs");
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -3 6 6")
      .attr("refX", 14)
      .attr("refY", 0)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-3L6,0L0,3")
      .attr("fill", "#3d4a5c");

    // Glow filter
    const glowFilter = defs.append("filter").attr("id", "node-glow");
    glowFilter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = glowFilter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    this.linkGroup = this.svg.append("g");
    this.nodeGroup = this.svg.append("g");
    this.labelGroup = this.svg.append("g");

    this.simulation = d3.forceSimulation<AgentNode, IdeaLink>()
      .force("link", d3.forceLink<AgentNode, IdeaLink>().id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-80))
      .force("center", d3.forceCenter(this.width / 2, this.height / 2))
      .force("collision", d3.forceCollide().radius(20))
      .alphaDecay(0.008)
      .on("tick", () => this.tick());

    // Handle resize
    const observer = new ResizeObserver(() => {
      const newRect = svgEl.parentElement!.getBoundingClientRect();
      this.width = newRect.width;
      this.height = newRect.height - 24;
      this.svg.attr("width", this.width).attr("height", this.height);
      (this.simulation.force("center") as d3.ForceCenter<AgentNode>)
        .x(this.width / 2).y(this.height / 2);
    });
    observer.observe(svgEl.parentElement!);
  }

  handleMessage(msg: WSMessage) {
    if (msg.type === "reset") {
      this.nodes = [];
      this.links = [];
      this.hypothesisOwner.clear();
      this.linkGroup.selectAll("*").remove();
      this.nodeGroup.selectAll("*").remove();
      this.labelGroup.selectAll("*").remove();
      this.simulation.nodes([]);
      (this.simulation.force("link") as d3.ForceLink<AgentNode, IdeaLink>).links([]);
      this.simulation.alpha(0).stop();
      return;
    }

    if (msg.type === "agent_joined") {
      this.addAgent(msg.agent_id, msg.agent_name);
    }

    if (msg.type === "hypothesis_proposed") {
      // Track ownership
      this.hypothesisOwner.set(msg.hypothesis_id, msg.agent_id);

      // Increment contribution count
      const node = this.nodes.find((n) => n.id === msg.agent_id);
      if (node) node.contributions++;

      // If this hypothesis has a parent, create a link
      if (msg.parent_hypothesis_id) {
        const parentAgentId = this.hypothesisOwner.get(msg.parent_hypothesis_id);
        if (parentAgentId && parentAgentId !== msg.agent_id) {
          this.addLink(parentAgentId, msg.agent_id);
        }
      }

      this.updateGraph();
    }

    if (msg.type === "experiment_published") {
      const node = this.nodes.find((n) => n.id === msg.agent_id);
      if (node) node.contributions++;
      this.updateGraph();
    }
  }

  private addAgent(id: string, name: string) {
    if (this.nodes.find((n) => n.id === id)) return;
    this.nodes.push({
      id,
      name,
      contributions: 0,
      color: getAgentColor(id),
      x: this.width / 2 + (Math.random() - 0.5) * 100,
      y: this.height / 2 + (Math.random() - 0.5) * 100,
    });
    this.updateGraph();
  }

  private addLink(sourceId: string, targetId: string) {
    // Don't duplicate
    const exists = this.links.some(
      (l) => l.sourceId === sourceId && l.targetId === targetId,
    );
    if (exists) return;

    this.links.push({ source: sourceId, target: targetId, sourceId, targetId });
  }

  private updateGraph() {
    this.simulation.nodes(this.nodes);
    (this.simulation.force("link") as d3.ForceLink<AgentNode, IdeaLink>)
      .links(this.links);
    this.simulation.alpha(0.3).restart();

    // Links
    const linkSel = this.linkGroup.selectAll("line")
      .data(this.links, (d: any) => `${d.sourceId}-${d.targetId}`);

    linkSel.enter()
      .append("line")
      .attr("stroke", "#3d4a5c")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0)
      .attr("marker-end", "url(#arrowhead)")
      .transition()
      .duration(600)
      .attr("stroke-opacity", 0.4);

    linkSel.exit().remove();

    // Nodes
    const nodeSel = this.nodeGroup.selectAll("circle")
      .data(this.nodes, (d) => d.id);

    nodeSel.enter()
      .append("circle")
      .attr("r", 0)
      .attr("fill", (d) => d.color)
      .attr("opacity", 0.8)
      .attr("filter", "url(#node-glow)")
      .transition()
      .duration(400)
      .attr("r", (d) => Math.max(6, Math.min(16, 6 + d.contributions * 1.5)));

    nodeSel
      .transition()
      .duration(300)
      .attr("r", (d) => Math.max(6, Math.min(16, 6 + d.contributions * 1.5)));

    nodeSel.exit().remove();

    // Labels
    const labelSel = this.labelGroup.selectAll("text")
      .data(this.nodes, (d) => d.id);

    labelSel.enter()
      .append("text")
      .attr("fill", "#7a869a")
      .attr("font-size", "8px")
      .attr("font-family", "var(--mono)")
      .attr("text-anchor", "middle")
      .attr("dy", 18)
      .text((d) => d.name)
      .attr("opacity", 0)
      .transition()
      .duration(400)
      .attr("opacity", 0.7);

    labelSel.exit().remove();
  }

  private tick() {
    this.linkGroup.selectAll("line")
      .attr("x1", (d: any) => d.source.x)
      .attr("y1", (d: any) => d.source.y)
      .attr("x2", (d: any) => d.target.x)
      .attr("y2", (d: any) => d.target.y);

    this.nodeGroup.selectAll("circle")
      .attr("cx", (d) => d.x!)
      .attr("cy", (d) => d.y!);

    this.labelGroup.selectAll("text")
      .attr("x", (d) => d.x!)
      .attr("y", (d) => d.y!);
  }
}
