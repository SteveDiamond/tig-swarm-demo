import type { WSMessage, RouteData, AllRouteData, LeaderboardEntry } from "./types";

const ADJECTIVES = [
  "swift", "bold", "keen", "bright", "sharp", "vivid", "fierce", "noble",
  "agile", "lucid", "cosmic", "astral", "quantum", "neural", "radiant",
  "golden", "silver", "blazing", "frozen", "obsidian",
];
const NOUNS = [
  "falcon", "wolf", "hawk", "lynx", "otter", "raven", "fox", "crane",
  "tiger", "eagle", "puma", "phoenix", "hydra", "nova", "pulse",
  "spark", "orbit", "prism", "nexus", "helix",
];

const HYPOTHESIS_TITLES = [
  "Apply 2-opt local search after construction",
  "Use nearest-neighbor insertion heuristic",
  "Implement or-opt move operator",
  "Try simulated annealing with adaptive cooling",
  "Apply savings algorithm (Clarke-Wright)",
  "Use sweep algorithm for initial routes",
  "Implement tabu search with short-term memory",
  "Try cross-exchange between routes",
  "Apply time-window relaxation then repair",
  "Decompose by geographic clusters",
  "Implement ALNS with destroy-repair operators",
  "Use spatial indexing for nearest lookups",
  "Try genetic algorithm with route crossover",
  "Apply ejection chain improvements",
  "Use constraint propagation for feasibility",
  "Implement relocate operator within routes",
  "Try large neighborhood search",
  "Apply greedy randomized construction",
  "Use regret-based insertion",
  "Implement record-to-record travel",
];

const STRATEGY_TAGS = [
  "construction", "local_search", "metaheuristic",
  "constraint_relaxation", "decomposition", "hybrid", "data_structure",
];

type Handler = (msg: WSMessage) => void;

interface MockAgent {
  id: string;
  name: string;
  bestScore: number;
  experiments: number;
  improvements: number;
  scoreSum: number;
}

export class MockDataGenerator {
  private handlers: Handler[] = [];
  private agents: MockAgent[] = [];
  // bestScore / baseline are set from the first emitted experiment — there is
  // no reference point before anything has been run.
  private bestScore: number | null = null;
  private baseline: number | null = null;
  private totalExperiments = 0;
  private totalHypotheses = 0;
  private hypIndex = 0;

  onMessage(handler: Handler) {
    this.handlers.push(handler);
  }

  private emit(msg: WSMessage) {
    this.handlers.forEach((h) => h(msg));
  }

  private now(): string {
    return new Date().toISOString();
  }

  private randomAgent(): MockAgent {
    return this.agents[Math.floor(Math.random() * this.agents.length)];
  }

  start() {
    // Register agents gradually
    let agentCount = 0;
    const registerInterval = setInterval(() => {
      if (agentCount >= 15) {
        clearInterval(registerInterval);
        return;
      }
      const adj = ADJECTIVES[agentCount % ADJECTIVES.length];
      const noun = NOUNS[agentCount % NOUNS.length];
      const agent: MockAgent = {
        id: `mock-${agentCount}`,
        name: `${adj}-${noun}`,
        bestScore: Infinity,
        experiments: 0,
        improvements: 0,
        scoreSum: 0,
      };
      this.agents.push(agent);
      agentCount++;

      this.emit({
        type: "agent_joined",
        agent_id: agent.id,
        agent_name: agent.name,
        timestamp: this.now(),
      });

      this.emitStats();
    }, randomBetween(2000, 5000));

    // Hypotheses
    setInterval(() => {
      if (this.agents.length === 0) return;
      const agent = this.randomAgent();
      const title = HYPOTHESIS_TITLES[this.hypIndex % HYPOTHESIS_TITLES.length];
      this.hypIndex++;
      this.totalHypotheses++;

      this.emit({
        type: "hypothesis_proposed",
        hypothesis_id: `hyp-${this.totalHypotheses}`,
        agent_name: agent.name,
        agent_id: agent.id,
        title,
        description: `Testing optimization approach: ${title}`,
        strategy_tag: STRATEGY_TAGS[Math.floor(Math.random() * STRATEGY_TAGS.length)],
        parent_hypothesis_id: this.totalHypotheses > 3 && Math.random() > 0.5
          ? `hyp-${Math.floor(Math.random() * this.totalHypotheses)}`
          : null,
        timestamp: this.now(),
      });

      this.emitStats();
    }, randomBetween(4000, 8000));

    // Experiments
    setInterval(() => {
      if (this.agents.length === 0) return;
      const agent = this.randomAgent();
      this.totalExperiments++;
      agent.experiments++;

      // First experiment seeds both baseline and bestScore — everything after
      // is measured against it.
      let score: number;
      if (this.baseline === null) {
        score = randomBetween(1800, 2000);
        this.baseline = score;
        this.bestScore = score;
      } else {
        const improvement = Math.random() > 0.3;
        const delta = improvement
          ? randomBetween(5, 80)
          : -randomBetween(10, 100);
        score = Math.max(800, this.bestScore! + delta);
      }
      const isNewBest = score < this.bestScore!;
      const prevBestForBroadcast = this.bestScore!;

      if (isNewBest) {
        this.bestScore = score;
        agent.improvements++;
      }
      if (score < agent.bestScore) {
        agent.bestScore = score;
      }
      agent.scoreSum += score;

      // Semantic % improvement vs prev best: positive = score dropped.
      const deltaVsBest =
        prevBestForBroadcast > 0 && prevBestForBroadcast !== score
          ? Number(
              (((prevBestForBroadcast - score) / prevBestForBroadcast) * 100).toFixed(6),
            )
          : null;
      const impPct = Number(
        (((this.baseline - score) / this.baseline) * 100).toFixed(2),
      );
      this.emit({
        type: "experiment_published",
        experiment_id: `exp-${this.totalExperiments}`,
        agent_name: agent.name,
        agent_id: agent.id,
        score,
        feasible: Math.random() > 0.1,
        improvement_pct: impPct,
        delta_vs_best_pct: deltaVsBest,
        num_instances: 8,
        is_new_best: isNewBest,
        hypothesis_id: `hyp-${Math.floor(Math.random() * Math.max(1, this.totalHypotheses))}`,
        notes: this.totalExperiments === 1 ? "Baseline established" : (score < prevBestForBroadcast ? "Improved routing efficiency" : "Score regressed"),
        timestamp: this.now(),
      });

      if (isNewBest) {
        this.emit({
          type: "new_global_best",
          experiment_id: `exp-${this.totalExperiments}`,
          agent_name: agent.name,
          agent_id: agent.id,
          score,
          improvement_pct: impPct,
          incremental_improvement_pct:
            prevBestForBroadcast > 0 && prevBestForBroadcast !== score
              ? Number((((prevBestForBroadcast - score) / prevBestForBroadcast) * 100).toFixed(2))
              : null,
          num_instances: 8,
          route_data: generateMockRoutes(),
          timestamp: this.now(),
        });
      }

      // Emit leaderboard
      const entries: LeaderboardEntry[] = this.agents
        .map((a) => ({
          agent: a,
          avg: a.experiments > 0 ? a.scoreSum / a.experiments : null,
        }))
        .sort((a, b) => {
          if (a.avg === null && b.avg === null) return 0;
          if (a.avg === null) return 1;
          if (b.avg === null) return -1;
          return a.avg - b.avg;
        })
        .map(({ agent: a, avg }, i) => ({
          rank: i + 1,
          agent_id: a.id,
          agent_name: a.name,
          runs: a.experiments,
          improvements: a.improvements,
          runs_since_improvement: 0,
          best_score: avg,
        }));

      this.emit({
        type: "leaderboard_update",
        entries,
        timestamp: this.now(),
      });

      this.emitStats();
    }, randomBetween(3000, 7000));
  }

  private emitStats() {
    const impPct =
      this.baseline !== null && this.bestScore !== null
        ? Number((((this.baseline - this.bestScore) / this.baseline) * 100).toFixed(2))
        : 0;
    this.emit({
      type: "stats_update",
      active_agents: this.agents.length,
      total_experiments: this.totalExperiments,
      hypotheses_count: this.totalHypotheses,
      best_score: this.bestScore,
      baseline_score: this.baseline,
      num_instances: 8,
      improvement_pct: impPct,
      timestamp: this.now(),
    });
  }
}

function randomBetween(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generateMockRoutes(): AllRouteData {
  const instances: AllRouteData = {};
  for (let i = 1; i <= 8; i++) {
    instances[`RC1_2_${i}.txt`] = generateMockInstance();
  }
  return instances;
}

function generateMockInstance(): RouteData {
  const depot = { x: 50, y: 50 };
  const numVehicles = randomBetween(5, 10);
  const routes: Array<{vehicle_id: number; path: Array<{x: number; y: number; customer_id: number}>}> = [];

  for (let v = 0; v < numVehicles; v++) {
    const numCustomers = randomBetween(3, 8);
    const angle = (v / numVehicles) * Math.PI * 2;
    const path: Array<{x: number; y: number; customer_id: number}> = [];

    for (let c = 0; c < numCustomers; c++) {
      const r = 15 + Math.random() * 30;
      const a = angle + ((c - numCustomers / 2) * 0.3);
      path.push({
        x: depot.x + Math.cos(a) * r + (Math.random() - 0.5) * 10,
        y: depot.y + Math.sin(a) * r + (Math.random() - 0.5) * 10,
        customer_id: v * 10 + c,
      });
    }
    routes.push({ vehicle_id: v, path });
  }

  return { depot, routes };
}
