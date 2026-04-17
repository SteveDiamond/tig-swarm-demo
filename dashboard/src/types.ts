export interface RoutePoint {
  x: number;
  y: number;
  customer_id: number;
}

export interface VehicleRoute {
  vehicle_id: number;
  path: RoutePoint[];
}

export interface RouteData {
  depot: { x: number; y: number };
  routes: VehicleRoute[];
}

// route_data from server: dict keyed by instance name
export type AllRouteData = Record<string, RouteData>;

export interface LeaderboardEntry {
  rank: number;
  agent_id: string;
  agent_name: string;
  runs: number;
  improvements: number;
  runs_since_improvement: number;
  // Best per-instance score the agent has achieved across feasible runs.
  // null when the agent has no feasible runs yet.
  best_score: number | null;
}

// WebSocket message types
export type WSMessage =
  | AgentJoined
  | AgentOffline
  | HypothesisProposed
  | HypothesisStatusChanged
  | ExperimentPublished
  | NewGlobalBest
  | LeaderboardUpdate
  | StatsUpdate
  | ChatMessage
  | KnowledgeUpdated
  | AdminBroadcastMsg
  | ResetMsg;

export interface AgentJoined {
  type: "agent_joined";
  agent_id: string;
  agent_name: string;
  timestamp: string;
}

export interface AgentOffline {
  type: "agent_offline";
  agent_id: string;
  agent_name: string;
  timestamp: string;
}

export interface HypothesisProposed {
  type: "hypothesis_proposed";
  hypothesis_id: string;
  agent_name: string;
  agent_id: string;
  title: string;
  description: string;
  strategy_tag: string;
  parent_hypothesis_id: string | null;
  timestamp: string;
}

export interface HypothesisStatusChanged {
  type: "hypothesis_status_changed";
  hypothesis_id: string;
  new_status: string;
  agent_name: string;
  timestamp: string;
}

export interface ExperimentPublished {
  type: "experiment_published";
  experiment_id: string;
  agent_name: string;
  agent_id: string;
  score: number;
  feasible: boolean;
  improvement_pct: number;
  // Semantic % improvement vs the previous global best:
  // `(prev_best - score) / prev_best * 100`. Positive = score dropped
  // (improvement), negative = score rose (regression). Null when there
  // is no previous best.
  delta_vs_best_pct: number | null;
  num_instances: number;
  is_new_best: boolean;
  hypothesis_id: string | null;
  notes: string;
  timestamp: string;
}

export interface NewGlobalBest {
  type: "new_global_best";
  experiment_id: string;
  agent_name: string;
  agent_id: string;
  score: number;
  improvement_pct: number;
  // % improvement over the previous global best (null if first ever)
  incremental_improvement_pct: number | null;
  num_instances: number;
  route_data: AllRouteData | null;
  timestamp: string;
}

export interface LeaderboardUpdate {
  type: "leaderboard_update";
  entries: LeaderboardEntry[];
  timestamp: string;
}

export interface StatsUpdate {
  type: "stats_update";
  active_agents: number;
  total_experiments: number;
  hypotheses_count: number;
  // Both per-instance averages. null until the first feasible experiment
  // lands — there is no reference point before then.
  best_score: number | null;
  baseline_score: number | null;
  num_instances: number;
  improvement_pct: number;
  timestamp: string;
}

export interface ChatMessage {
  type: "chat_message";
  message_id: string;
  agent_name: string;
  agent_id: string | null;
  content: string;
  msg_type: "agent" | "synthesis" | "milestone";
  timestamp: string;
}

export interface KnowledgeUpdated {
  type: "knowledge_updated";
  content: string;
  updated_by: string;
  timestamp: string;
}

export interface AdminBroadcastMsg {
  type: "admin_broadcast";
  message: string;
  priority: "normal" | "high";
  timestamp: string;
}

export interface ResetMsg {
  type: "reset";
  timestamp: string;
}

// Panel interfaces
export interface Panel {
  init(container: HTMLElement): void;
  handleMessage(msg: WSMessage): void;
}
