export const STRATEGY_COLORS: Record<string, string> = {
  construction: "#00e5ff",
  local_search: "#00e676",
  metaheuristic: "#b388ff",
  constraint_relaxation: "#ffab00",
  decomposition: "#00bfa5",
  hybrid: "#ff80ab",
  data_structure: "#40c4ff",
  other: "#7a869a",
};

export function getStrategyColor(tag: string): string {
  return STRATEGY_COLORS[tag] || STRATEGY_COLORS.other;
}

export const STRATEGY_LABELS: Record<string, string> = {
  construction: "Construction",
  local_search: "Local Search",
  metaheuristic: "Metaheuristic",
  constraint_relaxation: "Constraint Relax.",
  decomposition: "Decomposition",
  hybrid: "Hybrid",
  data_structure: "Data Structure",
  other: "Other",
};
