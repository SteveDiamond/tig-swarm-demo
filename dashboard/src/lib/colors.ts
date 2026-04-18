const PALETTE = [
  "#00e5ff", "#ff6d00", "#00e676", "#e040fb", "#ffea00",
  "#ff5252", "#40c4ff", "#69f0ae", "#ff80ab", "#ffd740",
  "#b388ff", "#00bfa5", "#ffab00", "#18ffff", "#ff9100",
  "#76ff03", "#ff4081", "#3d5afe", "#d500f9", "#ff3d00",
  "#1de9b6", "#c6ff00", "#ff1744", "#7c4dff",
];

export const ROUTE_COLORS = PALETTE.slice(0, 10);

const agentColorMap = new Map<string, string>();

// Deterministic color from the agent key. Using a stable hash instead of
// first-come-first-served means every panel resolves the same agent to the
// same palette slot regardless of which one rendered first, which kept the
// leaderboard dot, chart step, and diversity grid out of sync when they
// populated in different orders.
export function getAgentColor(agentId: string): string {
  const cached = agentColorMap.get(agentId);
  if (cached) return cached;
  // FNV-1a 32-bit
  let h = 0x811c9dc5;
  for (let i = 0; i < agentId.length; i++) {
    h ^= agentId.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) | 0;
  }
  const color = PALETTE[Math.abs(h) % PALETTE.length];
  agentColorMap.set(agentId, color);
  return color;
}

export function getRouteColor(vehicleIndex: number): string {
  return ROUTE_COLORS[vehicleIndex % ROUTE_COLORS.length];
}
