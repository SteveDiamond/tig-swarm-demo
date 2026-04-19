const PALETTE = [
  "#00e5ff", "#ff6d00", "#00e676", "#e040fb", "#ffea00",
  "#ff5252", "#40c4ff", "#69f0ae", "#ff80ab", "#ffd740",
  "#b388ff", "#00bfa5", "#ffab00", "#18ffff", "#ff9100",
  "#76ff03", "#ff4081", "#3d5afe", "#d500f9", "#ff3d00",
  "#1de9b6", "#c6ff00", "#ff1744", "#7c4dff",
];

export const ROUTE_COLORS = PALETTE.slice(0, 10);

const agentColorMap = new Map<string, string>();

// Agent → palette color. Cached in a module-level Map so every panel on the
// page resolves the same agent to the same slot regardless of render order —
// that's what keeps the leaderboard dot, chart step, and diversity grid in
// sync once a color is picked.
//
// The agent's *preferred* slot is the FNV-1a hash of its id mod palette size.
// This preserves stability across reloads in the common case. When the
// preferred slot is already claimed by a different agent, we walk forward
// through the palette and take the first free slot — so uniqueness is
// guaranteed for the first PALETTE.length agents. Beyond that the palette is
// exhausted and we accept the hashed collision.
export function getAgentColor(agentId: string): string {
  const cached = agentColorMap.get(agentId);
  if (cached) return cached;

  // FNV-1a 32-bit
  let h = 0x811c9dc5;
  for (let i = 0; i < agentId.length; i++) {
    h ^= agentId.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) | 0;
  }
  const preferred = Math.abs(h) % PALETTE.length;

  const used = new Set(agentColorMap.values());
  let color = PALETTE[preferred];
  if (used.size < PALETTE.length) {
    for (let i = 0; i < PALETTE.length; i++) {
      const slot = (preferred + i) % PALETTE.length;
      if (!used.has(PALETTE[slot])) {
        color = PALETTE[slot];
        break;
      }
    }
  }
  agentColorMap.set(agentId, color);
  return color;
}

export function getRouteColor(vehicleIndex: number): string {
  return ROUTE_COLORS[vehicleIndex % ROUTE_COLORS.length];
}
