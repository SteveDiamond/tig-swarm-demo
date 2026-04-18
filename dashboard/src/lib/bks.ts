// Average of the literature Best Known Solution (BKS) upper bounds across the
// 24 Homberger-Gehring 400-customer VRPTW instances used as benchmarks here.
// Source: tig-swarm-demo/datasets/vehicle_routing/bks.json, which is derived
// from tig-challenges/datasets/vehicle_routing/test/bks.csv (15 of 24 are
// Opt=yes — proven optima; the remaining 9 are best-known upper bounds).
// The swarm's reported score is already a per-instance average, so it is
// directly comparable to this value.
export const BKS_AVERAGE = 6679.775;
export const BKS_INSTANCE_COUNT = 24;

// Returns (score - BKS_AVERAGE) / BKS_AVERAGE * 100. Positive = gap above BKS.
export function bksGapPct(score: number): number {
  return ((score - BKS_AVERAGE) / BKS_AVERAGE) * 100;
}
