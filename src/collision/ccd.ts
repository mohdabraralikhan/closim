// Continuous Collision Detection stubs (Phase 1).
// Target: conservative advancement + cubic coplanarity root solve for
// vertex-triangle and edge-edge pairs, feeding a barrier energy.

export interface CcdPair { kind: "vt" | "ee"; ids: number[]; toi: number; }

export function vertexTriangleToi(): number {
  // TODO Phase 1
  return Infinity;
}

export function edgeEdgeToi(): number {
  // TODO Phase 1
  return Infinity;
}
