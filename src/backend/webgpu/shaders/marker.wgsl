// G3 device-execution marker — proves WGSL actually ran.
//
// G4C: mark_stage takes a STAGE MASK (not one bit): the driver accumulates
// stage bits on CPU and flushes one dispatch per evaluation. Mask semantics
// preserve the proof (bits set iff the stage dispatched since the epoch);
// [3] counts mark executions (coalesced flushes, not stages).
//
//   actual WGSL dispatch  (magic set, stage bits accumulate, count grows)
//        vs
//   JS FP32 mirror       (buffer stays zeroed)
//
// Layout (16 B, execMarker buffer):
//   [0] magic    : 0xC10A57 once any mark dispatch has executed
//   [1] epoch    : evaluation id, written by CPU at eval start (control path)
//   [2] stageMask: bit i set iff stage i dispatched since the epoch write
//   [3] dispatches: total mark_stage executions since buffer creation
//
// Stage ids (must match gpu-executor.ts STAGE_IDS):
//   0 predictor, 1 aabb, 2 morton, 3 sort, 4 lbvh, 5 traverse,
//   6 closestVT, 7 closestEE, 8 ccdVT, 9 ccdEE, 10 expand, 11 compact,
//   12 floor, 13 membrane, 14 bending, 15 assemble, 16 barrier, 17 friction,
//   18 newtonRHS, 19 jacobi, 20 hvp, 21 pcg, 22 applyStep, 23 diagnostics.

@group(0) @binding(0) var<storage, read_write> marker : array<atomic<u32>>;
@group(0) @binding(1) var<uniform> stageMask : u32;

@compute @workgroup_size(1)
fn mark_stage() {
  atomicStore(&marker[0], 0xC10A57u);
  atomicOr(&marker[2], stageMask);
  atomicAdd(&marker[3], 1u);
}
