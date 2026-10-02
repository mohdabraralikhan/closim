// G0 Jacobi diagonal: diag = M/h^2 + beta + contact curvature (§18).
// beta matches newton.ts: max(C00,C11,G)*thickness*0.1 + 1e-6.
// Contact part adds kappa*max(b''(d),0) projected on n per stencil weight —
// same formula as ContactSystem.addDiagEstimate. Clamp ONLY for safety:
//   diagSafe = max(diag, epsilon)
// Do NOT force indefinite diagonals positive and claim SPD: the solver keeps
// the CPU fallback (Jacobi-scaled steepest descent on non-descent).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> mass : array<f32>;
@group(0) @binding(2) var<storage, read> contactDiag : array<f32>; // n*3 barrier curvature add-on (0 when no contact)
@group(0) @binding(3) var<storage, read_write> diag : array<f32>;   // n*3
@group(0) @binding(4) var<uniform> beta : f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let vi = i / 3u;
  let d = mass[vi] * params.invDt2 + beta + contactDiag[i];
  diag[i] = max(d, 1e-12);
}
