// G0 PCG vector passes — portable workgroup reductions, NO subgroups (§16).
// Three kernels share one file for review convenience; pipelines.ts creates
// one pipeline per entry point.
//
// pcg_init: r = b - A*x0 (x0 = 0 on first Newton iter -> r = b),
//           z = r / diagSafe, p = z, rz = dot(r,z).
// pcg_update_xr: x += alpha*p, r -= alpha*Ap (alpha from CPU status reduction).
// pcg_update_p: z = r/diag, beta = rzNew/rz, p = z + beta*p.
//
// Dot products use a two-stage reduction: per-workgroup partial sums into
// reduceScratch, then a second dispatch over workgroup results. This file
// writes per-thread products; pcg-reduce.wgsl finishes the sum.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read> diag : array<f32>;
@group(0) @binding(3) var<storage, read_write> x : array<f32>;
@group(0) @binding(4) var<storage, read_write> r : array<f32>;
@group(0) @binding(5) var<storage, read_write> z : array<f32>;
@group(0) @binding(6) var<storage, read_write> p : array<f32>;
@group(0) @binding(7) var<storage, read> pinMask : array<u32>;
@group(0) @binding(8) var<storage, read_write> prod : array<f32>; // per-element r*z products for reduction

@compute @workgroup_size(64)
fn pcg_init(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let pinned = pinMask[i / 3u];
  x[i] = 0.0;
  var ri = b[i];
  if (pinned != 0u) { ri = 0.0; }
  r[i] = ri;
  var zi = ri;
  if (diag[i] > 1e-12) { zi = ri / diag[i]; }
  if (pinned != 0u) { zi = 0.0; }
  z[i] = zi;
  p[i] = zi;
  prod[i] = ri * zi;
}

@compute @workgroup_size(64)
fn pcg_apply_alpha(
  @builtin(global_invocation_id) gid : vec3u,
) {
  // alpha is broadcast via uniform below; separate binding keeps G0 simple.
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
}

// G3: alpha/beta arrive via STORAGE scalar slots (written by sdiv), so the
// whole PCG solve stays in one encoder with zero scalar readback mid-solve.
// (G0 used alphaU/betaU uniforms + CPU sync per iteration — superseded.)

@group(1) @binding(0) var<storage, read> alphaSlot : array<f32>;
@group(1) @binding(1) var<storage, read> Ap : array<f32>;

@compute @workgroup_size(64)
fn pcg_update_xr(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let pinned = pinMask[i / 3u];
  if (pinned != 0u) { return; }
  let alpha = alphaSlot[0];
  x[i] += alpha * p[i];
  r[i] -= alpha * Ap[i];
}

@group(1) @binding(2) var<storage, read> betaSlot : array<f32>;

// G3 split updates (device-side beta sequencing; the fused pcg_update_p
// above is retained for reference but unused by the device PCG loop):
//   pcg_update_z  : z = M^-1 r (filtered), prod = r*z   (no beta needed)
//   pcg_update_p2 : p = z + beta*p (filtered)
@compute @workgroup_size(64)
fn pcg_update_z(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let pinned = pinMask[i / 3u];
  var zi = r[i];
  if (diag[i] > 1e-12) { zi = r[i] / diag[i]; }
  if (pinned != 0u) { zi = 0.0; }
  z[i] = zi;
  prod[i] = r[i] * zi;
}

@compute @workgroup_size(64)
fn pcg_update_p2(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let pinned = pinMask[i / 3u];
  if (pinned != 0u) { p[i] = 0.0; return; }
  p[i] = z[i] + betaSlot[0] * p[i];
}

@compute @workgroup_size(64)
fn pcg_update_p(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let pinned = pinMask[i / 3u];
  var zi = r[i];
  if (diag[i] > 1e-12) { zi = r[i] / diag[i]; }
  if (pinned != 0u) { zi = 0.0; }
  z[i] = zi;
  p[i] = zi + betaSlot[0] * p[i];
  if (pinned != 0u) { p[i] = 0.0; }
  prod[i] = r[i] * zi;
}

// G5B block-Jacobi variants: z = blockInv[v] * r[v] (fallback pre-baked as a
// diagonal matrix, so no branch; pins filtered as usual).
@group(0) @binding(9) var<storage, read> blockInv : array<f32>; // n*9 row-major

@compute @workgroup_size(64)
fn pcg_init_bj(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let v = i / 3u;
  let a = i % 3u;
  let pinned = pinMask[v];
  x[i] = 0.0;
  var ri = b[i];
  if (pinned != 0u) { ri = 0.0; }
  r[i] = ri;
  let base = v * 9u + a * 3u;
  var zi = blockInv[base] * r[v * 3u] + blockInv[base + 1u] * r[v * 3u + 1u] + blockInv[base + 2u] * r[v * 3u + 2u];
  if (pinned != 0u) { zi = 0.0; }
  z[i] = zi;
  p[i] = zi;
  prod[i] = ri * zi;
}

@compute @workgroup_size(64)
fn pcg_update_z_bj(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let v = i / 3u;
  let a = i % 3u;
  let pinned = pinMask[v];
  let base = v * 9u + a * 3u;
  var zi = blockInv[base] * r[v * 3u] + blockInv[base + 1u] * r[v * 3u + 1u] + blockInv[base + 2u] * r[v * 3u + 2u];
  if (pinned != 0u) { zi = 0.0; }
  z[i] = zi;
  prod[i] = r[i] * zi;
}
