// G0 HVP: y = H(x) * v WITHOUT assembling a global sparse matrix (§17).
// y = M/h^2 * v + H_membrane_FD * v + H_barrier_frozen * v
// (bending kept in the gradient only — same inexact-Newton approximation as
// newton.ts; friction contributes no Hessian.)
//
// This file implements the two ANALYTIC parts (inertia + frozen barrier).
// The membrane FD part is produced by re-invoking membrane-gradient.wgsl at
// x +/- h*p (two extra dispatches, central difference) and combining here:
//   Hp_membrane = (g(x+h p) - g(x-h p)) / (2h)
// That keeps ONE membrane formula (no second derivation to drift).
//
// This kernel: y[i] = m_i*invDt2*v[i] + HpMembrane[i] + HpBarrier[i].
// Barrier term: kappa * max(b''(d),0) * (J v) * J^T with J = frozen projector.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> direction : array<f32>;   // p (n*3)
@group(0) @binding(2) var<storage, read> mass : array<f32>;        // n
@group(0) @binding(3) var<storage, read> hpMembrane : array<f32>;  // n*3 from FD combine
@group(0) @binding(4) var<storage, read> hpBarrier : array<f32>;   // n*3 from barrier-hvp pass
@group(0) @binding(5) var<storage, read_write> out : array<f32>;   // n*3

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let vi = i / 3u;
  out[i] = mass[vi] * params.invDt2 * direction[i] + hpMembrane[i] + hpBarrier[i];
}
