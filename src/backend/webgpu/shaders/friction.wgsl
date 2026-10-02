// G0 friction — lagged Coulomb, residual only (no Hessian contribution).
// Ports friction.ts coulombForce + ContactSystem.relativeSlip/addFrictionForce.
// Reads the SAME frozen contact set as barrier-gradient.wgsl plus:
//   slipFromStepStart (n*3, precomputed displacement x - xStep on GPU), and
//   laggedNormal (cap*4: nx,ny,nz,lambdaN from last accepted iterate).
// Writes frictionForce (n*3). assemble-gradient adds it into the total.
// Invariants preserved: dot(f_t, n) ~= 0, |f_t| <= mu*|lambdaN|.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> contactW : array<vec4f>;
@group(0) @binding(2) var<storage, read> contactN : array<vec4f>;
@group(0) @binding(3) var<storage, read> contactId : array<vec4u>;
@group(0) @binding(4) var<storage, read> laggedN : array<vec4f>; // nx,ny,nz,lambdaN
@group(0) @binding(5) var<storage, read> slip : array<vec4f>;    // per-vertex x - xStep (vec4 padded)
@group(0) @binding(6) var<storage, read_write> frictionScratch : array<vec4f>; // cap: (fx,fy,fz,activeFlag)
@group(0) @binding(7) var<storage, read> contactPrmF : array<vec4f>; // (dHat,kappa,*,*) — G6B inactivity guard

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  // G6B: cleared (never-appended) slots carry prm.x == 0 and must stay inert
  // under cap-bounded batch loops — otherwise stale laggedN + zero stencil
  // would forge phantom friction. Live records always carry dHat > 0, so
  // this guard never fires on live data (behavior-preserving by construction).
  if (contactPrmF[i].x <= 0.0) {
    frictionScratch[i] = vec4f(0.0, 0.0, 0.0, 0.0);
    return;
  }
  let w = contactW[i];
  let kind = contactN[i].w;
  let id = contactId[i];
  let lag = laggedN[i];
  let n = lag.xyz;
  let lambdaN = abs(lag.w);
  let mu = params.frictionMu;
  // relative slip of the frozen stencil from step start
  var u : vec3f;
  if (kind > 1.5) {
    u = slip[id.x].xyz;
  } else if (kind < 0.5) {
    u = slip[id.x].xyz - (w.x * slip[id.y].xyz + w.y * slip[id.z].xyz + w.z * slip[id.w].xyz);
  } else {
    u = w.x * slip[id.x].xyz + w.y * slip[id.y].xyz + w.z * slip[id.z].xyz + w.w * slip[id.w].xyz;
  }
  let un = dot(u, n);
  let ut = u - un * n;
  let utLen = length(ut);
  let eps : f32 = 1e-9;
  // smooth capped Coulomb: f = -mu*lambdaN * ut / sqrt(|ut|^2 + eps^2)
  let denom = sqrt(utLen * utLen + eps * eps);
  let mag = mu * lambdaN * utLen / denom;
  let f = -mag * select(vec3f(0.0), ut / max(utLen, 1e-30), utLen > 1e-30);
  // invariant: tangential by construction; cap by mu*lambdaN by construction.
  frictionScratch[i] = vec4f(f, 1.0);
}
