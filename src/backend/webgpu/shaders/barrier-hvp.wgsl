// G0 barrier HVP (frozen projector, analytic — matches ContactSystem.applyHvp):
//   b''(d) = -(2L + 4u/d - u^2/d^2), coeff = kappa * max(b''(d), 0)
//   out += coeff * (J v) * J^T   with J = frozen distance projector.
// Pass 1 (this file, per-contact): jv[i] = J_i . v, coeff[i].
// Pass 2 reuses the assemble pattern: per-vertex loop over contacts
// accumulates coeff[i]*jv[i]*J_i^T (atomic-free reference; colored in G2).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> direction : array<f32>;
@group(0) @binding(2) var<storage, read> contactW : array<vec4f>;
@group(0) @binding(3) var<storage, read> contactN : array<vec4f>;
@group(0) @binding(4) var<storage, read> contactId : array<vec4u>;
@group(0) @binding(5) var<storage, read> contactPrm : array<vec4f>;
@group(0) @binding(6) var<storage, read> contactDistNow : array<f32>;
@group(0) @binding(7) var<storage, read_write> jvOut : array<f32>;
@group(0) @binding(8) var<storage, read_write> coeffOut : array<f32>;
// pass-2 / diag bindings (own entries below; pruned from project layout)
@group(0) @binding(9) var<storage, read_write> hpBarrier : array<f32>; // n*3
@group(0) @binding(10) var<storage, read_write> contactDiag : array<f32>; // n*3

fn dirDotN(v : u32, n : vec3f) -> f32 {
  return n.x * direction[v * 3u] + n.y * direction[v * 3u + 1u] + n.z * direction[v * 3u + 2u];
}

@compute @workgroup_size(64)
fn barrier_hvp_project(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  let w = contactW[i];
  let nv = contactN[i];
  let n = nv.xyz;
  let kind = nv.w;
  let id = contactId[i];
  let dHat = contactPrm[i].x;
  let kappa = contactPrm[i].y;
  let d = max(contactDistNow[i], 1e-12);
  if (d >= dHat) { jvOut[i] = 0.0; coeffOut[i] = 0.0; return; }
  let L = log(d / dHat);
  let u = d - dHat;
  let b2 = -(2.0 * L + 4.0 * u / d - u * u / (d * d));
  let coeff = kappa * max(b2, 0.0);
  var jv = 0.0;
  if (kind > 1.5) {
    jv = direction[id.x * 3u + 1u];
  } else if (kind < 0.5) {
    jv = dirDotN(id.x, n);
    if (id.y < params.vertexCount) { jv -= w.x * dirDotN(id.y, n); }
    if (id.z < params.vertexCount) { jv -= w.y * dirDotN(id.z, n); }
    if (id.w < params.vertexCount) { jv -= w.z * dirDotN(id.w, n); }
  } else {
    if (id.x < params.vertexCount) { jv += w.x * dirDotN(id.x, n); }
    if (id.y < params.vertexCount) { jv += w.y * dirDotN(id.y, n); }
    if (id.z < params.vertexCount) { jv += w.z * dirDotN(id.z, n); }
    if (id.w < params.vertexCount) { jv += w.w * dirDotN(id.w, n); }
  }
  jvOut[i] = jv;
  coeffOut[i] = coeff;
}

// Pass 2: per-vertex O(n*C) atomic-free accumulation of
//   hpBarrier += coeff[i] * jv[i] * J_i^T
// with the frozen projector J (ports mirror distJvp + addDistGrad). Runs
// after barrier_hvp_project in the same encoder (pass boundary orders them).
@compute @workgroup_size(64)
fn assemble_barrier_hvp(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var hx = 0.0; var hy = 0.0; var hz = 0.0;
  for (var i = 0u; i < params.contactCount; i++) {
    let s = coeffOut[i] * jvOut[i];
    if (s == 0.0) { continue; }
    let nv = contactN[i];
    let n = nv.xyz;
    let kind = nv.w;
    let id = contactId[i];
    let w = contactW[i];
    if (kind > 1.5) {
      if (id.x == v) { hy += s; } // J^T = +y for floor
    } else if (kind < 0.5) {
      if (id.x == v) { hx += s * n.x; hy += s * n.y; hz += s * n.z; }
      if (id.y == v && id.y < params.vertexCount) { hx += -s * w.x * n.x; hy += -s * w.x * n.y; hz += -s * w.x * n.z; }
      if (id.z == v && id.z < params.vertexCount) { hx += -s * w.y * n.x; hy += -s * w.y * n.y; hz += -s * w.y * n.z; }
      if (id.w == v && id.w < params.vertexCount) { hx += -s * w.z * n.x; hy += -s * w.z * n.y; hz += -s * w.z * n.z; }
    } else {
      if (id.x == v && id.x < params.vertexCount) { hx += s * w.x * n.x; hy += s * w.x * n.y; hz += s * w.x * n.z; }
      if (id.y == v && id.y < params.vertexCount) { hx += s * w.y * n.x; hy += s * w.y * n.y; hz += s * w.y * n.z; }
      if (id.z == v && id.z < params.vertexCount) { hx += s * w.z * n.x; hy += s * w.z * n.y; hz += s * w.z * n.z; }
      if (id.w == v && id.w < params.vertexCount) { hx += s * w.w * n.x; hy += s * w.w * n.y; hz += s * w.w * n.z; }
    }
  }
  hpBarrier[v * 3u] = hx;
  hpBarrier[v * 3u + 1u] = hy;
  hpBarrier[v * 3u + 2u] = hz;
}

// Jacobi contact-curvature diagonal: diag += kappa*max(b''(d),0) projected
// per stencil weight (ports mirror addDiagEstimate / CPU addDiagEstimate).
// VT: k*n^2 on p, k*w^2*n^2 on tri; EE: k*w^2*n^2; floor: k on y.
@compute @workgroup_size(64)
fn contact_diag(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var dx = 0.0; var dy = 0.0; var dz = 0.0;
  for (var i = 0u; i < params.contactCount; i++) {
    let d = max(contactDistNow[i], 1e-12);
    let dHat = contactPrm[i].x;
    if (d >= dHat) { continue; }
    let kappa = contactPrm[i].y;
    let L = log(d / dHat);
    let u = d - dHat;
    let b2 = -(2.0 * L + 4.0 * u / d - u * u / (d * d));
    let k = kappa * max(b2, 0.0);
    if (k == 0.0) { continue; }
    let nv = contactN[i];
    let n2x = nv.x * nv.x; let n2y = nv.y * nv.y; let n2z = nv.z * nv.z;
    let kind = nv.w;
    let id = contactId[i];
    let w = contactW[i];
    if (kind > 1.5) {
      if (id.x == v) { dy += k; }
    } else if (kind < 0.5) {
      if (id.x == v) { dx += k * n2x; dy += k * n2y; dz += k * n2z; }
      if (id.y == v && id.y < params.vertexCount) { dx += k * w.x * w.x * n2x; dy += k * w.x * w.x * n2y; dz += k * w.x * w.x * n2z; }
      if (id.z == v && id.z < params.vertexCount) { dx += k * w.y * w.y * n2x; dy += k * w.y * w.y * n2y; dz += k * w.y * w.y * n2z; }
      if (id.w == v && id.w < params.vertexCount) { dx += k * w.z * w.z * n2x; dy += k * w.z * w.z * n2y; dz += k * w.z * w.z * n2z; }
    } else {
      if (id.x == v && id.x < params.vertexCount) { dx += k * w.x * w.x * n2x; dy += k * w.x * w.x * n2y; dz += k * w.x * w.x * n2z; }
      if (id.y == v && id.y < params.vertexCount) { dx += k * w.y * w.y * n2x; dy += k * w.y * w.y * n2y; dz += k * w.y * w.y * n2z; }
      if (id.z == v && id.z < params.vertexCount) { dx += k * w.z * w.z * n2x; dy += k * w.z * w.z * n2y; dz += k * w.z * w.z * n2z; }
      if (id.w == v && id.w < params.vertexCount) { dx += k * w.w * w.w * n2x; dy += k * w.w * w.w * n2y; dz += k * w.w * w.w * n2z; }
    }
  }
  contactDiag[v * 3u] = dx;
  contactDiag[v * 3u + 1u] = dy;
  contactDiag[v * 3u + 2u] = dz;
}
