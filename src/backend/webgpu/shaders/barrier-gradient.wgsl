// G0 barrier + friction (frozen-projection, residual-only friction).
// Ports ContactSystem.energyGrad term-for-term for the G0 path where the CPU
// still owns broadphase/pair generation (milestone G0) and uploads the frozen
// contact set. The GPU must NOT invent a penalty force.
//
// Per-contact record (64 B, see gpu-buffers.ts CONTACT_RECORD_BYTES):
//   w: vec4f  (VT: w0,w1,w2,unused | EE: wa,wb,wc,wd)
//   n: vec4f  (nx,ny,nz, kind: 0=VT,1=EE,2=floor)
//   id: vec4u (VT: p,a,b,c | EE: a,b,c,d | floor: p,~,~,~)
//   prm: vec4f (dHat, kappa, mu, epsFriction)
//
// One thread per contact writes:
//   contactEnergy[i], contactDist[i] (for diagnostics/min-distance)
//   per-contact force scratch contactScratch[i*8] (distributed by reduce below)
// Reduction to vertex forces uses the same O(n*C) reference scan as the FEM
// assemble pass — deterministic, atomic-free, replaced by coloring in G1/G2.
//
// Barrier (matches barrier.ts): B(d) = -(d-dHat)^2 * log(d/dHat), E = kappa*B.
// Friction (matches friction.ts lagged Coulomb): tangential force capped at
// mu*|lambdaN|, strictly dissipative, residual only.

struct ContactW { w : vec4f }
struct ContactN { n : vec4f }
struct ContactId { id : vec4u }
struct ContactPrm { prm : vec4f }

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> position : array<vec4f>;
@group(0) @binding(2) var<storage, read> contactW : array<vec4f>;
@group(0) @binding(3) var<storage, read> contactN : array<vec4f>;
@group(0) @binding(4) var<storage, read> contactId : array<vec4u>;
@group(0) @binding(5) var<storage, read> contactPrm : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> contactScratch : array<f32>; // cap*8: (vA,fx,fy,fz, vB,fx2,..) simplified: force per vertex-pair below
@group(0) @binding(7) var<storage, read_write> contactEnergy : array<f32>;
@group(0) @binding(8) var<storage, read_write> contactDist : array<f32>;
@group(0) @binding(9) var<storage, read_write> contactForce : array<f32>; // n*3 assembled (second pass in same dispatch via barrier)

fn frozenDist(kind : f32, w : vec4f, n : vec3f, id : vec4u, prm : vec4f) -> f32 {
  if (kind > 1.5) {
    // G2 D4: floor records carry floorY in prm.z (was: implicit position frame).
    return position[id.x].y - prm.z;
  } else if (kind < 0.5) {
    let p = position[id.x].xyz;
    let q = w.x * position[id.y].xyz + w.y * position[id.z].xyz + w.z * position[id.w].xyz;
    return dot(n, p - q);
  } else {
    let r = w.x * position[id.x].xyz + w.y * position[id.y].xyz + w.z * position[id.z].xyz + w.w * position[id.w].xyz;
    return dot(n, r);
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  let w = contactW[i];
  let nv = contactN[i];
  let n = nv.xyz;
  let kind = nv.w;
  let id = contactId[i];
  let prm = contactPrm[i];
  let dHat = prm.x; let kappa = prm.y;
  let dRaw = frozenDist(kind, w, n, id, prm);
  let d = max(dRaw, 1e-12);
  if (d >= dHat) {
    contactEnergy[i] = 0.0;
    for (var k = 0u; k < 8u; k++) { contactScratch[i * 8u + k] = 0.0; }
    // G6B: leave the clear-sentinel (1e30) in place for inactive slots so
    // cap-bounded batch loops see them as absent. Live slots always record
    // dRaw < dHat here (same state as compaction), so only the diagnostic
    // minimum's empty-set reading can change (to the 1e30 sentinel, which
    // decodes identically to "no contact").
    contactDist[i] = 1e30;
    return;
  }
  contactDist[i] = dRaw;
  let L = log(d / dHat);
  let u = d - dHat;
  let B = -(u * u) * L;
  contactEnergy[i] = kappa * B;
  // dE/dd = -kappa*(2u*L + u^2/d)  (negative = repulsive, matches barrierGradScalar)
  let g = -kappa * (2.0 * u * L + u * u / d);
  // scratch layout: [kind, g, nx, ny, nz, w0, w1, w2] — assembler expands to vertices.
  contactScratch[i * 8u] = kind;
  contactScratch[i * 8u + 1u] = g;
  contactScratch[i * 8u + 2u] = n.x;
  contactScratch[i * 8u + 3u] = n.y;
  contactScratch[i * 8u + 4u] = n.z;
  contactScratch[i * 8u + 5u] = w.x;
  contactScratch[i * 8u + 6u] = w.y;
  contactScratch[i * 8u + 7u] = w.z;
  // Friction residual is added by friction.wgsl using lagged lambdaN; this
  // kernel contributes the barrier part only (friction has no Hessian).
}
