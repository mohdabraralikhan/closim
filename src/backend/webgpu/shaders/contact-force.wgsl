// G2 contact-force assembly + lagged commit (fills the G0 pipeline gaps).
// Two entries, both fully GPU-side, no readback:
//
//   assemble_contact_force: one thread per VERTEX, O(n*C) atomic-free scan
//     over the compact set (same reference pattern as assemble-gradient).
//     Barrier part ports mirror addDistGrad exactly (g = scratch dE/dd,
//     VT: +g*n on p, -g*w*n on tri; EE: g*w*n with [w0,w1,-w2,-w3] pairing).
//     Friction part ports mirror addFrictionForce exactly: the RESIDUAL side
//     (-f physical) with the same VT/EE/floor distribution.
//     Writes contactForce (n*3), consumed by assemble-gradient.
//   commit_lagged: one thread per CONTACT; refreshes laggedN[i] = (n, lambdaN)
//     with lambdaN = max(0, -dE/dd) when dist < dHat, else LEAVES the stale
//     entry (exactly like CPU commit, which skips d >= dHat).
//     Runs once per ACCEPTED iterate (commit point, not per trial).

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
@group(0) @binding(2) var<storage, read> contactN : array<vec4f>; // (nx,ny,nz,kind)
@group(0) @binding(3) var<storage, read> contactId : array<vec4u>;
@group(0) @binding(4) var<storage, read> contactScratch : array<f32>; // cap*8 stride; lane 1 = dE/dd (barrier-gradient output)
@group(0) @binding(5) var<storage, read> frictionScratch : array<vec4f>; // physical (fx,fy,fz) per contact
@group(0) @binding(6) var<storage, read_write> contactForce : array<f32>; // n*3 residual
@group(0) @binding(7) var<storage, read> contactDist : array<f32>;
@group(0) @binding(8) var<storage, read> contactPrm : array<vec4f>; // (dHat,kappa,mu/floorY,eps)
@group(0) @binding(9) var<storage, read_write> laggedN : array<vec4f>; // (nx,ny,nz,lambdaN)
// velocity-filter binding (own entry below; pruned from other layouts)
@group(0) @binding(10) var<storage, read_write> velocity : array<vec4f>;

@compute @workgroup_size(64)
fn assemble_contact_force(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var fx = 0.0; var fy = 0.0; var fz = 0.0;
  for (var i = 0u; i < params.contactCount; i++) {
    let nv = contactN[i];
    let n = nv.xyz;
    let kind = nv.w;
    let id = contactId[i];
    let w = contactW[i];
    // --- barrier part (mirror addDistGrad; g = scratch lane 1 = dE/dd) ---
    let g = contactScratch[i * 8u + 1u];
    if (kind > 1.5) {
      // floor dd/dy = 1 and n = (0,1,0): residual force is (0, g, 0)
      if (id.x == v) { fy += g; }
    } else if (kind < 0.5) {
      if (id.x == v) { fx += g * n.x; fy += g * n.y; fz += g * n.z; }
      if (id.y == v && id.y < params.vertexCount) { fx += -g * w.x * n.x; fy += -g * w.x * n.y; fz += -g * w.x * n.z; }
      if (id.z == v && id.z < params.vertexCount) { fx += -g * w.y * n.x; fy += -g * w.y * n.y; fz += -g * w.y * n.z; }
      if (id.w == v && id.w < params.vertexCount) { fx += -g * w.z * n.x; fy += -g * w.z * n.y; fz += -g * w.z * n.z; }
    } else {
      // EE pairs [[a,w0],[b,w1],[c,-w2],[d,-w3]] with residual-side force
      let ws0 = w.x; let ws1 = w.y; let ws2 = w.z; let ws3 = w.w;
      if (id.x == v && id.x < params.vertexCount) { fx += g * ws0 * n.x; fy += g * ws0 * n.y; fz += g * ws0 * n.z; }
      if (id.y == v && id.y < params.vertexCount) { fx += g * ws1 * n.x; fy += g * ws1 * n.y; fz += g * ws1 * n.z; }
      if (id.z == v && id.z < params.vertexCount) { fx += g * ws2 * n.x; fy += g * ws2 * n.y; fz += g * ws2 * n.z; }
      if (id.w == v && id.w < params.vertexCount) { fx += g * ws3 * n.x; fy += g * ws3 * n.y; fz += g * ws3 * n.z; }
    }
    // --- friction part, residual side (mirror addFrictionForce with -f) ---
    let fp = frictionScratch[i].xyz;
    let gx = -fp.x; let gy = -fp.y; let gz = -fp.z;
    if (kind > 1.5) {
      if (id.x == v) { fx += gx; fy += gy; fz += gz; }
    } else if (kind < 0.5) {
      if (id.x == v) { fx += gx; fy += gy; fz += gz; }
      if (id.y == v && id.y < params.vertexCount) { fx += -gx * w.x; fy += -gy * w.x; fz += -gz * w.x; }
      if (id.z == v && id.z < params.vertexCount) { fx += -gx * w.y; fy += -gy * w.y; fz += -gz * w.y; }
      if (id.w == v && id.w < params.vertexCount) { fx += -gx * w.z; fy += -gy * w.z; fz += -gz * w.z; }
    } else {
      // EE friction pairs [[a,w0],[b,w1],[c,-w2],[d,-w3]] with (gx,gy,gz)
      if (id.x == v && id.x < params.vertexCount) { fx += gx * w.x; fy += gy * w.x; fz += gz * w.x; }
      if (id.y == v && id.y < params.vertexCount) { fx += gx * w.y; fy += gy * w.y; fz += gz * w.y; }
      if (id.z == v && id.z < params.vertexCount) { fx += gx * -w.z; fy += gy * -w.z; fz += gz * -w.z; }
      if (id.w == v && id.w < params.vertexCount) { fx += gx * -w.w; fy += gy * -w.w; fz += gz * -w.w; }
    }
  }
  contactForce[v * 3u] = fx;
  contactForce[v * 3u + 1u] = fy;
  contactForce[v * 3u + 2u] = fz;
}

@compute @workgroup_size(64)
fn commit_lagged(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  let d = max(contactDist[i], 1e-12);
  let prm = contactPrm[i];
  let dHat = prm.x; let kappa = prm.y;
  if (d >= dHat) { return; } // keep stale entry, exactly like CPU commit
  let t = d - dHat;
  // dE/dd = kappa * (-2t*log(d/dHat) - t^2/d); lambdaN = max(0, -dE/dd)
  let db = -2.0 * t * log(d / dHat) - (t * t) / d;
  let lambdaN = max(0.0, -(kappa * db));
  let n = contactN[i].xyz;
  laggedN[i] = vec4f(n, lambdaN);
}

// Restitution-0 impact velocity filter (ports newton.ts killApproachVelocity):
// removes approaching normal velocity at frozen contacts, single
// deterministic sweep; separating velocities untouched. Runs once per
// ACCEPTED step (not per trial), reading the accepted compact set.
@compute @workgroup_size(64)
fn velocity_filter(@builtin(global_invocation_id) gid : vec3u) {
  // NOTE: per-vertex formulation would race (pairs share verts); this entry
  // runs ONE thread per CONTACT and applies fractional impulses with the same
  // 0.5/0.25 splits as the CPU single sweep. Concurrent contacts sharing a
  // vertex accumulate via atomics-free... — CAUTION: plain read-modify-write
  // races here. G3 restricts this kernel to validation scenes and documents
  // the race as a G3.1 coloring target; parity tests use scenes where each
  // vertex appears in at most one deep contact (floor / head-on interface).
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  let nv = contactN[i];
  let n = nv.xyz;
  let kind = nv.w;
  let id = contactId[i];
  let w = contactW[i];
  if (kind > 1.5) {
    let vy = velocity[id.x].y;
    if (vy < 0.0) {
      velocity[id.x].y = 0.0;
    }
    return;
  }
  if (kind < 0.5) {
    let vp = velocity[id.x].xyz;
    var tx = 0.0; var ty = 0.0; var tz = 0.0;
    if (id.y < params.vertexCount) {
      tx += w.x * velocity[id.y].x; ty += w.x * velocity[id.y].y; tz += w.x * velocity[id.y].z;
    }
    if (id.z < params.vertexCount) {
      tx += w.y * velocity[id.z].x; ty += w.y * velocity[id.z].y; tz += w.y * velocity[id.z].z;
    }
    if (id.w < params.vertexCount) {
      tx += w.z * velocity[id.w].x; ty += w.z * velocity[id.w].y; tz += w.z * velocity[id.w].z;
    }
    let rel = (vp.x - tx) * n.x + (vp.y - ty) * n.y + (vp.z - tz) * n.z;
    if (rel < 0.0) {
      let j = -0.5 * rel;
      velocity[id.x] = velocity[id.x] + vec4f(j * n, 0.0);
      if (id.y < params.vertexCount) { velocity[id.y] -= vec4f(j * w.x * n, 0.0); }
      if (id.z < params.vertexCount) { velocity[id.z] -= vec4f(j * w.y * n, 0.0); }
      if (id.w < params.vertexCount) { velocity[id.w] -= vec4f(j * w.z * n, 0.0); }
    }
    return;
  }
  // EE: edge-ab average vs edge-cd average (0.25 split, CPU-identical)
  var ax = 0.0; var ay = 0.0; var az = 0.0;
  var cx = 0.0; var cy = 0.0; var cz = 0.0;
  if (id.x < params.vertexCount) {
    ax += velocity[id.x].x * 0.5; ay += velocity[id.x].y * 0.5; az += velocity[id.x].z * 0.5;
  }
  if (id.y < params.vertexCount) {
    ax += velocity[id.y].x * 0.5; ay += velocity[id.y].y * 0.5; az += velocity[id.y].z * 0.5;
  }
  if (id.z < params.vertexCount) {
    cx += velocity[id.z].x * 0.5; cy += velocity[id.z].y * 0.5; cz += velocity[id.z].z * 0.5;
  }
  if (id.w < params.vertexCount) {
    cx += velocity[id.w].x * 0.5; cy += velocity[id.w].y * 0.5; cz += velocity[id.w].z * 0.5;
  }
  let rel = (ax - cx) * n.x + (ay - cy) * n.y + (az - cz) * n.z;
  if (rel < 0.0) {
    let j = -0.25 * rel;
    if (id.x < params.vertexCount) { velocity[id.x] += vec4f(j * n, 0.0); }
    if (id.y < params.vertexCount) { velocity[id.y] += vec4f(j * n, 0.0); }
    if (id.z < params.vertexCount) { velocity[id.z] -= vec4f(j * n, 0.0); }
    if (id.w < params.vertexCount) { velocity[id.w] -= vec4f(j * n, 0.0); }
  }
}
