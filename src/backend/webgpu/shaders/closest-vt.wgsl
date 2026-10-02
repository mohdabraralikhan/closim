// G2 closest-point: vertex-triangle (Ericson 5.1.5).
// Exact port of closestPointVertexTriangle (closest-point.ts) into WGSL f32,
// including the region cascade and the (s,t) convention q = (1-s-t)*a+s*b+t*c.
//
// D1 (documented in gpu-contact.ts): degenerate (zero-area) triangles take a
// finite vertex-vertex fallback instead of risking 0/0 -> NaN. The CPU mesh
// builder rejects det < 1e-12, so real scenes never diverge.
//
// Thread mapping: one thread per VT PRIMITIVE. Primitive expansion from G1
// triangle pairs happens in contact-compact.wgsl; this module exposes both
// the callable `vt_closest` fn (used by ccd-vt.wgsl) and a standalone entry.

// @twin vt-core-begin
struct VtResult {
  s : f32,
  t : f32,
  dist : f32,
  rx : f32,
  ry : f32,
  rz : f32,
  degenerate : u32,
};

fn vt_pack(s : f32, t : f32, rx : f32, ry : f32, rz : f32) -> VtResult {
  return VtResult(s, t, length(vec3f(rx, ry, rz)), rx, ry, rz, 0u);
}

fn vt_closest(
  px : f32, py : f32, pz : f32,
  ax : f32, ay : f32, az : f32,
  bx : f32, by : f32, bz : f32,
  cx : f32, cy : f32, cz : f32,
) -> VtResult {
  let abx = bx - ax; let aby = by - ay; let abz = bz - az;
  let acx = cx - ax; let acy = cy - ay; let acz = cz - az;
  // D1: zero-area guard (compares |ab x ac|^2 against edge scale).
  let nx = aby * acz - abz * acy;
  let ny = abz * acx - abx * acz;
  let nz = abx * acy - aby * acx;
  let area2 = nx * nx + ny * ny + nz * nz;
  let edgeScale = abx * abx + aby * aby + abz * abz + acx * acx + acy * acy + acz * acz;
  if (!(area2 > 1e-30 * edgeScale)) {
    var best_s = 0.0; var best_t = 0.0;
    var best_rx = px - ax; var best_ry = py - ay; var best_rz = pz - az;
    var bd = length(vec3f(best_rx, best_ry, best_rz));
    let r1x = px - bx; let r1y = py - by; let r1z = pz - bz;
    let d1 = length(vec3f(r1x, r1y, r1z));
    if (d1 < bd) { bd = d1; best_s = 1.0; best_t = 0.0; best_rx = r1x; best_ry = r1y; best_rz = r1z; }
    let r2x = px - cx; let r2y = py - cy; let r2z = pz - cz;
    let d2 = length(vec3f(r2x, r2y, r2z));
    if (d2 < bd) { bd = d2; best_s = 0.0; best_t = 1.0; best_rx = r2x; best_ry = r2y; best_rz = r2z; }
    return VtResult(best_s, best_t, bd, best_rx, best_ry, best_rz, 1u);
  }
  let apx = px - ax; let apy = py - ay; let apz = pz - az;
  let d1 = abx * apx + aby * apy + abz * apz;
  let d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0.0 && d2 <= 0.0) { return vt_pack(0.0, 0.0, px - ax, py - ay, pz - az); }
  let bpx = px - bx; let bpy = py - by; let bpz = pz - bz;
  let d3 = abx * bpx + aby * bpy + abz * bpz;
  let d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0.0 && d4 <= d3) { return vt_pack(1.0, 0.0, px - bx, py - by, pz - bz); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    let den = d1 - d3; // = |ab|^2 > 0 (non-degenerate)
    let v = select(0.0, d1 / den, den > 0.0);
    return vt_pack(v, 0.0, px - (ax + v * abx), py - (ay + v * aby), pz - (az + v * abz));
  }
  let cpx = px - cx; let cpy = py - cy; let cpz = pz - cz;
  let d5 = abx * cpx + aby * cpy + abz * cpz;
  let d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0.0 && d5 <= d6) { return vt_pack(0.0, 1.0, px - cx, py - cy, pz - cz); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    let den = d2 - d6; // = |ac|^2 > 0
    let w = select(0.0, d2 / den, den > 0.0);
    return vt_pack(0.0, w, px - (ax + w * acx), py - (ay + w * acy), pz - (az + w * acz));
  }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    let den = (d4 - d3) + (d5 - d6);
    let w = select(0.0, (d4 - d3) / den, den > 0.0);
    let qx = bx + w * (cx - bx); let qy = by + w * (cy - by); let qz = bz + w * (cz - bz);
    return vt_pack(1.0 - w, w, px - qx, py - qy, pz - qz);
  }
  let denom = va + vb + vc;
  if (!(denom > 0.0)) {
    return VtResult(0.0, 0.0, length(vec3f(px - ax, py - ay, pz - az)), px - ax, py - ay, pz - az, 1u);
  }
  let v = vb / denom;
  let w = vc / denom;
  let qx = ax + abx * v + acx * w;
  let qy = ay + aby * v + acy * w;
  let qz = az + abz * v + acz * w;
  return vt_pack(v, w, px - qx, py - qy, pz - qz);
}
// @twin vt-core-end

// Standalone evaluation entry: reads one VT primitive per thread from the
// expanded primitive buffers, writes s/t/dist/r/degen to scratch.
@group(0) @binding(0) var<storage, read> primIds : array<vec4u>; // (p,a,b,c) extended ids
@group(0) @binding(1) var<storage, read> posTrial : array<vec4f>; // x1
@group(0) @binding(2) var<storage, read_write> outSTD : array<vec4f>; // (s,t,dist,0)
@group(0) @binding(5) var<storage, read_write> outR : array<vec4f>; // (rx,ry,rz,degen)
@group(0) @binding(6) var<storage, read> primTotal : array<u32>; // primCountVT (atomic view)

@compute @workgroup_size(64)
fn closest_vt(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= primTotal[0]) { return; }
  let id = primIds[i];
  let P = posTrial[id.x].xyz;
  let A = posTrial[id.y].xyz;
  let B = posTrial[id.z].xyz;
  let C = posTrial[id.w].xyz;
  let r = vt_closest(P.x, P.y, P.z, A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z);
  outSTD[i] = vec4f(r.s, r.t, r.dist, 0.0);
  outR[i] = vec4f(r.rx, r.ry, r.rz, f32(r.degenerate));
}
