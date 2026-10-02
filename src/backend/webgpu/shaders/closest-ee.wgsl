// G2 closest-point: edge-edge (Ericson 5.1.9).
// Exact port of closestPointEdgeEdge into WGSL f32, including the degenerate
// edge guards (a/e <= 1e-30) and the t-clamp re-solve cascade. Output
// convention matches the CPU: r = A(s) - C(t), s on ab, t on cd.
//
// Thread mapping mirrors closest-vt.wgsl: one thread per EE primitive, plus
// the callable `ee_closest` fn consumed by ccd-ee.wgsl.

// @twin ee-core-begin
struct EeResult {
  s : f32,
  t : f32,
  dist : f32,
  rx : f32,
  ry : f32,
  rz : f32,
  degenerate : u32,
};

fn ee_clamp(v : f32, lo : f32, hi : f32) -> f32 {
  return min(hi, max(lo, v));
}

fn ee_closest(
  ax : f32, ay : f32, az : f32,
  bx : f32, by : f32, bz : f32,
  cx : f32, cy : f32, cz : f32,
  dx : f32, dy : f32, dz : f32,
) -> EeResult {
  let d1x = bx - ax; let d1y = by - ay; let d1z = bz - az;
  let d2x = dx - cx; let d2y = dy - cy; let d2z = dz - cz;
  let rx = ax - cx; let ry = ay - cy; let rz = az - cz;
  let a = d1x * d1x + d1y * d1y + d1z * d1z;
  let e = d2x * d2x + d2y * d2y + d2z * d2z;
  let ff = d2x * rx + d2y * ry + d2z * rz;
  var s : f32;
  var t : f32;
  var deg = 0u;
  if (a <= 1e-30 && e <= 1e-30) {
    s = 0.0; t = 0.0; deg = 1u;
  } else if (a <= 1e-30) {
    s = 0.0; t = ee_clamp(ff / e, 0.0, 1.0); deg = 1u;
  } else {
    let c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= 1e-30) {
      t = 0.0; s = ee_clamp(-c / a, 0.0, 1.0); deg = 1u;
    } else {
      let b = d1x * d2x + d1y * d2y + d1z * d2z;
      let denom = a * e - b * b;
      s = select(0.0, ee_clamp((b * ff - c * e) / denom, 0.0, 1.0), denom > 1e-30);
      t = (b * s + ff) / e;
      if (t < 0.0) { t = 0.0; s = ee_clamp(-c / a, 0.0, 1.0); }
      else if (t > 1.0) { t = 1.0; s = ee_clamp((b - c) / a, 0.0, 1.0); }
    }
  }
  let axp = ax + d1x * s; let ayp = ay + d1y * s; let azp = az + d1z * s;
  let cxp = cx + d2x * t; let cyp = cy + d2y * t; let czp = cz + d2z * t;
  let rx2 = axp - cxp; let ry2 = ayp - cyp; let rz2 = azp - czp;
  return EeResult(s, t, length(vec3f(rx2, ry2, rz2)), rx2, ry2, rz2, deg);
}
// @twin ee-core-end

@group(0) @binding(0) var<storage, read> primIds : array<vec4u>; // (a,b,c,d) extended ids
@group(0) @binding(1) var<storage, read> posTrial : array<vec4f>; // x1
@group(0) @binding(2) var<storage, read_write> outSTD : array<vec4f>; // (s,t,dist,0)
@group(0) @binding(5) var<storage, read_write> outR : array<vec4f>; // (rx,ry,rz,degen)
@group(0) @binding(6) var<storage, read> primTotal : array<u32>; // primCountEE (atomic view)

@compute @workgroup_size(64)
fn closest_ee(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= primTotal[0]) { return; }
  let id = primIds[i];
  let A = posTrial[id.x].xyz;
  let B = posTrial[id.y].xyz;
  let C = posTrial[id.z].xyz;
  let D = posTrial[id.w].xyz;
  let r = ee_closest(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z, D.x, D.y, D.z);
  outSTD[i] = vec4f(r.s, r.t, r.dist, 0.0);
  outR[i] = vec4f(r.rx, r.ry, r.rz, f32(r.degenerate));
}
