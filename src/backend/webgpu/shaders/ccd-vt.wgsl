// G2 CCD: vertex-triangle time-of-impact over x(t) = x0 + t*(x1-x0), t in [0,1].
// Exact policy port of ccd-vt.ts:
//
//   1. resting: d(0) <= thickness -> TOI 0
//   2. relative-motion early-out (translation cancels in r(t))
//   3. cubic coplanarity V(t) = u.(v x w); roots validated by closest distance
//   4. conservative 16-sample fallback (grazing); Infinity only when clean
//
// Status encoding (outFlag): 0 = resting, 1 = impact, 2 = safe (Infinity),
// 3 = failure (non-finite intermediates — NEVER silently mapped to safe; the
// compact shader treats failure as trial-invalid, mirroring CPU checkTrial).
//
// D2: WGSL has no cbrt() — cubic roots use sign(x)*pow(|x|,1/3) with a zero
// guard (fcbrt). TOI tolerance absorbs the ulp-level difference.
//
// NOTE on duplication: WGSL has no module imports, so the closest-point core
// and cubic core are textually duplicated from closest-vt.wgsl (twin regions
// below). tests/webgpu twin-identity test enforces character equality.

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

// @twin cubic-core-begin
fn fcbrt(x : f32) -> f32 {
  if (x == 0.0) { return 0.0; }
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

// Solves a3 t^3 + a2 t^2 + a1 t + a0 = 0, writes up to 3 roots, returns count.
// Cardano + quadratic fallback + 4 Newton polish iterations (ports cubic.ts).
fn cubic01(a3 : f32, a2 : f32, a1 : f32, a0 : f32, out : ptr<function, array<f32, 3>>) -> u32 {
  let eps = 1e-12;
  var n = 0u;
  if (abs(a3) < eps) {
    // quadratic fallback
    if (abs(a2) < eps) {
      if (abs(a1) < eps) { return 0u; }
      let r = -a0 / a1;
      if (r >= 0.0 && r <= 1.0) { (*out)[0] = r; return 1u; }
      return 0u;
    }
    let disc = a1 * a1 - 4.0 * a2 * a0;
    if (disc < 0.0) { return 0u; }
    let sd = sqrt(disc);
    let r0 = (-a1 - sd) / (2.0 * a2);
    let r1 = (-a1 + sd) / (2.0 * a2);
    if (r0 >= 0.0 && r0 <= 1.0) { (*out)[n] = r0; n += 1u; }
    if (r1 >= 0.0 && r1 <= 1.0) { (*out)[n] = r1; n += 1u; }
    return n;
  }
  let A = a2 / a3; let B = a1 / a3; let C = a0 / a3;
  let sqA = A * A;
  let p = (3.0 * B - sqA) / 3.0;
  let q = (2.0 * sqA * A - 9.0 * A * B + 27.0 * C) / 27.0;
  var raw : array<f32, 3>;
  var nr = 0u;
  let disc = (q * q) / 4.0 + (p * p * p) / 27.0;
  let shift = A / 3.0;
  if (disc > eps) {
    let sd = sqrt(disc);
    let u = fcbrt(-q / 2.0 + sd);
    let v = fcbrt(-q / 2.0 - sd);
    raw[0] = u + v - shift; nr = 1u;
  } else if (abs(disc) <= eps) {
    let u = fcbrt(-q / 2.0);
    raw[0] = 2.0 * u - shift; raw[1] = -u - shift; nr = 2u;
  } else {
    let r = sqrt(-(p * p * p) / 27.0);
    let phi = acos(clamp(-q / (2.0 * r), -1.0, 1.0));
    let s = 2.0 * fcbrt(r);
    raw[0] = s * cos(phi / 3.0) - shift;
    raw[1] = s * cos((phi + 6.28318530718) / 3.0) - shift;
    raw[2] = s * cos((phi + 12.56637061436) / 3.0) - shift;
    nr = 3u;
  }
  for (var k = 0u; k < nr; k++) {
    var r = raw[k];
    if (r != r) { continue; } // NaN only: Inf roots die at the [0,1] range filter
    for (var it = 0u; it < 4u; it++) {
      let fv = ((a3 * r + a2) * r + a1) * r + a0;
      let df = (3.0 * a3 * r + 2.0 * a2) * r + a1;
      if (abs(df) < 1e-18) { break; }
      r -= fv / df;
    }
    if (r >= -1e-9 && r <= 1.0 + 1e-9) {
      let c = min(1.0, max(0.0, r));
      // dedupe (sorted push: roots arrive unsorted; insertion keeps order)
      var dup = false;
      for (var j = 0u; j < n; j++) {
        if (abs((*out)[j] - c) <= 1e-9) { dup = true; break; }
      }
      if (!dup && n < 3u) {
        // keep ascending: shift larger entries right
        var pos = n;
        while (pos > 0u && (*out)[pos - 1u] > c) {
          (*out)[pos] = (*out)[pos - 1u];
          pos -= 1u;
        }
        (*out)[pos] = c;
        n += 1u;
      }
    }
  }
  return n;
}
// @twin cubic-core-end

fn cross3(a : vec3f, b : vec3f) -> vec3f {
  return vec3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> posStart : array<vec4f>; // x0 (extended)
@group(0) @binding(2) var<storage, read> posTrial : array<vec4f>; // x1 (extended)
@group(0) @binding(3) var<storage, read> primIds : array<vec4u>;  // (p,a,b,c) per VT primitive
@group(0) @binding(4) var<storage, read_write> outTOI : array<f32>;
@group(0) @binding(5) var<storage, read_write> outFlag : array<u32>; // 0 resting, 1 impact, 2 safe, 3 failure
@group(0) @binding(6) var<storage, read> primTotal : array<u32>; // primCountVT (atomic view)
@group(0) @binding(7) var<uniform> thickness : f32; // dMin: CCD contact thickness

@compute @workgroup_size(64)
fn ccd_vt(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= primTotal[0]) { return; }
  let id = primIds[i];
  let p0 = posStart[id.x].xyz; let a0 = posStart[id.y].xyz;
  let b0 = posStart[id.z].xyz; let c0 = posStart[id.w].xyz;
  let p1 = posTrial[id.x].xyz; let a1 = posTrial[id.y].xyz;
  let b1 = posTrial[id.z].xyz; let c1 = posTrial[id.w].xyz;
  let cp0 = vt_closest(p0.x, p0.y, p0.z, a0.x, a0.y, a0.z, b0.x, b0.y, b0.z, c0.x, c0.y, c0.z);
  if (cp0.dist != cp0.dist) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
  if (cp0.dist <= thickness) { outTOI[i] = 0.0; outFlag[i] = 0u; return; }
  // relative-motion early-out
  let dp = p1 - p0; let da = a1 - a0; let db = b1 - b0; let dc = c1 - c0;
  let m = (dp + da + db + dc) / 4.0;
  var dev = 0.0;
  dev = max(dev, max(abs(dp.x - m.x), max(abs(dp.y - m.y), abs(dp.z - m.z))));
  dev = max(dev, max(abs(da.x - m.x), max(abs(da.y - m.y), abs(da.z - m.z))));
  dev = max(dev, max(abs(db.x - m.x), max(abs(db.y - m.y), abs(db.z - m.z))));
  dev = max(dev, max(abs(dc.x - m.x), max(abs(dc.y - m.y), abs(dc.z - m.z))));
  // NO_HIT sentinel is 2.0 (TOI domain is [0,1]; old Dawn rejects const Inf).
  if (cp0.dist - 4.0 * dev > thickness) { outTOI[i] = 2.0; outFlag[i] = 2u; return; }
  // cubic coplanarity V(t) = u.(v x w)
  let u0 = p0 - a0; let du = dp - da;
  let v0 = b0 - a0; let dv = db - da;
  let w0 = c0 - a0; let dw = dc - da;
  let n0 = cross3(v0, w0);
  let n1 = cross3(v0, dw) + cross3(dv, w0);
  let n2 = cross3(dv, dw);
  let c0c = dot(u0, n0);
  let c1c = dot(u0, n1) + dot(du, n0);
  let c2c = dot(u0, n2) + dot(du, n1);
  let c3c = dot(du, n2);
  // NaN coefficients -> failure (Inf is impossible at meter scales; NaN is the
  // only non-finite hazard, e.g. 0/0 in degenerate input the D1 guard missed).
  if ((c0c != c0c) || (c1c != c1c) || (c2c != c2c) || (c3c != c3c)) {
    outTOI[i] = 0.0; outFlag[i] = 3u; return;
  }
  var roots : array<f32, 3>;
  let nr = cubic01(c3c, c2c, c1c, c0c, &roots);
  for (var k = 0u; k < nr; k++) {
    let t = roots[k];
    let pp = p0 + dp * t; let aa = a0 + da * t;
    let bb = b0 + db * t; let cc = c0 + dc * t;
    let cp = vt_closest(pp.x, pp.y, pp.z, aa.x, aa.y, aa.z, bb.x, bb.y, bb.z, cc.x, cc.y, cc.z);
    if (cp.dist != cp.dist) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
    if (cp.dist <= thickness) { outTOI[i] = t; outFlag[i] = 1u; return; }
  }
  var minD = 1e30;
  var minT = 0.0;
  for (var k = 0u; k <= 16u; k++) {
    let t = f32(k) / 16.0;
    let pp = p0 + dp * t; let aa = a0 + da * t;
    let bb = b0 + db * t; let cc = c0 + dc * t;
    let d = vt_closest(pp.x, pp.y, pp.z, aa.x, aa.y, aa.z, bb.x, bb.y, bb.z, cc.x, cc.y, cc.z).dist;
    if (d != d) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
    if (d < minD) { minD = d; minT = t; }
  }
  if (minD <= thickness) { outTOI[i] = minT; outFlag[i] = 1u; return; }
  outTOI[i] = 2.0;
  outFlag[i] = 2u;
}
