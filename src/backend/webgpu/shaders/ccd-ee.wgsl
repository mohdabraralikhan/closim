// G2 CCD: edge-edge time-of-impact over x(t) = x0 + t*(x1-x0), t in [0,1].
// Exact policy port of ccd-ee.ts: resting check, deviation early-out, cubic
// coplanarity S(t) = r.(e1 x e2), root validation with the interior
// (s,t in [-1e-6, 1+1e-6]) requirement, conservative 16-sample fallback.
// Same status/flag encoding and D1/D2 notes as ccd-vt.wgsl.

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
@group(0) @binding(3) var<storage, read> primIds : array<vec4u>;  // (a,b,c,d) per EE primitive
@group(0) @binding(4) var<storage, read_write> outTOI : array<f32>;
@group(0) @binding(5) var<storage, read_write> outFlag : array<u32>; // 0 resting, 1 impact, 2 safe, 3 failure
@group(0) @binding(6) var<storage, read> primTotal : array<u32>; // primCountEE (atomic view)
@group(0) @binding(7) var<uniform> thickness : f32; // dMin: CCD contact thickness

@compute @workgroup_size(64)
fn ccd_ee(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= primTotal[0]) { return; }
  let id = primIds[i];
  let a0 = posStart[id.x].xyz; let b0 = posStart[id.y].xyz;
  let c0 = posStart[id.z].xyz; let d0 = posStart[id.w].xyz;
  let a1 = posTrial[id.x].xyz; let b1 = posTrial[id.y].xyz;
  let c1 = posTrial[id.z].xyz; let d1 = posTrial[id.w].xyz;
  let q0 = ee_closest(a0.x, a0.y, a0.z, b0.x, b0.y, b0.z, c0.x, c0.y, c0.z, d0.x, d0.y, d0.z);
  if (q0.dist != q0.dist) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
  if (q0.dist <= thickness) { outTOI[i] = 0.0; outFlag[i] = 0u; return; }
  // relative-motion early-out
  let da = a1 - a0; let db = b1 - b0; let dc = c1 - c0; let dd = d1 - d0;
  let m = (da + db + dc + dd) / 4.0;
  var dev = 0.0;
  dev = max(dev, max(abs(da.x - m.x), max(abs(da.y - m.y), abs(da.z - m.z))));
  dev = max(dev, max(abs(db.x - m.x), max(abs(db.y - m.y), abs(db.z - m.z))));
  dev = max(dev, max(abs(dc.x - m.x), max(abs(dc.y - m.y), abs(dc.z - m.z))));
  dev = max(dev, max(abs(dd.x - m.x), max(abs(dd.y - m.y), abs(dd.z - m.z))));
  // NO_HIT sentinel is 2.0 (TOI domain is [0,1]; old Dawn rejects const Inf).
  if (q0.dist - 4.0 * dev > thickness) { outTOI[i] = 2.0; outFlag[i] = 2u; return; }
  // cubic coplanarity S(t) = r.(e1 x e2), e1 = b-a, e2 = d-c, r = c-a
  let e10 = b0 - a0; let de1 = db - da;
  let e20 = d0 - c0; let de2 = dd - dc;
  let r0 = c0 - a0; let dr = dc - da;
  let n0 = cross3(e10, e20);
  let n1 = cross3(e10, de2) + cross3(de1, e20);
  let n2 = cross3(de1, de2);
  let c0c = dot(r0, n0);
  let c1c = dot(r0, n1) + dot(dr, n0);
  let c2c = dot(r0, n2) + dot(dr, n1);
  let c3c = dot(dr, n2);
  // NaN coefficients -> failure (Inf is impossible at meter scales; NaN is the
  // only non-finite hazard, e.g. 0/0 in degenerate input the D1 guard missed).
  if ((c0c != c0c) || (c1c != c1c) || (c2c != c2c) || (c3c != c3c)) {
    outTOI[i] = 0.0; outFlag[i] = 3u; return;
  }
  var roots : array<f32, 3>;
  let nr = cubic01(c3c, c2c, c1c, c0c, &roots);
  for (var k = 0u; k < nr; k++) {
    let t = roots[k];
    let A = a0 + da * t; let B = b0 + db * t;
    let C = c0 + dc * t; let D = d0 + dd * t;
    let cp = ee_closest(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z, D.x, D.y, D.z);
    if (cp.dist != cp.dist) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
    if (cp.dist <= thickness && cp.s >= -1e-6 && cp.s <= 1.0 + 1e-6
      && cp.t >= -1e-6 && cp.t <= 1.0 + 1e-6) {
      outTOI[i] = t; outFlag[i] = 1u; return;
    }
  }
  var minD = 1e30;
  var minT = 0.0;
  for (var k = 0u; k <= 16u; k++) {
    let t = f32(k) / 16.0;
    let A = a0 + da * t; let B = b0 + db * t;
    let C = c0 + dc * t; let D = d0 + dd * t;
    let d = ee_closest(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z, D.x, D.y, D.z).dist;
    if (d != d) { outTOI[i] = 0.0; outFlag[i] = 3u; return; }
    if (d < minD) { minD = d; minT = t; }
  }
  if (minD <= thickness) { outTOI[i] = minT; outFlag[i] = 1u; return; }
  outTOI[i] = 2.0;
  outFlag[i] = 2u;
}
