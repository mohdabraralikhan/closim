// G0.5 bending gradient — hinge-local pass with CPU-identical FD stencil.
// Ports src/physics/bending.ts: W = 0.5*k*(theta-theta0)^2*edgeLen^2/areaSum,
// gradient via central differences (eps = 1e-7 m in f64 on CPU; f32 eps here).
// One thread per hinge writes 12 f32 into hingeGradient + 1 energy.
// NOTE: deliberately FD-based so GPU matches the CPU reference bit-near.
// Analytic hinge Hessian is a post-G0 optimization, NOT a G0 behavior change.

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
@group(0) @binding(2) var<storage, read> hinges : array<vec4u>;   // v0,v1,v2,v3
@group(0) @binding(3) var<storage, read> hingeMeta : array<vec4f>; // restAngle, edgeLen, areaSum, k
@group(0) @binding(4) var<storage, read_write> hingeGradient : array<f32>; // h*12
@group(0) @binding(5) var<storage, read_write> hingeEnergyOut : array<f32>;  // h

fn cross3(a : vec3f, b : vec3f) -> vec3f {
  return vec3f(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

fn dihedralOf(p0 : vec3f, p1 : vec3f, p2 : vec3f, p3 : vec3f) -> f32 {
  let e0 = p1 - p0;
  let e1 = p2 - p0;
  let e2 = p3 - p0;
  let n1 = cross3(e0, e1);
  let n2 = cross3(e0, e2);
  let l1 = max(length(n1), 1e-30);
  let l2 = max(length(n2), 1e-30);
  let cosT = clamp(dot(n1, n2) / (l1 * l2), -1.0, 1.0);
  return acos(cosT);
}

fn hingeEnergyAt(p0 : vec3f, p1 : vec3f, p2 : vec3f, p3 : vec3f,
                 rest : f32, edgeLen : f32, areaSum : f32, k : f32) -> f32 {
  let th = dihedralOf(p0, p1, p2, p3);
  let d = th - rest;
  let w = edgeLen * edgeLen / max(areaSum, 1e-12);
  return 0.5 * k * d * d * w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let hh = gid.x + gid.y * 4194240u;
  if (hh >= params.hingeCount) { return; }
  let hv = hinges[hh];
  let hm = hingeMeta[hh];
  let rest = hm.x; let edgeLen = hm.y; let areaSum = hm.z; let k = hm.w;
  var p : array<vec3f, 4>;
  p[0] = position[hv.x].xyz; p[1] = position[hv.y].xyz;
  p[2] = position[hv.z].xyz; p[3] = position[hv.w].xyz;
  let e = hingeEnergyAt(p[0], p[1], p[2], p[3], rest, edgeLen, areaSum, k);
  hingeEnergyOut[hh] = e;
  let eps : f32 = 1e-5; // f32 FD step (CPU uses 1e-7 in f64; G0 tolerance absorbs the difference)
  let base = hh * 12u;
  for (var corner = 0u; corner < 4u; corner++) {
    for (var ax = 0u; ax < 3u; ax++) {
      var pp = p[corner];
      if (ax == 0u) { pp.x += eps; } else if (ax == 1u) { pp.y += eps; } else { pp.z += eps; }
      var q0 = p[0]; var q1 = p[1]; var q2 = p[2]; var q3 = p[3];
      if (corner == 0u) { q0 = pp; } else if (corner == 1u) { q1 = pp; }
      else if (corner == 2u) { q2 = pp; } else { q3 = pp; }
      let ep = hingeEnergyAt(q0, q1, q2, q3, rest, edgeLen, areaSum, k);
      var pm = p[corner];
      if (ax == 0u) { pm.x -= eps; } else if (ax == 1u) { pm.y -= eps; } else { pm.z -= eps; }
      var r0 = p[0]; var r1 = p[1]; var r2 = p[2]; var r3 = p[3];
      if (corner == 0u) { r0 = pm; } else if (corner == 1u) { r1 = pm; }
      else if (corner == 2u) { r2 = pm; } else { r3 = pm; }
      let em = hingeEnergyAt(r0, r1, r2, r3, rest, edgeLen, areaSum, k);
      hingeGradient[base + corner * 3u + ax] = (ep - em) / (2.0 * eps);
    }
  }
}
