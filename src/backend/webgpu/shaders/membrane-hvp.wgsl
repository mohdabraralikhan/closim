// G4A analytic membrane HVP — y = H_membrane(x) * v WITHOUT finite differences.
// Ports src/physics/membrane-hvp.ts term-for-term (f32):
//   v -> dDs -> dF -> dE -> dS -> dP -> dgrad
// with the tangent moduli differentiating the stress lines exactly:
//   dS00 = C00*dE00 + C01*dE11, dS11 = C11*dE11 + C01*dE00, dS01 = 2*G*dE01
// preserving S01 = 2*G*E01 (NOT 4*G*E01 — see membrane.ts comment).
//
// Two entries, dispatched in order (both GPU->GPU, no readback between):
//   membrane_hvp : one thread per triangle; reads position (vec4f) + direction
//                  (packed f32 n*3, same layout as pcgSearch); writes per-tri
//                  dgrad (9 f32) into elementHVP.
//   assemble_hvp : one thread per vertex; reference O(n*m) corner scan over
//                  elementHVP into hpMembrane (same pattern as
//                  assemble-gradient main; G4B replaces both with CSR gather).
// Membrane-only (bending excluded from Hessian, kept in gradient) — same
// inexact-Newton approximation as the FD oracle it replaces.

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
@group(0) @binding(2) var<storage, read> triangles : array<u32>;
@group(0) @binding(3) var<storage, read> dmInv : array<vec4f>; // (a,b,c,d) in xyzw
@group(0) @binding(4) var<storage, read> restArea : array<f32>;
@group(0) @binding(5) var<storage, read> direction : array<f32>; // n*3 packed
@group(0) @binding(6) var<storage, read_write> elementHVP : array<f32>; // m*9
@group(0) @binding(7) var<storage, read_write> hpMembrane : array<f32>; // n*3

// G0 material uniforms (same slots as membrane-gradient).
@group(0) @binding(8) var<uniform> matC00 : f32;
@group(0) @binding(9) var<uniform> matC11 : f32;
@group(0) @binding(10) var<uniform> matC01 : f32;
@group(0) @binding(11) var<uniform> matG : f32;
@group(0) @binding(12) var<uniform> matThickness : f32;

@compute @workgroup_size(64)
fn membrane_hvp(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x + gid.y * 4194240u;
  if (t >= params.triangleCount) { return; }
  let i0 = triangles[t * 3u];
  let i1 = triangles[t * 3u + 1u];
  let i2 = triangles[t * 3u + 2u];
  let x0 = position[i0].xyz;
  let x1 = position[i1].xyz;
  let x2 = position[i2].xyz;
  let v0 = vec3f(direction[i0 * 3u], direction[i0 * 3u + 1u], direction[i0 * 3u + 2u]);
  let v1 = vec3f(direction[i1 * 3u], direction[i1 * 3u + 1u], direction[i1 * 3u + 2u]);
  let v2 = vec3f(direction[i2 * 3u], direction[i2 * 3u + 1u], direction[i2 * 3u + 2u]);
  let e1 = x1 - x0;
  let e2 = x2 - x0;
  let w1 = v1 - v0;
  let w2 = v2 - v0;
  let a = dmInv[t].x; let b = dmInv[t].y; let c = dmInv[t].z; let d = dmInv[t].w;
  let F00 = a * e1.x + c * e2.x;
  let F10 = a * e1.y + c * e2.y;
  let F20 = a * e1.z + c * e2.z;
  let F01 = b * e1.x + d * e2.x;
  let F11 = b * e1.y + d * e2.y;
  let F21 = b * e1.z + d * e2.z;
  let dF00 = a * w1.x + c * w2.x;
  let dF10 = a * w1.y + c * w2.y;
  let dF20 = a * w1.z + c * w2.z;
  let dF01 = b * w1.x + d * w2.x;
  let dF11 = b * w1.y + d * w2.y;
  let dF21 = b * w1.z + d * w2.z;
  let E00 = 0.5 * (F00*F00 + F10*F10 + F20*F20 - 1.0);
  let E11 = 0.5 * (F01*F01 + F11*F11 + F21*F21 - 1.0);
  let E01 = 0.5 * (F00*F01 + F10*F11 + F20*F21);
  let dC00 = 2.0 * (F00*dF00 + F10*dF10 + F20*dF20);
  let dC11 = 2.0 * (F01*dF01 + F11*dF11 + F21*dF21);
  let dC01 = dF00*F01 + F00*dF01 + dF10*F11 + F10*dF11 + dF20*F21 + F20*dF21;
  let dE00 = 0.5 * dC00;
  let dE11 = 0.5 * dC11;
  let dE01 = 0.5 * dC01;
  // Stress (same as gradient) + tangent moduli — CRITICAL: dS01 = 2*G*dE01.
  let S00 = matC00*E00 + matC01*E11;
  let S11 = matC11*E11 + matC01*E00;
  let S01 = 2.0 * matG * E01;
  let dS00 = matC00*dE00 + matC01*dE11;
  let dS11 = matC11*dE11 + matC01*dE00;
  let dS01 = 2.0 * matG * dE01;
  let dP00 = dF00*S00 + F00*dS00 + dF01*S01 + F01*dS01;
  let dP10 = dF10*S00 + F10*dS00 + dF11*S01 + F11*dS01;
  let dP20 = dF20*S00 + F20*dS00 + dF21*S01 + F21*dS01;
  let dP01 = dF00*S01 + F00*dS01 + dF01*S11 + F01*dS11;
  let dP11 = dF10*S01 + F10*dS01 + dF11*S11 + F11*dS11;
  let dP21 = dF20*S01 + F20*dS01 + dF21*S11 + F21*dS11;
  let s = matThickness * restArea[t];
  let dg1x = s*(dP00*a + dP01*b); let dg1y = s*(dP10*a + dP11*b); let dg1z = s*(dP20*a + dP21*b);
  let dg2x = s*(dP00*c + dP01*d); let dg2y = s*(dP10*c + dP11*d); let dg2z = s*(dP20*c + dP21*d);
  let dg0x = -(dg1x+dg2x); let dg0y = -(dg1y+dg2y); let dg0z = -(dg1z+dg2z);
  let base = t * 9u;
  elementHVP[base] = dg0x;      elementHVP[base+1u] = dg0y; elementHVP[base+2u] = dg0z;
  elementHVP[base+3u] = dg1x;   elementHVP[base+4u] = dg1y; elementHVP[base+5u] = dg1z;
  elementHVP[base+6u] = dg2x;   elementHVP[base+7u] = dg2y; elementHVP[base+8u] = dg2z;
}

@compute @workgroup_size(64)
fn assemble_hvp(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var hx = 0.0; var hy = 0.0; var hz = 0.0;
  for (var t = 0u; t < params.triangleCount; t++) {
    for (var k = 0u; k < 3u; k++) {
      if (triangles[t * 3u + k] == v) {
        let base = t * 9u + k * 3u;
        hx += elementHVP[base]; hy += elementHVP[base + 1u]; hz += elementHVP[base + 2u];
      }
    }
  }
  hpMembrane[v * 3u] = hx; hpMembrane[v * 3u + 1u] = hy; hpMembrane[v * 3u + 2u] = hz;
}
