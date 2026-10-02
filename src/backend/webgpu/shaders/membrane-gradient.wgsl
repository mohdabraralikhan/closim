// G0.4 membrane gradient — element-local pass.
// One thread per triangle. Ports src/physics/membrane.ts term-for-term:
//   F = Ds * DmInv, C = F^T F, E = 0.5*(C - I)
//   psi = 0.5*C00*E00^2 + 0.5*C11*E11^2 + C01*E00*E11 + 2*G*E01^2
//   S01 = 2*G*E01  (NOT 4*G*E01 — see membrane.ts comment)
//   P = F * S, grad[x1] = s*(P_a*a + P_b*b), grad[x2] = s*(P_a*c + P_b*d)
// Writes 9 f32 per triangle into elementGradient (no atomics in G0).
// A separate assemble pass reduces elementGradient -> vertex gradient.

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
@group(0) @binding(6) var<storage, read_write> elementGradient : array<f32>; // m*9
@group(0) @binding(7) var<storage, read_write> elementEnergy : array<f32>;    // m
@group(0) @binding(13) var<storage, read_write> elementGradientB : array<f32>; // m*9 FD scratch
@group(0) @binding(14) var<uniform> outSel : u32; // 0 = elementGradient, 1 = elementGradientB (FD HVP)

// G0 material uniforms (full per-triangle materialId table is a G1 cleanup).
@group(0) @binding(8) var<uniform> matC00 : f32;
@group(0) @binding(9) var<uniform> matC11 : f32;
@group(0) @binding(10) var<uniform> matC01 : f32;
@group(0) @binding(11) var<uniform> matG : f32;
@group(0) @binding(12) var<uniform> matThickness : f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x + gid.y * 4194240u;
  if (t >= params.triangleCount) { return; }
  let i0 = triangles[t * 3u];
  let i1 = triangles[t * 3u + 1u];
  let i2 = triangles[t * 3u + 2u];
  let x0 = position[i0].xyz;
  let x1 = position[i1].xyz;
  let x2 = position[i2].xyz;
  let e1 = x1 - x0;
  let e2 = x2 - x0;
  let a = dmInv[t].x; let b = dmInv[t].y; let c = dmInv[t].z; let d = dmInv[t].w;
  let F00 = a * e1.x + c * e2.x;
  let F10 = a * e1.y + c * e2.y;
  let F20 = a * e1.z + c * e2.z;
  let F01 = b * e1.x + d * e2.x;
  let F11 = b * e1.y + d * e2.y;
  let F21 = b * e1.z + d * e2.z;
  let C00 = F00*F00 + F10*F10 + F20*F20;
  let C11 = F01*F01 + F11*F11 + F21*F21;
  let C01 = F00*F01 + F10*F11 + F20*F21;
  let E00 = 0.5 * (C00 - 1.0);
  let E11 = 0.5 * (C11 - 1.0);
  let E01 = 0.5 * C01;
  let psi = 0.5*matC00*E00*E00 + 0.5*matC11*E11*E11 + matC01*E00*E11 + 2.0*matG*E01*E01;
  let area = restArea[t];
  let s = matThickness * area;
  elementEnergy[t] = s * psi;
  // Stress — CRITICAL: S01 = 2*G*E01.
  let S00 = matC00*E00 + matC01*E11;
  let S11 = matC11*E11 + matC01*E00;
  let S01 = 2.0 * matG * E01;
  let P00 = F00*S00 + F01*S01;
  let P10 = F10*S00 + F11*S01;
  let P20 = F20*S00 + F21*S01;
  let P01 = F00*S01 + F01*S11;
  let P11 = F10*S01 + F11*S11;
  let P21 = F20*S01 + F21*S11;
  // corner gradients, xyz per corner
  let g1x = s*(P00*a + P01*b); let g1y = s*(P10*a + P11*b); let g1z = s*(P20*a + P21*b);
  let g2x = s*(P00*c + P01*d); let g2y = s*(P10*c + P11*d); let g2z = s*(P20*c + P21*d);
  let g0x = -(g1x+g2x); let g0y = -(g1y+g2y); let g0z = -(g1z+g2z);
  let base = t * 9u;
  if (outSel == 0u) {
    elementGradient[base] = g0x;     elementGradient[base+1u] = g0y; elementGradient[base+2u] = g0z;
    elementGradient[base+3u] = g1x;  elementGradient[base+4u] = g1y;  elementGradient[base+5u] = g1z;
    elementGradient[base+6u] = g2x;  elementGradient[base+7u] = g2y;  elementGradient[base+8u] = g2z;
  } else {
    elementGradientB[base] = g0x;     elementGradientB[base+1u] = g0y; elementGradientB[base+2u] = g0z;
    elementGradientB[base+3u] = g1x;  elementGradientB[base+4u] = g1y;  elementGradientB[base+5u] = g1z;
    elementGradientB[base+6u] = g2x;  elementGradientB[base+7u] = g2y;  elementGradientB[base+8u] = g2z;
  }
}
