// G0 assemble: elementGradient (m*9) + hingeGradient (h*12) -> vertex gradient.
// G0 REFERENCE implementation: one thread per vertex scans all triangles and
// hinges. O(n*(m+h)) — acceptable for validation meshes (<= ~2k tris) and
// trivially correct without an adjacency table. G1 replaces this with a
// prefix-sum gather; the numerics do not change, only the complexity.
//
// Separate output buffers avoid aliasing hazards:
//
//   elementGradient --triangle pass--> | + hingeGradient --hinge pass-->
//                                     v
//                        assemble (this file) --> gradient (n*3 f32)
// Inertia term M*(x-yHat)/h^2 is added by newton-rhs.wgsl, not here.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> triangles : array<u32>;
@group(0) @binding(2) var<storage, read> elementGradient : array<f32>;
@group(0) @binding(3) var<storage, read> hinges : array<vec4u>;
@group(0) @binding(4) var<storage, read> hingeGradient : array<f32>;
@group(0) @binding(5) var<storage, read> contactForce : array<f32>; // n*3 (barrier+friction residual), may be zero-filled
@group(0) @binding(6) var<storage, read_write> gradient : array<f32>; // n*3
@group(0) @binding(7) var<storage, read_write> gradientAlt : array<f32>; // n*3 FD scratch
@group(0) @binding(8) var<uniform> outSel : u32; // 0 = gradient, 1 = gradientAlt (FD HVP)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var gx = 0.0; var gy = 0.0; var gz = 0.0;
  for (var t = 0u; t < params.triangleCount; t++) {
    for (var c = 0u; c < 3u; c++) {
      if (triangles[t * 3u + c] == v) {
        gx += elementGradient[t * 9u + c * 3u];
        gy += elementGradient[t * 9u + c * 3u + 1u];
        gz += elementGradient[t * 9u + c * 3u + 2u];
      }
    }
  }
  for (var hh = 0u; hh < params.hingeCount; hh++) {
    let hv = hinges[hh];
    if (hv.x == v) {
      gx += hingeGradient[hh * 12u];     gy += hingeGradient[hh * 12u + 1u]; gz += hingeGradient[hh * 12u + 2u];
    } else if (hv.y == v) {
      gx += hingeGradient[hh * 12u + 3u]; gy += hingeGradient[hh * 12u + 4u]; gz += hingeGradient[hh * 12u + 5u];
    } else if (hv.z == v) {
      gx += hingeGradient[hh * 12u + 6u]; gy += hingeGradient[hh * 12u + 7u]; gz += hingeGradient[hh * 12u + 8u];
    } else if (hv.w == v) {
      gx += hingeGradient[hh * 12u + 9u]; gy += hingeGradient[hh * 12u + 10u]; gz += hingeGradient[hh * 12u + 11u];
    }
  }
  gx += contactForce[v * 3u]; gy += contactForce[v * 3u + 1u]; gz += contactForce[v * 3u + 2u];
  if (outSel == 0u) {
    gradient[v * 3u] = gx;
    gradient[v * 3u + 1u] = gy;
    gradient[v * 3u + 2u] = gz;
  } else {
    gradientAlt[v * 3u] = gx;
    gradientAlt[v * 3u + 1u] = gy;
    gradientAlt[v * 3u + 2u] = gz;
  }
}
