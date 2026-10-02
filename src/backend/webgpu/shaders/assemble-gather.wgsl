// G4B gather assembly — CSR vertex->element lists replace the O(n*m/h)
// reference scans (assemble-gradient main, assemble_hvp).
// Incidence maps are STATIC (topology): vertexElement{Offsets,Ids,Corners}
// (3m records, corners 0..2) and vertexHinge{Offsets,Ids,Corners} (4h
// records, corners 0..3, first-match-wins — see incidence.ts).
// Records are element-ordered, so gather sums match direct scatter bit-for-bit
// (f32 on device: same order, same roundings as the scan path).
//
// Three entries (fixed bindings per entry):
//   gather_membrane_grad : gradient[v]  = sum over incident tris of
//                          elementGradient[t*9+c*3..] (WRITE).
//   gather_membrane_hvp  : hpMembrane[v] = sum over incident tris of
//                          elementHVP[t*9+c*3..] (WRITE).
//   gather_hinge_grad    : gradient[v] += sum over incident hinges of
//                          hingeGradient[h*12+c*3..] (ADDITIVE read_write;
//                          each thread owns its vertex — no atomics).
// Contact residual (contactForce, n*3) is added by blas add_into; barrier HVP
// assembly stays O(n*C) over the compact active set (bounded, not scanned).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> offsets : array<u32>;
@group(0) @binding(2) var<storage, read> ids : array<u32>;
@group(0) @binding(3) var<storage, read> corners : array<u32>;
@group(0) @binding(4) var<storage, read> elemSrc : array<f32>;
@group(0) @binding(5) var<storage, read_write> vertDst : array<f32>;

@compute @workgroup_size(64)
fn gather_membrane_grad(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var gx = 0.0; var gy = 0.0; var gz = 0.0;
  for (var k = offsets[v]; k < offsets[v + 1u]; k++) {
    let base = ids[k] * 9u + corners[k] * 3u;
    gx += elemSrc[base]; gy += elemSrc[base + 1u]; gz += elemSrc[base + 2u];
  }
  vertDst[v * 3u] = gx; vertDst[v * 3u + 1u] = gy; vertDst[v * 3u + 2u] = gz;
}

@compute @workgroup_size(64)
fn gather_membrane_hvp(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var hx = 0.0; var hy = 0.0; var hz = 0.0;
  for (var k = offsets[v]; k < offsets[v + 1u]; k++) {
    let base = ids[k] * 9u + corners[k] * 3u;
    hx += elemSrc[base]; hy += elemSrc[base + 1u]; hz += elemSrc[base + 2u];
  }
  vertDst[v * 3u] = hx; vertDst[v * 3u + 1u] = hy; vertDst[v * 3u + 2u] = hz;
}

@compute @workgroup_size(64)
fn gather_hinge_grad(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  var gx = 0.0; var gy = 0.0; var gz = 0.0;
  for (var k = offsets[v]; k < offsets[v + 1u]; k++) {
    let base = ids[k] * 12u + corners[k] * 3u;
    gx += elemSrc[base]; gy += elemSrc[base + 1u]; gz += elemSrc[base + 2u];
  }
  vertDst[v * 3u] += gx; vertDst[v * 3u + 1u] += gy; vertDst[v * 3u + 2u] += gz;
}
