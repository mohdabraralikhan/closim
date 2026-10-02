// G0 Newton RHS: g = M*(x-yHat)/h^2 + gradInternal + gradContact.
// Runs after assemble-gradient; pins filtered by zeroing pinned rows here so
// every downstream kernel can assume filtered inputs.

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
@group(0) @binding(2) var<storage, read> positionRef : array<vec4f>; // y_hat
@group(0) @binding(3) var<storage, read> mass : array<f32>;
@group(0) @binding(4) var<storage, read> gradInternal : array<f32>; // n*3 assembled FEM+contact
@group(0) @binding(5) var<storage, read> pinMask : array<u32>;
@group(0) @binding(6) var<storage, read_write> rhs : array<f32>;     // g (n*3)
@group(0) @binding(7) var<storage, read_write> negRhs : array<f32>;  // b = -g (n*3)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let v = i / 3u;
  let k = i % 3u;
  var pv : f32;
  var rv : f32;
  if (k == 0u) { pv = position[v].x; rv = positionRef[v].x; }
  else if (k == 1u) { pv = position[v].y; rv = positionRef[v].y; }
  else { pv = position[v].z; rv = positionRef[v].z; }
  var g = mass[v] * (pv - rv) * params.invDt2 + gradInternal[i];
  if (pinMask[v] != 0u) { g = 0.0; }
  rhs[i] = g;
  negRhs[i] = -g;
}
