// G0 Newton trial + acceptance helpers.
// apply_step: xTrial = x + alpha*dx (pins re-imposed by zeroing dx at pinned
// rows upstream; enforcePins mirror runs on CPU once per accepted step).
// trust_scale is folded on the CPU (cheap max-scan over the status buffer is
// NOT in the hot loop) — this kernel applies the final scaled alpha.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> xBase : array<f32>;
@group(0) @binding(2) var<storage, read> dx : array<f32>;
@group(0) @binding(3) var<storage, read_write> xTrial : array<f32>;
// vec4 path bindings (own entry below; pruned from main's layout)
@group(0) @binding(4) var<storage, read> xBase4 : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> xTrial4 : array<vec4f>;
@group(0) @binding(6) var<storage, read> pinMask : array<u32>;
@group(0) @binding(7) var<storage, read> pinPos : array<vec4f>; // STATIC pinned targets

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  xTrial[i] = xBase[i] + params.lineSearchAlpha * dx[i];
}

// G3 vec4 path: xTrial4[v] = xBase4[v] + alpha*dx[3v..] with exact pins.
// dx is already zero at pinned rows (PCG filter); pinPos restore makes pins
// bit-exact regardless of upstream rounding.
@compute @workgroup_size(64)
fn apply_step_vec4(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  let a = params.lineSearchAlpha;
  var p = xBase4[v].xyz + a * vec3f(dx[v * 3u], dx[v * 3u + 1u], dx[v * 3u + 2u]);
  if (pinMask[v] != 0u) { p = pinPos[v].xyz; }
  xTrial4[v] = vec4f(p, 0.0);
}

// G3 pins for the predictor output: restores exact pin positions into a vec4
// state buffer (used once per step after predictor; Newton then preserves).
@compute @workgroup_size(64)
fn enforce_pins(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  if (pinMask[v] != 0u) { xTrial4[v] = pinPos[v]; }
}
