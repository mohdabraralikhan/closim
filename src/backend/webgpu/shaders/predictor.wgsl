// G0.3 predictor: y_hat = x + h*v + h^2*g  (pins enforced on CPU after dispatch
// via pinMask, or by a follow-up masked copy — pins stay exact).
// One thread per vertex. Portable @workgroup_size(64), no subgroups.

struct SimParams {
  dt : f32,
  invDt2 : f32,
  gravityX : f32,
  gravityY : f32,
  gravityZ : f32,
  _pad0 : f32,
  vertexCount : u32,
  triangleCount : u32,
  hingeCount : u32,
  contactCount : u32,
  newtonIteration : u32,
  pcgIteration : u32,
  lineSearchAlpha : f32,
  trustRegion : f32,
  barrierActivation : f32,
  frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> position0 : array<vec4f>;
@group(0) @binding(2) var<storage, read> velocity : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> position : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> xReference : array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.vertexCount) { return; }
  let x0 = position0[i];
  let v = velocity[i];
  let h = params.dt;
  let h2 = h * h;
  let g = vec3f(params.gravityX, params.gravityY, params.gravityZ);
  let pred = x0.xyz + h * v.xyz + h2 * g;
  position[i] = vec4f(pred, 0.0);
  xReference[i] = vec4f(pred, 0.0);
}
