// G1.1 Morton keys from swept centroids.
// 30-bit keys (10 bits/axis) normalized against the scene centroid bbox.
// One thread per triangle. Deterministic: identical centroids -> key ties are
// broken by triangle id in the sort payload, matching the CPU mirror.
//
// Scene bbox (DYNAMIC uniform, recomputed per build from triCentroid via the
// existing pcg-reduce min/max pattern, or maintained on CPU from the 64 B
// status path — never a position readback).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> triCentroid : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> mortonKeys : array<u32>;
@group(0) @binding(3) var<storage, read_write> mortonPayload : array<u32>; // identity tri ids (sort input)
// Scene centroid bbox as two uniforms (min, max).
@group(0) @binding(4) var<uniform> sceneMin : vec4f;
@group(0) @binding(5) var<uniform> sceneMax : vec4f;

fn expand10(v : u32) -> u32 {
  // Spread 10 low bits to every 3rd bit: ------98 76543210 -> ...b9..b0 pattern.
  var x = v & 1023u;
  x = (x | (x << 16u)) & 0x030000FFu;
  x = (x | (x << 8u)) & 0x0300F00Fu;
  x = (x | (x << 4u)) & 0x030C30C3u;
  x = (x | (x << 2u)) & 0x09249249u;
  return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x + gid.y * 4194240u;
  if (t >= params.triangleCount) { return; }
  let c = triCentroid[t].xyz;
  let span = max(sceneMax.xyz - sceneMin.xyz, vec3f(1e-9));
  let n = clamp((c - sceneMin.xyz) / span, vec3f(0.0), vec3f(1.0));
  let ix = u32(n.x * 1023.0);
  let iy = u32(n.y * 1023.0);
  let iz = u32(n.z * 1023.0);
  mortonKeys[t] = (expand10(ix) << 2u) | (expand10(iy) << 1u) | expand10(iz);
  mortonPayload[t] = t;
}
