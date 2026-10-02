// G1.0 swept triangle AABBs — GPU positions -> per-triangle swept bounds.
//
// Newton-segment semantics IDENTICAL to CPU sweptTriAabb (src/collision/aabb.ts):
//   min/max over the 6 corner points (3 verts x {x_start, x_trial}),
//   then expand by pad on all sides.
// x_start = position0 (step start), x_trial = position (current Newton trial).
// One thread per triangle. Portable @workgroup_size(64), no subgroups.
//
// Static vs dynamic split: triangle indices are STATIC (uploaded once);
// triAabb + triCentroid are DYNAMIC (rewritten every Newton iteration without
// touching any static structure).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> position0 : array<vec4f>; // x_start
@group(0) @binding(2) var<storage, read> position : array<vec4f>;  // x_trial
@group(0) @binding(3) var<storage, read> triangles : array<u32>;   // STATIC
@group(0) @binding(4) var<storage, read_write> triAabb : array<f32>; // DYNAMIC m*6: minX..maxZ
@group(0) @binding(5) var<storage, read_write> triCentroid : array<vec4f>; // DYNAMIC m
@group(0) @binding(6) var<uniform> pad : f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x + gid.y * 4194240u;
  if (t >= params.triangleCount) { return; }
  let i0 = triangles[t * 3u];
  let i1 = triangles[t * 3u + 1u];
  let i2 = triangles[t * 3u + 2u];
  // 6-point swept min/max: same segment the CPU CCD validates.
  var mn = min(min(position0[i0].xyz, position[i0].xyz),
           min(min(position0[i1].xyz, position[i1].xyz),
               min(position0[i2].xyz, position[i2].xyz)));
  var mx = max(max(position0[i0].xyz, position[i0].xyz),
           max(max(position0[i1].xyz, position[i1].xyz),
               max(position0[i2].xyz, position[i2].xyz)));
  mn -= vec3f(pad);
  mx += vec3f(pad);
  triAabb[t * 6u] = mn.x;     triAabb[t * 6u + 1u] = mn.y; triAabb[t * 6u + 2u] = mn.z;
  triAabb[t * 6u + 3u] = mx.x; triAabb[t * 6u + 4u] = mx.y; triAabb[t * 6u + 5u] = mx.z;
  // Swept centroid: 6-point average, same as TriBvh.build (x0 AND x1).
  let c = (position0[i0].xyz + position[i0].xyz
         + position0[i1].xyz + position[i1].xyz
         + position0[i2].xyz + position[i2].xyz) / 6.0;
  triCentroid[t] = vec4f(c, 0.0);
}
