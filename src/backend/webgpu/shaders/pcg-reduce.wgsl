// G0 PCG dot-product reduction — portable two-stage workgroup sum.
// Stage 1 (reduce_stage1): each workgroup sums 64 inputs from `src` into one
// partial in `partial[workgroupId]` using workgroup-shared memory.
// Stage 2 (reduce_stage2): single workgroup sums `partial[0..numGroups)`.
// No subgroups, no atomics on the fast path. numGroups <= 4096 for G0 scenes.

var<workgroup> tile : array<f32, 64>;

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> partial : array<f32>;
@group(0) @binding(2) var<uniform> count : u32;

@compute @workgroup_size(64)
fn reduce_stage1(
  @builtin(global_invocation_id) gid : vec3u,
  @builtin(local_invocation_id) lid : vec3u,
  @builtin(workgroup_id) wid : vec3u,
) {
  let i = gid.x + gid.y * 4194240u;
  var v = 0.0;
  if (i < count) { v = src[i]; }
  tile[lid.x] = v;
  workgroupBarrier();
  // tree reduction in shared memory (sequential addressing, bank-safe)
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { tile[lid.x] += tile[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[wid.x] = tile[0]; }
}

@group(0) @binding(3) var<uniform> numGroups : u32;

@compute @workgroup_size(64)
fn reduce_stage2(
  @builtin(local_invocation_id) lid : vec3u,
) {
  var acc = 0.0;
  for (var g = lid.x; g < numGroups; g += 64u) {
    acc += partial[g];
  }
  tile[lid.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { tile[lid.x] += tile[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[0] = tile[0]; }
}

// G3 max-reduction (trust-region cap): same two-stage pattern with max.
// Out-of-range lanes seed -1e30 (all geometries are far above it).
@compute @workgroup_size(64)
fn reduce_max_stage1(
  @builtin(global_invocation_id) gid : vec3u,
  @builtin(local_invocation_id) lid : vec3u,
  @builtin(workgroup_id) wid : vec3u,
) {
  let i = gid.x + gid.y * 4194240u;
  var v = -1e30;
  if (i < count) { v = src[i]; }
  tile[lid.x] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { tile[lid.x] = max(tile[lid.x], tile[lid.x + s]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[wid.x] = tile[0]; }
}

@compute @workgroup_size(64)
fn reduce_max_stage2(
  @builtin(local_invocation_id) lid : vec3u,
) {
  var acc = -1e30;
  for (var g = lid.x; g < numGroups; g += 64u) {
    acc = max(acc, partial[g]);
  }
  tile[lid.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { tile[lid.x] = max(tile[lid.x], tile[lid.x + s]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[0] = tile[0]; }
}
