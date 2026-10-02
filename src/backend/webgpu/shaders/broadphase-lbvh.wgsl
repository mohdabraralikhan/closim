// G1.1 LBVH construction from sorted Morton keys (Karras 2012 radix tree).
// Reads: sorted mortonKeys + mortonPayload (leaf tri ids), triAabb.
// Writes: lbvhNodes — internal nodes [0, m-1) then leaves [m-1, 2m-2].
// Node record (64 B): minXYZ (12 B) + maxXYZ (12 B) + left/right/leafTri (12 B)
//   + padding. Leaf nodes store their triangle's swept AABB directly.
//
// Three dispatches, all GPU->GPU:
//   lbvh_seed_leaves: one thread per sorted leaf; copies its tri AABB.
//   lbvh_build: one thread per internal node; Karras range + split, links
//     children (child = leaf slot or internal id).
//   lbvh_refit: bottom-up bound merge, one dispatch per tree level high->low
//     (level count derives from CPU-known m — a sequence length, NOT a
//     readback). Production replaces this with a single atomic-refcount pass.
//
// Duplicate keys (coincident centroids) use the sorted-POSITION tiebreak in
// `deltaPos`, exactly as the CPU mirror does — every delta differs, so the
// tree is always a valid binary radix tree.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> mortonKeys : array<u32>;
@group(0) @binding(2) var<storage, read> mortonPayload : array<u32>;
@group(0) @binding(3) var<storage, read> triAabb : array<f32>; // m*6
@group(0) @binding(4) var<storage, read_write> lbvhMin : array<vec4f>; // node minXYZ
@group(0) @binding(5) var<storage, read_write> lbvhMax : array<vec4f>; // node maxXYZ
@group(0) @binding(6) var<storage, read_write> lbvhChild : array<vec4u>; // left, right, leafTri, isLeaf
@group(0) @binding(7) var<storage, read_write> lbvhRange : array<vec4u>; // lo, hi, 0, 0 (sorted-leaf range)
@group(0) @binding(8) var<storage, read_write> lbvhRoot : array<u32>; // root internal id (single)

// Longest-common-prefix with sorted-position tiebreak. Out-of-range -> -1
// (lower than any valid delta 0..63) so ranges terminate at the ends.
fn deltaPos(p : i32, q : i32, n : i32) -> i32 {
  if (q < 0 || q >= n) { return -1; }
  let kp = mortonKeys[u32(p)];
  let kq = mortonKeys[u32(q)];
  if (kp != kq) { return i32(countLeadingZeros(kp ^ kq)); }
  return 32 + i32(countLeadingZeros(u32(p) ^ u32(q)));
}

@compute @workgroup_size(64)
fn lbvh_seed_leaves(@builtin(global_invocation_id) gid : vec3u) {
  let m = params.triangleCount;
  let s = gid.x + gid.y * 4194240u;
  if (s >= m) { return; }
  let tri = mortonPayload[s];
  let slot = m - 1u + s;
  lbvhMin[slot] = vec4f(triAabb[tri * 6u], triAabb[tri * 6u + 1u], triAabb[tri * 6u + 2u], 0.0);
  lbvhMax[slot] = vec4f(triAabb[tri * 6u + 3u], triAabb[tri * 6u + 4u], triAabb[tri * 6u + 5u], 0.0);
  lbvhChild[slot] = vec4u(0xFFFFFFFFu, 0xFFFFFFFFu, tri, 1u);
}

@compute @workgroup_size(64)
fn lbvh_build(@builtin(global_invocation_id) gid : vec3u) {
  let m = i32(params.triangleCount);
  let i = i32(gid.x + gid.y * 4194240u);
  if (i >= m - 1) { return; }
  // Karras determineRange: expand toward the GREATER adjacent delta; dMin is
  // the delta on the side OPPOSITE expansion (near-side dMin collapses every
  // range to length 1 and no node covers [0, m-1] — tree left rootless).
  let dNext = deltaPos(i, i + 1, m);
  let dPrev = deltaPos(i, i - 1, m);
  var d = 1;
  var dMin = dPrev;
  if (dPrev > dNext) { d = -1; dMin = dNext; }
  var lMax = 2;
  while (deltaPos(i, i + d * lMax, m) > dMin) {
    lMax *= 2;
    if (lMax > m) { break; }
  }
  var l = 0;
  var t = lMax / 2;
  while (t > 0) {
    if (deltaPos(i, i + d * (l + t), m) > dMin) { l += t; }
    t /= 2;
  }
  let j = i + d * l; // Karras 2012: i + d*l (not l+1)
  let lo = min(i, j);
  let hi = max(i, j);
  // Karras findSplit: max s in [lo,hi) with delta(lo,s) > delta(lo,hi).
  // Strides MUST be powers of two (largest <= span); other sequences land on
  // a wrong split whose child ranges no longer nest -> disconnected forest.
  let dNode = deltaPos(lo, hi, m);
  var split = lo;
  var stride = 1;
  while (stride * 2 <= hi - lo) { stride *= 2; }
  while (stride > 0) {
    let mid = split + stride;
    if (mid < hi && deltaPos(lo, mid, m) > dNode) { split = mid; }
    stride /= 2;
  }
  let mu = u32(m);
  var left : u32;
  var right : u32;
  if (split == lo) { left = mu - 1u + u32(lo); } else { left = u32(split); }
  if (split + 1 == hi) { right = mu - 1u + u32(split + 1); } else { right = u32(split + 1); }
  lbvhChild[u32(i)] = vec4u(left, right, 0xFFFFFFFFu, 0u);
  lbvhRange[u32(i)] = vec4u(u32(lo), u32(hi), 0u, 0u);
}

@compute @workgroup_size(64)
fn lbvh_find_root(@builtin(global_invocation_id) gid : vec3u) {
  // The Karras root is the unique internal node with range [0, m-1].
  // Fully GPU-resident: no key/tree readback to discover it.
  let m = params.triangleCount;
  let i = gid.x + gid.y * 4194240u;
  if (m <= 1u || i >= m - 1u) { return; }
  let r = lbvhRange[i];
  if (r.x == 0u && r.y == m - 1u) {
    lbvhRoot[0] = i;
  }
}

@compute @workgroup_size(64)
fn lbvh_refit(@builtin(global_invocation_id) gid : vec3u) {
  let m = params.triangleCount;
  let i = gid.x + gid.y * 4194240u;
  if (i >= m - 1u) { return; }
  let ch = lbvhChild[i];
  let l = ch.x;
  let r = ch.y;
  let mn = min(lbvhMin[l].xyz, lbvhMin[r].xyz);
  let mx = max(lbvhMax[l].xyz, lbvhMax[r].xyz);
  lbvhMin[i] = vec4f(mn, 0.0);
  lbvhMax[i] = vec4f(mx, 0.0);
}
