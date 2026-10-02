// G1.2 candidate generation — LBVH self-query with adjacency exclusion.
//
// One thread per QUERY LEAF (sorted order s in [0, m)). Each thread walks the
// LBVH with a private fixed-depth stack (64 entries — tree depth for G1 sizes
// is < 32; deeper trees are a non-issue after the depth assert in the mirror).
// At each leaf L != query leaf with payload triId greater than the query triId
// (dedupe: emit only a < b), test swept-AABB overlap (inclusive, same as CPU
// `overlaps`), then test the adjacency exclusion set.
//
// Exclusion encoding: sorted (lo,hi) u32 pairs (vec2u), binary-searched per
// component. No triCount ceiling (G2 supersedes the packed-u32 path).
//
// Two-stage output (fixed/oversized buffer + atomic counter):
//   idx = atomicAdd(pairCount, 1); if (idx < capacity) write pairs[2*idx];
//   else atomicMax(overflowFlag, 1). scannedCount counts every emitted overlap
//   regardless of capacity so overflow is EXPLICIT — never silent truncation.
//
// Deterministic: pairs are unordered by traversal; the CPU mirror sorts before
// comparison and the G1->G2 handoff compares ordering-independently.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> lbvhMin : array<vec4f>;
@group(0) @binding(2) var<storage, read> lbvhMax : array<vec4f>;
@group(0) @binding(3) var<storage, read> lbvhChild : array<vec4u>;
@group(0) @binding(4) var<storage, read> mortonPayload : array<u32>;
@group(0) @binding(5) var<storage, read> exclusionKeys : array<vec2u>; // sorted (lo,hi)
@group(0) @binding(6) var<uniform> exclusionCount : u32;
@group(0) @binding(7) var<storage, read_write> candidatePairs : array<u32>; // cap*2
@group(0) @binding(8) var<storage, read_write> pairCount : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> overflowFlag : array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> scannedCount : array<atomic<u32>>;
@group(0) @binding(11) var<uniform> pairCapacity : u32;
@group(0) @binding(12) var<storage, read> lbvhRoot : array<u32>; // from lbvh_find_root

fn isExcluded(a : u32, b : u32) -> bool {
  let lo = min(a, b);
  let hi = max(a, b);
  var l = 0u;
  var h = exclusionCount;
  while (l < h) {
    let mid = (l + h) / 2u;
    let kv = exclusionKeys[mid];
    if (kv.x == lo && kv.y == hi) { return true; }
    if (kv.x < lo || (kv.x == lo && kv.y < hi)) { l = mid + 1u; } else { h = mid; }
  }
  return false;
}

fn aabbOverlap(qMin : vec3f, qMax : vec3f, nMin : vec3f, nMax : vec3f) -> bool {
  return qMin.x <= nMax.x && qMax.x >= nMin.x
      && qMin.y <= nMax.y && qMax.y >= nMin.y
      && qMin.z <= nMax.z && qMax.z >= nMin.z;
}

@compute @workgroup_size(64)
fn traverse(@builtin(global_invocation_id) gid : vec3u) {
  let m = params.triangleCount;
  let s = gid.x + gid.y * 4194240u;
  if (s >= m) { return; }
  let qTri = mortonPayload[s];
  let qSlot = m - 1u + s;
  let qMin = lbvhMin[qSlot].xyz;
  let qMax = lbvhMax[qSlot].xyz;
  var stack : array<u32, 64>;
  var sp = 0u;
  if (m == 1u) { return; } // single triangle: no pairs possible
  stack[0] = lbvhRoot[0]; // Karras root from lbvh_find_root (NOT node 0)
  sp = 1u;
  while (sp > 0u) {
    sp -= 1u;
    let node = stack[sp];
    let ch = lbvhChild[node];
    // node is internal here by construction (leaves are never pushed)
    let left = ch.x;
    let right = ch.y;
    // visit left
    if (aabbOverlap(qMin, qMax, lbvhMin[left].xyz, lbvhMax[left].xyz)) {
      if (lbvhChild[left].w == 1u) {
        let tL = lbvhChild[left].z;
        if (tL > qTri && !isExcluded(qTri, tL)) {
          atomicAdd(&scannedCount[0], 1u);
          let idx = atomicAdd(&pairCount[0], 1u);
          if (idx < pairCapacity) {
            candidatePairs[idx * 2u] = qTri;
            candidatePairs[idx * 2u + 1u] = tL;
          } else {
            atomicMax(&overflowFlag[0], 1u);
          }
        }
      } else if (sp < 64u) {
        stack[sp] = left;
        sp += 1u;
      }
    }
    // visit right
    if (aabbOverlap(qMin, qMax, lbvhMin[right].xyz, lbvhMax[right].xyz)) {
      if (lbvhChild[right].w == 1u) {
        let tR = lbvhChild[right].z;
        if (tR > qTri && !isExcluded(qTri, tR)) {
          atomicAdd(&scannedCount[0], 1u);
          let idx = atomicAdd(&pairCount[0], 1u);
          if (idx < pairCapacity) {
            candidatePairs[idx * 2u] = qTri;
            candidatePairs[idx * 2u + 1u] = tR;
          } else {
            atomicMax(&overflowFlag[0], 1u);
          }
        }
      } else if (sp < 64u) {
        stack[sp] = right;
        sp += 1u;
      }
    }
  }
}
