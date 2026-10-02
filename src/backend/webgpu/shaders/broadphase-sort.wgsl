// G1.1 key sort — bitonic compare-swap over (mortonKeys, mortonPayload).
// Keeps the ENTIRE build GPU-resident: no key readback for a CPU sort.
// Standard Batcher network with XOR partners: stage k builds runs of length
// 2^k (block = 1<<stage), sub-passes compare distance 2^j (dist = 1<<substage)
// for j = k-1 down to 0. Portable (no subgroups, no atomics).
//
// Arbitrary triangle counts ride POWER-OF-TWO padded lanes: mortonKeys and
// mortonPayload are allocated for P = nextPow2(m) with the tail initialized
// once to key 0xFFFFFFFF / payload 0xFFFFFFFF. The network sorts all P lanes;
// +INF lanes sink to [m, P) and LBVH consumes only the first m (real, sorted)
// lanes. n uniform carries P; the kernel guards nothing (all P lanes valid).
//
// G1 validation sizes (<= ~2k tris) sort in microseconds; production scenes
// graduate to a radix sort without changing downstream consumers (sorted
// keys + payload in, LBVH build reads them).

@group(0) @binding(0) var<storage, read_write> mortonKeys : array<u32>;
@group(0) @binding(1) var<storage, read_write> mortonPayload : array<u32>;
@group(0) @binding(2) var<uniform> n : u32;      // key capacity P (padded pow2)
@group(0) @binding(3) var<uniform> stage : u32;  // k: run length 2^k
@group(0) @binding(4) var<uniform> substage : u32; // j: compare distance 2^j
// G6C.1 indexed path: pass parameters from a static table (built once per
// scene — depends only on P) stepped by a GPU-side cursor. Zero per-pass CPU
// uniform writes, so the whole sort stays in one encoder with no added
// submits. Same Batcher math as bitonic_sort_step (shared helper below).
@group(0) @binding(5) var<storage, read> sortParams : array<vec4u>; // (P, stage, sub, 0) per pass t
@group(0) @binding(6) var<storage, read_write> sortCursor : array<u32>; // [t]: sort_next advances, sort_step reads (t-1)

@compute @workgroup_size(64)
fn bitonic_sort_step(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= n) { return; }
  let block = 1u << stage;
  let dist = 1u << substage;
  let partner = i ^ dist;
  if (partner <= i || partner >= n) { return; } // lower thread acts, in range
  let ascending = ((i / block) % 2u) == 0u;
  let a = mortonKeys[i];
  let b = mortonKeys[partner];
  // strict weak order with id tiebreak (payloads unique incl. 0xFFFFFFFF
  // pad lanes): swap iff out of order. Comparisons parenthesized: old Tint
  // parses bare a<b inside select() as a template list.
  var swap = false;
  if (a != b) {
    swap = select((a < b), (a > b), ascending);
  } else {
    let pa = mortonPayload[i];
    let pb = mortonPayload[partner];
    swap = select((pa < pb), (pa > pb), ascending);
  }
  if (swap) {
    mortonKeys[i] = b;
    mortonKeys[partner] = a;
    let pa = mortonPayload[i];
    mortonPayload[i] = mortonPayload[partner];
    mortonPayload[partner] = pa;
  }
}

fn compareSwap(i : u32, nIn : u32, stageIn : u32, subIn : u32) {
  if (i >= nIn) { return; }
  let block = 1u << stageIn;
  let dist = 1u << subIn;
  let partner = i ^ dist;
  if (partner <= i || partner >= nIn) { return; } // lower thread acts, in range
  let ascending = ((i / block) % 2u) == 0u;
  let a = mortonKeys[i];
  let b = mortonKeys[partner];
  var swap = false;
  if (a != b) {
    swap = select((a < b), (a > b), ascending);
  } else {
    let pa = mortonPayload[i];
    let pb = mortonPayload[partner];
    swap = select((pa < pb), (pa > pb), ascending);
  }
  if (swap) {
    mortonKeys[i] = b;
    mortonKeys[partner] = a;
    let pa = mortonPayload[i];
    mortonPayload[i] = mortonPayload[partner];
    mortonPayload[partner] = pa;
  }
}

@compute @workgroup_size(64)
fn sort_next(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let t = sortCursor[0];
  sortCursor[0] = t + 1u;
}

@compute @workgroup_size(64)
fn sort_step_indexed(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  // The matching sort_next already advanced the cursor past our pass.
  let t = sortCursor[0] - 1u;
  let prm = sortParams[t];
  compareSwap(i, prm.x, prm.y, prm.z);
}
