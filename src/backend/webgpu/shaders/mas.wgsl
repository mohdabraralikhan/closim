// G5D two-level MAS coarse correction (single damped-Jacobi sweep, GPU mirror
// of mas.ts buildMasTwoLevel/applyMasTwoLevel — NO extra HVP):
//   z = Schwarz1(r) + P (omega * Dc^-1 (R r)),  Dc = diag(R diag(H) P)
// Kernels (all group 0):
//   mas_restrict      : coarse[dc] = sum_{v in d} fine[v*3+a]/sqrt(|d|)
//   mas_coarse_scale  : cz[i] = omega * cr[i]/cd[i] (guarded)
//   mas_prolongate_add: fine[v*3+a] += coarse[d*3+a]/sqrt(|d|), pin-filtered
//   mas_contact_span  : counts active contacts spanning >1 aggregate (atomic)
// Vector buffers are rebound per use (diag->coarseDiag once per solve;
// residual->coarseR every PCG iteration). Coarse vectors have no pins; the
// prolongation filters them back out.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> masFine : array<f32>; // n*3 input
@group(0) @binding(2) var<storage, read_write> masCoarse : array<f32>; // nDoms*3 output
@group(0) @binding(3) var<storage, read> masVerts : array<u32>; // nDoms*8
@group(0) @binding(4) var<uniform> masCount : u32; // coarse dof count (nDoms*3)

@compute @workgroup_size(64)
fn mas_restrict(@builtin(global_invocation_id) gid : vec3u) {
  let dc = gid.x + gid.y * 4194240u;
  if (dc >= masCount) { return; }
  let d = dc / 3u;
  let a = dc % 3u;
  var cnt = 0u;
  for (var l = 0u; l < 8u; l++) {
    if (masVerts[d * 8u + l] != 0xffffffffu) { cnt++; }
  }
  var s = 0.0;
  if (cnt > 0u) { s = 1.0 / sqrt(f32(cnt)); }
  var acc = 0.0;
  for (var l = 0u; l < 8u; l++) {
    let w = masVerts[d * 8u + l];
    if (w == 0xffffffffu) { continue; }
    acc += masFine[w * 3u + a];
  }
  masCoarse[dc] = acc * s;
}

@group(0) @binding(10) var<uniform> masOmega : f32;
@group(0) @binding(11) var<storage, read> masCr : array<f32>;
@group(0) @binding(12) var<storage, read> masCd : array<f32>;
@group(0) @binding(13) var<storage, read_write> masCz : array<f32>;
@group(0) @binding(14) var<uniform> masCountS : u32;

@compute @workgroup_size(64)
fn mas_coarse_scale(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= masCountS) { return; }
  let dd = masCd[i];
  var v = 0.0;
  if (dd > 1e-12) { v = masOmega * masCr[i] / dd; }
  masCz[i] = v;
}

@group(0) @binding(20) var<storage, read> pCoarse : array<f32>; // nDoms*3
@group(0) @binding(21) var<storage, read_write> pFine : array<f32>; // n*3 (z accumulates)
@group(0) @binding(22) var<storage, read> pVerts : array<u32>; // nDoms*8 (|d| scan)
@group(0) @binding(23) var<storage, read> pPin : array<u32>; // n
@group(0) @binding(24) var<storage, read> pDomain : array<u32>; // n vertex -> domain

// True coarse diagonal from the assembled local matrices:
// Dc[(d,a)] = sum of component-a entries of schwarzMat[d] / |d|.
// schwarzMat already holds membrane + embedded diag (symmetrized), so this
// equals diag(R A P) including intra-aggregate membrane curvature. Padded
// (identity) rows are skipped.
@group(0) @binding(40) var<storage, read_write> cDiag : array<f32>; // nDoms*3
@group(0) @binding(41) var<storage, read> cMat : array<f32>; // nDoms*576
@group(0) @binding(42) var<storage, read> cVerts : array<u32>; // nDoms*8
@group(0) @binding(43) var<uniform> cCount : u32; // nDoms*3

@compute @workgroup_size(64)
fn mas_coarse_diag(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= cCount) { return; }
  let d = i / 3u;
  let a = i % 3u;
  var cnt = 0u;
  for (var l = 0u; l < 8u; l++) {
    if (cVerts[d * 8u + l] != 0xffffffffu) { cnt++; }
  }
  var s = 0.0;
  for (var l1 = 0u; l1 < 8u; l1++) {
    if (cVerts[d * 8u + l1] == 0xffffffffu) { continue; }
    for (var l2 = 0u; l2 < 8u; l2++) {
      if (cVerts[d * 8u + l2] == 0xffffffffu) { continue; }
      s += cMat[d * 576u + (l1 * 3u + a) * 24u + (l2 * 3u + a)];
    }
  }
  var v = 0.0;
  if (cnt > 0u) { v = s / f32(cnt); }
  cDiag[i] = v;
}

// Cross-aggregate contact coverage: one thread per contact slot; counts the
// contact when its cloth verts (id < vertexCount excludes the static tail)
// span more than one Schwarz aggregate. Floor contacts involve one vert and
// never span. Run once per rebuild when the coarse path is enabled; the CPU
// reads the 4 B counter alongside contactCount (no extra sync point).
@group(0) @binding(50) var<storage, read> spN : array<vec4f>; // (nx,ny,nz,kind)
@group(0) @binding(51) var<storage, read> spId : array<vec4u>;
@group(0) @binding(52) var<storage, read> spDomain : array<u32>; // n vertex -> domain
@group(0) @binding(53) var<storage, read_write> spCount : array<atomic<u32>>; // masContactSpan
@group(0) @binding(54) var<storage, read_write> spActive : array<atomic<u32>>; // contactCount (loaded only)

@compute @workgroup_size(64)
fn mas_contact_span(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= atomicLoad(&spActive[0])) { return; }
  let kind = spN[i].w;
  let id = spId[i];
  var d0 = 0xffffffffu;
  var spanned = false;
  // check up to 4 stencil verts; static-tail ids (>= vertexCount) are skipped
  for (var k = 0u; k < 4u; k++) {
    var vv = 0xffffffffu;
    if (k == 0u) { vv = id.x; }
    else if (k == 1u) { vv = id.y; }
    else if (k == 2u) { vv = id.z; }
    else { vv = id.w; }
    if (vv >= params.vertexCount) { continue; }
    // floor contacts carry a single vert (id.y/z/w are 0xffffffff)
    if (kind > 1.5 && k > 0u) { continue; }
    let dd = spDomain[vv];
    if (d0 == 0xffffffffu) { d0 = dd; }
    else if (dd != d0) { spanned = true; break; }
  }
  if (spanned) {
    atomicAdd(&spCount[0], 1u);
  }
}

@compute @workgroup_size(64)
fn mas_prolongate_add(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let v = i / 3u;
  if (pPin[v] != 0u) { return; }
  let a = i % 3u;
  let d = pDomain[v];
  var cnt = 0u;
  for (var l = 0u; l < 8u; l++) {
    if (pVerts[d * 8u + l] != 0xffffffffu) { cnt++; }
  }
  var s = 0.0;
  if (cnt > 0u) { s = 1.0 / sqrt(f32(cnt)); }
  pFine[i] += pCoarse[d * 3u + a] * s;
}
