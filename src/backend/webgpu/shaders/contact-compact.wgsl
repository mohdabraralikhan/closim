// G2 contact compaction: candidate pairs + per-primitive CCD scratch ->
// compact 64 B frozen records for the barrier, plus counters.
//
// Three entries, dispatched in order (all GPU->GPU, no readback between):
//
//   expand_pairs  : one thread per G1 triangle pair; writes up to 6 VT ids +
//                   up to 9 EE ids (endpoint-sharing EE skipped, same rule as
//                   CPU expandTriPair) into primIdsVT/primIdsEE via atomics.
//   compact_pairs : one thread per EXPANDED primitive; reads closest scratch
//                   (s/t/dist/r at x1) + CCD scratch (toi/flag); classifies
//                   1e-12 <= dist < dHat; appends the 64 B record (w/n/id/prm)
//                   + toi/dist sidecars via atomic append. flag == 3 (failure)
//                   is NEVER appended and raises failFlag (trial-invalid,
//                   mirroring CPU checkTrial on NaN toi).
//   compact_floor : one thread per cloth vertex; d = y - floorY; active when
//                   d < dHat (penetrating included, barrier pushes them out);
//                   floorY rides in prm.z (D4).
//
// Overflow: append idx >= capacity -> contactOverflow = 1 via atomicMax, no
// write past capacity. scannedCount counts every evaluated primitive so the
// overflow ratio is exact. Never silently truncate.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
// topology (STATIC)
@group(0) @binding(1) var<storage, read> triangles : array<u32>;
@group(0) @binding(2) var<storage, read> candidatePairs : array<u32>; // G1 pairs, 2 u32 each
@group(0) @binding(3) var<storage, read> pairTotal : array<u32>; // pairCount (atomic view)
// primitive id buffers (DYNAMIC, sized pairCapacity * 15)
@group(0) @binding(4) var<storage, read_write> primIdsVT : array<vec4u>;
@group(0) @binding(5) var<storage, read_write> primIdsEE : array<vec4u>;
@group(0) @binding(6) var<storage, read_write> primCountVT : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> primCountEE : array<atomic<u32>>;
// closest scratch at x1 (from closest-vt/ee.wgsl)
@group(0) @binding(8) var<storage, read> vtSTD : array<vec4f>; // (s,t,dist,0) at x1
@group(0) @binding(11) var<storage, read> vtR : array<vec4f>;
@group(0) @binding(12) var<storage, read> eeSTD : array<vec4f>;
@group(0) @binding(15) var<storage, read> eeR : array<vec4f>;
// CCD scratch (from ccd-vt/ee.wgsl)
@group(0) @binding(16) var<storage, read> vtTOI : array<f32>;
@group(0) @binding(17) var<storage, read> vtFlag : array<u32>;
@group(0) @binding(18) var<storage, read> eeTOI : array<f32>;
@group(0) @binding(19) var<storage, read> eeFlag : array<u32>;
// compact outputs: 64 B frozen records + sidecars
@group(0) @binding(20) var<storage, read_write> contactW : array<vec4f>;
@group(0) @binding(21) var<storage, read_write> contactN : array<vec4f>; // (nx,ny,nz,kind)
@group(0) @binding(22) var<storage, read_write> contactId : array<vec4u>;
@group(0) @binding(23) var<storage, read_write> contactPrm : array<vec4f>; // (dHat,kappa,mu/floorY,eps)
@group(0) @binding(24) var<storage, read_write> contactTOI : array<f32>;
@group(0) @binding(25) var<storage, read_write> contactDist : array<f32>;
@group(0) @binding(26) var<storage, read_write> contactCount : array<atomic<u32>>;
@group(0) @binding(27) var<storage, read_write> contactOverflow : array<atomic<u32>>;
@group(0) @binding(28) var<storage, read_write> contactScanned : array<atomic<u32>>;
@group(0) @binding(29) var<storage, read_write> contactFail : array<atomic<u32>>;
@group(0) @binding(30) var<uniform> contactCapacity : u32;
@group(0) @binding(31) var<uniform> dHat : f32;
@group(0) @binding(32) var<uniform> kappa : f32;
@group(0) @binding(33) var<uniform> mu : f32;
@group(0) @binding(34) var<uniform> fricEps : f32;
@group(0) @binding(35) var<storage, read> posTrial : array<vec4f>; // x1 (floor d)
@group(0) @binding(36) var<uniform> floorY : f32;
@group(0) @binding(37) var<uniform> floorOn : u32;
@group(0) @binding(38) var<uniform> dMin : f32; // CCD thickness (resting iff d <= dMin)
@group(0) @binding(39) var<storage, read> vtTotal : array<u32>; // primCountVT (atomic view)
@group(0) @binding(40) var<storage, read> eeTotal : array<u32>; // primCountEE (atomic view)

// Appends one classified record. Callers bump contactScanned exactly once per
// evaluated primitive (top of each entry); appends only bump contactCount.
fn append_record(
  kind : f32, w : vec4f, n : vec3f,
  i0 : u32, i1 : u32, i2 : u32, i3 : u32,
  prmZ : f32, toi : f32, dist : f32,
) {
  let idx = atomicAdd(&contactCount[0], 1u);
  if (idx >= contactCapacity) {
    atomicMax(&contactOverflow[0], 1u);
    return;
  }
  contactW[idx] = w;
  contactN[idx] = vec4f(n, kind);
  contactId[idx] = vec4u(i0, i1, i2, i3);
  contactPrm[idx] = vec4f(dHat, kappa, prmZ, fricEps);
  contactTOI[idx] = toi;
  contactDist[idx] = dist;
}

@compute @workgroup_size(64)
fn expand_pairs(@builtin(global_invocation_id) gid : vec3u) {
  let q = gid.x + gid.y * 4194240u;
  if (q >= pairTotal[0]) { return; }
  let tA = candidatePairs[q * 2u];
  let tB = candidatePairs[q * 2u + 1u];
  let A0 = triangles[tA * 3u]; let A1 = triangles[tA * 3u + 1u]; let A2 = triangles[tA * 3u + 2u];
  let B0 = triangles[tB * 3u]; let B1 = triangles[tB * 3u + 1u]; let B2 = triangles[tB * 3u + 2u];
  // 6 VT: A verts vs B tri, then B verts vs A tri (CPU expandTriPair order)
  let vA : array<u32, 3> = array<u32, 3>(A0, A1, A2);
  let vB : array<u32, 3> = array<u32, 3>(B0, B1, B2);
  for (var k = 0u; k < 3u; k++) {
    let i = atomicAdd(&primCountVT[0], 1u);
    primIdsVT[i] = vec4u(vA[k], B0, B1, B2);
  }
  for (var k = 0u; k < 3u; k++) {
    let i = atomicAdd(&primCountVT[0], 1u);
    primIdsVT[i] = vec4u(vB[k], A0, A1, A2);
  }
  // up to 9 EE, skipping endpoint-sharing pairs
  let eA : array<u32, 6> = array<u32, 6>(A0, A1, A1, A2, A2, A0);
  let eB : array<u32, 6> = array<u32, 6>(B0, B1, B1, B2, B2, B0);
  for (var ea = 0u; ea < 3u; ea++) {
    for (var eb = 0u; eb < 3u; eb++) {
      let a = eA[ea * 2u]; let b = eA[ea * 2u + 1u];
      let c = eB[eb * 2u]; let d = eB[eb * 2u + 1u];
      if (a == c || a == d || b == c || b == d) { continue; }
      let i = atomicAdd(&primCountEE[0], 1u);
      primIdsEE[i] = vec4u(a, b, c, d);
    }
  }
}

@compute @workgroup_size(64)
fn compact_pairs(@builtin(global_invocation_id) gid : vec3u) {
  // One dispatch over the VT primitive range [0, vtTotal[0]).
  let i = gid.x + gid.y * 4194240u;
  if (i >= vtTotal[0]) { return; }
  atomicAdd(&contactScanned[0], 1u);
  let vst = vtSTD[i];
  let dist = vst.z;
  let flag = vtFlag[i];
  if (flag == 3u) {
    atomicMax(&contactFail[0], 1u);
    return;
  }
  if (!(dist >= 1e-12) || dist >= dHat) { return; }
  let r = vtR[i].xyz;
  let n = r / dist;
  let s = vst.x; let t = vst.y;
  let w0 = 1.0 - s - t;
  let id = primIdsVT[i];
  append_record(0.0, vec4f(w0, s, t, 0.0), n, id.x, id.y, id.z, id.w, mu, vtTOI[i], dist);
}

@compute @workgroup_size(64)
fn compact_pairs_ee(@builtin(global_invocation_id) gid : vec3u) {
  // One dispatch over the EE primitive range [0, eeTotal[0]).
  let i = gid.x + gid.y * 4194240u;
  if (i >= eeTotal[0]) { return; }
  atomicAdd(&contactScanned[0], 1u);
  let est = eeSTD[i];
  let dist = est.z;
  let flag = eeFlag[i];
  if (flag == 3u) {
    atomicMax(&contactFail[0], 1u);
    return;
  }
  if (!(dist >= 1e-12) || dist >= dHat) { return; }
  let r = eeR[i].xyz;
  let n = r / dist;
  let s = est.x; let t = est.y;
  let id = primIdsEE[i];
  append_record(1.0, vec4f(1.0 - s, s, -(1.0 - t), -t), n, id.x, id.y, id.z, id.w, mu, eeTOI[i], dist);
}

@compute @workgroup_size(64)
fn compact_floor(@builtin(global_invocation_id) gid : vec3u) {
  if (floorOn == 0u) { return; }
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  let d = posTrial[v].y - floorY;
  atomicAdd(&contactScanned[0], 1u);
  if (d >= dHat) { return; }
  // penetrating vertices included (d may be negative); barrier pushes out.
  let idx = atomicAdd(&contactCount[0], 1u);
  if (idx >= contactCapacity) {
    atomicMax(&contactOverflow[0], 1u);
    return;
  }
  contactW[idx] = vec4f(0.0, 0.0, 0.0, 0.0);
  contactN[idx] = vec4f(0.0, 1.0, 0.0, 2.0);
  contactId[idx] = vec4u(v, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  contactPrm[idx] = vec4f(dHat, kappa, floorY, fricEps); // D4: floorY in prm.z
  contactTOI[idx] = select(2.0, 0.0, d <= dMin); // 2.0 = NO_HIT sentinel
  contactDist[idx] = d;
}
