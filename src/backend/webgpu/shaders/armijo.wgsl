// G6B GPU-batched Armijo line search.
//
// One batch evaluates K alpha candidates (K=2/4/8) with zero per-trial CPU
// traffic: apply_k -> G1/G2/FEM/barrier/friction/diagnostics (existing passes,
// cap-bounded with sentinel-cleared records) -> armijo_record -> ... ->
// armijo_select -> ONE 64 B status readback. The CPU then commits at the
// selected alpha (existing applyTrial/evaluate/accept path) or schedules the
// next batch. Rejected trials never commit state or friction (commit only
// runs on the accepted re-evaluation, exactly like the CPU solver).
//
// Why indexed apply_0..7 entries: WebGPU has no per-dispatch parameters
// except uniforms, and every uniform write flushes the encoder (a submit).
// Hardcoded candidate indices keep the whole batch in flight with zero
// uniform rewrites. armijoCur (written by apply_k, thread 0) tells the
// generic record kernel which slot just ran.
//
// Candidate record (8 f32): [alpha, energy, gradNorm, armijoOk,
//   minDist, minToi, finite01, overflow01].
// armijoStatus (16 f32): [acceptedFlag, selectedAlpha, selectedIndex,
//   trialsEvaluated, armijoFails, ccdFails, barrierFails, overflowFails,
//   finite, energy, gradNorm, pcgBreakdown, newtonConverged, minDist, minToi,
//   0]. Lanes 13-14 carry the selected (or fallback) row's distance/TOI so
// commit bookkeeping needs no re-evaluation read.
// On reject-all: selectedIndex = 0xffffffff, energy/gradNorm = min-finite-
// energy row (diagnostic context only; the Newton iterate then fails exactly
// like the CPU path exhausting its trial budget).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(2) var<storage, read> searchDirection : array<f32>; // n*3 dx
@group(0) @binding(4) var<storage, read> position : array<vec4f>; // xAccepted
@group(0) @binding(5) var<storage, read_write> xTrial : array<vec4f>;
@group(0) @binding(6) var<storage, read> pinMask : array<u32>;
@group(0) @binding(7) var<storage, read> pinPos : array<vec4f>;
@group(0) @binding(8) var<storage, read> armijoAlphas : array<f32>; // K effective alphas (trust folded)
@group(0) @binding(9) var<storage, read_write> armijoCur : array<u32>;

@group(0) @binding(10) var<storage, read> trustScale : array<f32>; // G6C.2 trust mirror (1.0 in G6B flows)

fn applyIndexed(gid : vec3u, k : u32) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  if (v == 0u) { armijoCur[0] = k; }
  // Trust folds here (not on CPU): G6B writes effective alphas with
  // trustScale == 1.0 so xTrial is bit-identical to the folded path.
  let a = armijoAlphas[k] * trustScale[0];
  var p = position[v].xyz + a * vec3f(
    searchDirection[v * 3u], searchDirection[v * 3u + 1u], searchDirection[v * 3u + 2u]);
  if (pinMask[v] != 0u) { p = pinPos[v].xyz; }
  xTrial[v] = vec4f(p, 0.0);
}

@compute @workgroup_size(64)
fn apply_0(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 0u); }
@compute @workgroup_size(64)
fn apply_1(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 1u); }
@compute @workgroup_size(64)
fn apply_2(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 2u); }
@compute @workgroup_size(64)
fn apply_3(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 3u); }
@compute @workgroup_size(64)
fn apply_4(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 4u); }
@compute @workgroup_size(64)
fn apply_5(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 5u); }
@compute @workgroup_size(64)
fn apply_6(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 6u); }
@compute @workgroup_size(64)
fn apply_7(@builtin(global_invocation_id) gid : vec3u) { applyIndexed(gid, 7u); }

// Per-candidate contact-state clear, split in two: Dawn caps storage buffers
// at 16 per stage, so records (9) and counters (9) clear separately.
// Records: sentinel-clears so cap-bounded batch loops see un-appended slots
// as absent. Sentinels: dist=1e30 (min-safe, barrier-skipped), toi=2.0
// (NO_HIT), everything else 0 (prm.x==0 marks inactive for friction guard).
// Counters: plain atomic stores (single thread, ordered by pass boundary).
@group(0) @binding(20) var<storage, read_write> clW : array<vec4f>;
@group(0) @binding(21) var<storage, read_write> clN : array<vec4f>;
@group(0) @binding(22) var<storage, read_write> clId : array<vec4u>;
@group(0) @binding(23) var<storage, read_write> clPrm : array<vec4f>;
@group(0) @binding(24) var<storage, read_write> clDist : array<f32>;
@group(0) @binding(25) var<storage, read_write> clTOI : array<f32>;
@group(0) @binding(26) var<storage, read_write> clEnergy : array<f32>;
@group(0) @binding(27) var<storage, read_write> clScratch : array<f32>; // cap*8
@group(0) @binding(28) var<storage, read_write> clFriction : array<vec4f>;

@compute @workgroup_size(64)
fn armijo_clear_records(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= params.contactCount) { return; }
  clW[i] = vec4f(0.0);
  clN[i] = vec4f(0.0);
  clId[i] = vec4u(0u);
  clPrm[i] = vec4f(0.0);
  clDist[i] = 1e30;
  clTOI[i] = 2.0;
  clEnergy[i] = 0.0;
  for (var k = 0u; k < 8u; k++) { clScratch[i * 8u + k] = 0.0; }
  clFriction[i] = vec4f(0.0);
}

@group(0) @binding(29) var<storage, read_write> cPair : array<atomic<u32>>;
@group(0) @binding(30) var<storage, read_write> cPairOverflow : array<atomic<u32>>;
@group(0) @binding(31) var<storage, read_write> cPairScanned : array<atomic<u32>>;
@group(0) @binding(32) var<storage, read_write> cPrimVT : array<atomic<u32>>;
@group(0) @binding(33) var<storage, read_write> cPrimEE : array<atomic<u32>>;
@group(0) @binding(34) var<storage, read_write> cContact : array<atomic<u32>>;
@group(0) @binding(35) var<storage, read_write> cContactOverflow : array<atomic<u32>>;
@group(0) @binding(36) var<storage, read_write> cContactScanned : array<atomic<u32>>;
@group(0) @binding(37) var<storage, read_write> cContactFail : array<atomic<u32>>;

@compute @workgroup_size(64)
fn armijo_clear_counts(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  atomicStore(&cPair[0], 0u);
  atomicStore(&cPairOverflow[0], 0u);
  atomicStore(&cPairScanned[0], 0u);
  atomicStore(&cPrimVT[0], 0u);
  atomicStore(&cPrimEE[0], 0u);
  atomicStore(&cContact[0], 0u);
  atomicStore(&cContactOverflow[0], 0u);
  atomicStore(&cContactScanned[0], 0u);
  atomicStore(&cContactFail[0], 0u);
}

// Record one candidate's verdict into its slot + accumulate batch counters.
// Validity mirrors the CPU Armijo gate exactly (finite && ccdSafe &&
// minDistance > dMin && !(0 < toi < 1) && energy finite && no overflow),
// plus the sufficient-decrease inequality. NaN comparisons are false, so
// nonfinite states reject exactly like the CPU `continue` path. Overflow is
// GPU-stricter than the CPU path by design (conservative; documented).
@group(0) @binding(40) var<storage, read> rStatus : array<f32>; // 16 solverStatus lanes
@group(0) @binding(41) var<storage, read> rOverflow : array<u32>;
@group(0) @binding(42) var<storage, read_write> rCandidates : array<f32>; // 8*8
@group(0) @binding(43) var<storage, read_write> rBatch : array<f32>; // 16 armijoStatus
// NOTE: current candidate index comes from armijoCur (binding 9, shared with
// the apply entries) — no separate binding here.
@group(0) @binding(45) var<storage, read> rE0s : array<f32>; // e0Store mirror (GPU-maintained)
@group(0) @binding(46) var<storage, read> rGtdxs : array<f32>; // gtdxStore mirror (GPU-maintained)
@group(0) @binding(47) var<uniform> rDMin : f32;
@group(0) @binding(48) var<storage, read> rTrust : array<f32>; // trustScaleStore mirror

@compute @workgroup_size(64)
fn armijo_record(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let k = armijoCur[0];
  // Effective alpha folds trust here (was CPU-folded in G6B flows; the G6B
  // batch writes trustScaleStore=1.0 so xTrial stays bit-identical).
  let alpha = armijoAlphas[k] * rTrust[0];
  let rE0 = rE0s[0];
  let rGtdx = rGtdxs[0];
  let e = rStatus[0];
  let gnorm = rStatus[2];
  let dist = rStatus[4];
  let toi = rStatus[5];
  let fin = rStatus[6] > 0.5;
  let ccd = rStatus[7] > 0.5;
  let eOk = (e == e);
  let finiteOk = fin && eOk;
  let ccdOk = ccd && !(toi > 0.0 && toi < 1.0 - 1e-9);
  let barrierOk = dist > rDMin;
  let overflow = rOverflow[0] != 0u;
  let geomOk = finiteOk && ccdOk && barrierOk && !overflow;
  let ineq = geomOk && (e <= rE0 + 1e-4 * alpha * rGtdx);
  rCandidates[k * 8u] = alpha;
  rCandidates[k * 8u + 1u] = e;
  rCandidates[k * 8u + 2u] = gnorm;
  if (ineq) { rCandidates[k * 8u + 3u] = 1.0; } else { rCandidates[k * 8u + 3u] = 0.0; }
  rCandidates[k * 8u + 4u] = dist;
  rCandidates[k * 8u + 5u] = toi;
  if (finiteOk) { rCandidates[k * 8u + 6u] = 1.0; } else { rCandidates[k * 8u + 6u] = 0.0; }
  if (overflow) { rCandidates[k * 8u + 7u] = 1.0; } else { rCandidates[k * 8u + 7u] = 0.0; }
  rBatch[3] += 1.0; // trialsEvaluated
  if (geomOk && !ineq) { rBatch[4] += 1.0; } // armijoFails
  if (!ccdOk) { rBatch[5] += 1.0; } // ccdFails
  if (!barrierOk) { rBatch[6] += 1.0; } // barrierFails
  if (overflow) { rBatch[7] += 1.0; } // overflowFails
  if (!finiteOk) { rBatch[8] = 0.0; } // finite lane (init 1.0 per batch)
}

// First-valid-wins selection (largest alpha first — candidates run in order).
// Writes the compact batch status; the CPU commits at selectedAlpha via the
// existing applyTrial/evaluate/accept path or schedules the next batch.
@group(0) @binding(50) var<storage, read> sCandidates : array<f32>; // 8*8
@group(0) @binding(51) var<storage, read_write> sBatch : array<f32>; // 16
@group(0) @binding(52) var<uniform> sK : u32;
@group(0) @binding(53) var<uniform> sPcgBd : f32;
@group(0) @binding(54) var<uniform> sNewtonConv : f32;

@compute @workgroup_size(64)
fn armijo_select(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  var best = 0xffffffffu;
  for (var k = 0u; k < sK; k++) {
    if (sCandidates[k * 8u + 3u] > 0.5) { best = k; break; }
  }
  if (best != 0xffffffffu) {
    sBatch[0] = 1.0;
    sBatch[1] = sCandidates[best * 8u];
    sBatch[2] = f32(best);
    sBatch[9] = sCandidates[best * 8u + 1u];
    sBatch[10] = sCandidates[best * 8u + 2u];
    sBatch[13] = sCandidates[best * 8u + 4u];
    sBatch[14] = sCandidates[best * 8u + 5u];
  } else {
    // Reject-all: min-finite-energy row as diagnostic context (the Newton
    // iterate then fails exactly like the CPU budget-exhaustion path).
    sBatch[0] = 0.0;
    sBatch[1] = 0.0;
    sBatch[2] = 4294967295.0;
    var be = 1e30;
    var bg = 0.0;
    var bd = 1e30;
    var bt = 2.0;
    for (var k = 0u; k < sK; k++) {
      let fe = sCandidates[k * 8u + 6u] > 0.5;
      let ek = sCandidates[k * 8u + 1u];
      if (fe && ek < be) {
        be = ek; bg = sCandidates[k * 8u + 2u];
        bd = sCandidates[k * 8u + 4u]; bt = sCandidates[k * 8u + 5u];
      }
    }
    if (be >= 1e30) { be = 0.0; }
    sBatch[9] = be;
    sBatch[10] = bg;
    sBatch[13] = bd;
    sBatch[14] = bt;
  }
  sBatch[11] = sPcgBd;
  sBatch[12] = sNewtonConv;
}
