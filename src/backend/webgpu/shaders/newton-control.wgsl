// G6C.2 GPU Newton control: per-round decisions without scalar readbacks.
// The driver still unrolls rounds/batches in JS (bounded statics only); every
// *decision* below runs on device and the CPU reads ONE compact newtonStatus
// per round (plus the once-per-step E0 eval and explicit debug snapshots).
// Submits stay roughly flat (same passes); syncs collapse to ~1 per round.
//
// Lanes (see NewtonStatusLane / NewtonCtlLane in gpu-buffers.ts):
//   newtonStatus[20]: iter, converged, failure, dirValid, pcgBd, accepted,
//     batchIdx, alpha, trialIdx, gradNorm, stepNorm, merit, energy, minDist,
//     minToi, overflow, ccdFail, barrierFail, residual, spare.
//   newtonCtl[16]: done, fallback, nbAccepted, doCommit, nbAlpha, nbBatch,
//     nbAcceptBatch, nbTrialBase, ...spare.
// Convention: single-thread entries (x:1) own all flag transitions; vector
// kernels never branch on flags except the two predicated commit entries.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;

// ---- pcg_report: residual norm + breakdown latch into status (no readback)
@group(0) @binding(10) var<storage, read> prReduce : array<f32>; // reduceScratch[0] = r.r
@group(0) @binding(11) var<storage, read> prBreak : array<f32>; // breakFlag[0]
@group(0) @binding(12) var<storage, read_write> prStatus : array<f32>; // newtonStatus

@compute @workgroup_size(64)
fn pcg_report(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  prStatus[18] = sqrt(max(prReduce[0], 0.0));
  var bd = 0.0;
  if (prBreak[0] != 0.0) { bd = 1.0; }
  prStatus[4] = bd;
}

// ---- trust_compute: trust scale from the device-side max (no readback)
// NaN-safe exactly like the CPU ternary (NaN > trust is false -> 1.0).
@group(0) @binding(20) var<storage, read> trReduce : array<f32>; // reduceScratch[0] = max|dx|
@group(0) @binding(21) var<storage, read_write> trScale : array<f32>; // trustScaleStore
@group(0) @binding(22) var<storage, read_write> trStatus : array<f32>; // newtonStatus (stepNorm lane)
@group(0) @binding(23) var<uniform> trTrust : f32; // NewtonTrust persistent slot

@compute @workgroup_size(64)
fn trust_compute(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let maxDx = trReduce[0];
  var s = 1.0;
  if (maxDx > trTrust) { s = trTrust / maxDx; }
  trScale[0] = s;
  trStatus[10] = maxDx;
}

// ---- descent_check: fallback flag + directionValid lane (no readback)
// Mirrors the CPU gate exactly: !finite(gtdx) || gtdx >= 0 || breakdown.
@group(0) @binding(30) var<storage, read> dcGtdx : array<f32>; // gtdxStore
@group(0) @binding(31) var<storage, read> dcBreak : array<f32>; // breakFlag
@group(0) @binding(32) var<storage, read_write> dcCtl : array<f32>; // newtonCtl
@group(0) @binding(33) var<storage, read_write> dcStatus : array<f32>; // newtonStatus

@compute @workgroup_size(64)
fn descent_check(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let gtdx = dcGtdx[0];
  var fb = 0.0;
  if (!(gtdx == gtdx) || gtdx >= 0.0 || dcBreak[0] != 0.0) { fb = 1.0; }
  dcCtl[1] = fb;
  if (fb > 0.5) { dcStatus[3] = 0.0; } else { dcStatus[3] = 1.0; }
}

// ---- select_fallback: predicated copy descentDir -> searchDirection.
@group(0) @binding(40) var<storage, read> sfDescent : array<f32>; // descentDir n*3
@group(0) @binding(41) var<storage, read_write> sfSearch : array<f32>; // searchDirection n*3
@group(0) @binding(42) var<storage, read> sfCtl : array<f32>; // newtonCtl (fallback lane)
@group(0) @binding(43) var<uniform> sfParams : SimParams;

@compute @workgroup_size(64)
fn select_fallback(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= sfParams.vertexCount * 3u) { return; }
  if (sfCtl[1] > 0.5) { sfSearch[i] = sfDescent[i]; }
}

// ---- commit_arm: first-accept arbitration across this round's batches.
// Runs after every batch select; sets doCommit only for the newly accepted
// batch and latches accepted energy/min-dist/TOI + merit. Batch indexing is
// self-tracked (nbBatch++ per call) so no CPU writes are needed mid-round.
@group(0) @binding(50) var<storage, read> caBatch : array<f32>; // armijoStatus (this batch)
// NOTE: candidate rows are intentionally NOT bound here (minDist/minToi ride
// the armijoStatus lanes the select kernel fills); over-binding a pruned
// binding fails Dawn validation and poisons the whole encoder — see G6C notes.
@group(0) @binding(52) var<storage, read_write> caCtl : array<f32>; // newtonCtl
@group(0) @binding(53) var<storage, read_write> caE0 : array<f32>; // e0Store
@group(0) @binding(54) var<storage, read_write> caStatus : array<f32>; // newtonStatus
@group(0) @binding(55) var<uniform> caK : u32; // ArmijoK (this batch width)

@compute @workgroup_size(64)
fn commit_arm(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let b = caCtl[5]; // nbBatch counter (f32 holding u32, exact for small ints)
  let bi = u32(b + 0.5);
  let localAccept = caBatch[0] > 0.5;
  // sticky failure-flag ORs (per-batch counters would otherwise be lost)
  if (caBatch[7] > 0.0) { caStatus[15] = 1.0; }
  if (caBatch[5] > 0.0) { caStatus[16] = 1.0; }
  if (caBatch[6] > 0.0) { caStatus[17] = 1.0; }
  caCtl[3] = 0.0; // doCommit defaults off; set below only when newly accepted
  // A converged round (done set by newton_check) must not move position even
  // if a stale direction still passes Armijo — first-accept AND !done.
  if (localAccept && caCtl[2] == 0.0 && caCtl[0] < 0.5) {
    caCtl[2] = 1.0; // nbAccepted
    caCtl[3] = 1.0; // doCommit
    caCtl[4] = caBatch[1]; // nbAlpha (effective)
    caCtl[6] = f32(bi); // nbAcceptBatch
    let li = u32(caBatch[2] + 0.5); // local selected index
    caCtl[8] = f32(bi * caK + li); // global trial index
    // merit BEFORE overwriting e0 (round-start energy minus accepted)
    caStatus[11] = caE0[0] - caBatch[9];
    caE0[0] = caBatch[9]; // latch accepted energy for later rounds
    caStatus[12] = caBatch[9];
    caStatus[13] = caBatch[13];
    caStatus[14] = caBatch[14];
  }
  caCtl[5] = f32(bi + 1u);
  caCtl[7] = caCtl[7] + f32(caK); // nbTrialBase advance (exact for small ints)
}

// ---- commit_apply re-materializes xTrial at the LATCHED accepted alpha.
// The candidate loop leaves xTrial holding the LAST candidate, which equals
// the accepted one only by accident — without this, commit would land the
// wrong state. Runs unconditionally (harmless when idle: the next batch
// overwrites xTrial; position moves only via commit_copy_if).
@group(0) @binding(64) var<storage, read> caXBase : array<vec4f>; // position (accepted base)
@group(0) @binding(65) var<storage, read> caDx : array<f32>; // searchDirection n*3
@group(0) @binding(66) var<storage, read_write> caXTrial : array<vec4f>; // xTrial
@group(0) @binding(67) var<storage, read> caPin : array<u32>;
@group(0) @binding(68) var<storage, read> caPinPos : array<vec4f>;
@group(0) @binding(69) var<storage, read> caCtlA : array<f32>; // newtonCtl (nbAlpha lane 4)

@compute @workgroup_size(64)
fn commit_apply(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= arrayLength(&caPinPos)) { return; }
  let a = caCtlA[4];
  var p = caXBase[v].xyz + a * vec3f(
    caDx[v * 3u], caDx[v * 3u + 1u], caDx[v * 3u + 2u]);
  if (caPin[v] != 0u) { p = caPinPos[v].xyz; }
  caXTrial[v] = vec4f(p, 0.0);
}

// ---- predicated commit entries (mirror acceptTrial; old path untouched).
// Bounds come from simParams (contactCount is cap during GPU-control rounds;
// the copy covers cloth verts — the static tail never changes).
@group(0) @binding(60) var<storage, read> ccTrial : array<vec4f>; // xTrial
@group(0) @binding(61) var<storage, read_write> ccPos : array<vec4f>; // position
@group(0) @binding(62) var<storage, read> ccCtl : array<f32>; // newtonCtl (doCommit lane)
@group(0) @binding(63) var<uniform> ccParams : SimParams;

@compute @workgroup_size(64)
fn commit_copy_if(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= ccParams.vertexCount) { return; }
  if (ccCtl[3] < 0.5) { return; }
  ccPos[v] = ccTrial[v];
}

@group(0) @binding(70) var<storage, read> clN : array<vec4f>;
@group(0) @binding(71) var<storage, read> clDist : array<f32>;
@group(0) @binding(72) var<storage, read> clPrm : array<vec4f>;
@group(0) @binding(73) var<storage, read_write> clLag : array<vec4f>; // laggedN
@group(0) @binding(74) var<storage, read> clCtl : array<f32>; // newtonCtl (doCommit lane)
@group(0) @binding(75) var<uniform> clParams : SimParams;

@compute @workgroup_size(64)
fn commit_lagged_if(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= clParams.contactCount) { return; }
  if (clCtl[3] < 0.5) { return; }
  // Same body as contact-force commit_lagged (d>=dHat keeps stale entry).
  let d = max(clDist[i], 1e-12);
  let prm = clPrm[i];
  let dHat = prm.x; let kappa = prm.y;
  if (d >= dHat) { return; }
  let t = d - dHat;
  let db = -2.0 * t * log(d / dHat) - (t * t) / d;
  let lambdaN = max(0.0, -(kappa * db));
  let n = clN[i].xyz;
  clLag[i] = vec4f(n, lambdaN);
}

// ---- newton_check: convergence + done from the fresh Xk status.
// Runs after the round's Xk refresh (rhsJacobi + diagnostics, no reads).
// done = converged-at-start (the round still runs its passes — bounded waste
// of at most one extra round; commit is predicated on !done via commit_arm
// reading this lane... see note) or a hard failure is impossible here:
// failure (!accepted) is decided by round_report from nbAccepted.
@group(0) @binding(80) var<storage, read> ncStatus : array<f32>; // solverStatus (fresh Xk)
@group(0) @binding(81) var<storage, read_write> ncCtl : array<f32>; // newtonCtl
@group(0) @binding(82) var<storage, read_write> ncRound : array<f32>; // newtonStatus
@group(0) @binding(83) var<uniform> ncTol : f32; // NewtonTol persistent slot

@compute @workgroup_size(64)
fn newton_check(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  let gnorm = ncStatus[2];
  var conv = 0.0;
  if (gnorm < ncTol) { conv = 1.0; }
  ncRound[1] = conv;
  ncRound[9] = gnorm;
  if (conv > 0.5) { ncCtl[0] = 1.0; }
}

// ---- round_report: assemble the compact round status (single read/round).
// Lanes from: newtonCtl (done/fallback/accepted/alpha/batch/trial),
// pcg_report (breakdown/residual), trust_compute (stepNorm), commit-arm
// latched energy/minDist/minToi/merit, batch status (accepted flag source).
@group(0) @binding(90) var<storage, read> rrCtl : array<f32>; // newtonCtl
@group(0) @binding(91) var<storage, read_write> rrRound : array<f32>; // newtonStatus
@group(0) @binding(92) var<storage, read> rrGtdx : array<f32>; // gtdxStore (for the report lane)
@group(0) @binding(93) var<storage, read> rrE0 : array<f32>; // e0Store (accepted-or-round-start energy)
@group(0) @binding(94) var<storage, read> rrBatch : array<f32>; // armijoStatus (last batch lanes)

@compute @workgroup_size(64)
fn round_report(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  // iteration lane [0]: CPU-written per round, preserved here.
  rrRound[2] = 0.0; // failure filled below
  rrRound[3] = 1.0; // directionValid filled below
  rrRound[5] = rrCtl[2]; // armijoAccepted (sticky nbAccepted)
  rrRound[6] = rrCtl[6]; // accepting batch index
  rrRound[7] = rrCtl[4]; // selectedAlpha (effective)
  rrRound[8] = rrCtl[8]; // selected global trial index
  if (rrCtl[2] < 0.5) {
    rrRound[2] = 1.0; // no commit this round = failed Newton iterate
    rrRound[5] = 0.0;
    rrRound[7] = 0.0;
    rrRound[8] = 4294967295.0;
  }
  if (rrCtl[1] > 0.5) { rrRound[3] = 0.0; }
  rrRound[19] = rrGtdx[0];
  // Energy + distance lanes: e0Store holds the accepted (or round-start)
  // energy; the last batch status carries the freshest minDist/minToi.
  rrRound[12] = rrE0[0];
  rrRound[13] = rrBatch[13];
  rrRound[14] = rrBatch[14];
}
