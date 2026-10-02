// G3 diagnostics — the ONLY hot-loop readback source (§19).
// Computes the full 64 B SolverStatus on device in two dispatches:
//   diag_stage1: per-workgroup partials of 8 stats into diagScratch.
//   diag_stage2: single workgroup finishes sums/mins into statusOut[16].
//
// statusOut layout (matches decodeSolverStatus):
//   [0] total energy (inertia + membrane + bending + barrier)
//   [1] barrier energy
//   [2] gradient norm (sqrt on device)
//   [3] directionDotGradient (NOT computed here: CPU writes it via a 4 B
//       control writeBuffer after its own dot reduce — see gpu-executor.ts)
//   [4] minDistance over active contacts (1e30 sentinel when none)
//   [5] minTOI over contacts (2.0 NO_HIT sentinel when none)
//   [6] finite (0/1: any NaN lane poisons it; old Dawn has no isFinite)
//   [7] ccdSafe (contactFail == 0)
//   [8] barrierSafe (= finite)
//   [9] pcgBreakdown, [10] converged: CPU-written control fields
//   [11..15] reserved (written 0)
//
// CPU maps only the staging copy per Newton iteration:
//   GPU storage --copyBufferToBuffer--> staging --mapAsync(READ)--> CPU
// Never map the storage buffer directly while GPU commands are in flight.

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> gradient : array<f32>; // n*3 at trial
@group(0) @binding(2) var<storage, read> position : array<vec4f>; // xTrial
@group(0) @binding(3) var<storage, read> positionRef : array<vec4f>; // y_hat
@group(0) @binding(4) var<storage, read> mass : array<f32>; // n
@group(0) @binding(5) var<storage, read> elementEnergy : array<f32>; // m
@group(0) @binding(6) var<storage, read> hingeEnergyOut : array<f32>; // h
@group(0) @binding(7) var<storage, read> contactEnergy : array<f32>; // contacts
@group(0) @binding(8) var<storage, read> contactDist : array<f32>;
@group(0) @binding(9) var<storage, read> contactTOI : array<f32>; // 2.0 = NO_HIT
@group(0) @binding(10) var<storage, read> contactFail : array<u32>;
@group(0) @binding(11) var<storage, read_write> diagScratch : array<f32>; // groups*8
@group(0) @binding(12) var<storage, read_write> statusOut : array<f32>; // 16 f32

var<workgroup> tE : array<f32, 64>; // energy sum
var<workgroup> tB : array<f32, 64>; // barrier sum
var<workgroup> tG : array<f32, 64>; // gradSq sum
var<workgroup> tD : array<f32, 64>; // min distance
var<workgroup> tT : array<f32, 64>; // min TOI
var<workgroup> tF : array<f32, 64>; // finite flag (min: 0 if any NaN lane)
var<workgroup> tK : array<f32, 64>; // fail flag (max: 1 if any fail)
var<workgroup> tH : array<f32, 64>; // hinge energy sum

fn is_nan(x : f32) -> bool { return x != x; }

@compute @workgroup_size(64)
fn diag_stage1(
  @builtin(global_invocation_id) gid : vec3u,
  @builtin(local_invocation_id) lid : vec3u,
  @builtin(workgroup_id) wid : vec3u,
) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  var e = 0.0; var b = 0.0; var g2 = 0.0;
  var dm = 1e30; var tm = 2.0; var fin = 1.0; var fail = 0.0; var h = 0.0;
  if (i < n3) {
    let g = gradient[i];
    if (is_nan(g)) { fin = 0.0; } else { g2 = g * g; }
  }
  if (i < params.triangleCount) {
    let ee = elementEnergy[i];
    if (is_nan(ee)) { fin = 0.0; } else { e += ee; }
  }
  if (i < params.hingeCount) {
    let he = hingeEnergyOut[i];
    if (is_nan(he)) { fin = 0.0; } else { h += he; }
  }
  if (i < params.contactCount) {
    let ce = contactEnergy[i];
    if (is_nan(ce)) { fin = 0.0; } else { b += ce; }
    let dd = contactDist[i];
    if (is_nan(dd)) { fin = 0.0; } else if (dd < dm) { dm = dd; }
    let tt = contactTOI[i];
    if (is_nan(tt)) { fin = 0.0; } else if (tt < tm) { tm = tt; }
  }
  // inertia 0.5*m*(x-yHat)^2/h^2 needs per-VERTEX indexing: thread i covers
  // component i%3 of vertex i/3 — accumulate per-component then sum.
  if (i < n3) {
    let v = i / 3u;
    let k = i % 3u;
    var pv : f32;
    var rv : f32;
    if (k == 0u) { pv = position[v].x; rv = positionRef[v].x; }
    else if (k == 1u) { pv = position[v].y; rv = positionRef[v].y; }
    else { pv = position[v].z; rv = positionRef[v].z; }
    let dd2 = pv - rv;
    if (is_nan(dd2) || is_nan(mass[v])) { fin = 0.0; }
    else { e += 0.5 * mass[v] * dd2 * dd2 * params.invDt2; }
  }
  if (contactFail[0] != 0u) { fail = 1.0; }
  tE[lid.x] = e; tB[lid.x] = b; tG[lid.x] = g2; tD[lid.x] = dm;
  tT[lid.x] = tm; tF[lid.x] = fin; tK[lid.x] = fail; tH[lid.x] = h;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      tE[lid.x] += tE[lid.x + s];
      tB[lid.x] += tB[lid.x + s];
      tG[lid.x] += tG[lid.x + s];
      tD[lid.x] = min(tD[lid.x], tD[lid.x + s]);
      tT[lid.x] = min(tT[lid.x], tT[lid.x + s]);
      tF[lid.x] = min(tF[lid.x], tF[lid.x + s]);
      tK[lid.x] = max(tK[lid.x], tK[lid.x + s]);
      tH[lid.x] += tH[lid.x + s];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    diagScratch[wid.x * 8u] = tE[0];
    diagScratch[wid.x * 8u + 1u] = tB[0];
    diagScratch[wid.x * 8u + 2u] = tG[0];
    diagScratch[wid.x * 8u + 3u] = tD[0];
    diagScratch[wid.x * 8u + 4u] = tT[0];
    diagScratch[wid.x * 8u + 5u] = tF[0];
    diagScratch[wid.x * 8u + 6u] = tK[0];
    diagScratch[wid.x * 8u + 7u] = tH[0];
  }
}

@group(0) @binding(14) var<uniform> numGroups : u32;

@compute @workgroup_size(64)
fn diag_stage2(@builtin(local_invocation_id) lid : vec3u) {
  var e = 0.0; var b = 0.0; var g2 = 0.0;
  var dm = 1e30; var tm = 2.0; var fin = 1.0; var fail = 0.0; var h = 0.0;
  for (var g = lid.x; g < numGroups; g += 64u) {
    e += diagScratch[g * 8u];
    b += diagScratch[g * 8u + 1u];
    g2 += diagScratch[g * 8u + 2u];
    dm = min(dm, diagScratch[g * 8u + 3u]);
    tm = min(tm, diagScratch[g * 8u + 4u]);
    fin = min(fin, diagScratch[g * 8u + 5u]);
    fail = max(fail, diagScratch[g * 8u + 6u]);
    h += diagScratch[g * 8u + 7u];
  }
  tE[lid.x] = e; tB[lid.x] = b; tG[lid.x] = g2; tD[lid.x] = dm;
  tT[lid.x] = tm; tF[lid.x] = fin; tK[lid.x] = fail; tH[lid.x] = h;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      tE[lid.x] += tE[lid.x + s];
      tB[lid.x] += tB[lid.x + s];
      tG[lid.x] += tG[lid.x + s];
      tD[lid.x] = min(tD[lid.x], tD[lid.x + s]);
      tT[lid.x] = min(tT[lid.x], tT[lid.x + s]);
      tF[lid.x] = min(tF[lid.x], tF[lid.x + s]);
      tK[lid.x] = max(tK[lid.x], tK[lid.x + s]);
      tH[lid.x] += tH[lid.x + s];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    statusOut[0] = tE[0] + tH[0];
    statusOut[1] = tB[0];
    statusOut[2] = sqrt(tG[0]);
    statusOut[3] = 0.0; // dirDotGradient: CPU-written control field
    statusOut[4] = tD[0];
    statusOut[5] = tT[0];
    statusOut[6] = tF[0];
    statusOut[7] = 1.0 - tK[0]; // ccdSafe
    statusOut[8] = tF[0]; // barrierSafe
    statusOut[9] = 0.0; // pcgBreakdown: CPU-written
    statusOut[10] = 0.0; // converged: CPU-written
    statusOut[11] = 0.0;
    statusOut[12] = 0.0;
    statusOut[13] = 0.0;
    statusOut[14] = 0.0;
    statusOut[15] = 0.0;
  }
}
