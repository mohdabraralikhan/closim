// G5C one-level additive Schwarz: fixed-<=8-vertex domains (padded to 8),
// per-domain 24x24 dense local matrices, factored once per Newton solve.
//
// Layout (all group 0):
//   schwarz_assemble (one thread per domain row, 24/domain):
//     row(lr) of A_d from incident triangles (CSR) via the corner-pair
//     analytic membrane entries (same closed form as membrane-blocks.ts
//     triangleHessian9, restricted to intra-domain corner pairs) + the Jacobi
//     diagonal embedded on the row diagonal. Padded rows -> identity.
//   schwarz_factor (one thread per domain):
//     symmetrized load, dense Cholesky; success -> explicit 24x24 inverse,
//     fail -> symmetric Jacobi-diagonal fallback (flag 0).
//   schwarz_apply (one thread per fine dof):
//     z = Sinv * r with domain gather, pin-filtered, prod = r*z.
// Init reuses pcg_init (scalar throwaway) + schwarz_apply + blas copy(p=z);
// see DeviceNewtonDriver.pcgSolve. Bending stays out (inexact Newton).
// The stress term carries the delta(b,a) guard (dF_ik/dx_B = delta(b,i)K).

struct SimParams {
  dt : f32, invDt2 : f32,
  gravityX : f32, gravityY : f32, gravityZ : f32, _pad0 : f32,
  vertexCount : u32, triangleCount : u32, hingeCount : u32, contactCount : u32,
  newtonIteration : u32, pcgIteration : u32,
  lineSearchAlpha : f32, trustRegion : f32,
  barrierActivation : f32, frictionMu : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> position : array<vec4f>;
@group(0) @binding(2) var<storage, read> triangles : array<u32>;
@group(0) @binding(3) var<storage, read> dmInv : array<vec4f>; // (a,b,c,d)
@group(0) @binding(4) var<storage, read> restArea : array<f32>;
@group(0) @binding(5) var<storage, read> elemOffsets : array<u32>;
@group(0) @binding(6) var<storage, read> elemIds : array<u32>;
@group(0) @binding(7) var<storage, read> elemCorners : array<u32>;
@group(0) @binding(8) var<storage, read> diagBuf : array<f32>; // n*3 Jacobi diag
@group(0) @binding(9) var<storage, read_write> schwarzMat : array<f32>; // nDoms*576
@group(0) @binding(10) var<storage, read> schwarzDomain : array<u32>; // n
@group(0) @binding(11) var<storage, read> schwarzLocal : array<u32>; // n
@group(0) @binding(12) var<storage, read> schwarzVerts : array<u32>; // nDoms*8

@group(0) @binding(13) var<uniform> matC00 : f32;
@group(0) @binding(14) var<uniform> matC11 : f32;
@group(0) @binding(15) var<uniform> matC01 : f32;
@group(0) @binding(16) var<uniform> matG : f32;
@group(0) @binding(17) var<uniform> matThickness : f32;
@group(0) @binding(18) var<uniform> schwarzNDomsA : u32;

fn Dmod(k : u32, j : u32, l : u32, m : u32) -> f32 {
  if ((k == l && j == m) || (k == m && j == l)) {
    if (k == j) {
      if (k == 0u) { return matC00; }
      return matC11;
    }
    return matG;
  }
  if (k == j && l == m) { return matC01; }
  return 0.0;
}

// Analytic membrane entry H[(cu,a)][(cv,b)] for one triangle state.
// Kcu/Kcv: F-coefficient rows of the two corners; Fa/Fb: F columns a/b.
fn triPairEntry(
  Kcu0 : f32, Kcu1 : f32, Kcv0 : f32, Kcv1 : f32,
  Fa0 : f32, Fa1 : f32, Fb0 : f32, Fb1 : f32,
  F00 : f32, F10 : f32, F20 : f32, F01 : f32, F11 : f32, F21 : f32,
  S00 : f32, S11 : f32, S01 : f32, s : f32, a : u32, b : u32,
) -> f32 {
  var entry = 0.0;
  for (var j = 0u; j < 2u; j++) {
    var Kcu = 0.0;
    if (j == 0u) { Kcu = Kcu0; } else { Kcu = Kcu1; }
    for (var kk = 0u; kk < 2u; kk++) {
      var Kcvk = 0.0;
      if (kk == 0u) { Kcvk = Kcv0; } else { Kcvk = Kcv1; }
      if (a == b) {
        var Skj = 0.0;
        if (kk == 0u && j == 0u) { Skj = S00; }
        else if (kk == 1u && j == 1u) { Skj = S11; }
        else { Skj = S01; }
        entry += Kcu * (Kcvk * Skj);
      }
      var Fak = 0.0;
      if (kk == 0u) { Fak = Fa0; } else { Fak = Fa1; }
      if (Fak != 0.0) {
        var dsum = 0.0;
        for (var l = 0u; l < 2u; l++) {
          var Kl = 0.0;
          if (l == 0u) { Kl = Kcv0; } else { Kl = Kcv1; }
          for (var m = 0u; m < 2u; m++) {
            var Km = 0.0;
            if (m == 0u) { Km = Kcv0; } else { Km = Kcv1; }
            var dd = Dmod(kk, j, l, m);
            if (dd != 0.0) {
              var Fbl = 0.0; var Fbm = 0.0;
              if (l == 0u) { Fbl = Fb0; } else { Fbl = Fb1; }
              if (m == 0u) { Fbm = Fb0; } else { Fbm = Fb1; }
              dsum += dd * 0.5 * (Kl * Fbm + Fbl * Km);
            }
          }
        }
        entry += Kcu * Fak * dsum;
      }
    }
  }
  return s * entry;
}

fn cornerK(c : u32, ia : f32, ib : f32, ic : f32, id : f32) -> vec2f {
  if (c == 0u) { return vec2f(-(ia + ic), -(ib + id)); }
  if (c == 1u) { return vec2f(ia, ib); }
  return vec2f(ic, id);
}

@compute @workgroup_size(64)
fn schwarz_assemble(@builtin(global_invocation_id) gid : vec3u) {
  let thread = gid.x + gid.y * 4194240u;
  let d = thread / 24u;
  if (d >= schwarzNDomsA) { return; }
  let lr = thread % 24u;
  let lv = lr / 3u;
  let a = lr % 3u;
  // Domain count bound: threads past nDoms*24 exit (caller sizes exactly).
  // Padded vertex rows -> identity (decoupled, pivot 1).
  let v = schwarzVerts[d * 8u + lv];
  var row = array<f32, 24>();
  for (var j = 0u; j < 24u; j++) { row[j] = 0.0; }
  if (v == 0xffffffffu) {
    row[lr] = 1.0;
    for (var j = 0u; j < 24u; j++) { schwarzMat[d * 576u + lr * 24u + j] = row[j]; }
    return;
  }
  for (var k = elemOffsets[v]; k < elemOffsets[v + 1u]; k++) {
    let t = elemIds[k];
    let cu = elemCorners[k];
    let i0 = triangles[t * 3u];
    let i1 = triangles[t * 3u + 1u];
    let i2 = triangles[t * 3u + 2u];
    let x0 = position[i0].xyz;
    let x1 = position[i1].xyz;
    let x2 = position[i2].xyz;
    let e1 = x1 - x0;
    let e2 = x2 - x0;
    let ia = dmInv[t].x; let ib = dmInv[t].y; let ic = dmInv[t].z; let id = dmInv[t].w;
    let F00 = ia * e1.x + ic * e2.x;
    let F10 = ia * e1.y + ic * e2.y;
    let F20 = ia * e1.z + ic * e2.z;
    let F01 = ib * e1.x + id * e2.x;
    let F11 = ib * e1.y + id * e2.y;
    let F21 = ib * e1.z + id * e2.z;
    let C00 = F00*F00 + F10*F10 + F20*F20;
    let C11 = F01*F01 + F11*F11 + F21*F21;
    let C01 = F00*F01 + F10*F11 + F20*F21;
    let E00 = 0.5 * (C00 - 1.0);
    let E11 = 0.5 * (C11 - 1.0);
    let E01 = 0.5 * C01;
    let S00 = matC00*E00 + matC01*E11;
    let S11 = matC11*E11 + matC01*E00;
    let S01 = 2.0 * matG * E01;
    let s = matThickness * restArea[t];
    let Kcu = cornerK(cu, ia, ib, ic, id);
    var Fa0 = 0.0; var Fa1 = 0.0;
    if (a == 0u) { Fa0 = F00; Fa1 = F01; }
    else if (a == 1u) { Fa0 = F10; Fa1 = F11; }
    else { Fa0 = F20; Fa1 = F21; }
    for (var cj = 0u; cj < 3u; cj++) {
      var w = 0u;
      if (cj == 0u) { w = i0; } else if (cj == 1u) { w = i1; } else { w = i2; }
      if (schwarzDomain[w] != d) { continue; }
      let lw = schwarzLocal[w];
      let Kcv = cornerK(cj, ia, ib, ic, id);
      for (var bb = 0u; bb < 3u; bb++) {
        var Fb0 = 0.0; var Fb1 = 0.0;
        if (bb == 0u) { Fb0 = F00; Fb1 = F01; }
        else if (bb == 1u) { Fb0 = F10; Fb1 = F11; }
        else { Fb0 = F20; Fb1 = F21; }
        row[lw * 3u + bb] += triPairEntry(
          Kcu.x, Kcu.y, Kcv.x, Kcv.y, Fa0, Fa1, Fb0, Fb1,
          F00, F10, F20, F01, F11, F21, S00, S11, S01, s, a, bb);
      }
    }
  }
  row[lr] += diagBuf[v * 3u + a];
  for (var j = 0u; j < 24u; j++) { schwarzMat[d * 576u + lr * 24u + j] = row[j]; }
}

// ---- factor + apply use a second module scope (own bindings) ----
@group(0) @binding(20) var<storage, read> fMat : array<f32>; // nDoms*576 assembled
@group(0) @binding(21) var<storage, read_write> fInv : array<f32>; // nDoms*576 inverse
@group(0) @binding(22) var<storage, read_write> fFlag : array<u32>; // nDoms
@group(0) @binding(23) var<storage, read> fDiag : array<f32>; // n*3 (fallback)
@group(0) @binding(24) var<storage, read> fVerts : array<u32>; // nDoms*8
@group(0) @binding(25) var<uniform> schwarzNDomsF : u32;

@compute @workgroup_size(64)
fn schwarz_factor(@builtin(global_invocation_id) gid : vec3u) {
  let d = gid.x + gid.y * 4194240u;
  if (d >= schwarzNDomsF) { return; }
  var A = array<f32, 576>();
  for (var i = 0u; i < 24u; i++) {
    for (var j = 0u; j < 24u; j++) {
      A[i * 24u + j] = 0.5 * (fMat[d * 576u + i * 24u + j] + fMat[d * 576u + j * 24u + i]);
    }
  }
  var tr = 0.0;
  for (var i = 0u; i < 24u; i++) {
    let av = abs(A[i * 24u + i]);
    tr += av;
  }
  let tiny = 1e-20 * (1.0 + tr);
  var ok = true;
  for (var k = 0u; k < 24u; k++) {
    if (!ok) { break; }
    var sum = A[k * 24u + k];
    for (var s = 0u; s < k; s++) { sum -= A[k * 24u + s] * A[k * 24u + s]; }
    if (!(sum > tiny)) { ok = false; break; }
    A[k * 24u + k] = sqrt(sum);
    for (var i = k + 1u; i < 24u; i++) {
      var s2 = A[i * 24u + k];
      for (var s = 0u; s < k; s++) { s2 -= A[i * 24u + s] * A[k * 24u + s]; }
      A[i * 24u + k] = s2 / A[k * 24u + k];
    }
  }
  if (!ok) {
    // Symmetric Jacobi-diagonal fallback (keeps the preconditioner SPD).
    for (var i = 0u; i < 24u; i++) {
      for (var j = 0u; j < 24u; j++) { fInv[d * 576u + i * 24u + j] = 0.0; }
    }
    for (var l = 0u; l < 8u; l++) {
      let vv = fVerts[d * 8u + l];
      if (vv == 0xffffffffu) {
        for (var aa = 0u; aa < 3u; aa++) {
          fInv[d * 576u + (l * 3u + aa) * 24u + (l * 3u + aa)] = 1.0;
        }
        continue;
      }
      for (var aa = 0u; aa < 3u; aa++) {
        let dd = fDiag[vv * 3u + aa];
        var ival = 0.0;
        if (dd > 1e-12) { ival = 1.0 / dd; }
        fInv[d * 576u + (l * 3u + aa) * 24u + (l * 3u + aa)] = ival;
      }
    }
    fFlag[d] = 0u;
    return;
  }
  // Explicit inverse via 24 triangular solves (L kept in lower of A).
  for (var i = 0u; i < 24u; i++) {
    for (var j = i + 1u; j < 24u; j++) { A[i * 24u + j] = 0.0; }
  }
  var y = array<f32, 24>();
  var xx = array<f32, 24>();
  for (var col = 0u; col < 24u; col++) {
    for (var i = 0u; i < 24u; i++) {
      var acc = 0.0;
      if (i == col) { acc = 1.0; }
      for (var j = 0u; j < i; j++) { acc -= A[i * 24u + j] * y[j]; }
      y[i] = acc / A[i * 24u + i];
    }
    for (var ii = 0u; ii < 24u; ii++) {
      var i2 = 23u - ii;
      var acc2 = y[i2];
      for (var j = i2 + 1u; j < 24u; j++) { acc2 -= A[j * 24u + i2] * xx[j]; }
      xx[i2] = acc2 / A[i2 * 24u + i2];
    }
    for (var i = 0u; i < 24u; i++) { fInv[d * 576u + i * 24u + col] = xx[i]; }
  }
  fFlag[d] = 1u;
}

@group(0) @binding(30) var<storage, read> aRes : array<f32>; // n*3 residual
@group(0) @binding(31) var<storage, read_write> aZ : array<f32>; // n*3 output
@group(0) @binding(32) var<storage, read> aPin : array<u32>; // n
@group(0) @binding(33) var<storage, read_write> aProd : array<f32>; // n*3 r*z
@group(0) @binding(34) var<storage, read> aInv : array<f32>; // nDoms*576
@group(0) @binding(35) var<storage, read> aVerts : array<u32>; // nDoms*8
@group(0) @binding(36) var<storage, read> aDomain : array<u32>; // n
@group(0) @binding(37) var<storage, read> aLocal : array<u32>; // n

@compute @workgroup_size(64)
fn schwarz_apply(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  let n3 = params.vertexCount * 3u;
  if (i >= n3) { return; }
  let v = i / 3u;
  let a = i % 3u;
  let pinned = aPin[v];
  let d = aDomain[v];
  let l = aLocal[v];
  let lr = l * 3u + a;
  // gather domain residual (padded dofs read r=0 contribution via identity rows)
  var rl = array<f32, 24>();
  for (var j = 0u; j < 24u; j++) {
    let lv2 = j / 3u;
    let aa2 = j % 3u;
    let w = aVerts[d * 8u + lv2];
    var rv = 0.0;
    if (w != 0xffffffffu) { rv = aRes[w * 3u + aa2]; }
    rl[j] = rv;
  }
  var zi = 0.0;
  for (var j = 0u; j < 24u; j++) { zi += aInv[d * 576u + lr * 24u + j] * rl[j]; }
  if (pinned != 0u) { zi = 0.0; }
  aZ[i] = zi;
  aProd[i] = aRes[i] * zi;
}
