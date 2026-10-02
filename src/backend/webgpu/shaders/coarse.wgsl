// G5.5 assembled coarse block-CSR solve (Ac = R A P, 3x3 blocks).
//
// coarse_assemble_values (one thread per coarse row d1):
//   row blocks from incident triangles (fine CSR) via the corner-pair analytic
//   membrane entries (same closed form as membrane-blocks.ts triangleHessian9
//   and schwarz.wgsl triPairEntry, incl. the delta(b,a) stress guard) scaled
//   by 1/sqrt(|d1||d2|), plus the Jacobi-diagonal spread on the self block.
//   Raw (unsymmetrized) storage; the OPERATOR is symmetrized at read time
//   (spmv/z-step average B with B^T), so no transpose race and no temp buffer.
// coarse_spmv: y = sym(Ac) x with on-the-fly transpose average.
// c_init / c_update_xr / c_update_z / c_update_p2: fixed-K inner coarse PCG
//   with per-row diagonal-block adjugate (Jacobi-diagonal fallback), all
//   counts from the solve-persistent CoarseCount slot (zero uniform writes
//   inside the loop => zero added submits).
// coarse_reduce_stage1/2: pcg-reduce twins on CoarseCount/CoarseGroups slots.
//
// Membrane + diag-spread only in Ac (barrier/contact stay in the exact fine
// HVP). Coarse dofs have no pins (pin filtering happens at prolongation).

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
@group(0) @binding(9) var<storage, read> cDomain : array<u32>; // n vertex -> domain
@group(0) @binding(10) var<storage, read> cVerts : array<u32>; // nDoms*8 members
@group(0) @binding(11) var<storage, read> cRowOffsets : array<u32>; // nDoms+1
@group(0) @binding(12) var<storage, read> cColIndices : array<u32>; // nnz
@group(0) @binding(13) var<storage, read_write> cBlockValues : array<f32>; // nnz*9

@group(0) @binding(14) var<uniform> matC00 : f32;
@group(0) @binding(15) var<uniform> matC11 : f32;
@group(0) @binding(16) var<uniform> matC01 : f32;
@group(0) @binding(17) var<uniform> matG : f32;
@group(0) @binding(18) var<uniform> matThickness : f32;
@group(0) @binding(19) var<uniform> cNDomsA : u32;

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

fn domainSize(d : u32) -> u32 {
  var cnt = 0u;
  for (var l = 0u; l < 8u; l++) {
    if (cVerts[d * 8u + l] != 0xffffffffu) { cnt++; }
  }
  return cnt;
}

@compute @workgroup_size(64)
fn coarse_assemble_values(@builtin(global_invocation_id) gid : vec3u) {
  let d1 = gid.x + gid.y * 4194240u;
  if (d1 >= cNDomsA) { return; }
  let rowLo = cRowOffsets[d1];
  let rowHi = cRowOffsets[d1 + 1u];
  let deg = rowHi - rowLo;
  // Row pattern into private memory (kernel cap; CPU asserts degree at build).
  var cols = array<u32, 32>();
  for (var q = 0u; q < deg; q++) { cols[q] = cColIndices[rowLo + q]; }
  var acc = array<f32, 288>();
  for (var q = 0u; q < deg * 9u; q++) { acc[q] = 0.0; }
  let size1 = domainSize(d1);
  var fsize1 = 1.0;
  if (size1 > 0u) { fsize1 = sqrt(f32(size1)); }
  for (var lv = 0u; lv < 8u; lv++) {
    let v = cVerts[d1 * 8u + lv];
    if (v == 0xffffffffu) { continue; }
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
      for (var cj = 0u; cj < 3u; cj++) {
        var w = 0u;
        if (cj == 0u) { w = i0; } else if (cj == 1u) { w = i1; } else { w = i2; }
        let d2 = cDomain[w];
        // slot of d2 in this row (must exist: shared triangle => block edge)
        var slot = 0xffffffffu;
        for (var q = 0u; q < deg; q++) {
          if (cols[q] == d2) { slot = q; break; }
        }
        if (slot == 0xffffffffu) { continue; }
        let size2 = domainSize(d2);
        var pair = 0.0;
        if (size1 > 0u && size2 > 0u) { pair = 1.0 / (fsize1 * sqrt(f32(size2))); }
        let Kcv = cornerK(cj, ia, ib, ic, id);
        for (var a = 0u; a < 3u; a++) {
          var Fa0 = 0.0; var Fa1 = 0.0;
          if (a == 0u) { Fa0 = F00; Fa1 = F01; }
          else if (a == 1u) { Fa0 = F10; Fa1 = F11; }
          else { Fa0 = F20; Fa1 = F21; }
          for (var b = 0u; b < 3u; b++) {
            var Fb0 = 0.0; var Fb1 = 0.0;
            if (b == 0u) { Fb0 = F00; Fb1 = F01; }
            else if (b == 1u) { Fb0 = F10; Fb1 = F11; }
            else { Fb0 = F20; Fb1 = F21; }
            acc[slot * 9u + a * 3u + b] += pair * triPairEntry(
              Kcu.x, Kcu.y, Kcv.x, Kcv.y, Fa0, Fa1, Fb0, Fb1,
              F00, F10, F20, F01, F11, F21, S00, S11, S01, s, a, b);
          }
        }
      }
    }
  }
  // Jacobi-diagonal spread on the self block diagonal.
  var selfSlot = 0xffffffffu;
  for (var q = 0u; q < deg; q++) {
    if (cols[q] == d1) { selfSlot = q; break; }
  }
  if (selfSlot != 0xffffffffu && size1 > 0u) {
    let inv = 1.0 / f32(size1);
    for (var lv = 0u; lv < 8u; lv++) {
      let v = cVerts[d1 * 8u + lv];
      if (v == 0xffffffffu) { continue; }
      for (var a = 0u; a < 3u; a++) {
        acc[selfSlot * 9u + a * 3u + a] += inv * diagBuf[v * 3u + a];
      }
    }
  }
  for (var q = 0u; q < deg; q++) {
    for (var e = 0u; e < 9u; e++) {
      cBlockValues[(rowLo + q) * 9u + e] = acc[q * 9u + e];
    }
  }
}

// ---- inner-PCG scope (own bindings; counts from persistent slots) ----
@group(0) @binding(20) var<storage, read> sRows : array<u32>;
@group(0) @binding(21) var<storage, read> sCols : array<u32>;
@group(0) @binding(22) var<storage, read> sVals : array<f32>;
@group(0) @binding(23) var<storage, read> sX : array<f32>;
@group(0) @binding(24) var<storage, read_write> sY : array<f32>;
@group(0) @binding(25) var<uniform> sCount : u32; // coarse dofs (3D)

fn blockRowOf(i : u32) -> u32 { return i / 3u; }

@compute @workgroup_size(64)
fn coarse_spmv(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= sCount) { return; }
  let d1 = blockRowOf(i);
  let a = i % 3u;
  var y = 0.0;
  for (var p = sRows[d1]; p < sRows[d1 + 1u]; p++) {
    let d2 = sCols[p];
    // mirror block q = (d2,d1) for the transpose average (exact symmetry)
    var q = 0xffffffffu;
    for (var r = sRows[d2]; r < sRows[d2 + 1u]; r++) {
      if (sCols[r] == d1) { q = r; break; }
    }
    for (var b = 0u; b < 3u; b++) {
      var fwd = sVals[p * 9u + a * 3u + b];
      var bwd = fwd;
      if (q != 0xffffffffu) { bwd = sVals[q * 9u + b * 3u + a]; }
      y += 0.5 * (fwd + bwd) * sX[d2 * 3u + b];
    }
  }
  sY[i] = y;
}

// Diagonal-block inverse with the G5 Sylvester latch; fallback = 1/fb.
fn diagBlockInv(
  B00 : f32, B01 : f32, B02 : f32, B11 : f32, B12 : f32, B22 : f32,
  fb0 : f32, fb1 : f32, fb2 : f32,
  out00 : ptr<function, f32>, out01 : ptr<function, f32>, out02 : ptr<function, f32>,
  out11 : ptr<function, f32>, out12 : ptr<function, f32>, out22 : ptr<function, f32>,
) {
  var tr = abs(B00) + abs(B11) + abs(B22);
  var tiny = 1e-20 * (1.0 + tr);
  var ok = true;
  if (!(B00 > tiny)) { ok = false; }
  if (ok && !(B00 * B11 - B01 * B01 > tiny * (1.0 + abs(B00) + abs(B11)))) { ok = false; }
  var det = B00 * (B11 * B22 - B12 * B12) - B01 * (B01 * B22 - B12 * B02)
    + B02 * (B01 * B12 - B11 * B02);
  if (ok && !(det > tiny * (1.0 + tr) * (1.0 + tr))) { ok = false; }
  if (ok) {
    var s = 1.0 / det;
    *out00 = (B11 * B22 - B12 * B12) * s;
    *out01 = (B02 * B12 - B01 * B22) * s;
    *out02 = (B01 * B12 - B11 * B02) * s;
    *out11 = (B00 * B22 - B02 * B02) * s;
    *out12 = (B01 * B02 - B00 * B12) * s;
    *out22 = (B00 * B11 - B01 * B01) * s;
    return;
  }
  if (fb0 > 1e-12) { *out00 = 1.0 / fb0; } else { *out00 = 0.0; }
  *out01 = 0.0;
  *out02 = 0.0;
  if (fb1 > 1e-12) { *out11 = 1.0 / fb1; } else { *out11 = 0.0; }
  *out12 = 0.0;
  if (fb2 > 1e-12) { *out22 = 1.0 / fb2; } else { *out22 = 0.0; }
}

@group(0) @binding(30) var<storage, read> zB : array<f32>; // inner b (masCoarseR)
@group(0) @binding(31) var<storage, read_write> zX : array<f32>;
@group(0) @binding(32) var<storage, read_write> zR : array<f32>;
@group(0) @binding(33) var<storage, read_write> zZ : array<f32>;
@group(0) @binding(34) var<storage, read_write> zP : array<f32>;
@group(0) @binding(35) var<storage, read_write> zProd : array<f32>;
@group(0) @binding(36) var<storage, read> zDiag : array<f32>; // true Ac diagonal
@group(0) @binding(37) var<uniform> zCount : u32;

// z = Dblock^-1 r with on-the-fly symmetrized diagonal adjugate.
// Reads sRows/sCols/sVals (bindings 20-22): every entry calling this helper
// must bind the coarse pattern/value buffers at 20-22 (auto-layout merges
// them into the caller's bind-group layout).
fn applyDiagBlock(r0 : f32, r1 : f32, r2 : f32, d : u32, zc : ptr<function, vec3f>) {
  var slot = 0xffffffffu;
  for (var p = sRows[d]; p < sRows[d + 1u]; p++) {
    if (sCols[p] == d) { slot = p; break; }
  }
  var B00 = 0.0; var B01 = 0.0; var B02 = 0.0;
  var B11 = 0.0; var B12 = 0.0; var B22 = 0.0;
  if (slot != 0xffffffffu) {
    B00 = sVals[slot * 9u];
    B01 = 0.5 * (sVals[slot * 9u + 1u] + sVals[slot * 9u + 3u]);
    B02 = 0.5 * (sVals[slot * 9u + 2u] + sVals[slot * 9u + 6u]);
    B11 = sVals[slot * 9u + 4u];
    B12 = 0.5 * (sVals[slot * 9u + 5u] + sVals[slot * 9u + 7u]);
    B22 = sVals[slot * 9u + 8u];
  }
  var I00 = 0.0; var I01 = 0.0; var I02 = 0.0;
  var I11 = 0.0; var I12 = 0.0; var I22 = 0.0;
  diagBlockInv(B00, B01, B02, B11, B12, B22,
    zDiag[d * 3u], zDiag[d * 3u + 1u], zDiag[d * 3u + 2u],
    &I00, &I01, &I02, &I11, &I12, &I22);
  *zc = vec3f(
    I00 * r0 + I01 * r1 + I02 * r2,
    I01 * r0 + I11 * r1 + I12 * r2,
    I02 * r0 + I12 * r1 + I22 * r2);
}

@compute @workgroup_size(64)
fn c_init(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= zCount) { return; }
  let d = blockRowOf(i);
  let a = i % 3u;
  // One thread owns dof i end-to-end (no cross-thread writes): the 3-vector
  // is re-read (not shared) and each thread keeps only its component.
  zX[i] = 0.0;
  let ri = zB[i];
  zR[i] = ri;
  var zc = vec3f(0.0);
  applyDiagBlock(zB[d * 3u], zB[d * 3u + 1u], zB[d * 3u + 2u], d, &zc);
  var zi = 0.0;
  if (a == 0u) { zi = zc.x; } else if (a == 1u) { zi = zc.y; } else { zi = zc.z; }
  zZ[i] = zi;
  zP[i] = zi;
  zProd[i] = ri * zi;
}

@group(0) @binding(40) var<storage, read_write> uX : array<f32>;
@group(0) @binding(41) var<storage, read_write> uR : array<f32>;
@group(0) @binding(42) var<storage, read> uP : array<f32>;
@group(0) @binding(43) var<storage, read> uAp : array<f32>;
@group(0) @binding(44) var<storage, read> uAlpha : array<f32>;
@group(0) @binding(45) var<uniform> uCount : u32;

@compute @workgroup_size(64)
fn c_update_xr(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= uCount) { return; }
  let alpha = uAlpha[0];
  uX[i] += alpha * uP[i];
  uR[i] -= alpha * uAp[i];
}

@compute @workgroup_size(64)
fn c_update_z(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= zCount) { return; }
  let d = blockRowOf(i);
  let a = i % 3u;
  let r0 = zR[d * 3u];
  let r1 = zR[d * 3u + 1u];
  let r2 = zR[d * 3u + 2u];
  var zc = vec3f(0.0);
  applyDiagBlock(r0, r1, r2, d, &zc);
  var zi = 0.0;
  if (a == 0u) { zi = zc.x; } else if (a == 1u) { zi = zc.y; } else { zi = zc.z; }
  zZ[i] = zi;
  zProd[i] = zR[i] * zi;
}

@group(0) @binding(50) var<storage, read> pZ : array<f32>;
@group(0) @binding(51) var<storage, read_write> pP : array<f32>;
@group(0) @binding(52) var<storage, read> pBeta : array<f32>;
@group(0) @binding(53) var<uniform> pCount : u32;

// Elementwise multiply on the persistent count slot (blas mul needs the
// shared BlasParams slot, whose write would flush the encoder and add a
// submit per inner iteration — this keeps the inner loop submit-free).
@group(0) @binding(54) var<storage, read> mX : array<f32>;
@group(0) @binding(55) var<storage, read> mY : array<f32>;
@group(0) @binding(56) var<storage, read_write> mOut : array<f32>;
@group(0) @binding(57) var<uniform> mCount : u32;

@compute @workgroup_size(64)
fn c_mul(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= mCount) { return; }
  mOut[i] = mX[i] * mY[i];
}

// Inner breakdown latch: coarseBreak = 1 when pAp <= 1e-30 (hardcoded,
// matching the default outer pApTol; the latch is diagnostic — the inner
// solve simply yields fewer effective iterations, and sdiv's own guard keeps
// every lane finite). No uniforms => no added submits.
@group(0) @binding(58) var<storage, read> bPap : array<f32>; // reduceScratch[0]
@group(0) @binding(59) var<storage, read_write> bFlag : array<f32>; // coarseBreak

@compute @workgroup_size(64)
fn c_break(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x + gid.y * 4194240u != 0u) { return; }
  if (bPap[0] <= 1e-30) {
    bFlag[0] = 1.0;
  }
}

@compute @workgroup_size(64)
fn c_update_p2(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x + gid.y * 4194240u;
  if (i >= pCount) { return; }
  pP[i] = pZ[i] + pBeta[0] * pP[i];
}

// pcg-reduce twins on persistent coarse slots (no per-iteration uniform write,
// hence no added submits inside the inner loop).
var<workgroup> cTile : array<f32, 64>;

@group(0) @binding(60) var<storage, read> cSrc : array<f32>;
@group(0) @binding(61) var<storage, read_write> cPartial : array<f32>;
@group(0) @binding(62) var<uniform> cCount : u32;

@compute @workgroup_size(64)
fn coarse_reduce_stage1(
  @builtin(global_invocation_id) gid : vec3u,
  @builtin(local_invocation_id) lid : vec3u,
  @builtin(workgroup_id) wid : vec3u,
) {
  let i = gid.x + gid.y * 4194240u;
  var v = 0.0;
  if (i < cCount) { v = cSrc[i]; }
  cTile[lid.x] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { cTile[lid.x] += cTile[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { cPartial[wid.x] = cTile[0]; }
}

@group(0) @binding(63) var<uniform> cNumGroups : u32;

@compute @workgroup_size(64)
fn coarse_reduce_stage2(
  @builtin(local_invocation_id) lid : vec3u,
) {
  var acc = 0.0;
  for (var g = lid.x; g < cNumGroups; g += 64u) {
    acc += cPartial[g];
  }
  cTile[lid.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { cTile[lid.x] += cTile[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { cPartial[0] = cTile[0]; }
}
