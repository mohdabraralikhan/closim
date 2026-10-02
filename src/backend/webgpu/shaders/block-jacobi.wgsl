// G5B block-Jacobi preconditioner data: per-vertex membrane 3x3 + Jacobi
// diagonal, factored once per Newton iteration (not per PCG iteration).
//
// bj_build_factor (one thread per vertex):
//   1. membrane 3x3 Hvv from incident triangles (CSR), analytic entries
//      restricted to the incident corner (same closed form as
//      membrane-hvp.wgsl extended one more product rule — see
//      src/physics/membrane-blocks.ts, verified against FD).
//   2. B = Hvv + diagEmbed, where diagEmbed comes from the `diag` buffer
//      (Jacobi diagonal: M/h^2 + beta + contact curvature, already built).
//   3. Sylvester PD check; success -> blockInv = adj(B)/det, flag 1;
//      fail -> blockInv = diag(1/d) (Jacobi fallback pre-baked), flag 0.
// The PCG kernels (pcg_init_bj / pcg_update_z_bj) then do z = blockInv * r
// with NO branch — fallback is data, and pins are filtered as usual.
// Bending stays out (inexact Newton, same as the HVP operator).

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
@group(0) @binding(9) var<storage, read_write> blockInv : array<f32>; // n*9
@group(0) @binding(10) var<storage, read_write> blockFlag : array<u32>; // n

@group(0) @binding(11) var<uniform> matC00 : f32;
@group(0) @binding(12) var<uniform> matC11 : f32;
@group(0) @binding(13) var<uniform> matC01 : f32;
@group(0) @binding(14) var<uniform> matG : f32;
@group(0) @binding(15) var<uniform> matThickness : f32;

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

@compute @workgroup_size(64)
fn bj_build_factor(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x + gid.y * 4194240u;
  if (v >= params.vertexCount) { return; }
  // membrane 3x3 block (corner-restricted analytic entries, CSR gather)
  var H00 = 0.0; var H01 = 0.0; var H02 = 0.0;
  var H11 = 0.0; var H12 = 0.0; var H22 = 0.0;
  for (var k = elemOffsets[v]; k < elemOffsets[v + 1u]; k++) {
    let t = elemIds[k];
    let c = elemCorners[k];
    let i0 = triangles[t * 3u];
    let i1 = triangles[t * 3u + 1u];
    let i2 = triangles[t * 3u + 2u];
    let x0 = position[i0].xyz;
    let x1 = position[i1].xyz;
    let x2 = position[i2].xyz;
    let e1 = x1 - x0;
    let e2 = x2 - x0;
    let ia = dmInv[t].x; let ib = dmInv[t].y; let ic = dmInv[t].z; let id = dmInv[t].w;
    // K row for corner c: K0=(-(a+c),-(b+d)), K1=(a,b), K2=(c,d)
    var K0 = 0.0; var K1 = 0.0;
    if (c == 0u) { K0 = -(ia + ic); K1 = -(ib + id); }
    else if (c == 1u) { K0 = ia; K1 = ib; }
    else { K0 = ic; K1 = id; }
    var F00 = 0.0; var F10 = 0.0; var F20 = 0.0;
    var F01 = 0.0; var F11 = 0.0; var F21 = 0.0;
    // NOTE: F needs ALL corners' edges (global F, not corner-restricted).
    // Recompute edges per corner branch below via full-triangle F:
    F00 = ia * e1.x + ic * e2.x;
    F10 = ia * e1.y + ic * e2.y;
    F20 = ia * e1.z + ic * e2.z;
    F01 = ib * e1.x + id * e2.x;
    F11 = ib * e1.y + id * e2.y;
    F21 = ib * e1.z + id * e2.z;
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
    // H_ab for a,b in 0..2 (symmetric): s * sum_j K[c][j] * dP_ajb.
    // Unrolled a<=b with explicit F-vector selection (Fa = column a of F).
    for (var a = 0u; a < 3u; a++) {
      var Fa0 = 0.0; var Fa1 = 0.0;
      if (a == 0u) { Fa0 = F00; Fa1 = F01; }
      else if (a == 1u) { Fa0 = F10; Fa1 = F11; }
      else { Fa0 = F20; Fa1 = F21; }
      for (var b = a; b < 3u; b++) {
        var Fb0 = 0.0; var Fb1 = 0.0;
        if (b == 0u) { Fb0 = F00; Fb1 = F01; }
        else if (b == 1u) { Fb0 = F10; Fb1 = F11; }
        else { Fb0 = F20; Fb1 = F21; }
        // dP_ajb = sum_k (K[c][k]*S_kj + Fa_k * sum_lm D_kjlm * dE_lmb)
        // with K[c] = (K0,K1) and S rows selected by j below.
        var entry = 0.0;
        for (var j = 0u; j < 2u; j++) {
          var Kj = 0.0;
          if (j == 0u) { Kj = K0; } else { Kj = K1; }
          for (var kk = 0u; kk < 2u; kk++) {
            var Kk = 0.0;
            if (kk == 0u) { Kk = K0; } else { Kk = K1; }
            // dF_ik/dx_B = delta(b,i)*K[cv][k] with i fixed to a by the outer
            // M[A] factor: the stress term only survives for a == b. (Without
            // this guard the off-diagonal block entries overcount under
            // stress; invisible at rest where S ~= 0.)
            if (a == b) {
              var Skj = 0.0;
              if (kk == 0u && j == 0u) { Skj = S00; }
              else if (kk == 1u && j == 1u) { Skj = S11; }
              else { Skj = S01; }
              entry += Kj * (Kk * Skj);
            }
            // Fa_k * sum_lm D_kjlm * dE_lmb, dE_lmb = 0.5*(K[c][l]*Fb_m + Fb_l*K[c][m])
            var Fak = 0.0;
            if (kk == 0u) { Fak = Fa0; } else { Fak = Fa1; }
            if (Fak != 0.0) {
              var dsum = 0.0;
              for (var l = 0u; l < 2u; l++) {
                var Kl = 0.0;
                if (l == 0u) { Kl = K0; } else { Kl = K1; }
                for (var m = 0u; m < 2u; m++) {
                  var Km = 0.0;
                  if (m == 0u) { Km = K0; } else { Km = K1; }
                  var dd = Dmod(kk, j, l, m);
                  if (dd != 0.0) {
                    var Fbl = 0.0; var Fbm = 0.0;
                    if (l == 0u) { Fbl = Fb0; } else { Fbl = Fb1; }
                    if (m == 0u) { Fbm = Fb0; } else { Fbm = Fb1; }
                    dsum += dd * 0.5 * (Kl * Fbm + Fbl * Km);
                  }
                }
              }
              entry += Kj * Fak * dsum;
            }
          }
        }
        let e = s * entry;
        if (a == 0u && b == 0u) { H00 += e; }
        else if (a == 0u && b == 1u) { H01 += e; }
        else if (a == 0u && b == 2u) { H02 += e; }
        else if (a == 1u && b == 1u) { H11 += e; }
        else if (a == 1u && b == 2u) { H12 += e; }
        else { H22 += e; }
      }
    }
  }
  // B = Hvv + diag-embedded Jacobi diagonal
  let d0 = diagBuf[v * 3u]; let d1 = diagBuf[v * 3u + 1u]; let d2 = diagBuf[v * 3u + 2u];
  let B00 = H00 + d0; let B11 = H11 + d1; let B22 = H22 + d2;
  let B01 = H01; let B02 = H02; let B12 = H12;
  // Sylvester PD check (relative to diagonal scale)
  let tr = abs(B00) + abs(B11) + abs(B22);
  let tiny = 1e-20 * (1.0 + tr);
  var ok = 1u;
  if (!(B00 > tiny)) { ok = 0u; }
  if (ok == 1u && !(B00 * B11 - B01 * B01 > tiny * (1.0 + abs(B00) + abs(B11)))) { ok = 0u; }
  let det = B00 * (B11 * B22 - B12 * B12) - B01 * (B01 * B22 - B12 * B02) + B02 * (B01 * B12 - B11 * B02);
  if (ok == 1u && !(det > tiny * (1.0 + tr) * (1.0 + tr))) { ok = 0u; }
  let base = v * 9u;
  if (ok == 1u) {
    let inv = 1.0 / det;
    blockInv[base]     = (B11 * B22 - B12 * B12) * inv;
    blockInv[base + 1u] = (B02 * B12 - B01 * B22) * inv;
    blockInv[base + 2u] = (B01 * B12 - B11 * B02) * inv;
    blockInv[base + 3u] = (B02 * B12 - B01 * B22) * inv;
    blockInv[base + 4u] = (B00 * B22 - B02 * B02) * inv;
    blockInv[base + 5u] = (B01 * B02 - B00 * B12) * inv;
    blockInv[base + 6u] = (B01 * B12 - B11 * B02) * inv;
    blockInv[base + 7u] = (B01 * B02 - B00 * B12) * inv;
    blockInv[base + 8u] = (B00 * B11 - B01 * B01) * inv;
    blockFlag[v] = 1u;
  } else {
    // Jacobi fallback pre-baked (z = r/d row-wise, no branch in PCG).
    var i0 = 0.0; var i1 = 0.0; var i2 = 0.0;
    if (d0 > 1e-12) { i0 = 1.0 / d0; }
    if (d1 > 1e-12) { i1 = 1.0 / d1; }
    if (d2 > 1e-12) { i2 = 1.0 / d2; }
    blockInv[base] = i0; blockInv[base + 1u] = 0.0; blockInv[base + 2u] = 0.0;
    blockInv[base + 3u] = 0.0; blockInv[base + 4u] = i1; blockInv[base + 5u] = 0.0;
    blockInv[base + 6u] = 0.0; blockInv[base + 7u] = 0.0; blockInv[base + 8u] = i2;
    blockFlag[v] = 0u;
  }
}
