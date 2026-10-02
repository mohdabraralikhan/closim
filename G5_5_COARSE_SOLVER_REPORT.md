# G5.5 — Assembled GPU Coarse Solve: Report

Hardware: GTX 1050 (Dawn `webgpu@0.6.1`). CPU numbers: Node headless, same
machine (deterministic iters; ms indicative). Production default unchanged
(Jacobi); all G5/G5.5 methods remain opt-in candidates behind flags.

## Verdict (decision rule applied)

**Coarse PCG at 8 inner iterations (c1-8) beats Schwarz-1 fairly on the
stiff regimes → promote c1-8 to production candidate.** The single-sweep
forms (mas2 damped-Jacobi AND C0 block-Jacobi) do NOT beat Schwarz-1 and
stay experimental. Recommendation: flip the default to c1-8 as a separate
follow-up with submit-cap re-baselining (locked tests assert current caps);
this change keeps every existing regression green by leaving defaults off.

Fair-comparison headline (identical 10k-10x Newton system, 20 outer iters):

| method | resNorm | submits/solve | passes/solve |
|--------|---------|---------------|--------------|
| schwarz1 | 7.52e+1 | 44 | 351 |
| c1-8 | 1.30e+1 | 46 (+2) | 2684 (+2333, same encoder) |

5.8× lower residual for +2 submits (+5%), zero added syncs. The +2333
passes are tiny coarse-vector ops inside the existing single encoder.

## 1. Architecture (invariants held)

- `Ap` = exact existing fine HVP in every solve (never replaced).
- Coarse system exists only inside M⁻¹: restrict → Ac solve → prolongate.
- Ac = R A P, membrane + Jacobi-diag spread only (approved scope);
  barrier/contact terms stay in the exact fine HVP.
- 3×3 block-CSR (`coarseRowOffsets/ColIndices/BlockRows/BlockValues`);
  pattern STATIC (built pre-layout, exact-sized, degree cap 32 asserted),
  values refreshed per Newton state.
- Raw blocks stored unsymmetrized; the OPERATOR is symmetrized at read
  time (spmv/z-step average B with Bᵀ) — no transpose race, no temp buffer.
- Inner loop uses dedicated coarse vectors/scalars/slots and persistent
  count uniforms: zero uniform rewrites ⇒ zero added submits or syncs
  (verified: strip 12→12, 50k +1; the 10k end-to-end +519 below is
  Newton-path line-search behavior, not structural — see §5).
- Pins: prolongation pin-filters; pinned DOFs read exactly 0 through the
  full correction (gated by test).
- Cross-aggregate contacts: dedicated 4 B atomic counter (`mas_contact_span`),
  read alongside the contact-count sync (no extra sync point). Fold scene:
  device 1425/1435 spanning — exactly matching the CPU mirror.

## 2. Correctness gates (all green)

| check | agreement |
|-------|-----------|
| pattern vs CPU (exact) | bit-identical incl. blockRows |
| block values, rest / stressed | 3.02e-7 / 3.87e-7 (domain-scale) |
| device SpMV vs CPU operator | 2.46e-7 |
| on-device uᵀAcv ≈ vᵀAcu (randomized) | 8.4e-8 (nonzero both sides) |
| CPU bilinear probe | < 1e-12 |
| C0 correction vs restrict/BJ/prolongate | 5.0e-7 |
| C1 K=4 / K=8 vs CPU fixed-K coarsePcg | 4.8e-7 / 3.5e-7 |
| coarse diag vs diag(RAP) | 6.1e-8 (prior, unchanged path) |
| coarse-PCG vs CPU Cholesky (transitive via above + `coarse-csr` oracle tests) | — |

Bugs found by measurement (fixed):
1. `mas.wgsl` rejects `array<atomic<u32>>` with `read` access (WGSL rule) —
   counter read binding must be `read_write` (load-only use).
2. Test-only: `Float64Array` probe vector uploaded into an f32 buffer
   (2× bytes, silent zeros on readback). No implementation impact; noted
   as a harness footgun.
3. Test-only: CPU parity references must apply the pin filter (GPU
   prolongation suppresses pinned DOFs by design — the gate working).

## 3. Solver ladder (the experiment)

CPU PCG iterations to 1e-6 (same system per cell; cap 200):

| cell | jac | blk | s1 | mas2 | masX | c0 | c1-4 | c1-8 | c1-32 |
|------|-----|-----|----|------|------|----|------|------|-------|
| 1x/none | 157 | 83 | 61 | 65 | 54 | 66 | 58 | 54 | 54 |
| 1x/floor | 73 | 64 | 38 | 38 | 36 | 43 | 44 | 36 | 36 |
| 2x/none | 194 | 102 | 79 | 85 | 63 | 83 | 76 | 63 | 63 |
| 2x/floor | 77 | 69 | 40 | 41 | 40 | 46 | 49 | 40 | 40 |
| 5x/none | 200✗ | 140 | 109 | 121 | 92 | 122 | 99 | 93 | 92 |
| 5x/floor | 97 | 71 | 42 | 46 | 43 | 49 | 69 | 44 | 43 |
| 10x/none | 200✗ | 171 | 120 | 136 | 111 | 145 | 122 | 113 | 112 |
| 10x/floor | 87 | 73 | 44 | 47 | 46 | 51 | 67 | 46 | 46 |
| 10xbend ≡ 1x exactly (bending not in Hessian — reproduced) |

Readings:
- **c1-8 ≡ masX on every cell**: 8 inner iterations already equal the exact
  coarse solve at these sizes; c1-32 adds nothing (saturation).
- c1-4 is partial and can hurt (5x/floor: 69 vs 42; device self-10x:
  breakdown latch fires, recovered at ≥8). Minimum sensible rung is 8.
- **c0 never beats schwarz1** (66 vs 61, 43 vs 38, 145 vs 120): a single
  block-Jacobi coarse correction, like the damped sweep, is too crude —
  both single-sweep forms stay experimental.
- Jacobi non-convergent ≥5x free flight (G5 regime reproduced).

Device ladder, strip scene, 20 outer iters (resNorm):

| memK | s1 | c0 | c1-4 | c1-8 | c1-16 | c1-32 | submits |
|------|----|----|------|------|-------|-------|---------|
| 1x | 1.16e-4 | 3.49e-4 | 1.06e-4 | 3.30e-5 | 3.29e-5 | 3.29e-5 | 85–87 flat |
| 10x | 2.05e-3 | 3.88e-3 | 1.34e-3 | 3.30e-4 | 3.30e-4 | 3.30e-4 | 85–86 flat |

3.5× (1x) / 6.2× (10x) residual win at c1-8 with flat submits; 16/32 add
nothing. c0 worse than schwarz1 in both rows.

## 4. Reduced decision matrix (device steps, ni=2/pcg=20)

| cell | s1 res | c1-8 res | s1/c1-8 submits | syncs | span |
|------|--------|----------|-----------------|-------|------|
| 10k none 1x | 6.08e-7 | 6.91e-8 (8.8×) | 1141/1142 | 7/7 | 0/0 |
| 10k none 10x | 1.04e-5 | 5.33e-6 (~2×) | 1395/1914 | 9/13 | 0/0 |
| 10k floor 1x/10x | ≡ none rows (floor unengaged in 2 steps) | | | | 0/0 |
| 10k self 1x | 2.22 | 6.6e-2 | 2715/3725 | 17/24 | 12 vs 0 |
| 10k self 10x | 0.58 | 0.54 | 2565/3727 | 16/24 | 0 vs 0 |
| 50k none 10x | 1.85e-5 | 1.19e-5 (c1-8, +1 submit) / 1.13e-5 (c1-32, +161) | 1696/1697/1857 | 9/9/10 | 0 |

All cells finite, no breakdown except self-10x c1-4 (latch works as
designed). Memory identical per size (preallocated; coarse adds ~nnz×36 B
values + 6 small vectors — ≈2.5 MB at 50k scale).

Two honest caveats, both measured:
1. **Contact cells diverge trajectories**: self-1x span reads 12 (s1) vs 0
   (c1-8) — different search directions walk different Newton paths within
   2 chaotic steps, so cross-method resNorm on contact cells compares
   different linear systems. Fixed-state comparisons (§3 structural,
   ladder, CPU table) are the fair signal; matrix contact cells gate
   finiteness + coverage, not ranking.
2. **End-to-end submit deltas mix preconditioner cost with line-search path
   dependence** (different dx ⇒ different Armijo trial counts). The
   isolated structural number is +2 submits/solve; the 10k +519 rides on
   Newton-path differences (50k shows +1 on the same code). Wall is never
   ranked (first-cell compile + CPU encode effects dominate; e.g. 11 s
   cold walls).

## 5. Regime guidance (no single score)

- Stiff free flight (the StiffGIPC regime, plain StVK): c1-8 is the method —
  biggest wins, saturates at 8 inner iters, zero sync cost, +2 submits/solve.
- Contact-localized (floor settling): one-level Schwarz captures nearly
  everything; coarse adds little (barrier springs localize). c1-8 harmless,
  not needed.
- Chaotic contact (self/fold/friction, few steps): compare fixed-state only;
  require K≥8 (c1-4 can trip the breakdown latch).
- Never ship single-sweep coarse in any form (damped-Jacobi or
  block-Jacobi): measured worse-than-nothing more often than not.
- Next branch if coarse stalls at larger scale: sparser/larger aggregates
  or adaptive hierarchy (AGIPC-class) — explicitly out of G5.5 per scope.

## 6. Deliverables / exit checklist

- [x] Block-CSR Ac: CPU reference (`src/solver/coarse-csr.ts`) + GPU
      assemble/SpMV/inner-PCG (`shaders/coarse.wgsl`) + span counter
      (`mas_contact_span`)
- [x] Ladder C0 (block-Jacobi) / C1 (4/8/16/32) / C2 (CPU dense oracle,
      pre-existing, reused)
- [x] Same-encoder inner loop: zero added submits (strip, 50k) and zero
      added syncs by construction (no uniform rewrites, no scalar reads)
- [x] Gates: pattern/value/symmetry/bilinear/R/P/cPCG-vs-oracle/pins/full
      correction/span/submits all green; stress cells finite
- [x] Jacobi default + descent fallback untouched; coarse flags default off
- [x] No AGIPC / HSC / barrier reformulation / adaptive hierarchy
- [x] Prior tests green, `tsc --noEmit` clean

Tests added: `coarse-csr` (9 CPU), `webgpu/coarse-parity` (5),
`webgpu/coarse-solve` (5: C0/C1 parity, span exact-match, structural +
submit regression), `webgpu/coarse-ladder` (1), `webgpu/coarse-matrix`
(3); `preconditioner-compare` extended with c0/c1-4/c1-8/c1-32.
New sources: `src/solver/coarse-csr.ts`, `shaders/coarse.wgsl`;
extended: `gpu-buffers` (pattern/value/vector/scalar/counter buffers,
CoarseCount/CoarseGroups slots), `gpu-pipelines`, `gpu-newton`
(useCoarseC0/useCoarsePcg/coarseIters + assemble/solve passes),
`gpu-solver` (pre-layout pattern build, upload, zeroing), `mas.wgsl`
(span counter).
