# G5 — Preconditioner / Subspace Investigation: Report

Hardware: GTX 1050 (Dawn `webgpu@0.6.1`), same scenes/runtimes throughout.
CPU numbers: Node headless on the same machine (indicative timings, deterministic iters).
Device numbers: real WebGPU execution. Production defaults unchanged
(Jacobi + analytic HVP + gather assembly); all G5 methods are opt-in candidates.

Exit gate (from the G5 brief): measurement matrix + block Jacobi + one-level
Schwarz implemented and benchmarked; two-level MAS added (stable); Jacobi and
descent fallback kept; no AGIPC / heterogeneous-subspace / reformulation work.

## 1. G5A — PCG profile (measurement first)

`tests/webgpu/pcg-matrix.test.ts` runs stiffness (1x/2x/5x/10x membrane,
10x bending) × contact (none/floor/self/fold/friction) × size (1k/10k/50k)
on the real device at fixed production budgets (ni=2, pcg=20 — iteration
counts are constant by design; the behavioral data are residual curves,
Newton convergence, submits, syncs, wall). `tests/webgpu/pcg-profile.test.ts`
covers the instrumentation itself (profiled ≡ unprofiled solve, timestamp
self-check). Timestamp-query is feature-detected with a wall-clock fallback:
on this backend all queries latched within one submit read the same value
(verified: inter-submit deltas track wall at ~1 ns, intra-submit deltas are
0), so pass-granularity GPU/overhead splits are NOT claimed here — wall +
submits + syncs are the primary signals.

Observed 50k cells (all finite, none converged under the ni=2 probe budget):

| cell | resNorm (20 iters) | relRzEnd | submits | syncs | wall ms/step |
|------|-------------------|----------|---------|-------|--------------|
| none/1x | — (see note) | — | — | — | — |
| none/5x | 1.40e-5 | — | 1692 | 9 | 501 |
| none/10x | 1.86e-5 | 6.77e-3 | 1368 | 7 | 342 |
| none/10xbend | 3.57e-5 | 6.97e-3 | 4770 | 28 | 1257 |
| floor/1x | 4.25e-6 | 9.77e-4 | 1372 | 7 | 2629 |
| floor/2x | 9.24e-6 | 2.63e-3 | 2825 | 16 | 714 |
| floor/5x | 1.40e-5 | — | 1692 | 9 | 522 |
| floor/10x | 1.65e-5 | 6.77e-3 | 1368 | 7 | 393 |
| floor/10xbend | 3.54e-5 | 6.97e-3 | 4770 | 28 | 1279 |

Notes:
- Residual after fixed 20 PCG iterations grows monotonically with stiffness
  (floor: 4.25e-6 → 9.24e-6 → 1.40e-5 → 1.65e-5 → 3.54e-5): the simulator
  DOES enter the stiffness/conditioning regime StiffGIPC targets. Jacobi
  deteriorates; the physics is unchanged, the solver is the bottleneck.
- Floor/5x, /10x, /10xbend duplicate the none-cell submits/syncs/energies:
  at 50k the cloth (5 cm hover) has not reached the floor in 2 steps, so the
  contact sets are empty and the trajectories coincide (wall differs by
  submission-latency noise). Floor-contact differentiation at 50k needs
  longer runs — the CPU harness (§3) covers the contact regime instead.
- Wall variance across cells is large on this backend (342–2629 ms); the
  run is submission-latency dominated, not FLOP dominated — consistent with
  the G4 finding (flat ~50 ms PCG across sizes). Do not rank methods by wall.
- Device memory at 50k: 1419 MB resident (see §5 for per-method deltas).

## 2. CPU regime comparison (PCG iterations per Newton system)

`tests/preconditioner-compare.test.ts`: one representative Newton linear
system per (stiffness × contact) cell on a 6×6 tensile patch (PD-ish),
RHS = deterministic sine field, tol 1e-6, cap 200. Exact same HVP operator
for every method (preconditioner-only comparison). masX = exact-coarse
reference (dense Ac + Cholesky, CPU only — headroom probe, not shippable).

PCG iterations (residuals all ≤ 4.2e-6 except Jacobi-nonconverged cells):

| cell | jacobi | block | schwarz1 | mas2 | masX |
|------|--------|-------|----------|------|------|
| 1x/none | 157 | 83 | 61 | 65 | 54 |
| 1x/floor | 73 | 64 | 38 | 38 | 36 |
| 2x/none | 194 | 102 | 79 | 85 | 63 |
| 2x/floor | 77 | 69 | 40 | 41 | 40 |
| 5x/none | 200✗ (2.1e-4) | 140 | 109 | 121 | 92 |
| 5x/floor | 97 | 71 | 42 | 46 | 43 |
| 10x/none | 200✗ (2.7e-3) | 171 | 120 | 136 | 111 |
| 10x/floor | 87 | 73 | 44 | 47 | 46 |
| 10xbend/none | 157 | 83 | 61 | 65 | 54 |
| 10xbend/floor | 73 | 64 | 38 | 38 | 36 |

✗ = hit the 200 cap without converging. 10xbend ≡ 1x bit-for-bit in
iteration counts: bending lives in the gradient only (not the Hessian), so
bending stiffness cannot affect PCG — confirmed, not assumed.

Readings:
- Jacobi FAILS to converge at 5x/10x free flight (the StiffGIPC regime,
  reached with plain membrane stiffness — no exotic materials needed).
- Block-Jacobi wins 12–47% of Jacobi iterations everywhere; XYZ coupling
  alone is worth roughly 1.2–2× in iteration count.
- Schwarz-1 wins substantially over block (61 vs 83 at 1x, 120 vs 171 at
  10x): connectivity-aware patches capture the membrane coupling Jacobi and
  block both drop.
- mas2 (single damped-Jacobi coarse sweep, ω=0.5) tracks schwarz1 ± noise
  and is sometimes slightly worse (121 vs 109, 136 vs 120): one sweep is too
  crude a coarse solver on these systems (see §4).
- masX (exact coarse) beats schwarz1 on every free-flight cell (54 vs 61,
  63 vs 79, 92 vs 109, 111 vs 120) and ties on floor cells (36/38, 40/40,
  43/42, 46/44): the coarse SPACE is good (constant-per-aggregate captures
  the soft rigid-like modes); floor-barrier springs localize the system and
  shrink the coarse win. Scale probe (separate, 10x free flight):
  grid6: 120 → 111, grid12 (11 domains): 183 → 147, grid20 (28 domains):
  240 → 174 — the exact-coarse gap WIDENS with domain count (classic MAS
  scaling: 8% → 20% → 28%).

## 3. Device solve comparison (strip scene, 20 fixed PCG iters, GTX 1050)

`tests/webgpu/preconditioner-solve-compare.test.ts`, same Newton state:

| method | resNorm | relRzEnd | submits |
|--------|---------|----------|---------|
| jacobi | 1.13e-2 | 1.61e-3 | 86 |
| block | 7.00e-3 | 6.73e-4 | 85 |
| schwarz1 | 1.16e-4 | 9.00e-8 | 85 |
| mas2 | 2.59e-4 | 2.91e-7 | 87 |

schwarz1 reaches ~100× lower residual than Jacobi for the same 20
iterations and the same submit count (factor builds stay inside the
single PCG encoder — zero extra syncs). mas2-single trails schwarz1 here,
consistent with the CPU finding. No breakdowns, no NaN/Inf anywhere.

## 4. Cost breakdown (where each method helps and what it costs)

CPU factor/apply timings are indicative (49-vert patch, JIT-warm variance);
the structural facts travel, the milliseconds don't:

| method | PCG iters (regime) | precond time | HVP time | sync | submits | memory |
|--------|-------------------|--------------|----------|------|---------|--------|
| jacobi | baseline; non-convergent ≥5x free | 0 (diag reuse) | 1× (unchanged) | baseline | baseline | 0 |
| block | 0.5–0.9× jacobi | one 3×3 factor/vert per solve; apply ~1 vector op | unchanged | +0 | +0 (same encoder) | n·9 f32 + n u32 |
| schwarz1 | 0.35–0.55× jacobi | one 24×24 Cholesky/domain per solve; apply = gather/matvec/scatter | unchanged | +0 | +0 (same encoder) | nDoms·576·2 f32 + maps |
| mas2 (1-sweep) | ≈ schwarz1 (±noise; occasionally worse) | + restrict/scale/prolongate per iter (small kernels) | unchanged | +0 | +2/solve (coarse-diag build rides the factor encoder) | +3 coarse vecs |
| masX (exact, CPU ref) | 0.75–0.92× schwarz1, gap widening with scale | dense Ac (small only) | unchanged | n/a (CPU) | n/a | dense (3D)² — NOT shippable at scale |

The exact HVP is byte-identical for all methods (never replaced, only
preconditioned against). Sync/submit counts are flat across methods on
device: every factor build and every apply lives inside the existing
single-encoder PCG solve.

## 5. Device parity (correctness evidence, not speedup claims)

| check | agreement |
|-------|-----------|
| bj_build_factor vs buildBlockFactors, rest | 3.77e-7 (flags exact) |
| bj_build_factor vs buildBlockFactors, 10% stretch + shear | 5.62e-7 (flags exact) |
| schwarz assemble+factor vs buildSchwarzFactors, rest | ~4e-7 domain-scale |
| schwarz assemble+factor vs buildSchwarzFactors, stressed | 4.36e-7 domain-scale |
| mas coarse diag vs diag(RAP) | 6.11e-8 |
| mas restrict vs restrictCoarse | 6.11e-8 |

## 6. Bugs found by G5 measurement (fixed)

1. `bj_build_factor` missed the δ(b,a) guard on the stress term
   (dF_ik/dx_B = δ(b,i)·K[cv][k] with i fixed to a): off-diagonal block
   entries were 86% wrong under 10%+ shear stress (9.74 vs 5.24 on a probe
   element). Invisible at rest (S≈0) — which is exactly why the stressed
   parity cell exists now. CPU reference was correct throughout.
2. `schwarz.wgsl` was written with the guard from the start (same formula
   family) — verified by the stressed parity cell, not by inspection.
3. WGSL `let` reassignment in `bj_build_factor` (compile error) and two
   over-bound `diag` bindings on the bj PCG kernels (Dawn auto-layout
   prunes unused bindings; over-binding fails validation) — both caught by
   the first device run of the new tests.

## 7. Regime guidance (no single score — per the brief)

- Soft / contact-dominated (floor settling at 1–2x): Jacobi converges but
  wastes ~2× iterations vs schwarz1; floor-barrier springs localize the
  system, so one-level Schwarz captures nearly everything and the coarse
  level adds little. Use schwarz1; skip the coarse level here.
- Stiff free flight (≥5x membrane,None — flag-like, sails, sails): Jacobi
  does not converge in 200 iterations; block-Jacobi converges slowly;
  schwarz1 is the first robust method; exact-coarse MAS extends the lead
  with problem size. This is the StiffGIPC regime and our measurements
  reproduce its premise on plain StVK cloth.
- Bending stiffness is irrelevant to the solver (Hessian excludes it):
  do not spend preconditioner budget on bending.
- Single-sweep damped-Jacobi coarse correction (mas2 as shipped on GPU) is
  NOT recommended as a default: it matches schwarz1 at best and degrades
  it at worst. The coarse space is validated (masX); the sweep is the weak
  link. Next step is an assembled coarse solve on GPU (sparse Ac + inner
  PCG à la Wu–Wang–Wang / StiffGIPC connectivity-enhanced MAS), explicitly
  NOT a bigger single-sweep ω search (ω ∈ {0.25, 0.5, 1.0} all trail).
- AGIPC (in-solve algebraic coarsening) and heterogeneous subspace
  corrections stay out: the evidence now points at the coarse SOLVER as
  the next bottleneck, which neither branch addresses first.

## 8. Deliverables / exit checklist

- [x] G5A measurement matrix + PCG profile instrumentation (device) +
      CPU stiffness×contact comparison harness
- [x] G5B block Jacobi: CPU reference (`src/solver/block-jacobi.ts`),
      GPU factor + PCG kernels, wired into `pcgSolve` (default OFF),
      device parity at rest + stressed
- [x] G5C one-level Schwarz: CPU reference (`src/solver/schwarz.ts`,
      deterministic 8–32-vert connectivity+Morton domains, dense Cholesky
      + symmetric Jacobi fallback), GPU assemble/factor/apply
      (`schwarz.wgsl`), wired into `pcgSolve` (default OFF), device parity
- [x] G5D two-level MAS: CPU single-sweep mirror + exact-coarse reference
      (`src/solver/mas.ts`, normalized aggregation, Ac=RAP verified
      symmetric, coarse-solve verified), GPU restrict/scale/prolongate +
      true-Ac-diagonal (`mas.wgsl`), wired into `pcgSolve` (default OFF)
- [x] Jacobi baseline + descent fallback untouched and still the default
- [x] No AGIPC / HSC / B-spline FEM / contact reformulation
- [x] Prior tests green (38 files / 170 tests, was 30 / 139 at G4 exit), `tsc --noEmit` clean

Tests added (all green): `block-jacobi` (5), `schwarz` (8), `mas` (7),
`preconditioner-compare` (1), `webgpu/block-jacobi-parity` (3),
`webgpu/schwarz-parity` (3), `webgpu/mas-parity` (3),
`webgpu/preconditioner-solve-compare` (1).
New sources: `src/solver/block-jacobi.ts`, `src/solver/schwarz.ts`,
`src/solver/mas.ts`, `shaders/schwarz.wgsl`, `shaders/mas.wgsl`;
`src/math/pcg.ts` gained an optional `applyPreconditioner` hook
(default path byte-identical).
