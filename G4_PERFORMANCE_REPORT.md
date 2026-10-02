# G4 — Solver Throughput: Performance Report

Hardware/runtime/scenes identical before/after: GTX 1050 (Dawn `webgpu@0.6.1`),
same 1k/10k/50k floor-settling scenes, ni=2/pcg=20. "Before" = G3 exit numbers.

## Before → after (wall ms/step)

| size | before | after | speedup |
|------|--------|-------|---------|
| 1k   | 87     | 80    | 1.1x    |
| 10k  | 5200   | 576   | 9.0x    |
| 50k  | 74000  | 960   | 77x     |

## Stage probes (same benchmark)

| size | eval before → after | pcg before → after | submits before → after | syncs before → after |
|------|---------------------|--------------------|------------------------|----------------------|
| 1k   | 67 → 72 ms          | 1074 → ~50 ms      | 106 → 81               | 1 → 1                |
| 10k  | 140 → 112 ms        | 1285 → 59 ms       | 3106 → 698             | 18 → 4.5             |
| 50k  | 866 → 140 ms        | 29430 → 55 ms      | 3425 → 927             | 16.5 → 5             |

Notes:
- After-pcg for 1k is the re-measured warm value (~50 ms; probes 1,2 after a
  cold first probe at ~830–975 ms). The benchmark's single pcg probe per size
  catches 1k cold (first probe in the process). Same applies to the before
  column: steady-state before-1k-pcg is overstated the same way — the 10k/50k
  ratios are the clean comparison.
- Fewer Armijo trials post-G4 (10k: 18 → 4.5 syncs): exact Newton directions
  converge in fewer trials. Trajectory energies consistent (10k: 3.85e-9 →
  4.26e-9; 50k: 1.16e-8 → 1.33e-8).
- Strip scene steady state: 402 → 205 submits/step (locked by test 7d, cap 240).

## What changed

- G4A analytic membrane HVP (StVK chain v→dDs→dF→dE→dS→dP→dgrad, S01=2·G·E01):
  CPU analytic vs FD oracle worst rel err 3.87e-9 over 200 randomized
  element/material/direction cases; device vs mirror 1.31e-7 (FD path measured
  4.74e-3 on the same state). Production default ON; FD kept as debug oracle.
- G4B CSR gather (vertexElement{Offsets,Ids,Corners}, vertexHinge{...}): CPU
  gather bit-exact vs scatter; device HVP gather bit-exact vs scan, gradient
  ulp-level (3-term parenthesization). Production default ON; scans kept as oracle.
- G4C batching: uniform write-coalescing (CPU-write-only UNIFORM state only —
  provably safe, shaders cannot write UNIFORM bindings) + mark-stage masks
  (one dispatch per eval instead of ~20). PCG was already single-encoder with
  GPU-side alpha/beta.
- G4D: already status-only (64 B + 16 B + 4 B scalars); zero forbidden
  readbacks asserted every step (test 13 + 7d).

## Parity

129/129 green (112 pre-existing + 17 new: 8 analytic-HVP, 4 incidence,
3b/3c kernel, 7b/7c/7d solver), `tsc --noEmit` clean.

## What G4 did NOT fix (measured, next)

1. PCG cost is now flat across sizes (~50 ms at 1k/10k/50k) — fixed
   sync/submit overhead dominates, not GPU compute. Splitting sync latency
   from dispatch cost needs timestamp queries.
2. Bitonic sort uniforms (stage/sub change every sub-pass) are the largest
   remaining per-rebuild submit source. Collapsing them needs per-pass uniform
   slots or shader-side step derivation — structural, deferred.
3. The O(n·m)/O(n·h) reference scans remain as debug-oracle code paths only
   (production uses gather). Barrier/contact assembly stays O(n·C) over the
   compact active set (bounded, not scanned).
4. Preconditioning untouched (Jacobi). If PCG iteration counts grow on stiffer
   scenes, the G5 investigation is preconditioner/subspace/coarsening — to be
   decided by measured PCG behavior, not in advance.
