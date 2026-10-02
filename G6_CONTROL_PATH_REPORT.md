# G6 — c1-8 Production Candidate + GPU-Resident Armijo: Report

Hardware: GTX 1050 (Dawn `webgpu@0.6.1`). Production default unchanged
(Jacobi); c1-8 and batched Armijo are explicit opt-ins. `timestamp-query`
stays optional (no change; still never a baseline requirement).

## Verdict

- **G6A complete**: `PreconditionerMode` (`jacobi`/`schwarz1`/`mas-c1-8`)
  wired with requested/actual/reason fallback accounting and a per-step
  report that splits fixed-state solver evidence from end-to-end trajectory
  data. Submit caps rebaselined per mode; Jacobi default untouched.
- **G6B complete**: GPU batched Armijo (K=2/4/8) evaluates candidates with
  **one status sync per batch**, selects first-valid on device, and commits
  through the identical code path as the sequential solver. Fixed-state
  selection matches the CPU loop **exactly** (alpha, index, merit,
  validity); full-step trajectories match **bit-exactly** (0.00e+0) on
  strip and floor. Per-trial CPU polling is gone.
- Do NOT flip any default yet (same rationale as G5.5): end-to-end submit
  counts remain line-search-path dependent. The data below is the rebaseline.

## 1. G6A — production candidate + accounting

Mode mapping is pure and unit-tested (`applyPreconditionerMode`,
`resolvePreconditioner`): `mas-c1-8` = Schwarz fine level + coarse-PCG K=8;
missing topology degrades any candidate toward Jacobi with a
`missing-topology` record; descent/biased directions record
`jacobi-descent` with `pcg-breakdown`/`non-descent-direction` reasons.
The step report carries requested/actual/fallback + submits/syncs/Newton/
fine/coarse iters/residual/energy + Armijo counters + per-Newton
submit/sync arrays. A `no-solve` marker distinguishes immediate convergence
from "ran Jacobi" (caught during testing: a settled floor scene converges
with zero solves).

Rebaseline, strip ni=2 hermetic resets (submit/syncs):

| mode | submits | syncs |
|------|---------|-------|
| jacobi | 365 | 3 |
| schwarz1 | 365 (+0) | 3 |
| mas-c1-8 | 367 (+2) | 3 |

Settled-state anchor remains 7d's 240 (transient vs settled documented in
the test). Fold c1-8: 2 Newton iters, 120 fine + 16 coarse iters, zero
fallbacks, 2897 contacts engaged, finite.

## 2. G6B — batched Armijo design (what was built)

Per batch of K (K≤8): CPU writes alphas + E0/gtdx/K + zeroed status (all
amortized, encoder-null, zero submits); ONE encoder runs K×
(clear → apply_k → G1/G2 → FEM → barrier/friction → diagnostics → record)
+ select; ONE 64 B status readback. Commit re-runs the accepted alpha
through apply + sync-free rebuild + existing accept (zero syncs; energy/
gradNorm bookkeeping comes from the batch status, bit-identical to a
re-read). Rejected trials never commit (commit only runs post-select).

Two load-bearing invariants make cap-bounded candidate loops exact:
1. Contact records sentinel-clear per candidate (dist=1e30, toi=2.0,
   prm=0); compaction appends live records (always dist<dHat) from zeroed
   counters. Two minimal kernel edits, both provably no-op on live data:
   barrier-gradient records dist only for active slots (1e30 otherwise),
   friction skips slots with `prm.x <= 0`.
2. Indexed `apply_0..7` entries (hardcoded k): WebGPU has no per-dispatch
   parameters except uniforms, and every uniform write flushes the encoder
   into a submit — static indices keep the batch in flight with zero
   uniform rewrites.

Overflow is GPU-stricter than the CPU path by design (invalid trial,
conservative): tiny-cap fold measures overflowFails=4, finite status,
graceful reject-all. Friction commits only via acceptTrial (kernel is
read-only on laggedN): accept changes laggedN, reject leaves it bitwise
identical.

## 3. Correctness gates (all green)

| check | result |
|-------|--------|
| fixed-state select, strip | li=0 both, alpha exact, 1 sync |
| per-candidate rows vs CPU trials | alpha/merit/validity exact |
| commit positions sequential vs batch | 0.00e+0 (bit-exact) |
| laggedN after both commits | bitwise identical |
| fold multi-batch parity (contact) | li=0 both, rows match |
| select unit: invalid-first / reject-all / all-valid / safe-Infinity | idx 1 / -1 / 0, flags correct |
| K=2/4/8 agreement | identical selection |
| friction commit vs reject | changed vs untouched |
| one-step trajectories strip/floor | 0.00e+0 positions, 0 energy rel |
| pins | preserved (existing gates) |

## 4. Cost evidence (honest split)

Fixed trial set, fold scene (accept-first): sequential 38 submits / 3 maps
vs batch 123 submits / 1 map. Maps 3→1 is the structural win; submits rise
by the speculative candidates. End-to-end submit deltas additionally mix
line-search path dependence (G5.5 §4 mechanism), so the report leads with
syncs and identical-state residuals, never with blended submit totals.
Per-trial maps collapse K×3+1 → 1 per batch; per-trial submits
(~30, dominated by sort-uniform flushes — the deferred G4 item) are
unchanged in kind.

## 5. Remaining bottlenecks (G6C and beyond)

1. Sort-uniform flush granularity dominates per-candidate submits
   (unchanged by batching; needs shader-side step derivation — deferred).
2. Newton-level CPU decisions remain (E0/gtdx/converge/commit scheduling);
   G6C removes them once batch stability is confirmed in broader runs.
3. Fixed K=4 speculates up to 3 wasted candidates on accept-first steps;
   the K=2/4/8 ladder gates make adaptive-K a safe follow-up (last trial
   count predicts the next batch width).
4. Overflow truncation order is atomic-race dependent (observed accept vs
   reject variance across runs at cap=8); validity stays conservative
   either way.

## 6. Deliverables / exit checklist

- [x] c1-8 production-wired (`PreconditionerMode`) with fallback accounting
      and fresh benchmark rebaseline; Jacobi remains default
- [x] GPU Armijo alpha selection; per-trial CPU polling gone (1 status
      readback per batch); fresh G1/G2 validity per alpha; rejected trials
      never commit; accepted state commits on GPU
- [x] Fixed-state parity (alpha/index/merit/validity/friction), K ladder,
      overflow/barrier/safe-Infinity gates, submit-reduction evidence,
      one-step stiff/floor/fold/friction parity (chaos rule respected)
- [x] No AGIPC / HSC / B-spline FEM / adaptive mesh / barrier reformulation
- [x] Prior tests green, `tsc --noEmit` clean

Tests added: `webgpu/g6a-modes` (6 unit + 4 device: report/fallback/
rebaseline), `webgpu/g6a-modes-stress` (floor/fold c1-8),
`webgpu/g6b-armijo-parity` (strip/fold/commit/select-unit),
`webgpu/g6b-armijo-ladder` (K ladder, overflow, friction, submit,
one-step parity).
New sources: `shaders/armijo.wgsl`, driver batch/select/rebuild methods,
solver batch search + instrumentation + report fields; buffers
(armijoAlphas/Candidates/Status/Cur) and ArmijoE0/Gtdx/K/PcgBd/NewtonConv
slots; two minimal kernel edits (barrier dist-sentinel, friction prm
guard) proven no-op on live data by the unchanged 193-test floor.
