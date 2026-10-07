# G6C Newton State-Machine Audit

## Scope and result

Audited the `validation-hardening` worktree on branch `validation-hardening`,
covering `gpu-newton.ts`, `gpu-solver.ts`, `newton-control.wgsl`, `armijo.wgsl`,
`diagnostics.wgsl`, buffer layouts, and the G6B/G6C device tests. The primary
checkout was left untouched.

Four correctness issues were reproduced and fixed:

1. **Fallback metrics described the discarded PCG direction.** After
   `select_fallback` replaced `searchDirection`, Armijo still used the old
   `gtdxStore` and `trustScaleStore`. The round now recomputes both from the
   selected direction before any trial.
2. **Converged rounds without a commit were reported as failures.**
   `round_report` now distinguishes `done` convergence from reject-all.
   A no-commit report uses alpha `0` and the invalid-index sentinel in either
   case; only a non-converged no-commit round is a failure.
3. **Batch latches could describe the wrong state.** Updating `e0Store` at
   first acceptance changed the Armijo reference for later speculative
   batches, while the final rejected batch could overwrite accepted-state
   distance/TOI. Round-start `E0` now stays fixed until `round_report`;
   accepted energy, gradient norm, distance, and TOI are latched at first
   acceptance. Current-state distance/TOI are seeded by `newton_check` for
   rounds without a commit.
4. **NaN-only finite checks admitted infinities.** In particular, `-Inf`
   energy could pass the Armijo inequality when the incoming finite bit was
   true, and `-Inf` `gtdx` could pass the descent gate. Diagnostics, Armijo,
   and the descent gate now reject NaN and both infinities using a
   Dawn-compatible finite-range check.

No adaptive batch-width policy was added or changed. Non-uniform widths are
covered to verify the existing cumulative-index behavior.

## State and buffer ownership

| State | Sources of truth | Sticky state | Reused/cleared state | Commit and rejection rules |
|---|---|---|---|---|
| `NEWTON_READY` | `position` is the accepted iterate; refreshed contact/FEM records, `gradient`, `rhs`, and diagnostics describe it. `e0Store` is the accepted energy at round start. | `laggedN` survives trials and rounds until an accepted update. | Per-round `newtonCtl` and `newtonStatus` are zeroed. `simParams.contactCount` is set to capacity for cap-bounded loops; the compacted `contactCount` counter remains the live count. | No candidate may mutate `position` or `laggedN`. |
| `DIRECTION_READY` | PCG writes `searchDirection`; `rhs` is the total residual (`g`) used for the descent dot. The initial dot is used only to decide whether to fall back. | PCG breakdown stays latched in `breakFlag`/the report lane for this solve. | `descentDir` is Jacobi fallback scratch. The selected direction is either PCG output or the copied fallback. `gtdxStore` and `trustScaleStore` are recomputed from that selected direction. | Neither direction is committed to position. Armijo slope and trust scaling must refer to the same selected vector. |
| `ARMIJO_BATCH` | `position` is the immutable batch base; `armijoCandidates` rows and `armijoStatus` describe the current batch. `selectedAlpha` is the effective alpha (trust folded); selected index is cumulative batch base plus local index. | Per-round failure flags OR overflow/CCD/barrier failures across evaluated candidates. They are aggregate line-search diagnostics, not proof that the committed iterate is unsafe. | `xTrial` is overwritten per candidate. Each candidate clears contact records to sentinels and resets contact/overflow counters before rebuilding. `armijoStatus` is reset per batch; only the active K candidate rows are read. `E0` remains the round-start value for every batch. | Rejected candidate records, gradients, RHS, contact counters, and friction state are scratch; they cannot advance position or lagged friction. |
| `COMMIT` | `newtonCtl[2]` is the sticky “some batch accepted” latch. On the first acceptance, control/status lanes latch alpha, batch, global index, energy, gradient norm, distance, and TOI. | First acceptance wins; later accepts cannot replace it. | `commit_apply` rematerializes `xTrial` at the latched alpha because the last trial may have overwritten it. `rebuildTrialPasses` clears and reconstructs contacts/FEM/RHS at that state. | Only after acceptance does `commit_copy_if` copy to `position`, and `commit_lagged_if` update `laggedN`. At report time, accepted energy is copied into `e0Store` for the next round. |
| `CONVERGENCE` | Fresh `solverStatus` at round start feeds `newton_check`; `newtonCtl[0]` and `newtonStatus[1]` record done/converged. The initial step also has a CPU-side zero-round fast path. | Convergence is sticky for the remainder of that round. | The bounded PCG/trial passes still execute in a converged GPU-controlled round; `commit_arm` gates acceptance on `!done`. | No position or friction commit occurs. The report is converged, not failed, with alpha zero and index `-1`. |
| `NEXT_NEWTON` | The compact status read after the report controls the CPU loop. After an accepted round, position and rebuilt records agree; E0 is the accepted energy. | Per-step trial history and fallback evidence stay on the solver for the next width decision/report. | The next round refreshes RHS/Jacobi and diagnostics from accepted state; per-round controls are cleared at entry. | Continue only if neither converged nor failed. |
| `FAILURE` | For reject-all, `newtonCtl[2]` remains zero and `round_report` sets failure unless the round was converged. Device validation errors are separate thrown execution failures. | Failure flags describe evaluated candidates; no rejected candidate is made current. | Reject-all rematerializes alpha zero and rebuilds records at the base state. Device error scopes are popped before the compact status read. | Reject-all leaves `position` and `laggedN` unchanged. Validation failure throws before stale/zero status can be decoded as convergence and follows the solver fallback path. |
| `DONE` | Final solver diagnostics and convergence/failure result. | Last accepted iterate remains authoritative even when the Newton iteration budget ends. | No extra candidate state survives past the round report. | Budget exhaustion after accepted work is finite but non-converged; reject-all is failure. Immediate convergence performs no PCG solve or round. |

### Field-specific invariants

- **`selectedAlpha` / `selectedTrialIndex`:** accepted alpha is trust-folded;
  index is the raw line-search trial index. Rejected or converged-without-commit
  reports alpha zero and decode the f32 `0xffffffff` sentinel as `-1`.
- **`E0`:** seeded once per controlled step; read unchanged by every batch in a
  round; replaced with accepted energy only after round selection completes.
- **`gtdx` / `trustScale`:** the first `gtdx` is only the PCG descent check.
  The final stored slope and scale are recomputed from the actual chosen
  direction, including fallback.
- **`newtonConverged`:** GPU-control commit gating uses `newtonCtl[0]`, set
  from the fresh Xk norm. The Armijo batch’s legacy convergence lane is not the
  source of truth for GPU-controlled rounds.
- **`contactCount` / `contactOverflow`:** compacted `contactCount` is the live
  record count. The uniform count is deliberately the capacity during
  cap-bounded trial/rebuild loops; inactive tail slots must be sentinel-cleared.
  Overflow is reset per candidate then ORed into round diagnostics.
- **`laggedN`:** survives every rejected candidate and reject-all round. It is
  updated only after accepted-state records have been rebuilt; the rebuild
  itself uses the round-start lagged values.
- **`contactDist`:** candidate-local scratch is cleared before contact
  generation. The end-of-round rebuild leaves it describing the accepted
  position, or the unchanged base position on reject/convergence.
- **`gradient` / `rhs`:** `gradient` is internal assembled force; `rhs` is the
  total residual including inertia and pin filtering. Candidate evaluation
  rebuilds both at `xTrial`; the final rebuild uses accepted `xTrial` or
  alpha-zero base state. Convergence diagnostics use the RHS norm.
- **`searchDirection`:** PCG output is scratch and may be replaced by
  `-rhs/diag`. Armijo and trust calculations occur only after that choice.

## Adversarial cases

| Case | Expected transition/result | Audit evidence |
|---|---|---|
| Accept first | First valid row latches; later batches cannot change the committed state. | `g6c-newton-parity.test.ts`, `g6c-trial-index.test.ts` |
| Accept after reject | Rejected batches do not mutate E0/position/laggedN; cumulative base advances by each actual width. | `g6c-trial-index.test.ts` |
| All-invalid batch | No commit; alpha zero, index sentinel, failure unless already converged; rejected trial state is rebuilt away. | `g6c-newton-e2e.test.ts`, `g6b-armijo-parity.test.ts` |
| Multiple batches | E0 stays fixed through the whole round; only first acceptance latches metrics; variable-width indices use cumulative widths. | `g6c-trial-index.test.ts` |
| `trustScale < 1` | Scale is `trust/max(abs(selected dx))`, including fallback; candidate alpha is raw alpha times this scale. | `g6c-trust.test.ts`, `g6c-newton-parity.test.ts` |
| PCG breakdown | Breakdown selects Jacobi descent; recomputed slope and trust reflect that replacement vector. | `g6c-newton-parity.test.ts` |
| Non-descent fallback | Nonnegative, NaN, or infinite dot falls back; negative infinity is explicitly rejected. | `g6c-newton-parity.test.ts` |
| Newton budget exhaustion | Last accepted state remains current; converged stays false unless a fresh convergence check passed. | `gpu-solver.ts:stepNewtonGpuControlled` and `stepGpuDevice` |
| Immediate/no-solve convergence | Initial gradient check exits before a Newton round/PCG solve. In-round convergence cannot commit stale candidates or report reject failure. | `gpu-solver.ts:stepNewtonGpuControlled`, `g6c-trial-index.test.ts` |
| Contact overflow | Candidate is rejected; per-round overflow diagnostics remain sticky, while reject-all leaves position/friction unchanged. | `g6b-armijo-ladder.test.ts`, `g6c-newton-e2e.test.ts` |
| CCD failure | Unsafe TOI or contact failure rejects the candidate; it cannot become the committed state. | `armijo.wgsl:armijo_record`, `g6b-armijo-ladder.test.ts` |
| Barrier failure | `minDistance <= dMin` rejects the candidate; failure flags are carried to the report. | `armijo.wgsl:armijo_record`, `g6b-armijo-ladder.test.ts` |
| Device validation failure | Scoped errors throw before status decode; the solver takes its established fallback path. | `validation-scope.test.ts` |

## Future batch-width constraints

- Candidate storage and indexed apply entries are fixed at eight rows/entries:
  every batch width must remain in `[1, 8]` unless those resources and entry
  points are expanded together.
- A batch’s global index is `sum(previous batch widths) + local index`; it is
  not `batchIndex * currentWidth`. The CPU-generated alpha and GPU control
  lane use this same cumulative base.
- The f32 control lanes can exactly represent integer indices only up to
  `2^24`; current Newton trial budgets are far below that bound.
- The controlled round still evaluates all statically scheduled batches after
  first acceptance. Later candidates cannot replace the accepted state, but
  sticky overflow/CCD/barrier bits remain aggregate evidence for speculative
  trials, including post-acceptance trials. Consumers must not interpret those
  aggregate flags as a committed-state-only diagnosis.
- Capacity-bounded loops rely on sentinel-cleared tail records. Changing the
  uniform back to live contact count without changing clear/loop bounds can
  re-expose stale contact data.

## Regression coverage and verification

Added device reproducers in `g6c-newton-parity.test.ts`,
`g6c-trial-index.test.ts`, and `g6b-armijo-ladder.test.ts` for fallback metrics,
negative-infinite descent, acceptance latches across later batches, convergence
without commit, infinite RHS/energy rejection, and current-state reporting.

Verified with `npm run build` and targeted Vitest runs covering:

- `g6b-armijo-ladder.test.ts`
- `g6c-newton-parity.test.ts`
- `g6c-trial-index.test.ts`
- `g6c-newton-e2e.test.ts`
- `g6c-state-sync.test.ts`
- `g6c-trust.test.ts`
- `validation-scope.test.ts`

