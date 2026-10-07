# G6C adversarial break report

**Branch/worktree:** `adversarial/g6c-break` at `162a689` in `C:\Users\Abrar\Desktop\closim-adversarial`  
**Device:** real Dawn/WebGPU execution  
**Production edits:** none

## Results

| Attack | Result | Evidence |
|---|---|---|
| Accepted-step / round-boundary state sync | **FAIL — broken** | A01 confirms the post-commit `commit_apply` calculates a second `alpha * dx`. In A02, an accepted round leaves `contactCount=98` where the initial evaluation had 49; re-derivation at unchanged position changes gradient by `2.097e-3`, contact distance by `1.055e-5`, and contact diagonal by `4.625`. |
| Global trial index for non-uniform batches | **FAIL — broken** | A08’s dynamic `[2,4,4]` physics scene rejected every candidate, so that scenario was inconclusive. New deterministic A13 shader probe seeds cumulative base 6, batch index 2, width 4, local index 1: shader reports 9, while the global index is 7. |
| Trust scaling | **PASS** | A09.3 sets trust scale to 0.5 and verifies raw alpha 1.0 is evaluated and reported as effective alpha 0.5. |
| Validation error propagation / zero status | **FAIL — broken** | Dawn reports invalid bind-group and layout errors in A11 when the test manually pushes a scope. The executor has no scoped or uncaptured-error handler. A13 feeds a zero status to the solver control fast path: `finite=0`, `gradNorm=0`, yet it returns `converged=true`. A07 also observes a validation-failed readback returning zeros. |
| Accepted status after trailing rejected batch | **FAIL — broken** | A09.4 accepts the first batch, rejects a later one, then `round_report` returns `minDistance=Infinity` and `minToi=Infinity` from the rejected batch. |

The A01/A02 and A13 cases are reproducers for the failures above. The G6C source in this worktree still contains the affected paths: `commit_apply` is unconditional, `commit_arm` uses `batchIndex * currentWidth + localIndex`, and the solver convergence fast path checks the gradient norm without requiring a valid status. This checkout is at the same commit as local `master`, so the handover’s stated fixes were not present in the inspected source.

## Verification

- Focused G6C/device adversarial run: **5 files, 16 tests passed**; defect assertions confirm the listed breaks.
- New deterministic A13 probes: **2 tests passed**.
- Full suite: **64 files, 267 tests; 263 passed, 4 failed**. Three failures are in the existing `tests/adversarial/a12-sort-stress.test.ts` legacy-sort stress assertions; one is `tests/webgpu/g6c-newton-e2e.test.ts` Gate B contact-count assertion. These failures are outside this task’s edits.
- No production source was changed. The worktree adds only `tests/adversarial/a13-g6c-index-status.test.ts` and this report.

## Follow-up

The defects are confirmed and should be repaired by the primary implementer before treating the G6C control path as hardened. The separate G6C.1 bitonic-sort prototype remains on its own branch/worktree.
