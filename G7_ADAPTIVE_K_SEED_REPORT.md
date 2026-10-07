# G7 Seed: Adaptive-K Measurement

## Run

Ran `npx vitest run tests/webgpu/g6c-adaptive.test.ts` on the real WebGPU
device available to the `validation-hardening` worktree. All 4 tests passed.
This is an initial measurement, not a production-default recommendation.

## Observed trial histories

The distribution test ran three steps with two Newton iterations per regime,
comparing sequential and fixed-K batched paths:

| Regime | Sequential history | Batched history | Observation |
|---|---|---|---|
| strip, 1x stiffness | `[0,0,0,0,0,0]` | `[0,0,0,0,0,0]` | Six accept-first iterations on both paths. |
| strip, 10x stiffness | `[]` | `[]` | No trial history was recorded; this run provides no evidence about late-accept or reject behavior. |
| fold, 1x stiffness | `[0,0,0,0,0,0]` | `[0,0,0,0,0,0]` | Six accept-first iterations on both paths. |

The measured non-empty histories contain no late acceptance or exhausted
search. Therefore they support the accept-first case only; they do not yet
validate the policy's K=4/K=8 choices for late-accept or failure histories.

## Fixed-K4 versus adaptive-K

On one strip step with two Newton iterations, identical initial positions,
zeroed velocity, and batched Armijo enabled:

| Policy | Submits | Hot-loop syncs | Energy | Trial history |
|---|---:|---:|---:|---|
| Fixed K=4 | 315 | 3 | `3.4742658954201033e-6` | `[0,0]` |
| Adaptive K | 306 | 3 | `3.4742658954201033e-6` | `[0,0]` |

Adaptive-K reduced this single run by 9 submits (about 2.9%) with unchanged
sync count, energy, and accepted indices. This is consistent with shrinking
speculation after accept-first iterations, but is not a broad performance
claim. The test compares the batched-Armijo path; it does not measure adaptive
widths through the separate GPU-controlled Newton loop.

## G7 follow-up evidence needed

- Construct repeatable cases that accept at later trial indices and cases
  that exhaust the budget; record both paths rather than treating empty
  history as a sample.
- Measure more steps per regime and report per-regime sample counts,
  acceptance-index distributions, submit/sync deltas, and energy/trajectory
  parity.
- Exercise adaptive widths through both batched Armijo and GPU-controlled
  Newton, including non-uniform schedules, before changing defaults.

