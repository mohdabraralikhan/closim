# G6/G6C Adversarial Validation Report

**Scope:** READ-ONLY against production implementation. No file under `src/` was
modified. New artefacts are confined to `tests/adversarial/` (8 files, 23 tests)
and this report.

**Harness status:** `npx tsc --noEmit` clean. `npx vitest run tests/adversarial/`
→ **23/23 pass** against the current (post-concurrent-fix) tree.

Full-suite figure **251 passed / 1 failed (252 total)**, with the single failure
**pre-existing** (see §6) — reproduced on the untouched baseline before any
adversarial file existed. That number was taken *before* the concurrent G6C
agent landed its fixes and added `tests/webgpu/g6c-adaptive.test.ts` +
`tests/webgpu/g6c-state-sync.test.ts`; a full re-run afterwards exceeded the
10-minute budget, so **treat 251/252 as a lower bound, not a current count.**
Re-run the suite yourself once the tree settles.

> **Convention used throughout.** A test that *proves a bug* asserts the buggy
> behaviour, so the suite stays green while the defect is recorded. Flipping an
> assertion to its correct value is the fix-verification step. Every such test is
> marked `// REGRESSION:` below.

---

## 0. Environment caveat — read before trusting any number here

| Property | Observed |
|---|---|
| `adapter.info` | `{"subgroupMatrixConfigs":[]}` — **no vendor / device / architecture** |
| Device granted | `maxStorageBuffersPerShaderStage = 16`, `maxComputeWorkgroupsPerDimension = 65535`, `maxUniformBuffersPerShaderStage = 12` |
| Default (unrequested) device | `maxStorageBuffersPerShaderStage = 8` |

**This is not the GTX 1050 rig the G5.5/G6 reports cite.** The adapter returns no
device identity, so it is a fallback/software path. Two consequences:

1. Every submit/pass/sync **count** in this report is a control-plane fact and
   transfers. Every **timing** would not — none are reported here.
2. The G6 reports' claims that were measured on the GTX 1050 are *not*
   re-validated by this run. Where a pre-existing test fails here (fold produces
   0 contacts), it may be adapter-specific rather than a real defect.

Note also: the task brief says "214/214 tests green". The baseline in this tree
is **229** tests, not 214. The count is stale.

---

## 0b. STATUS UPDATE — concurrent G6C work landed mid-review

The parallel G6C agent committed fixes to `src/` **while this review was
running** (`git diff --stat src/`: 6 files, +210/−48, with in-code comments
referencing "S1" and "adaptive-K"). I did not modify production code, but my
harness was re-run against the updated tree and **verifies three of the
findings are now fixed**:

| Finding | Pre-fix measurement (this tree, earlier) | Post-fix measurement | Status |
|---|---|---|---|
| **F-02** round-boundary desync | `\|gradient\| = 1.436e-2` (22 % of 6.539e-2 scale), `contactCount = 98` vs 49 | `\|gradient\| = 0.000e+0`, `contactCount = 49` | **FIXED** |
| **F-03** end-to-end drift, `newtonIters` 6 / 10 | `5.056e-10 → 1.515e-9` per step | `0.000e+0` at every budget (2 / 6 / 10) | **FIXED** |
| **F-04** global trial index | `bi*K+li`, 22 mismatching `(bi,li)` pairs over 5 ladders | `caCtl[8] = f32(base + li)` with a cumulative `caCtl[7]` base | **FIXED** |
| **F-01** `commit_apply` double-step | kernel lands `x+2αd` | kernel unchanged; **driver** now issues a single end-of-round commit, so it cannot double-apply | **FIXED (driver)** |
| **F-05** no error scopes | validation errors silently discarded; `simParams`/`uniformBank` readback returns zeros | unchanged | **OPEN** |
| **F-06** `sortCursor` underflow | silent no-op confirmed | unchanged | **OPEN** |
| **F-07** limits observability | 2 of 3 load-bearing limits unrecorded | unchanged | **OPEN** |

The landed fix moves `commit_apply` + `rebuildTrialPasses` + `commit_copy_if` +
`commit_lagged_if` **out of the batch loop** into a single end-of-round commit
("Position moves at most once per round, so records stay in sync with position
by construction"), and re-points `commit_copy_if` at the sticky `nbAccepted`
lane (2) instead of the per-batch `doCommit` lane (3). That is exactly the
remedy §1 proposed, and it also removes the wasted full G1+G2 rebuild per
rejected batch.

Two consequences for this report:

* **`a02-state-sync.test.ts` is now a permanent regression guard.** Its assertions
  were flipped from "prove the bug" to "enforce the invariant"
  (`dGrad < 1e-6`, `countAsLeft === ev.contactCount`). Re-running it against a
  regressed driver fails loudly.
* **`a01-commit-apply.test.ts` was relabelled** as a *kernel characterization*
  test. `commit_apply` itself is unchanged and remains a pure function of
  (`position`, `nbAlpha`, `dx`) — it will re-introduce the double-step the moment
  any caller invokes it more than once per round without `position` advancing.
  The test is now a tripwire for exactly that, not an assertion of a live bug.

**Still open and unaddressed by the concurrent work: F-05, F-06, F-07.** F-05 is
the one that matters most — while validation errors are silently discarded, a
bind-set mistake in the new end-of-round commit (which adds bindings across
`commit_arm` / `commit_apply` / `commit_copy_if` / `commit_lagged_if`) would fail
*quietly*, and every finiteness assertion in the suite would still pass because a
zeroed `newtonStatus` decodes as `energy = 0, gradNorm = 0`.


---

## 1. Confirmed defects

### F-01 — `commit_apply` re-applies the sticky alpha on an already-advanced `position`

| | |
|---|---|
| **Severity** | **S1** — wrong physics, silent |
| **Determinism** | Fully deterministic |
| **File / function** | `src/backend/webgpu/gpu-newton.ts` → `newtonRound()` L2018–2071; `src/backend/webgpu/shaders/newton-control.wgsl` → `commit_apply` |
| **Tests** | `tests/adversarial/a01-commit-apply.test.ts`, `a02-state-sync.test.ts` |

**Mechanism.** `newtonRound` always runs the *whole* Armijo ladder —
`batchKsFor()` never truncates on accept, so with K=4 / `armijoIters=10` every
round issues `[4,4,2]`. Only `commit_copy_if` and `commit_lagged_if` are
predicated on `newtonCtl[3]` (`doCommit`). `commit_apply` and
`rebuildTrialPasses` are **not**. `commit_apply` computes

```wgsl
let a = caCtlA[4];                              // nbAlpha — STICKY after an accept
var p = caXBase[v].xyz + a * vec3f(caDx[...]);   // caXBase = position
```

In the accepting batch `position` is still `x_k`, so this is correct. But
`commit_copy_if` then advances `position` to `x_k + α·dx`. In **every
subsequent rejecting batch** the same expression re-runs against the advanced
buffer:

```
xTrial = (x_k + α·dx) + α·dx = x_k + 2α·dx      ← phantom state
```

and `rebuildTrialPasses()` immediately rebuilds the entire contact record set,
`gradient`, `contactDiag` and `rhs` **at that phantom state**.

**Measured (tiny unpinned 3×2 grid, 6 verts, deterministic):**

```
|xTrial − (x + 2αd)| = 2.334e-8      ← rounding only; this is the real answer
|xTrial − (x +  αd)| = 1.000e-3      ← equals the full step α·dx = 5e-4 ×2
maxStep                              = 5.000e-4
```

**Smallest reproduction.** `a01-commit-apply.test.ts`, first case: write
`position = x`, `searchDirection = dx`, `newtonCtl = [0,0,1,0,0.5,…]` (sticky
alpha, `doCommit=0`), encode `commit_apply`, read `xTrial`; then write
`position = xTrial` (simulating the commit) and encode `commit_apply` again.

**Suggested fix.** Predicate `commit_apply` on `caCtlA[3] >= 0.5`, or (better)
stop the batch loop on the first accept. The unpredicated rebuild is also pure
wasted work: one full G1+G2+FEM per rejected batch.

**Would existing tests catch it?** **No.** The one reject-path gate
(`g6c-newton-e2e` "reject-all round") leaves `nbAlpha == 0`, so `commit_apply`
degenerates to `xTrial = position` — correct *by accident*. The accept-path
gates (`g6b-armijo-ladder`, `g6b-armijo-parity`) use the **G6B** path, which has
no batch loop and no `commit_apply`.

---

### F-02 — End-of-round state desync: `position` ≠ the buffers that describe it

| | |
|---|---|
| **Severity** | **S1** — wrong physics at the round boundary |
| **Determinism** | Deterministic |
| **File / function** | Same site as F-01; `rhsJacobi()` L811 is the consumer |
| **Tests** | `tests/adversarial/a02-state-sync.test.ts` |

**Contract under test.** After `newtonRound` returns, every buffer the *next*
round's `rhsJacobi` reads must describe `position`. `rhsJacobi` is
`contactDiagPass(); rhsAt("position"); jacobiOnly(beta)` — it **consumes**
`gradient`, `contactW/N/Id/Prm/Dist` and never rebuilds them.

**Measured** (resting-floor scene, 49 live contacts, round 0 accepted at trial 0,
`batchKs=[4,4,2]`):

```
|position|    = 0.000e+0     (sanity: re-derivation does not move it)
|contactW|     = 0.000e+0
|contactDist|  = 4.272e-8
|gradient|     = 1.436e-2     ← vs |gradientRef|max = 6.539e-2  ⇒  22 % of scale
|contactDiag|  = 0.000e+0
contactCount left in buffer = 98   (true live count = 49)
```

Two distinct corruptions:

* **`gradient` is ~22 % wrong.** `contactDiag` and the contact record set are
  *identical*, so the error is entirely the internal (membrane/hinge) gradient
  evaluated at `x_k + 2α·dx` instead of `x_k + α·dx`.
* **`contactCount` is left doubled (98 vs 49).** `rebuildTrialPasses` appends via
  `atomicAdd` onto a non-zero counter because `newtonRound` never calls
  `zeroContactCounters`. Slots `[49,98)` are **stale duplicate records** carrying
  the pre-commit state with non-zero `contactPrm`. They are then processed by
  `frictionPass`, `barrier-gradient`, `commit_lagged_if` and the `diagnostics`
  min-reduce, because `params.contactCount == cap` during GPU-control rounds.
  The `prm.x <= 0` and `dist >= dHat` guards are what keep them inert — they are
  not inert by design, only by the sentinel convention.

**Control that passes:** `batchKs = [1]` (accept on the *final* batch) leaves the
buffers in sync — `|gradient| = 0`. This is precisely why the existing
reject-all gate misses F-01/F-02.

**Suggested fix.** (a) predicate `commit_apply`/`rebuildTrialPasses` on
`doCommit`, or break the loop on accept (kills both); and (b) call
`zeroContactCounters()` at the top of `rebuildTrialPasses`, mirroring
`evaluateNewtonState`.

**Would existing tests catch it?** **No.** Gate B's "tight" scenes
(strip / floor / 5x / 10x) have **no engaged contact** at the start of a step, so
the active sets are empty in both states and the gradient difference is
unobservable. `fold` is the only contact scene and it is explicitly excluded from
the position comparison (`tight: false`).

---

### F-03 — F-02 is reachable end-to-end, but ~5 orders below the existing gate

| | |
|---|---|
| **Severity** | **S2** — latent, currently small |
| **Determinism** | Deterministic |
| **Tests** | `tests/adversarial/a06-e2e-contact.test.ts` |

Resting-floor scene, sequential vs GPU-control, both started from **one**
captured state, 3 steps:

| `newtonIters` | Newton rounds actually run | `trialHistory` | max position drift per step |
|---|---|---|---|
| 2 | 1 | `[-1]` (reject-all) | `0.000e+0, 0.000e+0, 0.000e+0` |
| 6 | 2 | `[8, -1]` | `5.056e-10, 1.011e-9, 1.515e-9` |
| 10 | 2 | `[8, -1]` | `5.056e-10, 1.011e-9, 1.515e-9` |

The desync **does** reach `stepGpuDevice`, but the observable drift is ~1e-9 —
**five orders of magnitude below Gate B's 1e-4 bar**. The magnitude is small
because round 1 begins near-converged (energy ≈ 1.5e-11), so a stale gradient has
little left to perturb.

Contact-free `stripScene` reproduces Gate B exactly (`0.000e+0`).

**Conclusion for the team:** F-02 is not currently a visible accuracy bug at
these budgets, but it is a real invariant violation whose blast radius grows with
(a) Newton budget, (b) how far from convergence a round starts, and (c) how much
the trial step moves the internal energy. Fix it before raising any of those.

**Suggested test to add to the main suite:** the F-02 invariant itself
(rebuild-at-`position` must be bitwise-equal), not a drift threshold.

---

### F-04 — `selectedTrialIndex` is wrong for non-uniform batch widths (adaptive-K landmine)

| | |
|---|---|
| **Severity** | **S2** — wrong telemetry, silently |
| **Determinism** | Deterministic |
| **File** | `src/backend/webgpu/shaders/newton-control.wgsl` → `commit_arm` |
| **Tests** | `tests/adversarial/a08-trial-index.test.ts` |

**Mechanism.**

```wgsl
caCtl[8] = f32(bi * caK + li);     // bi = batch index, caK = THIS batch's width
```

The true global index is `trialBase(bi) + li` where
`trialBase(b) = Σ_{j<b} K_j`. The identity only holds when **every batch has the
same width**.

It currently does not fail because `batchKsFor()` emits uniform widths except
possibly the last, *and* the last batch's narrower `ArmijoK` reads a **stale**
uniform. Measured: with `batchKs=[4,4,2]` the reported index is `8`
(= 2·4 + 0) while the truth is also 8 (= `trialBase(2)` + 0) — **correct by
coincidence**, via a different route.

**Pure-arithmetic proof over the ladders an adaptive scheme would emit:**

```
ladder [4,4,2]    bi=2 li=1 → reported=5  truth=5   (ok)
ladder [4,4,4]    all bi,li → consistent
ladder [8,2,4,4]  bi=1 li=0 → reported=2  truth=8   (Δ−6)
ladder [2,4,4]    bi=1 li=0 → reported=4  truth=2   (Δ+2)
ladder [8,1,1]    bi=1 li=0 → reported=1  truth=8   (Δ−7)
total mismatching (bi, li) pairs = 22
```

**Why this matters now:** `selectedTrialIndex` / `trialHistory` is exactly the
signal the documented G6C follow-up (**adaptive K**) is designed from ("last trial
count predicts the next batch width"). Shipping adaptive K before fixing this
corrupts the telemetry used to design the adaptive rule — a circular failure.

**Suggested fix.** Carry an explicit `trialBase` accumulator in `newtonCtl`
alongside `nbBatch` (the driver already maintains `trialBase` in JS at
`newtonRound`), and compute `caCtl[8] = f32(caCtl[trialBaseLane] + li)`.

**Would existing tests catch it?** **No.** All current widths are uniform, and no
existing test asserts the index against ground truth.

---

### F-05 — No error scopes: validation failures silently corrupt results

| | |
|---|---|
| **Severity** | **S1 systemic** — makes every future change silently unsafe |
| **Determinism** | Deterministic |
| **Files** | Whole stack; `gpu-executor.ts` `runPass` / `getPipeline` / `bindGroups` |
| **Tests** | `tests/adversarial/a07-memory-control.test.ts` A7.1–A7.3 |

Zero occurrences of `pushErrorScope` / `popErrorScope` / `onuncapturederror` in
`src/` (verified). Every pipeline uses `layout: "auto"`, so **both over-binding
and under-binding are validation errors**, and Dawn reports them as *uncaptured*:
the offending command buffer is dropped, every later submit and `mapAsync` still
succeeds, and readbacks return stale or zeroed data.

**Demonstrated live:**

```
A7.1  deliberately bound binding 99 on sort_next
      → captured: "In entries[1], binding index 99 not present in the bind
                  group layout."      (discarded by the current stack)

A7.2  readBufferDebug("simParams")   → allZero = true, len = 16
      readBufferDebug("uniformBank") → allZero = true, len = 4096
      forbiddenReadbacks delta       = 0        (not even counted)
      captured: "[Buffer \"g3/simParams\"] usage (BufferUsage::(CopyDst|Uniform))
                  doesn't include BufferUsage::CopySrc."

A7.3  control: readBufferDebug("position") → 3.5, validation error = NONE
```

Root cause of the A7.2 zeros: `gpu-solver.ts` `initDevice` allocates
`simParams` and `uniformBank` as `UNIFORM | COPY_DST` with **no `COPY_SRC`**, so
any attempt to read them back is an illegal copy.

**Why it passed small tests:** a zeroed `solverStatus` / `newtonStatus` decodes
as `energy = 0, gradNorm = 0`, which satisfies every
`expect(Number.isFinite(d.energy)).toBe(true)` and `expect(d.finite).toBe(1)`
assertion in the suite. The `no-solve` marker added in G6A is the only guard
against this specific decode, and it checks `pcgSolves`, not the status payload.

**Suggested fix.** `device.pushErrorScope("validation")` around each
`submitBatch`, or install `device.onuncapturederror` once in `GpuExecutor.create`
and count into the existing `SyncLedger`. Add `COPY_SRC` to the two uniform
buffers if they are ever to be inspected. Both are small, and both convert an
entire class of silent failures into loud ones.

---

### F-06 — `sortCursor` underflow silently no-ops the indexed sort

| | |
|---|---|
| **Severity** | **S3** — silent no-op, currently unreachable |
| **Determinism** | Deterministic |
| **File** | `src/backend/webgpu/shaders/broadphase-sort.wgsl` L97 |
| **Tests** | `a07-memory-control.test.ts` A7.7 |

```wgsl
fn sort_step_indexed(...) {
  let t = sortCursor[0] - 1u;      // underflows to 0xFFFFFFFF if cursor == 0
  let prm = sortParams[t];         // OOB storage read
```

WebGPU clamps OOB storage reads to zero, so `prm = (0,0,0,0)` and
`compareSwap` returns immediately on `i >= nIn`. The sort does **nothing** — no
error, no exception.

**Measured** (P = 64 lanes seeded descending, cursor deliberately left at 0):

```
cursor-underflow: P=64 stillSorted=false out[0]=64 out[63]=1
```

Unreachable today because `sortPassesIndexed` emits `sort_next` /
`sort_step_indexed` in lockstep, but there is no assertion protecting that
coupling, and Strategy B from the sort review (replacing the CPU
`sortCursor` reset with a `sort_reset` pass) removes the last CPU write and
makes the ordering purely implicit.

**Suggested fix.** Clamp `let t = min(sortCursor[0] - 1u, arrayLength(&sortParams) - 1u);`
and/or add a `sort_reset` entry point so the counter is provably 0 on entry.

---

### F-07 — Executor records only 3 of the limits it depends on

**Severity S4 (observability).** `GpuExecutor.create` copies only
`maxStorageBuffersPerShaderStage` and `maxStorageBufferBindingSize` into
`facts.limits`. Measured:

```
granted maxStorageBuffersPerShaderStage = 16
maxComputeWorkgroupsPerDimension       = undefined
maxUniformBuffersPerShaderStage        = undefined
```

The `16` request is load-bearing — the default is `8` and `contact-compact`
binds 15 storage buffers, `broadphase-traverse` 10, `diagnostics` 11. The other
two limits are load-bearing too (the `NO_SPLIT` guard hard-codes 65535) but are
invisible in the recorded facts, so a future adapter that grants a different cap
would not be visible in any report.

---

## 2. Verified clean (checked, not assumed)

These were probed adversarially and found sound. Recording them so they are not
re-litigated.

| Area | Probe | Result |
|---|---|---|
| Row-split / dispatch overflow | `NO_SPLIT` dispatch at 70 000 workgroups | **Throws loudly**: `broadphase-sort:sort_step_indexed needs 70000 workgroups, past the 65535 row-splittable cap for a reduction/sort`. No silent truncation. |
| Storage-buffer limit | adapter limits | `16` granted via the explicit `requestDevice`; default `8` insufficient for the 15-binding kernels. The request is correctly load-bearing. |
| Trial-state FEM, **with** contact | `a04-nan-bisect` full pass-by-pass bisection | `gradient`, `rhs`, `diag`, `contactDiag`, `contactForce`, `contactEnergy`, `elementGradient`, `elementEnergy`, `hingeGradient`, `xTrial`, `dmInv`, `restArea` — **0 non-finite at every stage**; `solverStatus.finite = 1`, `gradNorm = 5.0385e-1`. |
| Trial-state FEM, **without** contact | same, `floorOn = 0` | `finite = 1`, `gradNorm = 1.2381e-1`. |
| Membrane kernel in isolation | `a05-membrane-isolate` @ position and @ xTrial, plus forced (un-coalesced) uniform writes | `elementEnergy 0/72`, `elementGradient 0/648` in all four configurations. |
| Contact record hygiene | `a04` record scan | `count=49 live=49 minLiveDist=1.100e-3 overflow=0 zeroDHatSlots=0 nonfinitePrm=0` — the `prm.x<=0` and `dist>=dHat` guards never fire on live data, as documented. |
| Barrier / friction sentinels | same | `contactForce`, `contactEnergy` finite throughout the candidate evaluation. |
| PCG + trust + descent in a GPU round | `a03` instrumented round | `gtdx = −5.7812e-2` (descent), `stepNorm = 6.2152e-2`, `trustScale = 3.2179e-2`, `residual = 3.8478e-8`, `dirValid = true`, `merit = 1.7963e-3` — all finite and mutually consistent. |
| Armijo verdict integrity | `a03` candidate rows | `fin = 1`, no spurious overflow, correct first-valid-wins selection at `trialIdx = 0`. |
| `commit_lagged_if` vs `commit_lagged` | source diff | Byte-identical bodies, including `d >= dHat → keep stale` and the `max(d, 1e-12)` floor. |
| Safe-Infinity sentinels | `decodeSolverStatus` / `decodeArmijoStatus` / `decodeNewtonStatus` | `1e30` → `Infinity`, `2.0` → `Infinity` consistently across all three decoders. |
| Contact-free parity | `a06` vs Gate B | `0.000e+0` — reproduces Gate B exactly. |

---

## 3. Coverage against the brief

| Requested area | Status |
|---|---|
| Batched Armijo: reject-then-accept, accept-first, all-invalid, CCD-safe ∞, barrier reject, overflow, stale records, stale friction, wrong alpha, later-candidate overwrite | **Partially covered.** `a02`/`a03` exercise accept-first + trailing-reject and all-invalid. **Not yet covered:** CCD-safe ∞, tiny-capacity overflow (existing G6B gate covers it), duplicate contact keys, and explicit friction commit/reject under the GPU-control path (only the G6B path has that gate). See §5. |
| GPU Newton control: PCG breakdown, non-descent, trust region, budget, no-solve, nonfinite diagnostics, device failure flags | **Partially covered.** Budget and no-solve are measured in `a06`; trust/descent/residual measured in `a03`. **Not covered:** forced PCG breakdown latch and forced non-descent direction — `descent_check` is verified by source review only. |
| GPU memory/control: stale buffers, incomplete clears, atomic reuse, row overflow, storage limits, bind mismatch, race-sensitive counters | **Covered** (`a02` incomplete clears + counter doubling, `a07` row overflow / storage limit / bind mismatch / cursor underflow / COPY_SRC). **Not covered:** race-sensitive counter nondeterminism under overflow — see §5. |
| Contact: repeated keys, duplicate primitives, fold, near-dHat, hard-core epsilon, tiny capacity, deterministic vs nondeterministic overflow | **Not covered.** §5. |
| Sizes: tiny / 1k / 10k | **tiny (3×2, 6 verts) and 1k-class (6×6 cells = 49 verts) covered.** **10k/50k not covered** — the desync is state-level and size-independent, but the overflow-race question genuinely needs scale. |

---

## 4. Two false positives I produced and corrected

Recording these because they are the most transferable part of the run.

**FP-1 — "the gradient goes all-NaN and every candidate rejects."**
`solver.materialNow()` already returns the *mapped* form
`{c00, c11, c01, g, thickness}`. I re-mapped it with `mat.stretchWarp`, which is
`undefined` → `Float32Array[0] = NaN` → NaN material uniforms → NaN
`elementGradient`/`gradient` → `diagnostics` reports `(gradNorm = 0, finite = 0)`
→ all candidates rejected. A bisection (`a04`, pass-by-pass) localised it to
`femPasses`, and `a05` localised it further to my own argument rather than to the
kernel. **The production path is NaN-free** at position and at xTrial, with and
without contact.

The `(gradNorm = 0, finite = 0)` signature is worth remembering independently:
per `diagnostics.wgsl`, `if (is_nan(g)) { fin = 0.0 } else { g2 = g*g }` — a NaN
gradient yields `finite = 0` **and** `gradNorm = sqrt(0) = 0` simultaneously. Any
future NaN will present that way.

**FP-2 — "resting-floor drift is 8.175e-3, 82× Gate B's bar."**
`solver.getPositions()` refreshes `scene.positions` (snapshot on step). Re-reading
it before each run started the *second* path from the *first* path's output. The
true drift is `0.000e+0` at `newtonIters=2` and ~1e-9 at 6/10 (F-03). Gate B
captures `x0` once and is correct; the first revision of `a06` did not.

---

## 5. Gaps I did not close (recommended next)

| # | Gap | Why it matters |
|---|---|---|
| G1 | **Contact overflow race determinism.** The G6 report already notes "overflow truncation order is atomic-race dependent (observed accept vs reject variance at cap=8)". `a07` A7.6 was inconclusive on a contact-free strip (count 0). Needs a scene with live contacts and `cap ∈ {exact, 2×, 0.5×}`, repeated N times, asserting bitwise reproducibility. `determinism.test.ts` covers the CPU reference only. |
| G2 | **Forced PCG breakdown / non-descent direction.** `descent_check` and the `jacobi-descent` fallback are only verified by source review. Both are reachable by writing `breakFlag = 1` or by injecting a non-descent `searchDirection` before a round. |
| G3 | **CCD-safe ∞ and near-`dHat` boundary.** The `toi = 2.0 → Infinity` and `d ≥ dHat` paths have no device test at the boundary. |
| G4 | **Duplicate/repeated contact keys** and **tiny `contactCapacity`** under the *GPU-control* path (as opposed to the G6B path, which `g6b-armijo-ladder` does cover). |
| G5 | **10k / 50k coverage.** Everything here is ≤49 verts. F-01/F-02 are state-level and size-independent, but G1 is not. |
| G6 | **Error-scope coverage as a suite-wide gate.** Once F-05 is fixed, a single `expect(validationErrors === 0)` across all device tests would immediately surface any bind-set drift introduced by the concurrent G6C work. |

---

## 6. Pre-existing failure (not caused by this work)

```
FAIL tests/webgpu/g6c-newton-e2e.test.ts > Gate B scene coverage under GPU control
  AssertionError: expected 0 to be greater than 0
  at line 256:  expect(ev.contactCount).toBeGreaterThan(0);   // fold scene
```

The `fold` scene produces **0 contacts** on this adapter, so the Gate-B contact
assertion fails. Reproduced on the untouched baseline before any adversarial file
was created. Likely adapter-dependent (the fold scene is the only self-contact
case and needs LBVH/traverse to find pairs). **Flagging, not claiming** — it may
pass on the GTX 1050 rig.

---

## 7. Recommended order of work (updated)

1. ~~**F-01 + F-02**~~ — **done by the concurrent G6C work**, verified by
   `a02` (`|gradient| = 0`, `contactCount = 49`) and `a06` (drift `0.000e+0`).
3. ~~**F-04**~~ — **done**, `trialBase` is now cumulative.
2. **F-05** (error scopes) — **now the top item.** The new end-of-round commit
   spreads bindings across four entry points; without an error scope a bind-set
   mistake in any of them is silent, and the suite's finiteness assertions cannot
   detect a zeroed status. Half a day.
4. **F-06 + F-07** — cheap hardening. F-06 is a prerequisite for the `sort_reset`
   change already contemplated, since that makes the cursor ordering implicit.
5. **G1–G6** — fill the coverage gaps above (§5), then re-run.

---

## Appendix — files added

| File | Purpose |
|---|---|
| `tests/adversarial/a01-commit-apply.test.ts` | F-01: kernel-level double-step proof |
| `tests/adversarial/a02-state-sync.test.ts` | F-02: round-boundary invariant + control |
| `tests/adversarial/a03-probe.test.ts` | Round instrumentation (gtdx / trust / candidates / status) |
| `tests/adversarial/a04-nan-bisect.test.ts` | Pass-by-pass NaN bisection (contact vs none) |
| `tests/adversarial/a05-membrane-isolate.test.ts` | Membrane/bending isolation, uniform-coalescing probe |
| `tests/adversarial/a06-e2e-contact.test.ts` | F-03: end-to-end reachability + Newton-budget sweep |
| `tests/adversarial/a07-memory-control.test.ts` | F-05/F-06/F-07, NO_SPLIT, storage limits, COPY_SRC |
| `tests/adversarial/a08-trial-index.test.ts` | F-04: global trial-index formula |

All tests skip cleanly when no WebGPU device is present, matching the existing
`device-setup.ts` convention.