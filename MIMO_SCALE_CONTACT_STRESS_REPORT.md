# MIMO Scale-Contact Stress Report — GPU contact handling & overflow at 1k / 10k / 50k

Branch: `mimo-scale-contact-stress`. Suite: `tests/adversarial/scale/`
(`scale-helpers.ts`, `scale-overflow.test.ts`, `scale-newton.test.ts`,
`scale-frames.test.ts`). **No production change** (zero `src/` edits in this
task; `git status` shows only the inherited G6C working tree plus test files).

Verdict: **35/35 scale tests green; full repo suite 298/298 green (67 files);
`tsc --noEmit` clean.** No engine defect reproduced. Two test-expectation bugs
found during calibration (fixed, both mine). One open nondeterminism question
documented below (observation, not corruption — every corruption gate passes).

---

## 1. Mesh sizes, scenes, repetitions, capacities

| class | fold (v/t) | headon (v/t) | floor (v/t) | dense 3-layer (v/t) |
|---|---|---|---|---|
| 1k | 1089 / 2048 | 1058 / 1936 | 1089 / 2048 | 3267 / 6144 |
| 10k | 10201 / 20000 | 10082 / 19600 | 10201 / 20000 | 30603 / 60000 |
| 50k | 50625 / 100352 | 50562 / 99856 | 50625 / 100352 | 151875 / 301056 |

Scenes: (1) fold — half-sheet reflected + 1.5 mm lift; (2) head-on patches —
1 mm gap + ±0.3 m/s closing velocities; (3) resting floor — sheet 1.2 mm above
y=0 (`dHat` = 2 mm); (4) dense self-contact — 3-layer stack 1 mm apart;
(5) tiny contact capacity (forces overflow); (6) oversized contact capacity;
(7) alternating contact/no-contact frames (1.2 mm ↔ 50 mm teleport, plus a
floor-kind toggle variant at 10k).

Repetitions per config: 1k × 8, 10k × 4, 50k × 2 (`REPS` table; identical
rebuilds, counters/sets compared across reps to expose ordering races).

Contact caps per (scene, class) — calibrated from measured need (see §2),
e.g. 1k: fold 131072, headon/floor 8192, dense 524288; tiny: 128 / 512 / 2048.
Pair caps: 1k 65536 (dense 196608), 10k 196608, 50k 131072.

## 2. Measured contact/pair need (attempted appends/emissions, exact counters)

| config | contacts | pairs | contact ov. | pair ov. |
|---|---|---|---|---|
| 1k fold | 59,818 | 16,147 | no (131072) | no |
| 1k headon | 1,136 | 4,750 | no (8192) | no |
| 1k floor | 1,089 | 4,867 | no (8192) | no |
| 1k dense | 404,344 | 120,633 | no (524288) | no (196608) |
| 10k headon | 1,713 | 352,302 | no (65536) | **yes** |
| 10k floor | ~144,000 | 818,283 | no (262144) | **yes** |
| 10k fold | ~120,000 | 1,211,819 | **yes** | **yes** |
| 10k dense | ~80,000 | 8,134,977 | **yes** | **yes** |
| 50k floor | ~53,000 | 15,825,091 | no (262144) | **yes** |
| 50k headon | ~4,500 | 7,252,046 | no (131072) | **yes** |
| 50k fold | ~3,000 | 20,761,491 | no (65536) | **yes** |
| 50k dense | ~3,400 | 146,318,073 | no (65536) | **yes** |

Two structural findings:
- **Pair truncation hides contact volume.** 1k/dense with pair cap 65536
  showed ~180k contacts; with full pair expansion (196608) the true need is
  404,344. Attempted counts stay exact either way, so the mask is measurable.
- **Prim scratch (15× pairs) is the binding memory ceiling.** 10k+ pair
  volumes (352k–146M) cannot run overflow-free on a 2 GB card (15× pairs ×
  ~112 B/prim). Exact-set gates therefore live at 1k; at 10k/50k the suite
  asserts iff-latches, kept-record validity, exact-duplicates, and
  finiteness. Pair-truncated trials remain *valid* by design on all paths
  (see §5 observation).

## 3. Invariants asserted (all green) and failure counts

Per evaluation, all reps, all configs — zero violations:
- `pairCount == pairScanned`; `contactCount <= contactScanned`;
  `primVT == 6 × expandedPairs`; `primEE <= 9 × expandedPairs`
- `overflowFlag == (pairCount > pairCap)` and
  `contactOverflow == (contactCount > contactCap)` — **iff, exact**
- every live record: kind ∈ {0,1,2}, finite W/N/Prm/Dist/TOI/Energy,
  dist < dHat (1e30 barrier-inactive sentinel allowed), kind≠2 ⇒ dist ≥ 1e-12
- same-identity records bitwise-identical payloads (196,544 duplicate groups
  on 1k/dense/big alone, max multiplicity 6, zero inexact)
- device live multiset == CPU-mirror (FP32 `GpuContactSystem`) multiset
  **element-wise incl. multiplicity**: dense 404344/404344, fold 59818/59818,
  headon 1136/1136, boundary-band flips 0/0
- batch candidate rows == sequential trials (alpha/energy/verdict exact,
  selection agrees) on 1k fold + headon, all 8 trials
- reject-all round: position + laggedN bitwise unchanged, failure flagged,
  energy finite, step completes
- accepted round: S1 holds (1k/floor: count 1089/1089, |grad| = 0.000e+0);
  round-1 Xk refresh reproduces the fresh evaluation bitwise
- 10k/50k truncation-immune gates: position == xTrial bitwise post-commit,
  laggedN tail untouched, everything finite, next round finite
- pins exact bitwise (101 pinned verts, 10k), zero non-finite in
  gradient/rhs/laggedN/contactEnergy/searchDirection
- alternating frames: engaged multisets stable (1k), clear frames match the
  never-contacted control bitwise (energy + gradNorm); 10k floor-kind toggle
  10201 ↔ 0 exact across 6 frames; rejected trials never touch
  position/laggedN (0.00e+0); accepted laggedN survives a clear frame;
  re-engagement finite

Failure counts: final **0 failures (35/35 scale, 298/298 full suite)**.
During calibration, 11 expectation failures total, all diagnosed as
test-expectation bugs (caps sized from un-truncated need estimates;
sequential gate missing the documented batch-strictness on overflow;
10k "clear" frame not empty — coplanar self-contact; 50k "saturated"
contacts fitting). Zero engine defects.

## 4. Nondeterminism observations (class A — never corruption)

Rule used throughout: never require identical contact ORDER; canonicalize
`(kind,id)` identities; multiset-compare. Findings:
- **Pair-emission order races under pair truncation (proven).** Direct
  candidate-pair multiset comparison across two identical 50k/floor rebuilds:
  same size (131072 kept slots), different members; pairCount exactly
  15825091 both times; contacts 53151 vs 53057. First-cap-wins truncation of
  a race-ordered emission list. All counter/validity/exactness gates pass on
  both sides.
- **Contact-count variation under truncation is bounded and explained:**
  dense-1k ±90/180k (pair-truncated), floor-50k ±85/53k, headon-50k ±18,
  fold-50k ±18. With fitting caps, counts are bitwise stable (1k/dense/big
  404344 ×8, 1k/fold 59818 ×8).
- **Stability contrast (open question).** Pair-truncated sets are sometimes
  bitwise stable across reps (dense-50k 3485 ×2, headon-10k 1713 ×4) and
  sometimes not (floor-50k, headon-50k) — scheduling-dependent, logged per
  config. Either way the kept set is validated, never trusted.
- **Duplicate `(kind,id)` tuples are documented multiset semantics**, not
  defects (`compareContactMultisets`: "the CPU active set repeats a key when
  one primitive is reachable via several triangle pairs (barrier sums every
  entry)"). The suite's gate is payload-exactness within each identity group:
  0 inexact groups in ~300k duplicate groups checked.
- **Benign 4e-8 dist outlier (1 in ~4 runs):** slot permutation among
  symmetric floor contacts (W/diag/gradient bitwise; dist multiset equal).
  Dist is compared order-insensitively for exactly this reason.

## 5. Minimal reproducer (for the headline race)

1k/dense, contact cap 128, pair cap 65536, 8 identical
`evaluateNewtonState` rebuilds: attempted counts vary rep-to-rep
(180177–180316) while `pairCount` stays exactly 120633 and all iff/validity
gates pass. Bumping the pair cap to 196608 (full expansion) makes counts
bitwise stable (404344 ×8). This isolates pair-truncation race order as the
sole variation source — no production change needed.

## 6. Production patch

**None.** No `src/` modification was required or made. Deliberately
out-of-scope observations (shared CPU/GPU behavior, changing either side
would be redesign):
- Pair overflow does not invalidate trials on any path (sequential gate and
  CPU golden ignore it; only contact overflow rejects, and only on batched
  paths — documented conservative asymmetry).
- `expand_pairs` dispatch covers `ceil(pairCap/64)*64` threads; with a
  `pairCap % 64 != 0` under pair overflow, up to 63 threads could read past
  `candidatePairs`. All production caps are multiples of 64 — noted, not
  patched (defect not reproduced, per task rules).

## 7. Cost

Scale suite: 35 tests, ~10 min (1k ~4 min, 10k ~3 min, 50k ~2.5 min,
mirror ~1 min). Full repo suite with scale included: 67 files / 298 tests,
all green. Reproduce: `npx vitest run --exclude '**/.worktrees/**'
tests/adversarial/scale/ --reporter=basic` (the exclude skips a nested
foreign worktree whose stale copies otherwise double-run).
