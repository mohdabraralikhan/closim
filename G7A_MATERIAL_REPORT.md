# G7A Material Abstraction + Calibration — Report

Branch: `feature/g7a-material`. New code: `src/physics/material-model.ts`
(model layer only). New tests: `tests/material/material-model.test.ts`,
`tests/material/material-params.test.ts` (25 tests, all CPU, ~2 s).

Verdict: orthotropic StVK is wrapped behind a backend-neutral
`ClothMaterialModel` interface with **bitwise-identical numerics**
(E=G=H=0 gap over 40 randomized materials × 4 states), SI-validated physical
parameters, Fabric-101-shaped calibration metadata (elastic stored;
hysteresis/plasticity reserved with a structural not-implemented guard), and
versioned JSON round-trip. No plasticity implemented. No changes to
ContactSystem, CCD, Newton, PCG, preconditioners, topology, UI, or rendering.

## 1. What was built

`ClothMaterialModel` provides exactly `energy`, `gradient`, `hessianVector`
(mesh-wide, `Float64Array`, no GPU types). `OrthotropicStVKMaterial`
implements it by **delegating to the verified kernels** (`evalInternal` /
`internalEnergyOnly` from `fem.ts`, `evalMembraneHvp` from
`membrane-hvp.ts`), so parity holds by construction and the tests pin it:
- energy/gradient include membrane + bending (legacy `evalInternal` semantics);
- `hessianVector` is membrane-only analytic (legacy inexact-Newton
  approximation: bending in gradient, excluded from Hessian — same as the
  `evalMembraneHvp`/FD-oracle pair, whose established agreement bar is 1e-7;
  measured model-vs-oracle worst 1.34e-10 over 20 cases).

Parameter split (SI, validated at construction):
- inertia: `arealDensityKgM2` (kg/m², mass only — thickness never enters inertia)
- `thicknessM` (m), membrane moduli in **Pa**, bending in **N·m**,
  `dampingRatio` dimensionless in [0,1] (engine band 0..0.1; the one
  semi-artistic knob, range-validated not hidden)
- Coulomb friction stays contact-owned (`ContactParams`), recorded as an
  explicit ownership note, not duplicated.

Validation rejects: non-finite anywhere; thickness/density ≤ 0; any negative
modulus; `C00·C11 − C01² < 0` (positive-definiteness); damping outside [0,1];
zero-stiffness membranes explicitly allowed; PSD-boundary equality allowed.
`MaterialValidationError` carries a machine-readable `code`.

Calibration: `MaterialCalibration{ elastic{dataset, specimenId?, protocol?,
measuredAt?, moduliPa?, notes?}, hysteresis?, plasticity?, provenanceNotes? }`
with `model: "none"` literals — any other dissipative-model tag throws
`not-implemented` at construction. Serialization is versioned
(`orthotropic-stvk` / v1; unknown kind/version rejected). Instances are
immutable snapshots (`clone()` deep-copies; params frozen).

`fromLegacy`/`toLegacy` is the 1:1 bridge (validates on entry; legacy path
itself untouched). No consumer was rewired — scene/solver/GPU upload keep
reading `ClothMaterial`; adoption is a later track's decision with this
parity gate in place.

## 2. Correctness gates (all green)

| check | result |
|---|---|
| default material, 4 states: energy/grad/HVP vs legacy | bitwise (0) |
| 40 randomized VALID materials × 4 states (rest/stretch/shear/jitter) | E=G=H=0 |
| model HVP vs legacy FD oracle (20 cases) | worst 1.34e-10 (< 1e-7 bar) |
| Float32 inputs + out-buffer reuse | bitwise |
| 13 invalid-parameter cases + malformed shapes | each throws the documented code |
| legacy bridge: DEFAULT ok; unphysical legacy throws; nothing legacy changed | pass |
| calibration: dataset required; non-"none" hysteresis/plasticity throw | pass |
| JSON string + object round-trip, kind/version tags, no-calibration case | bitwise behavior restored |
| clone: no shared references, equal values, frozen params | pass |
| SI unit table (9 rows, exact) | pass |

Randomized materials use `C01 = ρ·0.95·√(C00·C11)` (PSD-safe by
construction). Note: the pre-existing `membrane-hvp.test.ts` generator can
emit PSD-violating couplings (e.g. C01=5000 with C00=C11=1e3) — that is why
PSD validation lives only in the new constructor, never in legacy paths.

## 3. Full-suite evidence

`npx tsc --noEmit` clean. Full `vitest` run: **69 files / 323 tests green**
(298 pre-existing + 25 G7A). No existing test modified; legacy physics,
contact, solver, and GPU paths untouched.

## 4. Non-goals honored; seams for later tracks

No plasticity/hysteresis models (guarded), no B-spline FEM, no AGIPC/HSC, no
solver rewrite. `membrane-blocks.ts` (G5 block-Jacobi factors) stays on
`ClothMaterial` — outside the three-op interface by design. G7B+ can adopt
via `fromLegacy`/`toLegacy` with these parity tests as the lock.
