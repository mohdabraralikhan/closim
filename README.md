# cloth-engine-v0 — physics-first FEM reference

Goal: a scientifically meaningful CPU reference for a Marvelous/CLO-style
garment engine. Triangular FEM, orthotropic StVK membrane, hinge bending,
variational implicit Euler + Newton/PCG/Armijo, hard pins, gravity,
**barrier contact + CCD + lagged friction** (Phase 1). No rendering, no
sewing, no GPU yet.

## Layout

```text
src/
  math/       vec + PCG
  mesh/       grid builder, FEM preprocessing (Dm^-1, area, MASS, hinges)
  physics/    types, membrane energy, bending, fem assembly, scene
  solver/     implicit Newton stepper (contact-integrated, trust region)
  backend/    ClothSolver interface + CpuSolver reference
  collision/  types, aabb, bvh, closest-point, cubic, ccd-vt/ee,
              barrier, friction, self-collision, contact-assembly
tests/        V0 smoke + gradient check + Phase 1 ladder (10 files)
examples/     strip-pull reference case
```

## Quick start

```bash
npm install
npm test            # full suite: 18 tests (~1-2 min)
npm run strip-pull  # pinned strip reference case
```

## Physics core

Per-triangle deformation gradient `F = Ds * Dm^-1` (3x2), Green strain
`E = 0.5*(F^T F - I)`, orthotropic StVK density:

```text
psi = 0.5*C00*E00^2 + 0.5*C11*E11^2 + C01*E00*E11 + 2*G*E01^2
```

`C00 = stretchWarp, C11 = stretchWeft, G = shear/2` (decoupled by default,
`C01 = 0`). Triangle energy `W = thickness * area * psi`.
Mass is lumped from AREAL density: `m_i = arealDensityKgM2 * (sum A_t / 3)`
(thickness enters energy only — see ENGINE_SPEC.md).

Implicit Euler as optimization with CONTACT in the same variational solve
(no post-projection):

```text
y_hat = x + h*v + h^2 * g
min_x  1/(2h^2) ||x - y_hat||_M^2 + E_membrane + E_bend + E_barrier  (- f_friction_lagged)
```

Newton + Jacobi-PCG + CCD-gated Armijo + trust region + bisector-validated
predictor + restitution-0 impact filter. HVP is analytic (membrane FD,
contact exact-frozen); friction is lagged (residual only, dissipative).

See `ENGINE_SPEC.md` for the full V0 + Phase 1 contract.
