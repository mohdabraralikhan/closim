# ENGINE_SPEC.md — V0 Contract (frozen)

## 1. Scope

V0.1 implements exactly:

```text
1 cloth piece + gravity + triangular FEM + anisotropic stretch/shear
+ hinge bending + implicit Euler + Newton + PCG + pin constraints.
No contact solve yet (stubs only). No rendering. No sewing. No WASM/GPU.
```

## 2. Data structures

### ClothMesh

```text
positions: Float32Array (n*3, current)
restPositions: Float32Array (n*3)
uv: Float32Array (n*2)          // material space
indices: Uint32Array (m*3)
```

### FemData (precomputed per triangle t)

```text
invDm: 2x2   // inverse of [u1-u0, u2-u0]
area: float  // material-space area
hinges: { v0,v1 (edge), v2, v3 (opposite), restAngle, edgeLen, areaSum }
mass: Float32Array (n) lumped: m_i = arealDensityKgM2 * (sum A_t / 3)  // AREA density; thickness enters energy only
```

Grid builder: flat cloth in XZ plane, `uv = (x, z)`, y up.

### ClothMaterial

```ts
{
  arealDensityKgM2: number; // kg/m^2, e.g. 0.12 (renamed Phase 1: was ambiguous `density`)
  thickness: number;    // m, e.g. 0.001
  stretchWarp: number;  // Pa-ish stiffness C00, e.g. 2e4
  stretchWeft: number;  // C11
  stretchCoupling: number; // C01, default 0
  shear: number;        // G, e.g. 5e3
  bendWarp: number;     // N·m, hinge stiffness
  bendWeft: number;
  damping: number;      // Rayleigh-ish velocity blend 0..1
}
```

## 3. Element equations

For triangle (x0,x1,x2), with `invDm = [[a,b],[c,d]]`:

```text
F_i0 = a*(x1i-x0i) + c*(x2i-x0i)
F_i1 = b*(x1i-x0i) + d*(x2i-x0i)   i in {0,1,2}
C = F^T F (2x2), E = 0.5*(C - I)
psi = 0.5*C00*E00^2 + 0.5*C11*E11^2 + C01*E00*E11 + 2*G*E01^2
W_t = thickness * area * psi
```

Analytic gradient dW/dx (9-vector) via chain rule — see `src/physics/membrane.ts`.
HVP: `(g(x+eps*p) - g(x-eps*p)) / (2*eps)` central difference, pins filtered.
Bending per hinge edge:

```text
theta = dihedral(x0..x3), W_b = 0.5 * k_b * (theta - theta0)^2 * edgeLen^2 / areaSum
grad_b via central finite differences (reference only, small meshes)
```

Total internal gradient = membrane + bending. Total energy = inertia + internal.

## 4. Timestep (implicit Euler / variational)

```text
y_hat = x_t + h * v_t + h^2 * g_vec     (gravity folded into predictor)
for k in 0..newtonIters:
    g = M*(x_k - y_hat)/h^2 + gradE(x_k)
    filter pins in g
    if ||g|| < tol: break
    solve (M/h^2 + H_E) dx = -g   via Jacobi-PCG (matrix-free HVP)
    backtrack alpha in {1, 1/2, 1/4, ...}: E(x_k + alpha dx) < E(x_k) + 1e-4 alpha g^T dx
    x_k += alpha dx; enforce pins
v_{t+1} = (x_new - x_t) / h   (+ optional damping)
```

PCG: Jacobi (diagonal of M/h^2 + diag estimate via HVP on basis? we use
`diag = M/h^2 + beta` with beta from material scale). Max 100 iters, tol 1e-6
relative. Filtered (pinned DOFs stay 0).

## 5. Backend API (stable across CPU/WASM/GPU)

```ts
interface ClothSolver {
  initialize(scene: ClothScene): void;
  step(dt: number): void;
  setMaterial(m: ClothMaterial): void;
  pinVertex(id: number, pos?: [number,number,number]): void;
  unpinVertex(id: number): void;
  getPositions(): Float64Array;
  getVelocities(): Float64Array;
}
```

`CpuSolver` is the reference. `WebGpuSolver` (later) must reproduce
`strip-pull` tip deflection within tolerance.

## 6. Collision — Phase 1 (IMPLEMENTED, 18/18 tests pass)

```text
BVH (median-split, rebuilt per evaluate, deterministic)
  -> narrow: closest-point VT / EE (lagged s,t per Newton iter)
  -> CCD: cubic coplanarity roots + conservative fallback, relative-motion
     early-outs; static colliders via zero-motion extended arrays
  -> barrier B(d) = -(d-dHat)^2 * log(d/dHat), kappaJ SI-documented,
     convex on (0,dHat) -> SPD frozen contact Hessians
  -> friction: lagged Coulomb, residual-only, dissipative
  -> Newton/Armijo: predictor CCD-bisection, CCD-gated trials, trust region,
     descent fallback, restitution-0 velocity filter
```

ContactSystem (`src/collision/contact-assembly.ts`) is the single
solver boundary (broadphase/narrow/CCD/barrier/friction hidden behind it).

P1.0 unit fix: `density` renamed `arealDensityKgM2`; mass no longer uses
thickness. No ambiguous `density` remains.

Verification ladder (tests/): barrier-check, ccd, floor-contact,
static-collider, self-contact (head-on + page-fold), friction,
determinism, smoke (V0 untouched logic), gradient-check.

Known limitations: resting contacts grind near the hard core under the
10-iteration budget (stable, penetration-free; needs more iterations or
better preconditioning later); per-contact (not area-weighted) barrier
scale; single-threaded CPU reference.

## 7. Reference test (must pass)

**strip-pull**: 0.2 x 0.1 m strip, 12x6 grid, arealDensity 0.15 kg/m^2, thickness 1mm,
warp 2e4 / weft 2e4 / shear 5e3, bend 1e-5. Pin left edge (x=0). Gravity
(0,-9.81,0). Step h=1/60 for 60 steps. Assert:

- no NaN, energy bounded
- free end sags: mean y of last column < -0.01
- stretch check: max Green strain E00 < 0.05 (no PBD-like over-stretch)
- gradient finite-diff check passes (rel err < 1e-4)

Later: tip deflection recorded as JSON baseline for GPU parity.
