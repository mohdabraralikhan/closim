# G7B r-Adaptive Cloth Prototype Report

## Scope

This branch adds an isolated, fixed-connectivity r-adaptation prototype. It relocates existing interior reference vertices in 2D SI coordinates. It does not add or remove vertices, modify connectivity, move boundary vertices, modify solver state, or change FEM, contact, constitutive, or other physical equations. No material or garment topology code is changed.

The target is a stand-alone prototype API, not a solver integration. Callers provide reference coordinates and may optionally provide current 3D coordinates, wrinkle indicators, contact indicators, and pinned vertices.

## Method

- **Element quality:** triangle mean ratio `sum(edge_length²) / (4√3 × area)`, equal to 1 for an equilateral triangle. A quadratic quality penalty activates above the configurable soft threshold; a configurable maximum is a hard rejection.
- **Feature indicator:** adjacent-face normal change divided by edge length estimates curvature from current 3D geometry. Curvature is combined with caller-supplied nonnegative wrinkle and contact indicators.
- **Target resolution:** each vertex starts from its local mean reference edge length. Feature strength reduces the target by `1 / sqrt(1 + featureGain × feature)`, with a configured lower scale bound.
- **Relocation:** each of four default bounded iterations proposes edge-length relaxation toward the target field. Automatically detected boundary vertices and explicitly pinned vertices do not move. Backtracking accepts only candidates that lower the quality/target energy and satisfy hard checks.
- **Rejection checks:** zero-area or inverted triangles, collapsed edges, too-small area, excessive mean ratio, and non-manifold edges reject input/candidates. Connectivity is returned unchanged.

This is a small deterministic prototype inspired by the resolution concentration and element-quality safeguards in [Variational r-Adaptive Cloth Simulation](https://arxiv.org/abs/2608.17833). It is not an implementation of that paper's full variational method.

## Tests

`tests/adaptation/r-adaptive.test.ts` adds six tests:

1. Smooth planar cloth without feature indicators stays unchanged.
2. A folded surface yields greater estimated curvature and finer target spacing at the ridge.
3. A contact indicator refines target spacing and reduces local edge lengths.
4. Degenerate input is rejected; aggressive refinement preserves positive triangle orientation and the quality cap.
5. Inferred boundary vertices and explicit pins remain exactly fixed.
6. Identical input and options produce deterministic results.

## Verification

- Targeted G7B + G6C end-to-end tests: 10 passed, including all six adaptation tests and all four GPU Newton end-to-end tests. Gate-B contact scenes now assert contacts at their initialized state. The plate fixture uses the GPU-supported floor primitive, and the friction case preserves and restores its opposing authored velocities for both solver modes.
- TypeScript build (`npm run build`): passed. The G6C solver source imports the already-defined `adaptiveBatchK` helper from `gpu-newton.ts`; behavior and solver mathematics are unchanged.
- The full repository suite was not rerun after these focused fixes.

The branch is `feature/g7b-r-adaptive`, in the isolated `.worktrees/g7b-r-adaptive` worktree.

## Performance and limitations

The prototype is opt-in and is not called from the production solver, so production simulation has no added cost. A call allocates adjacency, curvature, quality, and candidate buffers and traverses the mesh for up to four iterations plus bounded backtracking. This is intended for explicit adaptation steps, not every solver dispatch. The target indicator scale and relaxation parameters are prototype controls and need garment-specific calibration. Boundary relocation and topology changes are intentionally unsupported.
