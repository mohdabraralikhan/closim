# G7E Avatar Collider Layer Report

## Scope and branch

Implemented on `feature/g7e-avatar-collider` in the isolated `.worktrees/g7e-avatar-collider` worktree. This layer defines simulation colliders separately from any visible avatar mesh. It does not modify CCD, `ContactSystem`, solver behavior, rendering, skeleton solving, or collision mathematics.

## Implementation

- `AvatarCollider` holds an id, body-part label, simulation `ColliderGeometry`, local rigid transform, optional `MotionProvider`, and thickness in metres.
- `ColliderGeometry` supports static triangle meshes, spheres, capsules, boxes, and compounds. Primitive tessellation defaults are intentionally coarse; caller-authored triangle meshes can include one body-part label per triangle.
- `Transform` uses metre translation and an `(x,y,z,w)` quaternion. `sampleMotion` records t0/t1 transforms. `interpolateTransform` uses translation lerp and shortest-arc quaternion slerp, with sign canonicalization for deterministic results when equivalent quaternions represent a 180-degree rotation.
- `preprocessCollisionMesh` validates finite vertices and indices, exact-welds duplicate vertices, and removes degenerate and duplicate faces while retaining part labels. Primitive colliders generate simplified simulation geometry directly; visible mesh decimation remains the caller's responsibility.
- `toCollisionObject` transforms geometry into world-space `Float32Array` positions and `Uint32Array` indices. `attachCollisionObject` uses the existing public `setStaticMesh(positions, indices)` boundary.
- Thickness is propagated as `thicknessM` metadata in the adapter object. The existing `ContactSystem.setStaticMesh` API has no per-object thickness input, so this prototype does not apply thickness to collision distances or add collision physics.
- `horizontalPlaneHeight` recognizes horizontal triangle meshes and supports equivalence checks against the existing `setFloor(y)` representation.

## Tests and verification

`tests/avatar-collider.test.ts` has eight tests covering static triangle meshes, t0/t1 translation, rotation, compounds and labels, thickness/buffer attachment, deterministic interpolation, invalid collider rejection, and floor distance equivalence.

- Targeted tests: **8 passed**.
- Module-only TypeScript check: **passed**.
- Project build: blocked by the pre-existing unresolved `adaptiveBatchK` symbol at `src/backend/webgpu/gpu-solver.ts:864`. The collider module and test introduce no TypeScript diagnostics.

## Performance and limitations

Collider conversion and preprocessing occur when the adapter is called, not in solver dispatches. A conversion allocates transformed positions and indices; primitive tessellation and compound merging scale with generated triangle count. The current collision API stores one static triangle mesh at a time, so the adapter supplies sampled endpoint objects but does not add swept moving-collider CCD or a multi-object registry. Thickness remains metadata until an existing collision-object API can consume it.
