# G10 — Professional 3D Garment Workspace: Core Delivery Report

Tracks implemented in this pass: **G10A (viewport core), G10B (garment manipulation), G10C (2D↔3D sync core)** plus the browser app shell. G10D/G10E/final integration were handled outside this pass.

## Architecture

Two layers, deliberately separated:

- **`src/view/**` — pure TypeScript view core.** No DOM, no three.js, fully headless-testable under the existing vitest/tsc setup. The main `tsconfig.json` (no DOM lib) is untouched, so other agents' regression checks are unaffected.
  - `types.ts` — entities, visibility flags, `WorkspaceError` taxonomy, bounds math.
  - `camera.ts` — `OrbitCamera` (orbit/pan/zoom/frame/presets/persp↔ortho), column-major mat4 helpers, right-handed Y-up.
  - `selection.ts` — panel/seam/avatar/region selection with `pruneSelection` against live assemblies.
  - `pick.ts` — ray build from NDC, Möller–Trumbore panel picking, vertex/seam/avatar picking; all by stable IDs.
  - `viewport.ts` — `GarmentWorkspace`, the single render-state hub. `setGarment()` bumps a `garmentEpoch`, prunes selection, and fires `onGarmentReplaced` (pins dropped, drags cancelled, stale references never survive a rebuild).
  - `sim-session.ts` — `SimulationSession`: fixed-dt accumulator (max 8 substeps), NaN detection, last-stable-state snapshots, `restoreStable()`. The **only** path that steps the solver.
  - `manipulation.ts` — placement transforms (translate/yaw/recenter, JSON-clone + validate), `PinManager` (through `ClothSolver.pinVertex/unpinVertex` only), `DragController` (temporary pin + target updates).
  - `sync.ts` — `classifyPatternChange` → rebuild level (`none`/`geometry-refresh`/`panel-remesh`/`topology-rebuild`/`full-assembly`) and `rebuildWithPlan`.
- **`app/**` — thin three.js r186 + vite shell** (separate `app/tsconfig.json` with DOM lib). `renderer.ts` mirrors workspace state into GPU objects and never feeds back; `main.ts` wires input/HUD/frame loop.

## Invariants honored

- Camera/selection/HUD interaction can never trigger a solve (no code path connects them to `SimulationSession`).
- Solver→viewport transfer happens only through `publishSimPositions(Float64Array)`.
- Pins and drags integrate exclusively via the existing `ClothSolver` constraint interface; no second constraint system, no direct solver-state writes.
- Rebuilds invalidate every derived reference via `garmentEpoch`; selection/pins/drags prune or cancel deterministically.
- Simulation never auto-starts; the app boots paused.
- No production renderer ambitions: Lambert material, helper lines, grid — inspection-grade only.

## Verification

| Check | Result |
| --- | --- |
| `tsc -p tsconfig.json --noEmit` (main, headless) | clean |
| `tsc -p app/tsconfig.json --noEmit` (app, DOM) | clean |
| `vitest run tests/view` (7 files) | **73/73 pass** |
| `vite build` | OK (≈640 kB bundle, three.js) |
| Dev server smoke (`npm run dev`, curl) | HTML shell + module graph transform OK |
| Full repo suite (`npm test`) after integration | **98 files, 701/701 tests pass** |

Limitation: this environment has no browser, so no interactive click-through was performed. Run `npm run dev` and open http://localhost:5173 to inspect the workspace. Golden-path logic (camera math, picking, pin→solver integration, drag, rebuild classification, NaN recovery) is covered headlessly in `tests/view/**`.

## Notes for other agents

- Added devDeps `three`, `@types/three`, `vite` and scripts `dev`, `build:app`, `typecheck:app`.
- Added root `vitest.config.ts` — required because `vite.config.ts` sets `root: "app"` and vitest would otherwise re-root test discovery inside `app/`, breaking `npm test` repo-wide. Do not delete it.
- `dist-app/` is build output and is gitignored.
- Untouched: everything under `src/backend/webgpu/**` and `tests/webgpu/**`.
