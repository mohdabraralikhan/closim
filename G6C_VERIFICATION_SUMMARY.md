# G6C Verification Summary

**Result:** All G6C harness, Gate-A parity, and Gate-B end-to-end device tests passed on the local Dawn/WebGPU device. The run executed 55 tests with no skips.

**Device:** NVIDIA GeForce GTX 1050 (Pascal), Dawn Node binding (`webgpu@0.6.1`); adapter backend reported as `unknown`. Granted limits: 16 storage buffers per shader stage and 512 MiB maximum storage-buffer binding. Timestamp queries are available; subgroups are not.

| Gate | Result | Measurement |
|---|---|---|
| A — full-iteration parity | PASS | Accepted state max absolute position difference: `1.16e-10`. CPU/GPU selected alphas differed (`1.0` / `3.328e-2`), but final states remained within parity tolerance. |
| B — scene coverage | PASS | Strip, floor, plate, fold, friction, stiff 5x, and stiff 10x all completed on sequential and GPU-control paths with finite energy. Active contact counts before the step: plate 25, fold 2,902, friction 182. The broad plate fixture uses the GPU floor-plane contact primitive; this does not claim static-triangle contact coverage. |
| Readback/submit budget | PASS | Sequential: 21 map syncs / 304 submits. GPU control: 7 map syncs / 347 submits. GPU control reduced map syncs by 66.7%; submits remained below the configured 3x ceiling. |
| Reject-all | PASS | All-invalid Armijo trials failed without accepting a trial; position remained bitwise unchanged. |
| State/failure/stress harness | PASS | State consistency, failure injection, stress matrix, falsifiability, and adaptive-K readiness checks passed. |

**Command:** `npx vitest run --maxWorkers=1 --minWorkers=1 tests/g6c-verify tests/webgpu/g6c-newton-parity.test.ts tests/webgpu/g6c-newton-e2e.test.ts` — 7 files, 55 tests passed.

**Build caveat:** `npm run build` remains blocked by an unrelated existing production TypeScript error: `src/backend/webgpu/gpu-solver.ts:864` references undefined `adaptiveBatchK`. No production or physics code was changed for this harness task.
