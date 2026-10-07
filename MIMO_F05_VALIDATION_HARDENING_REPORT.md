# MIMO F05: WebGPU validation hardening

## Result

WebGPU validation failures, out-of-memory errors, and device loss now become explicit device-step failures. A failed GPU step advances through the CPU golden solver, but returns `failed: true`, `finite: 0`, `converged: 0`, and a compact reason. Its path accounting reports `requested: GPU`, `actual: CPU`, and `fallbackReason: validation-error` for validation failures.

The solver also rejects invalid or stale accepted-state status before testing the gradient convergence threshold. A zeroed 64-byte status decodes with `finite = 0` and `converged = 0`.

## Error propagation

1. `GpuExecutor.create()` attaches an uncaptured-error listener and observes `device.lost`.
2. In development/test mode, the whole device step runs under validation and out-of-memory error scopes. Existing Newton/Armijo scopes remain around their compact status reads. `validateAllShaders()` also captures pipeline validation when explicitly run.
3. Scope errors, uncaptured errors, and device loss become `GpuExecutionFailure` with one of `validation-error`, `out-of-memory`, or `device-lost`.
4. The step boundary catches that state before returning solver diagnostics, runs the CPU fallback, and sets failure and path accounting fields. Invalid initial or accepted-state status is rejected before convergence can be declared.

Pipelines are created lazily inside the scoped device step so ordinary initialization does not compile every registered entry point. The explicit all-shader validator remains available for initialization checks. Test-only invalid bind groups and layouts are created only inside test scopes.

## Performance

No scope or wait is added per dispatch. Development/test builds use one validation-scope pair around a complete device step, in addition to the existing per-round scopes. Production mode disables error scopes; the uncaptured-error listener remains as a safety net. Error-scope work is asynchronous and adds no queue-idle wait. The eager compile-all experiment made two 5-second marker tests time out, so it was removed from normal initialization.

## Files changed for this task

- `src/backend/webgpu/gpu-executor.ts` — error scopes, error classification, uncaptured-error listener, and device-lost tracking.
- `src/backend/webgpu/gpu-solver.ts` — device-step failure handling, CPU fallback accounting, stale-status convergence gate, and diagnostics fields.
- `tests/webgpu/validation-scope.test.ts` — invalid bind-group/layout injection, error capture, zero-status, and CPU fallback assertions.
- `tests/webgpu/device-solver.test.ts` — valid GPU execution path assertions.
- `MIMO_F05_VALIDATION_HARDENING_REPORT.md` — this report.

No cloth physics or FEM, CCD, barrier, friction, or constitutive equations changed for this task.

## Tests and type checking

- The GPU physics path and valid-device assertions use real Dawn/WebGPU device execution. Device tests skip only when the runtime has no WebGPU binding or adapter.
- `npm.cmd run build` (tsc): passed.
- Targeted validation, device-marker, and device-solver run: **24 passed**.
- Full `npm.cmd test`: **65 files, 295 tests; 289 passed and 6 failed**. The six failures were in `tests/adversarial/scale/scale-overflow.test.ts` and reproduced when that file ran alone. They involve overflow counters/contact-set counts in its scale stress cases and are outside the validation-handling files.
- The project had **214+ tests previously green** before later concurrent test additions. The pre-change snapshot for this task contained 291 cases; four validation tests added here bring that snapshot to 295. The full-suite result above includes six unrelated scale-overflow failures, so it is not a clean 295-pass result.

## Branch

Implemented in the isolated worktree on branch `validation-hardening`.
