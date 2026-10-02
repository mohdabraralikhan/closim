// Staging-buffer readback policy (Phase 2 §20).
//
// Normal simulation: NO hot-loop readback of positions/gradients/contacts.
// Debug/inspection: explicit `copyBufferToBuffer` into a MAP_READ staging
// buffer, then mapAsync(READ). Never map a buffer still needed by GPU commands
// — always copy first, then map the staging copy after queue submission.
//
// Every CPU<->GPU sync point in gpu-solver.ts calls through here so the
// synchronization cost is labeled and measurable (encode vs submit vs map).

export interface ReadbackTiming {
  copyEnqueueMs: number;
  mapMs: number;
  bytes: number;
}

export async function readbackBuffer(
  device: any,
  src: any,
  byteSize: number,
  label = "readback",
): Promise<{ data: ArrayBuffer; timing: ReadbackTiming }> {
  const t0 = performance.now();
  const staging = device.createBuffer({
    label: `${label}-staging`,
    size: byteSize,
    usage: 0x0001 | 0x0008, // MAP_READ | COPY_DST (numeric to avoid DOM enum dependency)
  });
  const encoder = device.createCommandEncoder({ label: `${label}-copy-encoder` });
  encoder.copyBufferToBuffer(src, 0, staging, 0, byteSize);
  device.queue.submit([encoder.finish()]);
  const t1 = performance.now();
  await staging.mapAsync(1); // READ mode
  const t2 = performance.now();
  const copy = staging.getMappedRange(0, byteSize).slice(0);
  staging.unmap();
  try { staging.destroy?.(); } catch { /* best effort */ }
  return { data: copy, timing: { copyEnqueueMs: t1 - t0, mapMs: t2 - t1, bytes: byteSize } };
}

/** Classify a buffer read as hot-loop (forbidden) or explicit-debug (allowed). */
export type ReadbackKind = "status-compact" | "positions-debug" | "snapshot-export" | "frame-output";

export function isHotLoopReadback(kind: ReadbackKind): boolean {
  // Only the compact SolverStatus may be inspected per Newton iteration.
  // Everything else must be explicit debug/snapshot/frame output.
  return kind !== "status-compact";
}
