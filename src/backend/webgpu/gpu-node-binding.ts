// Guarded loader for the Dawn-based `webgpu` npm package (devDependency).
//
// G3 needs a REAL WebGPU device in Node. The `webgpu` package (Dawn) provides
// one, but it is a 95 MB native module that must NEVER be a hard dependency:
// production bundles and no-GPU CI must work without it. This loader therefore
// never throws — it returns null when the binding (or any adapter) is missing,
// and device tests skip cleanly in that case.
//
// Side effects are contained: WebGPU globals (GPUBufferUsage, GPUMapMode,
// navigator-adjacent classes) are installed onto globalThis ONLY when this
// loader successfully runs, and only inside the calling process (vitest worker).

import { createRequire } from "node:module";

export interface NodeWebGPU {
  /** navigator.gpu-like object with requestAdapter(). */
  gpu: any;
  /** WebGPU classes + enum namespaces (GPUBufferUsage, GPUMapMode, ...). */
  globals: Record<string, any>;
}

type Cache = NodeWebGPU | { missing: true } | null;
let cached: Cache = null;

/** Load the Node WebGPU binding, or null when unavailable. Never throws. */
export function loadNodeWebGPU(): NodeWebGPU | null {
  if (cached) return "missing" in cached ? null : cached;
  try {
    const req = createRequire(import.meta.url);
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const w = req("webgpu") as {
      create: (flags: string[]) => any;
      globals: Record<string, any>;
    };
    if (!w || typeof w.create !== "function") {
      cached = { missing: true };
      return null;
    }
    try {
      Object.assign(globalThis as Record<string, unknown>, w.globals ?? {});
    } catch {
      // global install is best-effort; callers use `globals` directly.
    }
    const gpu = w.create([]);
    if (!gpu || typeof gpu.requestAdapter !== "function") {
      cached = { missing: true };
      return null;
    }
    cached = { gpu, globals: w.globals ?? {} };
    return cached;
  } catch {
    cached = { missing: true };
    return null;
  }
}

/** True when a node WebGPU binding is importable (adapter still required). */
export function hasNodeWebGPUBinding(): boolean {
  return loadNodeWebGPU() !== null;
}

/** Buffer-usage bits, preferring loaded globals, falling back to numerics. */
export function usageBits(g: Record<string, any> | null | undefined): Record<string, number> {
  const U = (g as any)?.GPUBufferUsage;
  const num = (v: unknown, fb: number): number =>
    typeof v === "number" ? v : fb;
  return {
    MAP_READ: num(U?.MAP_READ, 0x0001),
    MAP_WRITE: num(U?.MAP_WRITE, 0x0002),
    COPY_DST: num(U?.COPY_DST, 0x0008),
    COPY_SRC: num(U?.COPY_SRC, 0x0004),
    STORAGE: num(U?.STORAGE, 0x0080),
    UNIFORM: num(U?.UNIFORM, 0x0040),
    INDIRECT: num(U?.INDIRECT, 0x0100),
    QUERY_RESOLVE: num(U?.QUERY_RESOLVE, 0x1000),
  };
}

/** Map-mode bits with the same fallback policy. */
export function mapModeBits(g: Record<string, any> | null | undefined): { READ: number } {
  const M = (g as any)?.GPUMapMode;
  return { READ: typeof M?.READ === "number" ? M.READ : 0x0001 };
}
