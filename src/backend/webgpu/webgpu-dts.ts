// Minimal WebGPU type shims for a DOM-free tsconfig (lib: ES2022 only).
// The real browser types come from lib.dom / @webgpu/types; here we alias to
// `any` so the CPU reference build (Node/CI without WebGPU) still compiles.
// GPU-touching code must always go through `isWebGPUAvailable()` first and
// never assume these globals exist.

export type GPUDeviceLike = any;
export type GPUAdapterLike = any;
export type GPUBufferLike = any;
export type GPUQueueLike = any;
export type GPUComputePipelineLike = any;
export type GPUBindGroupLike = any;
export type GPUShaderModuleLike = any;
export type GPUQuerySetLike = any;

export interface WebGPULoader {
  // Narrow structural view of navigator.gpu.requestAdapter we actually use.
  requestAdapter: (options?: any) => Promise<GPUAdapterLike | null>;
}

/** True only when a WebGPU implementation is reachable from this runtime. */
export function isWebGPUAvailable(): boolean {
  const g = globalThis as Record<string, any>;
  const nav = g["navigator"] as { gpu?: WebGPULoader } | undefined;
  if (nav?.gpu?.requestAdapter) return true;
  // Node >= 20 with --experimental-webgpu exposes global navigator.gpu too;
  // Deno/Bun expose (globalThis as any).gpu.
  if (typeof g["gpu"] !== "undefined" && g["gpu"]?.requestAdapter) return true;
  return false;
}

/** Best-effort handle to the `navigator.gpu`-like object, or null. */
export function getGpuNamespace(): WebGPULoader | null {
  const g = globalThis as Record<string, any>;
  const nav = g["navigator"] as { gpu?: WebGPULoader } | undefined;
  if (nav?.gpu?.requestAdapter) return nav.gpu;
  if (g["gpu"]?.requestAdapter) return g["gpu"] as WebGPULoader;
  return null;
}

/** WebGPU feature toggle set relevant to Phase 2 G0. */
export const GPU_REQUIRED_FEATURES: string[] = [];

export const GPU_OPTIONAL_FEATURES = [
  "timestamp-query",
  "subgroups",
  "shader-f16",
] as const;
