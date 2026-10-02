// GPU capability detection (Phase 2 §22).
// Never hard-code high-end limits: query the adapter and validate the
// concrete buffer/workgroup requirements of the G0 pipeline instead.

import { getGpuNamespace, isWebGPUAvailable, GPU_OPTIONAL_FEATURES } from "./webgpu-dts.js";

export interface GpuCapabilityReport {
  available: boolean;
  reason: string;
  features: string[];
  limits: Record<string, number>;
  timestampQuery: boolean;
  subgroups: boolean;
  shaderF16: boolean;
  /** True when the adapter satisfies every G0 minimum in `validateG0Requirements`. */
  meetsG0Minimums: boolean;
  requirementNotes: string[];
}

/** Conservative G0 minimums (bytes / counts). Small on purpose. */
export const G0_MINIMUMS = {
  maxStorageBufferBindingSize: 16 * 1024 * 1024, // 16 MiB per buffer is plenty for G0
  maxBufferSize: 64 * 1024 * 1024, // 64 MiB total addressable
  maxComputeWorkgroupSizeX: 64,
  maxComputeInvocationsPerWorkgroup: 256,
  maxBindGroups: 4,
} as const;

export async function detectGpuCapabilities(): Promise<GpuCapabilityReport> {
  if (!isWebGPUAvailable()) {
    return {
      available: false,
      reason: "WebGPU unavailable in this runtime (no navigator.gpu). CPU reference remains golden.",
      features: [],
      limits: {},
      timestampQuery: false,
      subgroups: false,
      shaderF16: false,
      meetsG0Minimums: false,
      requirementNotes: ["skipped: no WebGPU namespace"],
    };
  }
  try {
    const ns = getGpuNamespace()!;
    const adapter = await ns.requestAdapter();
    if (!adapter) {
      return {
        available: false, reason: "requestAdapter() returned null",
        features: [], limits: {}, timestampQuery: false, subgroups: false,
        shaderF16: false, meetsG0Minimums: false, requirementNotes: ["adapter null"],
      };
    }
    const features: string[] = [...(adapter.features as Set<string> ?? [])];
    void GPU_OPTIONAL_FEATURES;
    const rawLimits = (adapter.limits ?? {}) as Record<string, number>;
    const limits: Record<string, number> = {};
    for (const [k, v] of Object.entries(rawLimits)) limits[k] = Number(v);
    const notes: string[] = [];
    let ok = true;
    const need = (key: string, min: number) => {
      const got = limits[key];
      if (typeof got !== "number" || !(got >= min)) {
        ok = false;
        notes.push(`${key}: got ${String(got)}, need >= ${min}`);
      } else {
        notes.push(`${key}: ok (${got} >= ${min})`);
      }
    };
    need("maxStorageBufferBindingSize", G0_MINIMUMS.maxStorageBufferBindingSize);
    need("maxBufferSize", G0_MINIMUMS.maxBufferSize);
    // invocations limit may be named differently across implementations; check both spellings.
    const inv = limits["maxComputeInvocationsPerWorkgroup"] ?? limits["maxComputeWorkgroupsPerDimension"];
    if (typeof inv === "number" && inv < G0_MINIMUMS.maxComputeInvocationsPerWorkgroup) {
      ok = false;
      notes.push(`maxComputeInvocationsPerWorkgroup: got ${inv}, need >= 256`);
    }
    return {
      available: true,
      reason: "adapter acquired",
      features,
      limits,
      timestampQuery: features.includes("timestamp-query"),
      subgroups: features.includes("subgroups"),
      shaderF16: features.includes("shader-f16"),
      meetsG0Minimums: ok,
      requirementNotes: notes,
    };
  } catch (err) {
    return {
      available: false,
      reason: `adapter request failed: ${err instanceof Error ? err.message : String(err)}`,
      features: [], limits: {}, timestampQuery: false, subgroups: false,
      shaderF16: false, meetsG0Minimums: false, requirementNotes: ["exception during requestAdapter"],
    };
  }
}

/**
 * Pure (testable, no-GPU) validator: does a limits dict satisfy G0 minimums?
 * Used by unit tests on CI machines without a GPU.
 */
export function validateG0Limits(limits: Record<string, number>): { ok: boolean; notes: string[] } {
  const notes: string[] = [];
  let ok = true;
  const check = (key: string, min: number) => {
    const got = limits[key];
    if (typeof got !== "number" || !(got >= min)) { ok = false; notes.push(`${key}: got ${String(got)}, need >= ${min}`); }
    else notes.push(`${key}: ok`);
  };
  check("maxStorageBufferBindingSize", G0_MINIMUMS.maxStorageBufferBindingSize);
  check("maxBufferSize", G0_MINIMUMS.maxBufferSize);
  return { ok, notes };
}
