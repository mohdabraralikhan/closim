// WebGPU device bootstrap (Phase 2 G0.1).
// Compute-only device is sufficient: physics never touches a render canvas.
// Timestamp-query is OPTIONAL and only enabled when the adapter exposes it.

import { getGpuNamespace, isWebGPUAvailable, type GPUAdapterLike, type GPUDeviceLike } from "./webgpu-dts.js";
import { detectGpuCapabilities, type GpuCapabilityReport } from "./gpu-capabilities.js";

export interface GpuDeviceHandle {
  adapter: GPUAdapterLike | null;
  device: GPUDeviceLike | null;
  capabilities: GpuCapabilityReport;
  /** True when a real GPU device was created (false on headless CI). */
  ready: boolean;
  timestampQueryEnabled: boolean;
}

export async function initGpuDevice(opts: {
  powerPreference?: "low-power" | "high-performance";
  forceFallbackAdapter?: boolean;
} = {}): Promise<GpuDeviceHandle> {
  const capabilities = await detectGpuCapabilities();
  if (!capabilities.available || !isWebGPUAvailable()) {
    return { adapter: null, device: null, capabilities, ready: false, timestampQueryEnabled: false };
  }
  try {
    const ns = getGpuNamespace()!;
    const adapter = await ns.requestAdapter({
      powerPreference: opts.powerPreference ?? "high-performance",
      forceFallbackAdapter: opts.forceFallbackAdapter ?? false,
    });
    if (!adapter) {
      return { adapter: null, device: null, capabilities, ready: false, timestampQueryEnabled: false };
    }
    const requiredFeatures: string[] = [];
    // Phase 2 rule: NEVER require subgroups / timestamp-query / f16 in G0.
    // Request timestamp-query only opportunistically.
    const feats: Set<string> = adapter.features as Set<string>;
    if (feats && feats.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    const device = await adapter.requestDevice({ requiredFeatures });
    const tsEnabled = (device?.features as Set<string>)?.has?.("timestamp-query") ?? false;
    return { adapter, device, capabilities, ready: !!device, timestampQueryEnabled: tsEnabled };
  } catch {
    return { adapter: null, device: null, capabilities, ready: false, timestampQueryEnabled: false };
  }
}

export function destroyGpuDevice(handle: GpuDeviceHandle): void {
  try { handle.device?.destroy?.(); } catch { /* best effort */ }
}
