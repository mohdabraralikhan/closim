// GpuExecutor — G3 real-device context: pipelines, buffers, dispatch batches,
// readback discipline, and synchronization accounting.
//
// SYNC-POINT POLICY (every CPU<-GPU transfer is labeled at the call site):
//   - 64 B status + 16 B marker: the ONLY hot-loop mappings (Armijo control).
//   - 4 B counters/scalars: compact counter pack, PCG scalars, trust max.
//     Small, counted, never physics state.
//   - Full buffers (positions/gradients/contacts): ONLY via explicit debug
//     APIs (readBufferDebug), which increment forbiddenReadbacks when called
//     from a hot-loop context. Device tests assert the hot-loop count is 0.
//   - CPU->GPU writeBuffer uploads (scene init, uniforms, scalar slots) are
//     control transfers, unlimited but logged by byte volume.
//
// yields: every dispatch runs inside an explicit batch (one encoder, N passes,
// one submit) unless the driver needs per-stage wall time (benchmark mode).

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadNodeWebGPU, usageBits, mapModeBits } from "./gpu-node-binding.js";
import { SHADER_ENTRY_POINTS, type ShaderName } from "./gpu-pipelines.js";
import { GpuUniformSlot, UNIFORM_SLOT_STRIDE } from "./gpu-buffers.js";

/** Stage ids for marker.wgsl mark_stage (must match its comment). */
export const STAGE_IDS = {
  predictor: 0, aabb: 1, morton: 2, sort: 3, lbvh: 4, traverse: 5,
  closestVT: 6, closestEE: 7, ccdVT: 8, ccdEE: 9, expand: 10, compact: 11,
  floor: 12, membrane: 13, bending: 14, assemble: 15, barrier: 16,
  friction: 17, newtonRHS: 18, jacobi: 19, hvp: 20, pcg: 21, apply: 22,
  diagnostics: 23,
} as const;

export interface DeviceFacts {
  adapterName: string;
  backend: string;
  features: string[];
  limits: Record<string, number>;
  timestampQuery: boolean;
  subgroups: boolean;
}

export interface BindEntry {
  binding: number;
  buffer: string;
  offset?: number;
  size?: number;
}

export interface SyncLedger {
  submits: number;
  passes: number;
  /** Bytes mapped CPU<-GPU, by category. */
  mappedStatusBytes: number;
  mappedScalarBytes: number;
  mappedDebugBytes: number;
  /** CPU->GPU control upload bytes (scene + uniforms). */
  uploadedBytes: number;
  /** Number of mapAsync synchronizations (status + scalars). */
  mapSyncs: number;
  /** Full-buffer debug mappings (must be 0 in hot loops). */
  forbiddenReadbacks: number;
  /** G4C: uniform writes skipped as identical (no flush, no upload). */
  coalescedWrites: number;
}

export function emptyLedger(): SyncLedger {
  return {
    submits: 0, passes: 0, mappedStatusBytes: 0, mappedScalarBytes: 0,
    mappedDebugBytes: 0, uploadedBytes: 0, mapSyncs: 0, forbiddenReadbacks: 0,
    coalescedWrites: 0,
  };
}

const wgslCache = new Map<string, string>();

/** Load WGSL source (single source of truth: shaders/*.wgsl on disk). */
export function loadWgslSource(shader: string): string {
  const hit = wgslCache.get(shader);
  if (hit !== undefined) return hit;
  const dir = fileURLToPath(new URL("./shaders/", import.meta.url));
  // NOTE: import.meta.url under vitest resolves to the SOURCE tree, so tests
  // exercise the same files the device path ships.
  const src = readFileSync(`${dir}${shader}.wgsl`, "utf8");
  wgslCache.set(shader, src);
  return src;
}

export class GpuExecutor {
  device: any;
  facts: DeviceFacts;
  usage: Record<string, number>;
  mapMode: { READ: number };
  buffers = new Map<string, any>();
  bufferBytes = new Map<string, number>();
  pipelines = new Map<string, any>();
  bindGroups = new Map<string, any>();
  ledger: SyncLedger = emptyLedger();
  private encoder: any = null;
  private batchPasses = 0;
  private batches = 0;
  /** G4C: last-written bytes per CPU-write-only uniform region (bank/simParams). */
  private uniformCache = new Map<string, Uint8Array>();
  // ---- G5A timestamp capture (optional feature; zero cost when inactive) ----
  private tsActive = false;
  private tsQuerySet: any = null;
  private tsResolve: any = null;
  private tsLabels: string[] = [];
  private tsMaxQueries = 0;

  private constructor(device: any, facts: DeviceFacts, usage: Record<string, number>, mapMode: { READ: number }) {
    this.device = device;
    this.facts = facts;
    this.usage = usage;
    this.mapMode = mapMode;
  }

  /** Create a device context, or null when no binding/adapter exists. */
  static async create(): Promise<GpuExecutor | null> {
    const loaded = loadNodeWebGPU();
    if (!loaded) return null;
    try {
      const adapter = await loaded.gpu.requestAdapter();
      if (!adapter) return null;
      const info = (adapter.info ?? {}) as Record<string, string>;
      const features: string[] = [...(adapter.features ?? [])];
      const limits: Record<string, number> = {};
      for (const [k, v] of Object.entries((adapter.limits ?? {}) as Record<string, number>)) {
        limits[k] = Number(v);
      }
      // G3 rule: never REQUIRE optional features; record them.
      // Storage-buffer headroom: traverse (10), compact (15), diagnostics
      // (11) exceed the default 8/storage-stage. Always TRY 16 first (this
      // adapter class offers it); fall back to defaults only when the request
      // itself is rejected. Kernels that still exceed the granted limit fail
      // loudly at pipeline creation (validateAllShaders), never silently.
      // G5A: enable timestamp-query when the adapter offers it (query sets
      // cannot be created without the device feature). Still optional: every
      // request below degrades gracefully when it is absent or rejected.
      const wantTs = features.includes("timestamp-query") ? ["timestamp-query"] : [];
      let device = null;
      try {
        // Binding-size headroom: the contact expand buffers (primIdsVT/EE)
        // exceed the 128 MiB default binding size past ~10k verts (50k needs
        // ~192 MiB). Request up to 512 MiB, clamped to the adapter maximum
        // (requesting exactly the adapter max is always grantable).
        const adapterMaxBinding = Number(
          ((adapter.limits ?? {}) as Record<string, number>).maxStorageBufferBindingSize ?? 134217728,
        );
        const wantBinding = Math.min(512 * 1024 * 1024, adapterMaxBinding);
        device = await adapter.requestDevice({
          requiredFeatures: wantTs,
          requiredLimits: {
            maxStorageBuffersPerShaderStage: 16,
            maxStorageBufferBindingSize: wantBinding,
          },
        });
      } catch {
        device = null;
      }
      if (!device) {
        try {
          device = await adapter.requestDevice({ requiredFeatures: [] });
        } catch {
          return null;
        }
      }
      if (!device) return null;
      try {
        const granted = (device.limits as Record<string, number> | undefined)?.maxStorageBuffersPerShaderStage;
        if (typeof granted === "number") limits["maxStorageBuffersPerShaderStage"] = granted;
        const grantedBinding = (device.limits as Record<string, number> | undefined)?.maxStorageBufferBindingSize;
        if (typeof grantedBinding === "number") limits["maxStorageBufferBindingSize"] = grantedBinding;
      } catch { /* limits introspection is best-effort */ }
      const facts: DeviceFacts = {
        adapterName: `${info.vendor ?? "?"} ${info.device ?? "?"} (${info.architecture ?? "?"})`,
        backend: info.backendType ?? (info as Record<string, string>).backend ?? "unknown",
        features,
        limits,
        // Device-granted facts (the request above may have fallen back).
        timestampQuery: (() => {
          try {
            const df = device.features as unknown as { has?: (f: string) => boolean } | string[];
            if (df && typeof (df as { has?: unknown }).has === "function") {
              return (df as { has: (f: string) => boolean }).has("timestamp-query");
            }
            if (Array.isArray(df)) return (df as string[]).includes("timestamp-query");
          } catch { /* fall through to adapter facts */ }
          return features.includes("timestamp-query");
        })(),
        subgroups: features.includes("subgroups"),
      };
      const exec = new GpuExecutor(device, facts, usageBits(loaded.globals), mapModeBits(loaded.globals));
      exec.attachValidationListener();
      return exec;
    } catch {
      return null;
    }
  }

  /** Attach the uncaptured-error listener (called once at create). */
  attachValidationListener(): void {
    const dev = this.device as any;
    if (!dev) return;
    this.validationScopesSupported =
      typeof dev.pushErrorScope === "function" && typeof dev.popErrorScope === "function";
    const record = (e: any): void => {
      try {
        const type = String(
          (e as any)?.error?.errorType ?? (e as any)?.type ?? "uncapturederror",
        );
        const message = String((e as any)?.error?.message ?? (e as any)?.message ?? e).slice(0, 400);
        this.uncapturedErrors.push({ type, message });
      } catch { /* a listener must never throw */ }
    };
    try {
      if (typeof dev.addEventListener === "function") {
        dev.addEventListener("uncapturederror", record);
      } else {
        try { dev.onuncapturederror = record; } catch { /* best effort */ }
      }
    } catch { /* best effort */ }
  }

  ensureBuffer(name: string, bytes: number, usage: number, label?: string): any {
    const have = this.buffers.get(name);
    if (have && (this.bufferBytes.get(name) ?? 0) >= bytes) return have;
    if (have) {
      this.bindGroups.clear();
      try { have.destroy(); } catch { /* best effort */ }
    }
    const buf = this.device.createBuffer({
      label: label ?? `g3/${name}`,
      size: Math.max(16, bytes),
      usage,
    });
    this.buffers.set(name, buf);
    this.bufferBytes.set(name, Math.max(16, bytes));
    return buf;
  }

  storage(name: string, bytes: number): any {
    return this.ensureBuffer(name, bytes, this.usage.STORAGE | this.usage.COPY_DST | this.usage.COPY_SRC);
  }

  uniform16(name: string): any {
    return this.ensureBuffer(name, 16, this.usage.UNIFORM | this.usage.COPY_DST);
  }

  atomic(name: string): any {
    return this.ensureBuffer(name, 16, this.usage.STORAGE | this.usage.COPY_DST | this.usage.COPY_SRC);
  }

  writeBuffer(name: string, data: ArrayBufferView, offset = 0): void {
    const buf = this.buffers.get(name);
    if (!buf) throw new Error(`GpuExecutor: unknown buffer ${name}`);
    // G4C write-coalescing: CPU-write-only uniform state (uniformBank,
    // simParams — WebGPU forbids shader writes to UNIFORM bindings, so no
    // GPU write can stale this cache) skips identical rewrites, letting the
    // open batch extend across passes. Everything else always flushes:
    // STORAGE buffers may hold GPU-written results a rewrite must not hide.
    // GLOBAL INVARIANT (G6C): no shader in this codebase writes a UNIFORM
    // buffer or bank slot. If one ever does, this cache silently returns
    // stale values — grep for `var<uniform>` write targets before adding any
    // storage-write to a uniform-bound buffer.
    const bytes = new Uint8Array((data as any).buffer, (data as any).byteOffset, (data as any).byteLength);
    if ((name === this.uniformBankName || name === "simParams") && bytes.byteLength <= 256) {
      const key = `${name}@${offset}`;
      const prev = this.uniformCache.get(key);
      if (prev && prev.byteLength === bytes.byteLength) {
        let same = true;
        for (let i = 0; i < bytes.byteLength; i++) {
          if (prev[i] !== bytes[i]) { same = false; break; }
        }
        if (same) {
          this.ledger.coalescedWrites++;
          return;
        }
      }
      this.uniformCache.set(key, Uint8Array.from(bytes));
    }
    // Flush first: writes are immediate, so any open batch must submit before
    // the new value lands (otherwise earlier passes would read it too).
    this.flushBatch(`write/${name}`);
    this.device.queue.writeBuffer(buf, offset, data as any);
    this.ledger.uploadedBytes += (data as any).byteLength;
  }

  writeU32(name: string, value: number): void {
    const a = new Uint32Array(4);
    a[0] = value >>> 0;
    this.writeBuffer(name, a);
  }

  writeF32(name: string, value: number): void {
    const a = new Float32Array(4);
    a[0] = value;
    this.writeBuffer(name, a);
  }

  writeVec4(name: string, x: number, y: number, z: number, w: number): void {
    this.writeBuffer(name, new Float32Array([x, y, z, w]));
  }

  /** Write the shared BLAS params slot (bank slot 0: count u32 + alpha/beta f32). */
  writeBlas(count: number, alpha: number, beta = 0): void {
    const a = new ArrayBuffer(16);
    new Uint32Array(a)[0] = count >>> 0;
    const f = new Float32Array(a);
    f[1] = alpha; f[2] = beta;
    this.writeBuffer(this.uniformBankName, new Uint8Array(a), 0);
  }

  // ---- pipelines ----

  getPipeline(shader: ShaderName, entry: string): any {
    const key = `${shader}:${entry}`;
    const hit = this.pipelines.get(key);
    if (hit) return hit;
    const eps = (SHADER_ENTRY_POINTS as Record<string, string[]>)[shader] ?? [];
    if (!eps.includes(entry)) {
      throw new Error(`GpuExecutor: unknown entry ${entry} for ${shader}`);
    }
    const code = loadWgslSource(shader);
    const module = this.device.createShaderModule({ label: `${shader}:${entry}`, code });
    const pipe = this.device.createComputePipeline({
      label: key,
      layout: "auto",
      compute: { module, entryPoint: entry },
    });
    this.pipelines.set(key, pipe);
    return pipe;
  }

  /** Eagerly compile every registered shader entry (init-time validation). */
  async validateAllShaders(): Promise<{ ok: boolean; failures: string[] }> {
    const failures: string[] = [];
    for (const [shader, entries] of Object.entries(SHADER_ENTRY_POINTS)) {
      for (const entry of entries) {
        try {
          const code = loadWgslSource(shader);
          const module = this.device.createShaderModule({ label: `validate/${shader}`, code });
          const info = await module.getCompilationInfo?.();
          const errs = ((info?.messages ?? []) as any[]).filter((m) => m.type === "error");
          if (errs.length > 0) {
            failures.push(`${shader}:${entry}: ${errs.map((e) => e.message).join(" | ").slice(0, 400)}`);
            continue;
          }
          this.device.createComputePipeline({
            label: `validate/${shader}:${entry}`,
            layout: "auto",
            compute: { module, entryPoint: entry },
          });
        } catch (err) {
          failures.push(`${shader}:${entry}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }
    return { ok: failures.length === 0, failures };
  }

  // ---- dispatch batching ----

  /**
   * Open a batch, submitting any passes already encoded (safe to call
   * redundantly; labels the new boundary for profiling).
   */
  beginBatch(label = "batch"): void {
    this.flushBatch();
    this.encoder = this.device.createCommandEncoder({ label: `g3-${label}` });
    this.batchPasses = 0;
    this.batches++;
  }

  /** Append one compute pass to the open batch (auto-opens one if needed). */
  runPass(spec: {
    shader: ShaderName; entry: string; groups: BindEntry[][];
    x: number; y?: number; z?: number; label?: string;
  }): void {
    if (!this.encoder) {
      this.encoder = this.device.createCommandEncoder({ label: "g3-auto" });
      this.batchPasses = 0;
      this.batches++;
    }
    const pipe = this.getPipeline(spec.shader, spec.entry);
    const groups = spec.groups.map((entries, gi) => {
      // Bind-group cache: PCG/HVP loops re-issue identical bindings hundreds
      // of times per Newton solve; Dawn group creation dominates otherwise.
      const key = `${spec.shader}:${spec.entry}/g${gi}/` +
        entries.map((e) => `${e.binding}=${e.buffer}+${e.offset ?? 0}:${e.size ?? "*"}`).join(",");
      const hit = this.bindGroups.get(key);
      if (hit) return hit;
      const layout = pipe.getBindGroupLayout(gi);
      const bg = this.device.createBindGroup({
        label: `${spec.shader}:${spec.entry}/g${gi}`,
        layout,
        entries: entries.map((e) => {
          const buf = this.buffers.get(e.buffer);
          if (!buf) throw new Error(`GpuExecutor: unknown buffer ${e.buffer}`);
          return {
            binding: e.binding,
            resource: { buffer: buf, offset: e.offset ?? 0, size: e.size ?? this.bufferBytes.get(e.buffer)! - (e.offset ?? 0) },
          };
        }),
      });
      this.bindGroups.set(key, bg);
      return bg;
    });
    // G5A timestamp capture: attach begin/end writes in the pass descriptor.
    let passDesc: any = undefined;
    if (this.tsActive && this.tsQuerySet && this.tsLabels.length * 2 + 2 <= this.tsMaxQueries) {
      const qi = this.tsLabels.length * 2;
      passDesc = {
        timestampWrites: {
          querySet: this.tsQuerySet,
          beginningOfPassWriteIndex: qi,
          endOfPassWriteIndex: qi + 1,
        },
      };
      this.tsLabels.push(`${spec.shader}:${spec.entry}`);
    }
    const pass = passDesc !== undefined
      ? this.encoder.beginComputePass(passDesc)
      : this.encoder.beginComputePass();
    pass.setPipeline(pipe);
    groups.forEach((bg, gi) => pass.setBindGroup(gi, bg));
    // Scale-correctness: WebGPU caps one dispatch dimension at 65535
    // workgroups (4,194,240 threads at workgroup 64). Counts past that
    // (contact expand at 50k dispatches 188160 groups) split across Y rows;
    // shaders compute the flat index as gid.x + gid.y * 4194240.
    // Reductions (pcg-reduce, diagnostics) are NOT row-splittable;
    // broadphase-sort computes flat index gid.x + gid.y * 4194240 and is row-splittable.
    const MAX_GROUPS_X = 65535;
    const NO_SPLIT: ReadonlySet<string> = new Set(
      ["pcg-reduce", "diagnostics"],
    );
    let dx = spec.x;
    let dy = spec.y ?? 1;
    if (dy === 1 && dx > MAX_GROUPS_X) {
      if (NO_SPLIT.has(spec.shader)) {
        throw new Error(
          `GpuExecutor: ${spec.shader}:${spec.entry} needs ${dx} workgroups, ` +
          `past the 65535 row-splittable cap for a reduction/sort`,
        );
      }
      dx = MAX_GROUPS_X;
      dy = Math.ceil(spec.x / MAX_GROUPS_X);
    }
    pass.dispatchWorkgroups(dx, dy, spec.z ?? 1);
    pass.end();
    this.batchPasses++;
    this.ledger.passes++;
  }

  /**
   * Submit any open batch WITHOUT waiting (queue order preserves correctness).
   * Called automatically by writeBuffer: uniform/control writes are immediate
   * (not recorded), so every write flushes prior dispatches first. Without
   * this, all passes in a batch would observe only the LAST uniform values —
   * silently corrupting sort stages, BLAS params, marks, and SimParams
   * refreshes. The extra submits are the honest synchronization cost.
   */
  flushBatch(_reason = "flush"): void {
    if (!this.encoder || this.batchPasses === 0) {
      // Nothing encoded: drop the empty encoder without submitting.
      this.encoder = null;
      return;
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.batchPasses = 0;
    this.ledger.submits++;
  }

  /** Submit the open batch (closing it) and optionally wait for idle. */
  async submitBatch(wait = false): Promise<void> {
    if (!this.encoder) return;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.ledger.submits++;
    if (wait) await this.device.queue.onSubmittedWorkDone();
  }

  /** One-pass convenience: open batch, run, submit (optionally await idle). */
  async dispatch(spec: {
    shader: ShaderName; entry: string; groups: BindEntry[][];
    x: number; y?: number; z?: number; label?: string; wait?: boolean;
  }): Promise<void> {
    this.beginBatch(spec.label ?? `${spec.shader}:${spec.entry}`);
    this.runPass(spec);
    await this.submitBatch(spec.wait ?? false);
  }

  // ---- readback (every call is a labeled SYNC POINT) ----

  private async mapCopy(srcName: string, bytes: number, label: string): Promise<ArrayBuffer> {
    const src = this.buffers.get(srcName);
    if (!src) throw new Error(`GpuExecutor: unknown buffer ${srcName}`);
    const staging = this.device.createBuffer({
      label: `staging/${label}`,
      size: bytes,
      usage: this.mapMode.READ | this.usage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder({ label: `readback/${label}` });
    enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    this.ledger.submits++;
    await staging.mapAsync(this.mapMode.READ);
    this.ledger.mapSyncs++;
    const out = (staging.getMappedRange(0, bytes) as ArrayBuffer).slice(0);
    staging.unmap();
    try { staging.destroy(); } catch { /* best effort */ }
    return out;
  }

  /** Compact status/marker/counter/scalar read (allowed hot-loop transfer). */
  async readSmall(srcName: string, bytes: number, label: string, category: "status" | "scalar"): Promise<ArrayBuffer> {
    const data = await this.mapCopy(srcName, bytes, label);
    if (category === "status") this.ledger.mappedStatusBytes += bytes;
    else this.ledger.mappedScalarBytes += bytes;
    return data;
  }

  /**
   * Full-buffer debug read (positions/gradients/contacts/vectors).
   * FORBIDDEN in hot loops: increments forbiddenReadbacks so device tests can
   * prove the stepping path never calls it. Explicit debug/snapshot use only.
   */
  async readBufferDebug(srcName: string, label: string, hotLoop = true): Promise<ArrayBuffer> {
    const bytes = this.bufferBytes.get(srcName) ?? 0;
    const data = await this.mapCopy(srcName, bytes, `debug/${label}`);
    this.ledger.mappedDebugBytes += bytes;
    if (hotLoop) this.ledger.forbiddenReadbacks++;
    return data;
  }

  get uniformBankName(): string {
    return "uniformBank";
  }

  /** Uncaptured WebGPU errors observed on this device (dev/test mechanism:
   *  a poisoned queue can leave status buffers stale/zero, which decodes as
   *  convergence — these must surface loudly instead). Drained by
   *  popValidationScope / consumeUncapturedErrors. */
  uncapturedErrors: Array<{ type: string; message: string }> = [];
  /** True when device.pushErrorScope/popErrorScope exist (else no-op). */
  validationScopesSupported = false;
  private validationScopeDepth = 0;

  /** Push a `validation` error scope. No-op when the binding lacks scopes. */
  pushValidationScope(): void {
    this.validationScopeDepth++;
    try {
      (this.device as any)?.pushErrorScope?.("validation");
    } catch { /* scope tracking is best-effort; listener still records */ }
  }

  /**
   * Pop one validation scope: any error captured between push and pop is
   * recorded in uncapturedErrors AND thrown, so a poisoned batch can never
   * decode a stale/zero status as convergence. Also drains errors the
   * uncaptured listener observed in the window.
   */
  async popValidationScope(label: string): Promise<void> {
    if (this.validationScopeDepth > 0) this.validationScopeDepth--;
    let scopeErr: any = null;
    try {
      scopeErr = await (this.device as any)?.popErrorScope?.();
    } catch (err) {
      scopeErr = err;
    }
    if (scopeErr) {
      const message = String((scopeErr as any)?.message ?? scopeErr);
      this.uncapturedErrors.push({ type: "validation-scope", message: `${label}: ${message}` });
      throw new Error(`GpuExecutor validation error (${label}): ${message}`);
    }
    this.assertNoUncapturedErrors(label);
  }

  /** Throw if the uncaptured listener observed errors since the last drain. */
  assertNoUncapturedErrors(context: string): void {
    if (this.uncapturedErrors.length === 0) return;
    const errs = this.uncapturedErrors.splice(0, this.uncapturedErrors.length);
    throw new Error(
      `GpuExecutor uncaptured WebGPU error(s) (${context}): ` +
      errs.map((e) => `[${e.type}] ${e.message}`).join(" | ").slice(0, 800),
    );
  }

  /** Drain and return pending uncaptured errors without throwing (tests). */
  consumeUncapturedErrors(): Array<{ type: string; message: string }> {
    return this.uncapturedErrors.splice(0, this.uncapturedErrors.length);
  }

  /** Write one 16 B uniform-bank slot (f32[4] payload). */
  writeBankSlot(slot: number, f: [number, number, number, number]): void {
    this.writeBuffer(this.uniformBankName, new Float32Array(f), slot * UNIFORM_SLOT_STRIDE);
  }

  writeBankU32(slot: number, value: number): void {
    const a = new Uint32Array(4);
    a[0] = value >>> 0;
    this.writeBuffer(this.uniformBankName, a, slot * UNIFORM_SLOT_STRIDE);
  }

  /** Bind-entry shorthand for a bank slot. */
  bank(binding: number, slot: number, size = 16): BindEntry {
    void GpuUniformSlot;
    const offset = slot * UNIFORM_SLOT_STRIDE;
    if (offset % 256 !== 0) {
      // Dawn validation fails SILENTLY (uncaptured error poisons the queue);
      // catch misalignment here where the stack trace points at the cause.
      throw new Error(`GpuExecutor.bank: slot ${slot} offset ${offset} violates 256 B uniform alignment`);
    }
    return { binding, buffer: this.uniformBankName, offset, size };
  }

  destroy(): void {
    try { this.device.destroy(); } catch { /* best effort */ }
    this.buffers.clear();
    this.pipelines.clear();
    this.uniformCache.clear();
    try { this.tsQuerySet?.destroy?.(); } catch { /* best effort */ }
    try { this.tsResolve?.destroy?.(); } catch { /* best effort */ }
    this.tsQuerySet = null;
    this.tsResolve = null;
    this.tsActive = false;
    wgslCache.clear();
  }

  // ---- G5A timestamp capture ----
  // Optional `timestamp-query` feature; zero cost when inactive or
  // unsupported. Captures per-pass GPU begin/end ticks across submits in one
  // measurement window; endTimestampCapture resolves + reads them back.
  // Tick period is NOT exposed by this binding: callers calibrate in situ
  // (tick-delta vs wall-delta ratio; NVIDIA ~= 1ns) and report both.

  /** Open a capture window (maxPasses dispatches). False when unsupported. */
  beginTimestampCapture(maxPasses: number): boolean {
    if (!this.facts.timestampQuery || typeof this.device.createQuerySet !== "function") return false;
    try {
      const queries = Math.max(4, Math.min(Math.ceil(maxPasses) * 2, 8192));
      try { this.tsQuerySet?.destroy?.(); } catch { /* best effort */ }
      try { this.tsResolve?.destroy?.(); } catch { /* best effort */ }
      this.tsQuerySet = this.device.createQuerySet({ type: "timestamp", count: queries });
      this.tsResolve = this.device.createBuffer({
        label: "ts-resolve",
        size: queries * 8,
        usage: this.usage.QUERY_RESOLVE | this.usage.COPY_SRC,
      });
      this.tsLabels = [];
      this.tsMaxQueries = queries;
      this.tsActive = true;
      return true;
    } catch {
      this.tsActive = false;
      return false;
    }
  }

  /** Resolve + read the window. Returns per-pass {label, ticks} or null. */
  async endTimestampCapture(): Promise<{ labels: string[]; ticks: BigUint64Array | Uint32Array } | null> {
    if (!this.tsActive || !this.tsQuerySet || !this.tsResolve) {
      this.tsActive = false;
      return null;
    }
    this.tsActive = false;
    const count = this.tsLabels.length * 2;
    if (count === 0) return { labels: [], ticks: new BigUint64Array(0) };
    const labels = this.tsLabels;
    this.tsLabels = [];
    try {
      const enc = this.device.createCommandEncoder({ label: "ts-resolve" });
      enc.resolveQuerySet(this.tsQuerySet, 0, count, this.tsResolve, 0);
      this.device.queue.submit([enc.finish()]);
      this.ledger.submits++;
      const staging = this.device.createBuffer({
        label: "staging/ts",
        size: count * 8,
        usage: this.mapMode.READ | this.usage.COPY_DST,
      });
      const cenc = this.device.createCommandEncoder({ label: "ts-read" });
      cenc.copyBufferToBuffer(this.tsResolve, 0, staging, 0, count * 8);
      this.device.queue.submit([cenc.finish()]);
      this.ledger.submits++;
      await staging.mapAsync(this.mapMode.READ);
      const raw = (staging.getMappedRange(0, count * 8) as ArrayBuffer).slice(0);
      staging.unmap();
      try { staging.destroy(); } catch { /* best effort */ }
      return { labels, ticks: new BigUint64Array(raw) };
    } catch {
      return null;
    }
  }
}
