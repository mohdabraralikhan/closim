// G3 kernel parity on the real device: predictor / membrane / bending /
// barrier / contact-compaction, mirror<->device first, then CPU<->device.
// Small scenes, clear margins. Skips cleanly without WebGPU.
//
// NOTE: first device use in this file compiles ~40 WGSL pipelines on Dawn
// (seconds, one-time per worker); heavy tests carry explicit timeouts.
import { describe, it, expect, beforeAll } from "vitest";
import { CpuBroadPhase } from "../../src/collision/broadphase.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import {
  GpuContactSystem, compareContactMultisets, gpuVtKey, gpuEeKey,
} from "../../src/backend/webgpu/gpu-contact.js";
import { fp32TriangleEnergyGradient, fp32MembraneMesh } from "../../src/backend/webgpu/gpu-tolerances.js";
import { evalInternal, evalMembrane } from "../../src/physics/fem.js";
import { evalMembraneHvp } from "../../src/physics/membrane-hvp.js";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import {
  sharedDevice, resetDeviceState, stripScene, headOnScene, normTOI,
} from "./device-setup.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

async function readF32(
  ex: { readBufferDebug(n: string, l: string, h: boolean): Promise<ArrayBuffer> },
  name: string, floats: number,
): Promise<Float32Array> {
  const raw = await ex.readBufferDebug(name, `parity/${name}`, false);
  return new Float32Array(raw.slice(0, floats * 4));
}

/** Pack trial positions into the device vec4f layout (cloth only, no statics). */
function packVec4(X: Float64Array, n: number): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.fround(X[i * 3]);
    out[i * 4 + 1] = Math.fround(X[i * 3 + 1]);
    out[i * 4 + 2] = Math.fround(X[i * 3 + 2]);
  }
  return out;
}

let strip: DeviceFixture | null = null;
let headon: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("kernels-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
  const hs = headOnScene();
  headon = await sharedDevice(
    "kernels-headon",
    () => hs.scene,
    { contactCapacity: 1024, pairCapacity: 2048 },
  );
}, 180000);

describe("G3 device kernel parity", () => {
  it("2. predictor matches y_hat with exact pins", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    await driver.predictorPass();
    const n = scene.mesh.count;
    const pos = await readF32(ex, "position", n * 4);
    const h = 1 / 60;
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      const pinned = Math.abs(scene.positions[i * 3]) < 1e-9;
      for (let k = 0; k < 3; k++) {
        const got = pos[i * 4 + k];
        const g = k === 1 ? -9.81 : 0;
        const want = pinned
          ? scene.positions[i * 3 + k]
          : scene.positions[i * 3 + k] + h * scene.velocities[i * 3 + k] + h * h * g;
        maxErr = Math.max(maxErr, Math.abs(got - want));
      }
    }
    expect(maxErr).toBeLessThan(1e-6);
  });

  it("3. membrane element gradients match the FP32 mirror", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    // deterministically deformed state (shear + sag): nonzero strain per
    // triangle, so per-element comparison carries real signal.
    const n = scene.mesh.count;
    const x = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] += 0.002 * Math.sin(i);
      x[i * 3 + 1] -= 0.003 * (x[i * 3] / 0.2);
      x[i * 3 + 2] += 0.001 * Math.cos(i * 2);
    }
    resetDeviceState(strip, x);
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    const m = scene.mesh.triCount;
    const eg = await readF32(ex, "elementGradient", m * 9);
    const at = (v: number): [number, number, number] => [x[v * 3], x[v * 3 + 1], x[v * 3 + 2]];
    let maxAbs = 0;
    let maxRel = 0;
    for (let t = 0; t < m; t++) {
      const i0 = scene.mesh.indices[t * 3];
      const i1 = scene.mesh.indices[t * 3 + 1];
      const i2 = scene.mesh.indices[t * 3 + 2];
      const ref = fp32TriangleEnergyGradient(
        at(i0), at(i1), at(i2),
        [
          scene.mesh.invDm[t * 4], scene.mesh.invDm[t * 4 + 1],
          scene.mesh.invDm[t * 4 + 2], scene.mesh.invDm[t * 4 + 3],
        ],
        scene.mesh.areas[t], scene.material,
      );
      for (let k = 0; k < 9; k++) {
        const a = Math.abs(eg[t * 9 + k] - ref.grad[k]);
        maxAbs = Math.max(maxAbs, a);
        maxRel = Math.max(maxRel, a / Math.max(Math.abs(ref.grad[k]), 1e-9));
      }
    }
    expect(maxAbs).toBeLessThan(1e-4);
    expect(maxRel).toBeLessThan(1e-3);
  }, 120000);

  it("3b. analytic membrane HVP matches the mirror on device (G4A)", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    // same deformed state recipe as test 3 (nonzero strain per triangle)
    const x = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] += 0.002 * Math.sin(i);
      x[i * 3 + 1] -= 0.003 * (x[i * 3] / 0.2);
      x[i * 3 + 2] += 0.001 * Math.cos(i * 2);
    }
    resetDeviceState(strip, x);
    // deterministic direction (packed f32, same layout as pcgSearch)
    const p = new Float64Array(n * 3);
    for (let i = 0; i < n * 3; i++) p[i] = Math.fround(Math.sin(i * 12.9898) * 0.5);
    // evaluate first (pushes material uniforms + sets position), then HVP
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    ex.writeBuffer("pcgSearch", Float32Array.from(p));
    driver.analyticMembraneHvp();
    await ex.submitBatch(true);
    const hp = await readF32(ex, "hpMembrane", n * 3);
    // CPU analytic mirror on f32-rounded inputs (device sees f32 exactly)
    const xf = Float64Array.from(x, (v) => Math.fround(v));
    const mir = evalMembraneHvp(xf, Float64Array.from(p), scene.mesh, scene.material);
    let maxAbs = 0;
    const scale = Math.max(1e-12, ...Array.from(mir).map(Math.abs));
    for (let i = 0; i < n * 3; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(hp[i] - mir[i]));
    }
    expect(maxAbs / scale).toBeLessThan(2e-3);
    // FD oracle on the SAME device state (debug-path link): analytic must
    // agree with central differences far better than validation tolerance.
    driver.fdMembraneHvp();
    await ex.submitBatch(true);
    const fd = await readF32(ex, "hpMembrane", n * 3);
    let fdAbs = 0;
    for (let i = 0; i < n * 3; i++) {
      fdAbs = Math.max(fdAbs, Math.abs(fd[i] - mir[i]));
    }
    // eslint-disable-next-line no-console
    console.log(`[g4a-device] analytic rel=${(maxAbs / scale).toExponential(2)} fd rel=${(fdAbs / scale).toExponential(2)}`);
    expect(fdAbs / scale).toBeLessThan(2e-2);
  }, 120000);

  it("3c. gather assembly matches the scan on device (G4B)", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    const x = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] += 0.002 * Math.sin(i);
      x[i * 3 + 1] -= 0.003 * (x[i * 3] / 0.2);
      x[i * 3 + 2] += 0.001 * Math.cos(i * 2);
    }
    resetDeviceState(strip, x);
    const p = new Float64Array(n * 3);
    for (let i = 0; i < n * 3; i++) p[i] = Math.fround(Math.sin(i * 12.9898) * 0.5);
    const prev = driver.cfg.useGatherAssembly;
    try {
      // scan path (oracle): full eval + scan-assembled HVP
      driver.cfg.useGatherAssembly = false;
      await solver.evaluateNewtonState(false, 1, 1 / 60);
      ex.writeBuffer("pcgSearch", Float32Array.from(p));
      driver.analyticMembraneHvp();
      await ex.submitBatch(true);
      const gScan = await readF32(ex, "gradient", n * 3);
      const hpScan = await readF32(ex, "hpMembrane", n * 3);
      // gather path: identical sums in CSR (element) order
      driver.cfg.useGatherAssembly = true;
      await solver.evaluateNewtonState(false, 2, 1 / 60);
      ex.writeBuffer("pcgSearch", Float32Array.from(p));
      driver.analyticMembraneHvp();
      await ex.submitBatch(true);
      const gGather = await readF32(ex, "gradient", n * 3);
      const hpGather = await readF32(ex, "hpMembrane", n * 3);
      // HVP: single-sum, same order -> bit-exact
      let hpDiff = 0;
      for (let i = 0; i < n * 3; i++) hpDiff = Math.max(hpDiff, Math.abs(hpGather[i] - hpScan[i]));
      expect(hpDiff).toBe(0);
      // gradient: 3-term composition (membrane, hinge, contact) rounds the
      // partials in a different parenthesization -> ulp-level only
      let gDiff = 0;
      const scale = Math.max(1e-12, ...Array.from(gScan).map(Math.abs));
      for (let i = 0; i < n * 3; i++) gDiff = Math.max(gDiff, Math.abs(gGather[i] - gScan[i]));
      expect(gDiff / scale).toBeLessThan(1e-5);
    } finally {
      driver.cfg.useGatherAssembly = prev;
    }
  }, 120000);

  it("4. bending reaches the assembled gradient (device vs CPU split)", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    // genuinely bent state (middle lifted 5 mm): strong bending signal, so the
    // device-vs-CPU split comparison is meaningful (not noise-over-noise).
    // NOTE: resetDeviceState does NOT run the predictor, so the device
    // evaluates exactly this state; the CPU reference uses it verbatim.
    const x = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      const t = Math.sin((Math.PI * x[i * 3]) / 0.2);
      x[i * 3 + 1] += 0.005 * t;
    }
    resetDeviceState(strip, x);
    // free fall: no contacts, so device gradient = membrane + bending
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    const grad = await readF32(ex, "gradient", n * 3);
    const full = evalInternal(x, scene.mesh, scene.material);
    const mem = evalMembrane(x, scene.mesh, scene.material);
    const mir = fp32MembraneMesh(x, scene.mesh, scene.material);
    let maxAbs = 0;
    const scale = Math.max(1e-9, ...Array.from(full.grad).map(Math.abs));
    for (let i = 0; i < n * 3; i++) {
      const deviceBending = grad[i] - mir.grad[i];
      const cpuBending = full.grad[i] - mem.grad[i];
      maxAbs = Math.max(maxAbs, Math.abs(deviceBending - cpuBending));
    }
    expect(maxAbs / scale).toBeLessThan(1e-3);
  }, 120000);

  it("5. barrier energy matches the mirror on an active set", async () => {
    if (!headon) return;
    const { solver, ex } = headon;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const mesh = scene.mesh;
    const x0 = Float64Array.from(scene.positions);
    // shallow 1 mm approach: rebuild the G2-proven segment (mirror state)
    const nA = mesh.count / 2;
    const x1 = Float64Array.from(x0);
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    resetDeviceState(headon, x0, x1);
    const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
    expect(ev.contactCount).toBeGreaterThan(0);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    const ref = gcs.barrierEnergyGrad(x1, set);
    const ce = await readF32(ex, "contactEnergy", ev.contactCount);
    let devE = 0;
    for (let i = 0; i < ev.contactCount; i++) devE += ce[i];
    const relE = Math.abs(devE - ref.energy) / Math.max(Math.abs(ref.energy), 1e-12);
    expect(relE).toBeLessThan(2e-3);
  }, 120000);

  it("6. compact contact set matches the mirror (multiset + TOI)", async () => {
    if (!headon) return;
    const { solver, ex } = headon;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const mesh = scene.mesh;
    const x0 = Float64Array.from(scene.positions);
    const nA = mesh.count / 2;
    const x1 = Float64Array.from(x0);
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    resetDeviceState(headon, x0, x1);
    const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
    const cc = ev.contactCount;
    expect(cc).toBeGreaterThan(0);
    const NN = await readF32(ex, "contactN", cc * 4);
    const ID = new Uint32Array((await ex.readBufferDebug("contactId", "parity/id", false)).slice(0, cc * 16));
    const TOI = await readF32(ex, "contactTOI", cc);
    const DT = await readF32(ex, "contactDist", cc);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    const inBand = (dd: number): boolean => Math.abs(dd - P.dHatM) < 1e-6 || dd < 1e-9;
    const devKeys: string[] = [];
    const devTOI = new Map<string, number[]>();
    const devDist = new Map<string, number[]>();
    for (let i = 0; i < cc; i++) {
      const kind = NN[i * 4 + 3];
      const a = ID[i * 4], b = ID[i * 4 + 1], c = ID[i * 4 + 2], d = ID[i * 4 + 3];
      const key = kind < 0.5 ? gpuVtKey(a, b, c, d) : kind < 1.5 ? gpuEeKey(a, b, c, d) : `floor:${a}`;
      devKeys.push(key);
      if (!devTOI.has(key)) { devTOI.set(key, []); devDist.set(key, []); }
      devTOI.get(key)!.push(normTOI(TOI[i]));
      devDist.get(key)!.push(DT[i]);
    }
    const mirBand = (k: string): boolean => {
      const rs = set.contacts.filter((r) => r.key === k);
      return rs.length > 0 && rs.every((r) => inBand(r.dist));
    };
    const devBand = (k: string): boolean => {
      const ds = devDist.get(k) ?? [];
      return ds.length > 0 && ds.every(inBand);
    };
    const cmp = compareContactMultisets(
      set.contacts.map((r) => r.key).filter((k) => !mirBand(k)),
      devKeys.filter((k) => !devBand(k)),
    );
    expect(cmp.missing).toEqual([]);
    expect(cmp.extra).toEqual([]);
    // per-contact TOI agreement (nearest duplicate, 2.0 sentinel normalized)
    const mirTOI = new Map<string, number[]>();
    for (const r of set.contacts) {
      if (!mirTOI.has(r.key)) mirTOI.set(r.key, []);
      mirTOI.get(r.key)!.push(r.toi);
    }
    let maxTErr = 0;
    for (const [k, ds] of devTOI) {
      if (devBand(k)) continue;
      const ms = mirTOI.get(k) ?? [];
      for (const t of ds) {
        if (t === Infinity) {
          expect(ms.every((u) => u === Infinity)).toBe(true);
        } else {
          const finite = ms.filter((u) => u !== Infinity);
          expect(finite.length).toBeGreaterThan(0);
          maxTErr = Math.max(maxTErr, Math.min(...finite.map((u) => Math.abs(u - t))));
        }
      }
    }
    expect(maxTErr).toBeLessThan(1e-3);
  }, 120000);
});
