// ADVERSARIAL A09 — Armijo Matrix: candidate selection, rejection modes, trust scaling, and status latching.
//
// Probes the complete Armijo candidate evaluation and selection matrix:
//   - Candidate 0 accepted (alpha = 1.0)
//   - Candidate 1 accepted (alpha = 0.5)
//   - Candidate near end accepted
//   - All invalid (energy / geometry)
//   - Safe Infinity handling (dist = 1e30, toi = 2.0, non-finite energy)
//   - Barrier failure gating (dist < dMin)
//   - CCD failure gating (0 < toi < 1)
//   - Contact / pair overflow gating
//   - trustScale < 1: check effective alpha vs reported alpha
//   - Status overwrite bug: batch 0 accepts, batch 1 rejects -> does round_report
//     overwrite minDist/minToi with rejected batch 1 data?
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";
import { decodeArmijoStatus, decodeNewtonStatus } from "../../src/backend/webgpu/gpu-newton.js";
import { GpuUniformSlot, UNIFORM_SLOT_STRIDE } from "../../src/backend/webgpu/gpu-buffers.js";

type AdvInternals = {
  contactParamsNow(): {
    dHat: number; kappa: number; mu: number; fricEps: number;
    floorY: number; floorOn: number; dMin: number; contactCapacity: number;
  };
  materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
  pushSimParams(): void;
  contactCountNow: number;
  simImage: { newtonIteration: number };
  scene: { material: Record<string, number>; mesh: { count: number }; positions: Float64Array };
};
const IN = (solver: unknown): AdvInternals => solver as AdvInternals;

let fix: DeviceFixture | null = null;

function restingFloorScene() {
  const g = buildGrid(4, 4, 0.1, 0.1);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.002;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

beforeAll(async () => {
  fix = await requireDevice(restingFloorScene, { contactCapacity: 256, pairCapacity: 1024 });
}, 180000);

const B = (binding: number, buffer: string) => ({ binding, buffer });

describe("ADVERSARIAL A09 — Armijo candidate matrix & status latching", () => {
  it("A9.1 candidate 0 vs candidate 1 vs near-end acceptance", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const K = 4;

    // Helper to simulate armijo_record with synthetic candidate inputs
    const runSyntheticBatch = async (
      candidates: Array<{ e: number; dist: number; toi: number; fin: boolean; ccd: boolean; overflow: boolean }>,
      e0 = 10.0,
      gtdx = -1.0,
      trust = 1.0,
    ) => {
      ex.writeBuffer("armijoAlphas", new Float32Array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]));
      ex.writeBuffer("e0Store", new Float32Array([e0, 0, 0, 0]));
      ex.writeBuffer("gtdxStore", new Float32Array([gtdx, 0, 0, 0]));
      ex.writeBuffer("trustScaleStore", new Float32Array([trust, 0, 0, 0]));
      ex.writeBuffer("armijoCandidates", new Float32Array(64));
      const bInit = new Float32Array(16);
      bInit[8] = 1.0; // finite lane starts at 1
      ex.writeBuffer("armijoStatus", bInit);

      driver.bankF(GpuUniformSlot.DMin, 0.002);
      for (let k = 0; k < candidates.length; k++) {
        const c = candidates[k];
        ex.writeBuffer("armijoCur", new Uint32Array([k, 0, 0, 0]));
        const st = new Float32Array(16);
        st[0] = c.e;
        st[2] = 1.0; // gnorm
        st[4] = c.dist;
        st[5] = c.toi;
        st[6] = c.fin ? 1.0 : 0.0;
        st[7] = c.ccd ? 1.0 : 0.0;
        ex.writeBuffer("solverStatus", st);
        ex.writeBuffer("contactOverflow", new Uint32Array([c.overflow ? 1 : 0, 0, 0, 0]));

        ex.beginBatch(`adv-cand-${k}`);
        ex.runPass({
          shader: "armijo", entry: "armijo_record",
          groups: [[
            B(9, "armijoCur"), B(8, "armijoAlphas"), B(48, "trustScaleStore"),
            B(45, "e0Store"), B(46, "gtdxStore"), B(40, "solverStatus"),
            B(41, "contactOverflow"), B(42, "armijoCandidates"), B(43, "armijoStatus"),
            { binding: 47, buffer: "uniformBank", offset: GpuUniformSlot.DMin * UNIFORM_SLOT_STRIDE, size: 16 },
          ]],
          x: 1,
        });
        await ex.submitBatch(false);
      }

      driver.bankU(GpuUniformSlot.ArmijoK, candidates.length);
      driver.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
      driver.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
      ex.beginBatch("adv-select");
      driver.armijoSelectPass();
      await ex.submitBatch(false);

      const raw = await ex.readSmall("armijoStatus", 64, "adv-status", "status");
      return decodeArmijoStatus(raw);
    };

    // Case 1: candidate 0 accepted (sufficient decrease: E <= E0 + 1e-4 * alpha * gtdx)
    {
      const res = await runSyntheticBatch([
        { e: 9.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 8.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 7.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 6.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
      ]);
      expect(res.accepted).toBe(true);
      expect(res.selectedAlpha).toBeCloseTo(1.0);
      expect(res.selectedIndex).toBe(0);
    }

    // Case 2: candidate 0 fails decrease, candidate 1 accepts
    {
      const res = await runSyntheticBatch([
        { e: 10.5, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false }, // higher E -> rejects
        { e: 9.5, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },  // lower E -> accepts
        { e: 9.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 8.5, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
      ]);
      expect(res.accepted).toBe(true);
      expect(res.selectedAlpha).toBeCloseTo(0.5);
      expect(res.selectedIndex).toBe(1);
    }

    // Case 3: candidate 0, 1, 2 fail; candidate 3 (near end) accepts
    {
      const res = await runSyntheticBatch([
        { e: 11.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 10.8, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 10.2, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false },
        { e: 9.9, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: false }, // accepts
      ]);
      expect(res.accepted).toBe(true);
      expect(res.selectedAlpha).toBeCloseTo(0.125);
      expect(res.selectedIndex).toBe(3);
    }
  }, 180000);

  it("A9.2 all invalid: safe infinity, barrier fail, CCD fail, overflow, non-finite", async () => {
    if (!fix) return;
    const { ex, driver } = fix;

    const runCandidate = async (cand: { e: number; dist: number; toi: number; fin: boolean; ccd: boolean; overflow: boolean }) => {
      ex.writeBuffer("armijoAlphas", new Float32Array([1.0, 1, 1, 1, 1, 1, 1, 1]));
      ex.writeBuffer("e0Store", new Float32Array([10.0, 0, 0, 0]));
      ex.writeBuffer("gtdxStore", new Float32Array([-1.0, 0, 0, 0]));
      ex.writeBuffer("trustScaleStore", new Float32Array([1.0, 0, 0, 0]));
      ex.writeBuffer("armijoCandidates", new Float32Array(64));
      const bInit = new Float32Array(16);
      bInit[8] = 1.0;
      ex.writeBuffer("armijoStatus", bInit);
      ex.writeBuffer("armijoCur", new Uint32Array([0, 0, 0, 0]));

      const st = new Float32Array(16);
      st[0] = cand.e;
      st[2] = 1.0;
      st[4] = cand.dist;
      st[5] = cand.toi;
      st[6] = cand.fin ? 1.0 : 0.0;
      st[7] = cand.ccd ? 1.0 : 0.0;
      ex.writeBuffer("solverStatus", st);
      ex.writeBuffer("contactOverflow", new Uint32Array([cand.overflow ? 1 : 0, 0, 0, 0]));

      driver.bankF(GpuUniformSlot.DMin, 0.002);
      ex.beginBatch("adv-eval-single");
      ex.runPass({
        shader: "armijo", entry: "armijo_record",
        groups: [[
          B(9, "armijoCur"), B(8, "armijoAlphas"), B(48, "trustScaleStore"),
          B(45, "e0Store"), B(46, "gtdxStore"), B(40, "solverStatus"),
          B(41, "contactOverflow"), B(42, "armijoCandidates"), B(43, "armijoStatus"),
          { binding: 47, buffer: "uniformBank", offset: GpuUniformSlot.DMin * UNIFORM_SLOT_STRIDE, size: 16 },
        ]],
        x: 1,
      });
      await ex.submitBatch(false);

      driver.bankU(GpuUniformSlot.ArmijoK, 1);
      driver.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
      driver.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
      ex.beginBatch("adv-select-single");
      driver.armijoSelectPass();
      await ex.submitBatch(false);

      const raw = await ex.readSmall("armijoStatus", 64, "adv-status-single", "status");
      return decodeArmijoStatus(raw);
    };

    // 1. Barrier fail: dist = -0.001 <= dMin (0.002)
    {
      const res = await runCandidate({ e: 5.0, dist: -0.001, toi: 2.0, fin: true, ccd: true, overflow: false });
      expect(res.accepted).toBe(false);
      expect(res.barrierFails).toBe(1);
    }

    // 2. CCD fail: toi = 0.5 (in (0, 1))
    {
      const res = await runCandidate({ e: 5.0, dist: 0.01, toi: 0.5, fin: true, ccd: true, overflow: false });
      expect(res.accepted).toBe(false);
      expect(res.ccdFails).toBe(1);
    }

    // 3. Overflow fail: contactOverflow = 1
    {
      const res = await runCandidate({ e: 5.0, dist: 0.01, toi: 2.0, fin: true, ccd: true, overflow: true });
      expect(res.accepted).toBe(false);
      expect(res.overflowFails).toBe(1);
    }

    // 4. Non-finite / NaN energy
    {
      const res = await runCandidate({ e: NaN, dist: 0.01, toi: 2.0, fin: false, ccd: true, overflow: false });
      expect(res.accepted).toBe(false);
      expect(res.finite).toBe(false);
    }

    // 5. Safe Infinity: dist = 1e30, toi = 2.0 (no contacts at all), finite energy, decreased
    {
      const res = await runCandidate({ e: 5.0, dist: 1e30, toi: 2.0, fin: true, ccd: true, overflow: false });
      // dist 1e30 > dMin, toi 2.0 is not in (0, 1), decrease holds -> accepts
      expect(res.accepted).toBe(true);
      expect(res.minDist).toBeGreaterThanOrEqual(1e29);
      expect(res.minToi).toBeGreaterThanOrEqual(2.0);
    }
  }, 180000);

  it("A9.3 trustScale < 1 modulates alpha and is folded correctly", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const TRUST = 0.5;

    ex.writeBuffer("armijoAlphas", new Float32Array([1.0, 0.5, 0.25, 0.125, 0, 0, 0, 0]));
    ex.writeBuffer("e0Store", new Float32Array([10.0, 0, 0, 0]));
    ex.writeBuffer("gtdxStore", new Float32Array([-1.0, 0, 0, 0]));
    ex.writeBuffer("trustScaleStore", new Float32Array([TRUST, 0, 0, 0]));
    ex.writeBuffer("armijoCandidates", new Float32Array(64));
    const bInit = new Float32Array(16);
    bInit[8] = 1.0;
    ex.writeBuffer("armijoStatus", bInit);
    ex.writeBuffer("armijoCur", new Uint32Array([0, 0, 0, 0]));

    // Candidate 0 evaluated with trust = 0.5: effective alpha should be 1.0 * 0.5 = 0.5!
    const st = new Float32Array(16);
    st[0] = 9.0; // decreases
    st[2] = 1.0;
    st[4] = 0.01;
    st[5] = 2.0;
    st[6] = 1.0;
    st[7] = 1.0;
    ex.writeBuffer("solverStatus", st);
    ex.writeBuffer("contactOverflow", new Uint32Array([0, 0, 0, 0]));

    driver.bankF(GpuUniformSlot.DMin, 0.002);
    ex.beginBatch("adv-trust-test");
    ex.runPass({
      shader: "armijo", entry: "armijo_record",
      groups: [[
        B(9, "armijoCur"), B(8, "armijoAlphas"), B(48, "trustScaleStore"),
        B(45, "e0Store"), B(46, "gtdxStore"), B(40, "solverStatus"),
        B(41, "contactOverflow"), B(42, "armijoCandidates"), B(43, "armijoStatus"),
        { binding: 47, buffer: "uniformBank", offset: GpuUniformSlot.DMin * UNIFORM_SLOT_STRIDE, size: 16 },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);

    driver.bankU(GpuUniformSlot.ArmijoK, 1);
    driver.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
    driver.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
    ex.beginBatch("adv-trust-select");
    driver.armijoSelectPass();
    await ex.submitBatch(false);

    const raw = await ex.readSmall("armijoStatus", 64, "adv-trust-status", "status");
    const armStatus = decodeArmijoStatus(raw);
    expect(armStatus.accepted).toBe(true);
    // In armijo_record: alpha = armijoAlphas[0] * rTrust[0] = 1.0 * 0.5 = 0.5.
    expect(armStatus.selectedAlpha).toBeCloseTo(0.5);
  }, 180000);

  it("A9.4 status overwrite bug: round_report overwrites accepted batch minDist/minToi with trailing rejected batch", async () => {
    if (!fix) return;
    const { ex } = fix;

    // Simulate batch 0 accepting with close contact (dist = 0.001, toi = 0.2)
    // and batch 1 rejecting all candidates with safe infinity (dist = 1e30, toi = 2.0)
    const nCtl = new Float32Array(16);
    nCtl[2] = 1.0; // nbAccepted
    nCtl[4] = 0.5; // nbAlpha
    nCtl[5] = 2.0; // bi (2 batches ran)
    nCtl[6] = 0.0; // nbAcceptBatch = 0
    nCtl[8] = 1.0; // global trial index
    ex.writeBuffer("newtonCtl", nCtl);

    // E0 store holds accepted energy
    ex.writeBuffer("e0Store", new Float32Array([12.34, 0, 0, 0]));
    ex.writeBuffer("gtdxStore", new Float32Array([-0.5, 0, 0, 0]));

    // armijoStatus holds the LAST batch's output (Batch 1, rejected, empty/infinity)
    const lastBatchStatus = new Float32Array(16);
    lastBatchStatus[0] = 0.0; // rejected
    lastBatchStatus[9] = 100.0; // energy of last candidate
    lastBatchStatus[13] = 1e30; // minDist of last batch = Infinity
    lastBatchStatus[14] = 2.0;  // minToi of last batch = Infinity
    ex.writeBuffer("armijoStatus", lastBatchStatus);

    ex.writeBuffer("newtonStatus", new Float32Array(20));

    // Run round_report
    ex.beginBatch("adv-round-report-probe");
    ex.runPass({
      shader: "newton-control", entry: "round_report",
      groups: [[
        B(90, "newtonCtl"), B(91, "newtonStatus"), B(92, "gtdxStore"),
        B(93, "e0Store"), B(94, "armijoStatus"),
      ]],
      x: 1,
    });
    await ex.submitBatch(false);

    const raw = await ex.readSmall("newtonStatus", 80, "adv-nst", "status");
    const st = decodeNewtonStatus(raw);

    // eslint-disable-next-line no-console
    console.log(`[A09] round_report status: accepted=${st.armijoAccepted} alpha=${st.selectedAlpha} ` +
      `minDistance=${st.minDistance} minToi=${st.minToi} energy=${st.energy}`);

    // Accepted is correctly reported as true (from newtonCtl[2])
    expect(st.armijoAccepted).toBe(true);
    expect(st.energy).toBeCloseTo(12.34);
    // BUG DETECTED: minDistance and minToi were overwritten by Batch 1's Infinity!
    // The accepted state's geometric proximity is lost to caller telemetry!
    expect(st.minDistance).toBe(Infinity);
  }, 180000);
});
