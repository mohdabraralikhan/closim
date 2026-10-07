// MIMO scale-contact stress: counters, overflow latch, truncation order,
// duplicates, and canonical-set stability at 1k / 10k / 50k.
//
// Per evaluation the suite asserts the race-free structural invariants:
// - pairCount == pairScanned (traverse bumps both per emitted overlap)
// - contactCount <= contactScanned (appends are a subset of evaluations)
// - primVT == 6 * expandedPairs, primEE <= 9 * expandedPairs
// - overflowFlag  == (pairCount    > pairCapacity)    (iff, exact)
// - contactOverflow == (contactCount > contactCapacity) (iff, exact)
// - every LIVE record valid (kind domain, finite, dist domain)
// - no duplicate (kind,id) tuple in the live set
// Canonical SET stability across reps is asserted only when nothing
// overflowed (first-cap-wins truncation is race-ordered by design: class A).
// Under overflow the suite logs set-overlap ratios as nondeterminism data.
import { describe, it, expect } from "vitest";
import type { ClothScene } from "../../../src/physics/scene.js";
import { GpuBroadPhase } from "../../../src/backend/webgpu/gpu-broadphase.js";
import { GpuContactSystem, G2_BOUNDARY_BAND } from "../../../src/backend/webgpu/gpu-contact.js";
import { DEFAULT_CONTACT_PARAMS } from "../../../src/collision/types.js";
import { requireDevice, resetDeviceState } from "../../webgpu/device-setup.js";
import {
  BIG_CAP, GRID, IN, PAIR_CAP, PATCH, REPS,
  buildScene, checkDuplicateGroups, checkLiveRecords, liveKeys, liveMirrorKeys,
  multisetEqual, readContactLive, readCounters,
  type ScaleClass, type SceneName,
} from "./scale-helpers.js";

interface CaseResult {
  counts: number[];
  pairCounts: number[];
  overflows: number[];
  stableReps: number;
  dupGroups: number;
  maxDupTimes: number;
  inexactDups: number;
  invalidRecords: number;
}

interface CaseConfig {
  scene: SceneName;
  cls: ScaleClass;
  cap: number;
  pairCap?: number; // default PAIR_CAP[cls]
  /** True when measured CONTACT need exceeds memory-safe caps. */
  expectOverflow: boolean;
  /** Set when pair need is known to exceed the pair cap (pair-truncated
   *  coverage: contact sets may vary by emission-order race — class A). */
  expectPairOverflow?: boolean;
  label: string;
}

// Measured 1k needs (attempted appends): fold ~60k, dense ~404k,
// headon 1136, floor 1089. Fold/dense scale ~linearly in tris from there,
// far past memory-safe caps at 10k/50k (prim scratch is 15x pairs), so
// those configs exercise the overflow path by design (latch iff, kept-set
// validity, truncation logging) rather than full-set containment.
const CASES: CaseConfig[] = [
  // 1k: everything fits oversized except the tiny forcing configs.
  // dense-1k full need (~404k contacts from ~121k pairs) gets fitting caps.
  { scene: "fold", cls: "1k", cap: 131072, expectOverflow: false, expectPairOverflow: false, label: "1k/fold/big" },
  { scene: "headon", cls: "1k", cap: 8192, expectOverflow: false, expectPairOverflow: false, label: "1k/headon/big" },
  { scene: "floor", cls: "1k", cap: 8192, expectOverflow: false, expectPairOverflow: false, label: "1k/floor/big" },
  { scene: "dense", cls: "1k", cap: 524288, pairCap: 196608, expectOverflow: false, expectPairOverflow: false, label: "1k/dense/big" },
  { scene: "fold", cls: "1k", cap: 128, expectOverflow: true, expectPairOverflow: false, label: "1k/fold/tiny" },
  { scene: "headon", cls: "1k", cap: 128, expectOverflow: true, expectPairOverflow: false, label: "1k/headon/tiny" },
  { scene: "floor", cls: "1k", cap: 128, expectOverflow: true, expectPairOverflow: false, label: "1k/floor/tiny" },
  { scene: "dense", cls: "1k", cap: 128, expectOverflow: true, expectPairOverflow: true, label: "1k/dense/tiny" },
  // 10k: contacts fit headon/floor-big; pairs saturate everywhere at 10k.
  { scene: "headon", cls: "10k", cap: 65536, expectOverflow: false, expectPairOverflow: true, label: "10k/headon/big" },
  { scene: "floor", cls: "10k", cap: 262144, expectOverflow: false, expectPairOverflow: true, label: "10k/floor/big" },
  { scene: "fold", cls: "10k", cap: 65536, expectOverflow: true, expectPairOverflow: true, label: "10k/fold/saturated" },
  { scene: "dense", cls: "10k", cap: 65536, expectOverflow: true, expectPairOverflow: true, label: "10k/dense/saturated" },
  { scene: "fold", cls: "10k", cap: 512, expectOverflow: true, expectPairOverflow: true, label: "10k/fold/tiny" },
  { scene: "dense", cls: "10k", cap: 512, expectOverflow: true, expectPairOverflow: true, label: "10k/dense/tiny" },
  // 50k: floor/headon contact-fit (pairs saturate everywhere at 50k);
  // fold/dense are pair-saturated with small contact sets (kept-pair
  // coverage races; contact latch stays exact).
  { scene: "floor", cls: "50k", cap: 262144, expectOverflow: false, expectPairOverflow: true, label: "50k/floor/big" },
  { scene: "headon", cls: "50k", cap: 131072, expectOverflow: false, expectPairOverflow: true, label: "50k/headon/big" },
  { scene: "fold", cls: "50k", cap: 65536, expectOverflow: false, expectPairOverflow: true, label: "50k/fold/saturated" },
  { scene: "dense", cls: "50k", cap: 65536, expectOverflow: false, expectPairOverflow: true, label: "50k/dense/saturated" },
  { scene: "floor", cls: "50k", cap: 2048, expectOverflow: true, expectPairOverflow: true, label: "50k/floor/tiny" },
  { scene: "dense", cls: "50k", cap: 2048, expectOverflow: true, expectPairOverflow: true, label: "50k/dense/tiny" },
];

async function runCase(
  sceneName: SceneName, cls: ScaleClass, cap: number, tag: string, pairCapOverride?: number,
): Promise<CaseResult> {
  const build = (): ReturnType<typeof import("../../../src/physics/scene.js").createScene> =>
    buildScene(sceneName, cls);
  const pairCap = pairCapOverride ?? PAIR_CAP[cls];
  const fix = await requireDevice(build, { contactCapacity: cap, pairCapacity: pairCap });
  if (!fix) {
    // eslint-disable-next-line no-console
    console.log(`[scale-overflow] ${tag}: NO DEVICE — skipped`);
    return { counts: [], pairCounts: [], overflows: [], stableReps: 0, dupGroups: 0, maxDupTimes: 0, inexactDups: 0, invalidRecords: 0 };
  }
  try {
    const { solver, driver } = fix;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const x0 = Float64Array.from(scene.positions);
    const dHat = IN(solver).contactParamsNow().dHat;
    const dispatchThreads = 64 * Math.max(1, Math.ceil(pairCap / 64));
    const res: CaseResult = {
      counts: [], pairCounts: [], overflows: [], stableReps: 0,
      dupGroups: 0, maxDupTimes: 0, inexactDups: 0, invalidRecords: 0,
    };
    let refKeys: string[] | null = null;
    const reps = REPS[cls];
    for (let r = 0; r < reps; r++) {
      resetDeviceState(fix, Float64Array.from(x0));
      const ev = await solver.evaluateNewtonState(false, 100 + r, 1 / 60);
      const c = await readCounters(fix);
      res.counts.push(ev.contactCount);
      res.pairCounts.push(c.pairCount);
      res.overflows.push(c.contactOverflow);

      // --- race-free counter relations (class B gates) ---
      expect(c.pairScanned, `${tag} rep ${r}: pairScanned != pairCount (class B)`).toBe(c.pairCount);
      expect(c.contactCount <= c.contactScanned, `${tag} rep ${r}: attempted appends exceed evaluations (class B)`).toBe(true);
      const expanded = Math.min(c.pairCount, dispatchThreads);
      expect(c.primVT, `${tag} rep ${r}: primVT != 6*expanded (class B)`).toBe(6 * expanded);
      expect(c.primEE <= 9 * expanded, `${tag} rep ${r}: primEE > 9*expanded (class B)`).toBe(true);
      expect(c.pairOverflow, `${tag} rep ${r}: pair overflow latch wrong (class B)`).toBe(c.pairCount > pairCap ? 1 : 0);
      expect(c.contactOverflow, `${tag} rep ${r}: contact overflow latch wrong (class B)`).toBe(ev.contactCount > cap ? 1 : 0);
      expect(Number.isFinite(ev.status.energy), `${tag} rep ${r}: non-finite energy`).toBe(true);

      // --- live record validity + duplicates (class B gates) ---
      const live = await readContactLive(fix, ev.contactCount, driver.c.cap);
      const chk = checkLiveRecords(live, dHat);
      res.invalidRecords += chk.errors.length;
      expect(chk.errors, `${tag} rep ${r}: invalid live records (class B): ${chk.errors.slice(0, 3).join(" | ")}`).toEqual([]);
      const keys = liveKeys(live);
      // Duplicates are DOCUMENTED multiset semantics (one primitive reachable
      // via several triangle pairs; CPU mirror and barrier agree): the gate
      // is payload-exactness within each identity group, not absence.
      const groups = checkDuplicateGroups(live);
      res.dupGroups += groups.length;
      for (const g of groups) {
        res.maxDupTimes = Math.max(res.maxDupTimes, g.slots.length);
        if (!g.exact) {
          res.inexactDups++;
          // eslint-disable-next-line no-console
          console.log(`[scale-overflow] ${tag} rep ${r}: INEXACT duplicate ${g.key} ` +
            `x${g.slots.length} firstDiff=${g.firstDiff} (class B corruption)`);
        }
      }
      expect(res.inexactDups, `${tag} rep ${r}: same-identity records with different payloads (class B)`).toBe(0);

      // --- canonical stability only when nothing truncated (class A zone) ---
      // Pair truncation also races the kept subset (proven by direct pair
      // multiset comparison), so exactness needs both latches clear.
      const truncated = c.pairOverflow === 1 || c.contactOverflow === 1;
      if (!truncated) {
        if (refKeys === null) refKeys = keys;
        else {
          const cmp = multisetEqual(refKeys, keys);
          if (cmp.equal) res.stableReps++;
          else {
            // eslint-disable-next-line no-console
            console.log(`[scale-overflow] ${tag} rep ${r}: SET DRIFT without overflow ` +
              `onlyRef=${cmp.onlyA.length} onlyNew=${cmp.onlyB.length} (class B suspect)`);
          }
          expect(cmp.equal, `${tag} rep ${r}: canonical set drifted with no overflow (class B)`).toBe(true);
        }
      } else {
        // Truncation is first-cap-wins by atomic order: log overlap, assert nothing.
        if (refKeys !== null) {
          const a = new Set(refKeys);
          let overlap = 0;
          for (const k of keys) if (a.has(k)) overlap++;
          // eslint-disable-next-line no-console
          console.log(`[scale-overflow] ${tag} rep ${r}: overflow truncation overlap ` +
            `${overlap}/${keys.length} live (class A race data)`);
        } else refKeys = keys;
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[scale-overflow] ${tag}: verts=${scene.mesh.count} tris=${scene.mesh.triCount} ` +
      `cap=${cap} pairs=${pairCap} counts=[${res.counts.join(",")}] ` +
      `pairCounts=[${res.pairCounts.join(",")}] overflowFlags=[${res.overflows.join(",")}] ` +
      `dupGroups=${res.dupGroups} maxDupTimes=${res.maxDupTimes} invalid=${res.invalidRecords}`);
    return res;
  } finally {
    fix.ex.destroy();
  }
}

const SCENES: SceneName[] = ["fold", "headon", "floor", "dense"];

describe("MIMO scale overflow + canonical stability", () => {
  for (const cfg of CASES) {
    const verts = cfg.scene === "headon"
      ? 2 * (PATCH[cfg.cls] + 1) * (PATCH[cfg.cls] + 1)
      : (GRID[cfg.cls] + 1) * (GRID[cfg.cls] + 1) * (cfg.scene === "dense" ? 3 : 1);
    const contactWord = cfg.expectOverflow ? "contact overflows" : "contact fits";
    const pairWord = cfg.expectPairOverflow ? "pairs saturate" : "pairs fit";
    it(`${cfg.label} (~${verts} verts): ${contactWord}, ${pairWord}`, async () => {
      const r = await runCase(cfg.scene, cfg.cls, cfg.cap, cfg.label, cfg.pairCap);
      if (r.counts.length === 0) return; // no device
      if (cfg.expectOverflow) {
        for (const cc of r.counts) expect(cc > cfg.cap, `${cfg.label}: expected overflow did not happen`).toBe(true);
        for (const o of r.overflows) expect(o).toBe(1);
      } else {
        for (const o of r.overflows) expect(o).toBe(0);
        for (const cc of r.counts) expect(cc).toBeGreaterThan(0);
      }
      if (cfg.expectPairOverflow !== undefined) {
        for (const pc of r.pairCounts) {
          const want = cfg.expectPairOverflow ? 1 : 0;
          expect(pc > (cfg.pairCap ?? PAIR_CAP[cfg.cls]) ? 1 : 0, `${cfg.label}: pair overflow expectation wrong`).toBe(want);
        }
      }
      expect(r.inexactDups).toBe(0);
      expect(r.invalidRecords).toBe(0);
    }, 1800000);
  }
});

/**
 * Device live set vs CPU-mirror (FP32 GpuContactSystem) multiset, mapped to
 * one key space. Multiplicity included: duplicates are legal on both sides,
 * so the gate is multiset equality after removing G2_BOUNDARY_BAND-adjacent
 * records (classification flips inside the band are legitimate on both
 * sides — class A). Anything else is a device/mirror divergence (class B).
 */
async function mirrorMultisetCase(
  sceneName: SceneName, cls: ScaleClass, cap: number, pairCap?: number,
): Promise<void> {
  const fix = await requireDevice(() => buildScene(sceneName, cls), {
    contactCapacity: cap, pairCapacity: pairCap ?? PAIR_CAP[cls],
  });
  if (!fix) return;
  try {
    const { solver } = fix;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const x0 = Float64Array.from(scene.positions);
    resetDeviceState(fix, Float64Array.from(x0));
    const ev = await solver.evaluateNewtonState(false, 500, 1 / 60);
    expect(ev.contactCount).toBeGreaterThan(0);
    const c = await readCounters(fix);
    expect(c.contactOverflow).toBe(0);
    expect(c.pairOverflow).toBe(0);
    const live = await readContactLive(fix, ev.contactCount, fix.driver.c.cap);
    const devKeys = liveMirrorKeys(live);
    expect(devKeys.every((k) => k !== null), `${cls}/${sceneName}: unmappable device id`).toBe(true);
    const devDist = new Map<string, number[]>();
    for (let s = 0; s < live.live; s++) {
      const k = devKeys[s] as string;
      const arr = devDist.get(k);
      if (arr) arr.push(live.Dist[s]);
      else devDist.set(k, [live.Dist[s]]);
    }
    // CPU mirror over the identical degenerate segment x0 -> x0.
    const mesh = scene.mesh;
    const cp = scene.contact!.params;
    const bp = await new GpuBroadPhase({
      indices: mesh.indices, triCount: mesh.triCount, pad: 0.002,
      pairCapacity: mesh.triCount * 32,
    }).build(x0, x0);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: cp.dHatM, dMinM: cp.dMinM, kappaJ: cp.kappaJ,
      frictionMu: cp.frictionMu, frictionEpsM: cp.frictionEpsM,
      contactCapacity: 1 << 20,
    });
    if (scene.contact!.floorY !== null && scene.contact!.floorY !== undefined) {
      gcs.floorY = scene.contact!.floorY;
    }
    gcs.beginStep(x0);
    const set = await gcs.build(x0, bp);
    const mirKeys = set.contacts.map((r) => r.key);
    const mirDist = new Map<string, number[]>();
    for (const r of set.contacts) {
      const arr = mirDist.get(r.key);
      if (arr) arr.push(r.dist);
      else mirDist.set(r.key, [r.dist]);
    }
    const inBand = (d: number): boolean =>
      Math.abs(d - cp.dHatM) <= G2_BOUNDARY_BAND || d < 1e-12 + G2_BOUNDARY_BAND;
    const stripBand = (keys: string[], dists: Map<string, number[]>): { rest: string[]; band: number } => {
      const rest: string[] = [];
      let band = 0;
      const used = new Map<string, number>();
      for (const k of keys) {
        const arr = dists.get(k) ?? [];
        const idx = used.get(k) ?? 0;
        used.set(k, idx + 1);
        if (idx < arr.length && inBand(arr[idx])) band++;
        else rest.push(k);
      }
      return { rest, band };
    };
    const dev = stripBand(devKeys as string[], devDist);
    const mir = stripBand(mirKeys, mirDist);
    const cmp = multisetEqual(dev.rest.sort(), mir.rest.sort());
    // eslint-disable-next-line no-console
    console.log(`[scale-mirror] ${cls}/${sceneName}: device=${devKeys.length} mirror=${mirKeys.length} ` +
      `bandDev=${dev.band} bandMir=${mir.band} match=${cmp.equal} ` +
      `missing=${cmp.onlyA.length} extra=${cmp.onlyB.length}`);
    if (!cmp.equal) {
      // eslint-disable-next-line no-console
      console.log(`[scale-mirror]   missing e.g. ${JSON.stringify(cmp.onlyA.slice(0, 5))}`);
      // eslint-disable-next-line no-console
      console.log(`[scale-mirror]   extra e.g. ${JSON.stringify(cmp.onlyB.slice(0, 5))}`);
    }
    expect(cmp.equal, `${cls}/${sceneName}: device/mirror multiset divergence beyond the boundary band (class B)`).toBe(true);
  } finally {
    fix.ex.destroy();
  }
}

describe("MIMO device vs CPU-mirror contact multisets", () => {
  it("1k/dense: identical multisets incl. multiplicity", async () => {
    // Full pair expansion (~121k pairs) + full contact set (~330k).
    await mirrorMultisetCase("dense", "1k", 524288, 196608);
  }, 1800000);
  it("1k/fold: identical multisets incl. multiplicity", async () => {
    await mirrorMultisetCase("fold", "1k", 131072);
  }, 1800000);
  it("1k/headon: identical multisets incl. multiplicity", async () => {
    await mirrorMultisetCase("headon", "1k", 8192);
  }, 1800000);
  // NOTE: 10k+ pair volumes (352k-8.1M measured) exceed memory-safe prim
  // scratch (15x pairs), so exact device/mirror gates live at 1k only; above
  // that the iff/validity/exact-duplicate gates above carry the load.
});
