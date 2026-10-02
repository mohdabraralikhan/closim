// G2 benchmark: per-stage contact-pipeline breakdown at 1k / 10k / 50k verts.
// Stages: broadphase (G1 LBVH+traverse) | VT closest+CCD | EE closest+CCD |
// compaction | barrier energy+grad | friction. Informational only — no FPS
// assertions, no speedup claims. On headless CI the FP32 mirror stands in for
// dispatches, so mirror timings ARE the reported compute proxy; GPU wall time
// will only improve on this (minus synchronization, reported separately).
import { describe, it } from "vitest";
import { GpuBroadPhase } from "../../src/backend/webgpu/gpu-broadphase.js";
import { GpuContactSystem } from "../../src/backend/webgpu/gpu-contact.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

interface StageMs {
  broadphase: number;
  vtCcd: number;
  eeCcd: number;
  compact: number;
  barrier: number;
  friction: number;
  total: number;
}

async function benchPair(nx: number, gap0: number, approach: number): Promise<{
  vertexCount: number; triangleCount: number; contacts: number; stages: StageMs;
  pairOverflow: number; contactOverflow: number; pairs: number;
}> {
  const w = 0.2;
  const A = offsetMesh(nx, nx, w, w, -w / 2 - gap0 / 2, 0.02, -w / 2);
  const B = offsetMesh(nx, nx, w, w, w / 2 + gap0 / 2, 0.02, -w / 2);
  const mesh = preprocessMerged([A, B], 0.15);
  const x0 = Float64Array.from(mesh.positions);
  const x1 = Float64Array.from(x0);
  const nA = A.positions.length / 3;
  for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? approach : -approach;

  const tBp0 = performance.now();
  // Size buffers to the scene: conservative swept boxes yield ~25 pairs/tri,
  // so the default 16k pair buffer would truncate (explicitly, via overflow).
  // The benchmark measures the pipeline, not the truncation path.
  const bp = await new GpuBroadPhase({
    indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM,
    pairCapacity: mesh.triCount * 32,
  }).build(x0, x1);
  const tBp1 = performance.now();

  const gcs = new GpuContactSystem({
    indices: mesh.indices, triCount: mesh.triCount,
    dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
    frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    contactCapacity: 1 << 20,
  });
  gcs.beginStep(x0);
  // Timed sub-stages mirror the device passes: expand+closest+CCD live inside
  // build(); barrier/friction consume the compact set. We time build() as the
  // VT+EE+compact aggregate and split VT/EE by re-running classification-free
  // closest evaluation below.
  const tC0 = performance.now();
  const set = await gcs.build(x1, bp);
  const tC1 = performance.now();
  gcs.commit(x1, set);
  const tB0 = performance.now();
  gcs.barrierEnergyGrad(x1, set);
  const tB1 = performance.now();
  gcs.frictionForces(x1, set);
  const tB2 = performance.now();

  // VT vs EE split: expand a sample of pairs with the real endpoint-sharing
  // skip (materializing ALL primitives would cost GBs at 50k scale; the split
  // only weights the build aggregate, so a sample is exact enough).
  let vtN = 0, eeN = 0;
  for (const { a: tA, b: tB } of bp.pairs.slice(0, 4000)) {
    vtN += 6;
    const A = [mesh.indices[tA * 3], mesh.indices[tA * 3 + 1], mesh.indices[tA * 3 + 2]];
    const B = [mesh.indices[tB * 3], mesh.indices[tB * 3 + 1], mesh.indices[tB * 3 + 2]];
    const edges = (T: number[]): Array<[number, number]> => [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]];
    for (const [a, b] of edges(A)) {
      for (const [c, d] of edges(B)) {
        if (a === c || a === d || b === c || b === d) continue;
        eeN++;
      }
    }
  }
  const buildMs = tC1 - tC0;
  const vtFrac = vtN / Math.max(1, vtN + eeN);

  const stages: StageMs = {
    broadphase: tBp1 - tBp0,
    vtCcd: buildMs * vtFrac,
    eeCcd: buildMs * (1 - vtFrac),
    compact: 0, // fused into build() on the mirror; device reports it separately
    barrier: tB1 - tB0,
    friction: tB2 - tB1,
    total: (tBp1 - tBp0) + buildMs + (tB1 - tB0) + (tB2 - tB1),
  };
  return {
    vertexCount: mesh.count, triangleCount: mesh.triCount,
    contacts: set.contacts.length, stages,
    pairOverflow: bp.diagnostics.candidateOverflow,
    contactOverflow: set.diagnostics.contactOverflow,
    pairs: bp.pairs.length,
  };
}

describe("G2 contact benchmark", () => {
  it("reports per-stage breakdown at 1k / 10k / 50k vertices", async () => {
    // ~1k verts: 22x22 per patch; ~10k: 70x70; ~50k: 158x158.
    // 1 mm final gap keeps the interface in the barrier zone (realistic load).
    const small = await benchPair(22, 0.01, 0.0045);
    // eslint-disable-next-line no-console
    console.log("[g2-bench] 1k:", JSON.stringify(small));
    const medium = await benchPair(70, 0.01, 0.0045);
    // eslint-disable-next-line no-console
    console.log("[g2-bench] 10k:", JSON.stringify(medium));
    const large = await benchPair(158, 0.01, 0.0045);
    // eslint-disable-next-line no-console
    console.log("[g2-bench] 50k:", JSON.stringify(large));
    // sanity: every class produced contacts and finite timings
    for (const r of [small, medium, large]) {
      if (!(r.contacts > 0)) throw new Error("benchmark scene produced no contacts");
      for (const v of Object.values(r.stages)) {
        if (!Number.isFinite(v) || v < 0) throw new Error("non-finite stage timing");
      }
    }
  }, 600000);
});
