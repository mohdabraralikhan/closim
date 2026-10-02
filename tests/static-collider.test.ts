// Phase 1 ladder: cloth -> static triangle mesh (exercises the static
// collider path: negative-id verts, one-sided gradients, distance-only CCD).
import { describe, it, expect } from "vitest";
import { createScene } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { closestPointVertexTriangle } from "../src/collision/closest-point.js";
import { offsetMesh, preprocessMerged } from "./helpers.js";

function minClothStatic(
  x: ArrayLike<number>, nCloth: number,
  sPos: Float32Array, sIdx: Uint32Array,
): number {
  let minD = Infinity;
  const P = (v: number): [number, number, number] => [x[v * 3], x[v * 3 + 1], x[v * 3 + 2]];
  for (let v = 0; v < nCloth; v++) {
    const [px, py, pz] = P(v);
    for (let t = 0; t < sIdx.length / 3; t++) {
      const a = sIdx[t * 3], b = sIdx[t * 3 + 1], c = sIdx[t * 3 + 2];
      const d = closestPointVertexTriangle(
        px, py, pz,
        sPos[a * 3], sPos[a * 3 + 1], sPos[a * 3 + 2],
        sPos[b * 3], sPos[b * 3 + 1], sPos[b * 3 + 2],
        sPos[c * 3], sPos[c * 3 + 1], sPos[c * 3 + 2],
      ).dist;
      if (d < minD) minD = d;
    }
  }
  return minD;
}

describe("static triangle collider", () => {
  it("patch drapes over a static triangle without penetration", () => {
    const g = offsetMesh(4, 4, 0.08, 0.08, 0.01, 0.08, 0.01);
    const mesh = preprocessMerged([g], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
    // static platform: two triangles forming a 0.2x0.2 plate at y=0.03
    const sPos = new Float32Array([
      -0.05, 0.03, -0.05,
      0.15, 0.03, -0.05,
      0.15, 0.03, 0.15,
      -0.05, 0.03, 0.15,
    ]);
    const sIdx = new Uint32Array([0, 1, 2, 0, 2, 3]);
    scene.contact = new ContactSystem(
      { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.3 },
      mesh.indices,
    );
    scene.contact.setStaticMesh(sPos, sIdx);
    const solver = new CpuSolver();
    solver.initialize(scene);
    const h = 1 / 60;
    let maxActive = 0, maxVt = 0;
    for (let s = 0; s < 60; s++) {
      solver.step(h);
      for (const v of solver.getPositions()) expect(Number.isFinite(v)).toBe(true);
      const cd = solver.lastStats!.contact!;
      maxActive = Math.max(maxActive, cd.activePairs);
      maxVt = Math.max(maxVt, cd.vtPairs);
    }
    expect(maxActive).toBeGreaterThan(0); // static barrier engaged
    expect(maxVt).toBeGreaterThan(0); // vertex-vs-static-face pairs exist
    const minD = minClothStatic(solver.getPositions(), mesh.count, sPos, sIdx);
    expect(minD).toBeGreaterThan(5e-5); // no static penetration
    expect(minD).toBeLessThan(DEFAULT_CONTACT_PARAMS.dHatM); // resting in the barrier zone
    expect(Number.isFinite(solver.lastStats!.energy)).toBe(true);
  }, 240000);
});
