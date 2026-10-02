import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene, pinColumn } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { checkTriGradient } from "./gradient-check.js";
import { evalInternal } from "../src/physics/fem.js";

describe("membrane gradient", () => {
  it("matches finite differences", () => {
    const r = checkTriGradient();
    expect(r.maxRel).toBeLessThan(1e-4);
  });
});

describe("strip-pull reference (V0.1)", () => {
  it("sags under gravity without overstretch or NaN", () => {
    const { positions, uv, indices } = buildGrid(12, 6, 0.2, 0.1);
    const mesh = preprocess(positions, uv, indices, 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
    pinColumn(scene, (x) => x < 1e-9);
    expect(scene.pinned.size).toBeGreaterThan(0);
    const solver = new CpuSolver();
    solver.initialize(scene);
    const h = 1 / 60;
    for (let s = 0; s < 60; s++) {
      solver.step(h);
      for (const v of solver.getPositions()) expect(Number.isFinite(v)).toBe(true);
    }
    const pos = solver.getPositions();
    // free-end mean y must sag (material last column i==nx; tip x collapses when hanging)
    const NX = 12, NY = 6;
    let sumY = 0, cnt = 0;
    for (let j = 0; j <= NY; j++) {
      const id = j * (NX + 1) + NX;
      sumY += pos[id * 3 + 1]; cnt++;
    }
    const meanY = sumY / Math.max(cnt, 1);
    expect(meanY).toBeLessThan(-0.01);
    // strain bounded (no PBD-like overstretch)
    const ev = evalInternal(Float64Array.from(pos), mesh, scene.material);
    expect(ev.maxStrain).toBeLessThan(0.05);
    // pins held
    for (const [id, p] of scene.pinned) {
      expect(pos[id * 3]).toBeCloseTo(p[0], 9);
      expect(pos[id * 3 + 1]).toBeCloseTo(p[1], 9);
    }
  }, 60000);
});
