// Phase 1 ladder: deterministic repeatability on CPU.
import { describe, it, expect } from "vitest";
import { createScene } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { offsetMesh, preprocessMerged } from "./helpers.js";

function runOnce(): Float64Array {
  const g = offsetMesh(5, 5, 0.1, 0.1, 0, 0.04, 0.01);
  const mesh = preprocessMerged([g], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem(
    { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.3 },
    mesh.indices,
  );
  scene.contact.setFloor(0);
  const solver = new CpuSolver();
  solver.initialize(scene);
  const h = 1 / 60;
  for (let s = 0; s < 30; s++) solver.step(h);
  return Float64Array.from(solver.getPositions());
}

describe("determinism", () => {
  it("repeated CPU runs are bit-identical", () => {
    const a = runOnce();
    const b = runOnce();
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]);
  }, 180000);
});
