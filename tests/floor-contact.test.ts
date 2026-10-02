// Phase 1 ladder: cloth -> floor settling. No penetration, bounded energy, rest.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";

describe("floor contact", () => {
  it("patch falls, settles on floor without penetration", () => {
    const g = buildGrid(6, 6, 0.1, 0.1);
    for (let i = 0; i < g.positions.length / 3; i++) {
      g.positions[i * 3 + 1] += 0.05; // hover 5cm over floor y=0
    }
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
    const contact = new ContactSystem(
      { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.3 },
      mesh.indices,
    );
    contact.setFloor(0);
    scene.contact = contact;
    const solver = new CpuSolver();
    solver.initialize(scene);
    const h = 1 / 60;
    let fwork = 0;
    for (let s = 0; s < 90; s++) {
      solver.step(h);
      if (solver.lastStats?.contact) fwork += solver.lastStats.contact.frictionWork;
      for (const v of solver.getPositions()) expect(Number.isFinite(v)).toBe(true);
    }
    const pos = solver.getPositions();
    let minY = Infinity, meanV = 0;
    for (let i = 0; i < mesh.count; i++) {
      minY = Math.min(minY, pos[i * 3 + 1]);
      meanV += Math.hypot(
        scene.velocities[i * 3], scene.velocities[i * 3 + 1], scene.velocities[i * 3 + 2],
      );
    }
    meanV /= mesh.count;
    // never below hard core (dMin = 1e-4); rests on the barrier ~dHat above floor
    expect(minY).toBeGreaterThan(-5e-4);
    expect(minY).toBeLessThan(0.01);
    expect(meanV).toBeLessThan(0.5); // settled, not jittering
    expect(Number.isFinite(solver.lastStats!.energy)).toBe(true);
    expect(solver.lastStats!.contact!.activePairs).toBeGreaterThan(0);
    expect(fwork).toBeLessThanOrEqual(1e-9); // friction never creates energy
  }, 180000);
});
