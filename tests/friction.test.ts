// Phase 1 ladder: lagged friction — mu=0 slides, mu>0 resists; work <= 0;
// Coulomb invariants at the force level.
import { describe, it, expect } from "vitest";
import { createScene } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { coulombForce, checkInvariants } from "../src/collision/friction.js";
import { offsetMesh, preprocessMerged } from "./helpers.js";

function slideRun(mu: number): { disp: number; fwork: number } {
  const g = offsetMesh(4, 4, 0.08, 0.08, 0, 0.0025, 0);
  const mesh = preprocessMerged([g], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  for (let i = 0; i < mesh.count; i++) scene.velocities[i * 3] = 0.5; // tangential slip
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS, frictionMu: mu }, mesh.indices);
  scene.contact.setFloor(0);
  const solver = new CpuSolver();
  solver.initialize(scene);
  const h = 1 / 60;
  let fwork = 0;
  const xStart = Float64Array.from(scene.positions);
  for (let s = 0; s < 60; s++) {
    solver.step(h);
    if (solver.lastStats?.contact) fwork += solver.lastStats.contact.frictionWork;
  }
  let disp = 0;
  for (let i = 0; i < mesh.count; i++) disp += scene.positions[i * 3] - xStart[i * 3];
  disp /= mesh.count;
  return { disp, fwork };
}

describe("friction", () => {
  it("satisfies Coulomb invariants at force level", () => {
    const [fx, fy, fz] = coulombForce(0.01, 0.002, -0.004, 0, 1, 0, 2.5, 0.3, 1e-4);
    const chk = checkInvariants(fx, fy, fz, 0, 1, 0, 2.5, 0.3);
    expect(chk.tangential).toBe(true);
    expect(chk.inCone).toBe(true);
    const [zx, zy, zz] = coulombForce(0, 0, 0, 0, 1, 0, 2.5, 0.3, 1e-4);
    expect(Math.hypot(zx, zy, zz)).toBeLessThan(1e-9);
  });

  it("mu=0.6 resists sliding vs mu=0 and never creates energy", () => {
    const free = slideRun(0);
    const gripped = slideRun(0.6);
    expect(gripped.disp).toBeLessThan(free.disp);
    expect(gripped.fwork).toBeLessThanOrEqual(1e-9);
    expect(free.fwork).toBeLessThanOrEqual(1e-9);
    expect(Number.isFinite(gripped.disp)).toBe(true);
  }, 240000);
});
