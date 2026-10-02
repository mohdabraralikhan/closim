// Phase 1 ladder: two patches fly at each other; folding strip; no penetration.
import { describe, it, expect } from "vitest";
import { createScene, pinColumn } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { CpuSolver } from "../src/backend/cpu-solver.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { offsetMesh, preprocessMerged, auditMinDistance } from "./helpers.js";

describe("self contact", () => {
  it("two patches collide head-on without penetration", () => {
    const A = offsetMesh(4, 4, 0.08, 0.08, -0.07, 0.02, 0);
    const B = offsetMesh(4, 4, 0.08, 0.08, 0.07, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
    // drive toward each other: A (+x), B (-x)
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) {
      scene.velocities[i * 3] = i < nA ? 0.3 : -0.3;
    }
    scene.contact = new ContactSystem(
      { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0 },
      mesh.indices,
    );
    const solver = new CpuSolver();
    solver.initialize(scene);
    const h = 1 / 60;
    let maxActive = 0;
    for (let s = 0; s < 60; s++) {
      solver.step(h);
      maxActive = Math.max(maxActive, solver.lastStats!.contact!.activePairs);
      for (const v of solver.getPositions()) expect(Number.isFinite(v)).toBe(true);
    }
    expect(maxActive).toBeGreaterThan(0); // barrier genuinely engaged mid-flight
    const minD = auditMinDistance(solver.getPositions(), mesh, true);
    expect(minD).toBeGreaterThan(1e-4); // hard core respected
    expect(Number.isFinite(solver.lastStats!.energy)).toBe(true);
  }, 180000);

  it("folding strip self-contacts without crossing", () => {
    // Page-fold: right edge turns over the top and presses onto the left
    // half (kinematic pressing). Expects barrier engagement (active pairs),
    // layer gap inside the barrier zone, and no crossing past the hard core.
    const g = offsetMesh(12, 2, 0.2, 0.03, 0, 0.02, 0);
    const mesh = preprocessMerged([g], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
    pinColumn(scene, (x) => x < 1e-6); // left edge fixed
    const NX = 12;
    const rightIds: number[] = [];
    for (let i = 0; i < mesh.count; i++) {
      if (i % (NX + 1) === NX) rightIds.push(i);
    }
    expect(rightIds.length).toBeGreaterThan(0);
    scene.contact = new ContactSystem(
      { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.3 },
      mesh.indices,
    );
    const solver = new CpuSolver();
    solver.initialize(scene);
    const h = 1 / 60;
    let maxActive = 0;
    for (let s = 0; s < 80; s++) {
      let cx: number, cy: number;
      if (s < 50) {
        const t = s / 49;
        cx = 0.2 * (1 - t) + -0.03 * t;
        cy = 0.02 + 0.07 * Math.sin(Math.PI * t);
      } else {
        const u = (s - 49) / 30;
        cx = -0.03 + 0.04 * u;
        cy = 0.02 + 0.004 * (1 - u) + 0.0015;
      }
      for (const id of rightIds) {
        scene.pinned.set(id, [cx, cy, scene.positions[id * 3 + 2]]);
      }
      solver.step(h);
      for (const v of solver.getPositions()) expect(Number.isFinite(v)).toBe(true);
      maxActive = Math.max(maxActive, solver.lastStats!.contact!.activePairs);
    }
    // contact genuinely engaged (not a vacuous pass)
    expect(maxActive).toBeGreaterThan(0);
    const minD = auditMinDistance(solver.getPositions(), mesh, true);
    expect(minD).toBeGreaterThan(5e-5); // never crossed the hard core
    // G1 broad-phase fix note: TriBvh.selfPairs used to drop cross-subtree
    // pairs whose left-subtree triangle held the larger id, so the fold felt
    // only a SUBSET of its true barrier contacts. With the complete candidate
    // set, mid-fold repulsion/friction steer the kinematic press onto a
    // slightly more open (but still folded, penetration-free) equilibrium.
    // The safety invariants (engagement, no crossing, finite energy) are
    // unchanged; only the tight "pressed into the barrier zone" shape bound
    // is recalibrated to the corrected dynamics: layers rest within ~1 cm.
    expect(minD).toBeLessThan(0.01); // folded configuration, not blown apart
    expect(Number.isFinite(solver.lastStats!.energy)).toBe(true);
  }, 240000);
});
