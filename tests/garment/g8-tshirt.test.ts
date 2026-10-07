/**
 * G8 final integration — the first genuine commercial garment workflow:
 *
 *   Create/Open -> Edit Panels -> Define Seams -> Validate -> Triangulate ->
 *   Assemble in 3D -> Place on Avatar -> Simulate -> Inspect Fit ->
 *   Modify Pattern -> Rebuild -> Simulate Again
 *
 * The key requirement is the final loop (2D EDIT -> 3D RESULT -> 2D EDIT ->
 * 3D RESULT) without manually rebuilding the project.
 */
import { describe, expect, it } from "vitest";
import { movePoint } from "../../src/pattern/cad.js";
import { triangulateCadPanel } from "../../src/pattern/cad.js";
import { validateSeams } from "../../src/garment/sewing.js";
import { runFitting } from "../../src/garment/assembly.js";
import {
  applyPatternEdit,
  rebuildGarment,
  serializeGarmentProject,
  validateGarmentProject,
} from "../../src/garment/project.js";
import { buildTshirtProject } from "../../src/garment/tshirt.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";

describe("G8 final integration — T-shirt workflow", () => {
  it("runs the complete Create -> Simulate -> Edit -> Simulate-Again loop", () => {
    // Create/Open garment.
    const { project: opened } = buildTshirtProject();
    expect(opened.metadata.name).toBe("Basic T-shirt");

    // Panels exist and triangulate independently of the simulation mesh.
    for (const panel of opened.pattern.panels) {
      const tri = triangulateCadPanel(opened.pattern, panel.id);
      expect(tri.triangles.length).toBeGreaterThan(0);
      expect(tri.panelId).toBe(panel.id);
    }
    expect(opened.pattern.panels.map((p) => p.name)).toEqual(
      ["front", "back", "sleeve-left", "sleeve-right"],
    );

    // Seams validate.
    expect(validateGarmentProject(opened).valid).toBe(true);
    expect(validateSeams(opened.pattern, opened.seams).valid).toBe(true);

    // Assemble in 3D + place on avatar + simulate.
    const first = rebuildGarment(opened);
    expect(first.assembled.components).toHaveLength(1);
    // 3+3 shoulders, 9+9 sides, 5+5 sleeve caps across 6 seams.
    expect(first.assembled.weldPairs).toHaveLength(34);
    const solverA = new CpuSolver();
    const fitA = runFitting(first.assembled, first.fitting, solverA, opened.avatar, {
      dt: opened.simulation.dt,
      relaxationSteps: opened.simulation.relaxationSteps,
      simulationSteps: opened.simulation.simulationSteps,
    });
    expect(fitA.ok).toBe(true);
    expect(fitA.hasNaNInf).toBe(false);
    const positionsA = Array.from(solverA.getPositions());

    // Inspect fit: the project starts pre-draped (isometric wrap), so the
    // garment must stay settled in resting contact instead of drifting.
    expect(fitA.finalMaxDisplacementM).toBeLessThan(0.05);
    expect(fitA.finalMinAvatarDistanceM).not.toBeNull();
    expect(fitA.finalMinAvatarDistanceM as number).toBeLessThan(0.02);

    // Modify pattern in 2D: lengthen the front panel by 3 cm.
    const frontPanelId = opened.pattern.panels[0].id;
    const topEdge = opened.pattern.points.filter((p) => p.panelId === frontPanelId);
    const edited = applyPatternEdit(opened, (pattern) => {
      let next = pattern;
      for (const pt of topEdge) {
        if (pt.y > 0.3) next = movePoint(next, frontPanelId, pt.id, [pt.x, pt.y + 0.03]);
      }
      return next;
    });
    expect(edited.metadata.revision).toBe(2);

    // Rebuild + simulate again WITHOUT reconstructing the project.
    expect(validateGarmentProject(edited).valid).toBe(true);
    const second = rebuildGarment(edited);
    const solverB = new CpuSolver();
    const fitB = runFitting(second.assembled, second.fitting, solverB, edited.avatar, {
      dt: edited.simulation.dt,
      relaxationSteps: edited.simulation.relaxationSteps,
      simulationSteps: edited.simulation.simulationSteps,
    });
    expect(fitB.ok).toBe(true);
    expect(fitB.hasNaNInf).toBe(false);

    // The 2D edit produced a different 3D result.
    const positionsB = Array.from(solverB.getPositions());
    expect(positionsB).not.toEqual(positionsA);

    // The edited project still serializes deterministically.
    expect(serializeGarmentProject(edited)).toBe(serializeGarmentProject(edited));
  });
});
