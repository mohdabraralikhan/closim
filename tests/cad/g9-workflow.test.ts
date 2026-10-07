// G9 FINAL INTEGRATION — commercial acceptance test.
//
// A simple top is created entirely inside the editor workspace (panels,
// drafting, mirror, measure, constraints), validated, saved, reloaded, sent
// to the G8 pipeline, assembled, simulated, edited in 2D, reassembled, and
// simulated again. The 2D pattern stays authoritative throughout.
import { describe, expect, it } from "vitest";
import {
  createPatternDocument,
  movePoint,
  validatePatternDocument,
} from "../../src/pattern/cad.js";
import type { Vec2 } from "../../src/cad/geom.js";
import { EditorSession } from "../../src/cad/editor.js";
import { buildPanelFromRing, divideSegment, perpendicularAt } from "../../src/cad/draft.js";
import { mirrorPanelCopy } from "../../src/cad/ops.js";
import {
  addDistanceConstraint,
  createConstraintSet,
  measurePanelArea,
  measurePanelPerimeter,
  measurePointDistance,
  solveConstraints,
} from "../../src/cad/constraints.js";
import { getPoint } from "../../src/cad/queries.js";import {
  applyPatternEdit,
  createGarmentProject,
  rebuildGarment,
  serializeGarmentProject,
  validateGarmentProject,
} from "../../src/garment/project.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import { runFitting } from "../../src/garment/assembly.js";
import type { Seam } from "../../src/garment/sewing.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";

describe("G9 final integration — in-app T-shirt acceptance", () => {
  it("creates, drafts, constrains, saves, assembles, simulates, edits, re-simulates", () => {
    // 1. create project.
    const pattern0 = createPatternDocument("g9-top", "In-app top");
    const ed = new EditorSession(pattern0, {
      viewport: { center: [0.5, 0.3], scale: 400, widthPx: 800, heightPx: 600 },
    });

    // 2/3. front + back panels drawn as rings through the session (undoable).
    const frontRing: Vec2[] = [[0, 0], [0.46, 0], [0.46, 0.62], [0, 0.62]];
    const backRing: Vec2[] = frontRing.map(([x, y]) => [x + 0.6, y] as Vec2);
    const front = ed.cad.run("draw front", (d) => buildPanelFromRing(d, "front", "cotton", 0, frontRing));
    const back = ed.cad.run("draw back", (d) => buildPanelFromRing(d, "back", "cotton", 0, backRing));
    expect(ed.document.panels).toHaveLength(2);

    // 4. sleeve panel + mirrored copy (drafting-standard left/right pair).
    const sleeve = ed.cad.run("draw sleeve", (d) =>
      buildPanelFromRing(d, "sleeve-left", "cotton", 0, [[0.5, 0], [0.8, 0], [0.8, 0.25], [0.5, 0.25]]),
    );
    const mirrored = ed.cad.run("mirror sleeve", (d) => mirrorPanelCopy(d, sleeve.panelId, [0, -1], [0, 1]));
    expect(mirrored.panelId).not.toBe(sleeve.panelId);
    const sleeveRId = ed.document.panels.find((p) => p.name === "sleeve-left mirror")!.id;
    // Move the mirrored sleeve (sitting at negative x) next to the body: +1.6 m.
    for (const pt of ed.document.points.filter((p) => p.panelId === sleeveRId)) {
      ed.cad.run("arrange", (d) => movePoint(d, sleeveRId, pt.id, [pt.x + 1.6, pt.y]));
    }
    expect(ed.document.panels).toHaveLength(4);

    // 5. edit dimensions: lengthen the front by 3 cm at the top edge.
    const frontTop = ed.document.points.filter((p) => p.panelId === front.panelId && p.y > 0.3);
    expect(frontTop).toHaveLength(2);
    for (const pt of frontTop) {
      ed.cad.run("lengthen", (d) => movePoint(d, front.panelId, pt.id, [pt.x, pt.y + 0.03]));
    }

    // 6. drafting operations: divide + perpendicular construction.
    const frontLoop = ed.document.panels.find((p) => p.id === front.panelId)!.boundaryLoops[0];
    const bottomSeg = frontLoop.segmentIds[0];
    const div = ed.cad.run("divide", (d) => divideSegment(d, front.panelId, bottomSeg, 4));
    expect(div.pointIds).toHaveLength(3);
    const perp = ed.cad.run("perp", (d) => perpendicularAt(d, front.panelId, bottomSeg, 0.5, 0.1));
    expect(ed.document.segments.some((s) => s.id === perp.segmentId)).toBe(true);

    // 7. mirror already exercised in step 4 (mirror copy exists and validates).

    // 8. measure geometry.
    expect(measurePanelArea(ed.document, front.panelId)).toBeCloseTo(0.46 * 0.65, 9);
    expect(measurePanelPerimeter(ed.document, front.panelId)).toBeCloseTo(2 * (0.46 + 0.65), 9);
    const blId = ed.document.points.find((p) => p.panelId === front.panelId && p.x === 0 && p.y === 0)!.id;
    const brId = ed.document.points.find((p) => p.panelId === front.panelId && p.y === 0 && p.x > 0.4)!.id;
    expect(measurePointDistance(ed.document, front.panelId, blId, brId)).toBeCloseTo(0.46, 12);

    // 9. basic constraint: bottom edge is 460 mm; solve (already true → stable).
    let set = createConstraintSet();
    const dc = addDistanceConstraint(set, front.panelId, blId, brId, 0.46);
    set = dc.set;
    const solved = solveConstraints(ed.document, set);
    expect(solved.satisfied).toBe(true);
    ed.cad.run("apply constraint", () => solved.document);

    // 10. validate pattern.
    expect(validatePatternDocument(ed.document).valid).toBe(true);
    expect(ed.validateForAssembly().valid).toBe(true);

    // 11/12. save + reload (pattern, ids, constraints preserved).
    ed.constraints = set;
    const saved = ed.saveState();
    const ed2 = new EditorSession(createPatternDocument("blank"));
    ed2.loadState(saved);
    expect(ed2.document.panels.map((p) => p.id)).toEqual(ed.document.panels.map((p) => p.id));
    expect(ed2.constraints.constraints).toHaveLength(1);

    // 13. send to G8: seams over stable loop/segment ids.
    const loopOf = (pid: string): string =>
      ed2.document.panels.find((p) => p.id === pid)!.boundaryLoops[0].id;
    const segOf = (pid: string, k: number): string =>
      ed2.document.panels.find((p) => p.id === pid)!.boundaryLoops[0].segmentIds[k];
    const backId = back.panelId;
    const seams: Seam[] = [
      { id: "seam/shoulder", sideA: { panelId: front.panelId, loopId: loopOf(front.panelId), segmentIds: [segOf(front.panelId, 2)], reversed: false }, sideB: { panelId: backId, loopId: loopOf(backId), segmentIds: [segOf(backId, 2)], reversed: true }, stitchCount: 7 },
      { id: "seam/left", sideA: { panelId: front.panelId, loopId: loopOf(front.panelId), segmentIds: [segOf(front.panelId, 3)], reversed: false }, sideB: { panelId: backId, loopId: loopOf(backId), segmentIds: [segOf(backId, 3)], reversed: true }, stitchCount: 7 },
      { id: "seam/right", sideA: { panelId: front.panelId, loopId: loopOf(front.panelId), segmentIds: [segOf(front.panelId, 1)], reversed: false }, sideB: { panelId: backId, loopId: loopOf(backId), segmentIds: [segOf(backId, 1)], reversed: true }, stitchCount: 7 },
      { id: "seam/sleeve-l", sideA: { panelId: front.panelId, loopId: loopOf(front.panelId), segmentIds: [segOf(front.panelId, 3)], reversed: false }, sideB: { panelId: sleeve.panelId, loopId: loopOf(sleeve.panelId), segmentIds: [segOf(sleeve.panelId, 2)], reversed: false }, stitchCount: 5 },
      { id: "seam/sleeve-r", sideA: { panelId: front.panelId, loopId: loopOf(front.panelId), segmentIds: [segOf(front.panelId, 1)], reversed: false }, sideB: { panelId: sleeveRId, loopId: loopOf(sleeveRId), segmentIds: [segOf(sleeveRId, 2)], reversed: true }, stitchCount: 5 },
    ];
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.23, 0.95, 0] });
    const project = createGarmentProject("garment/g9-top", "In-app top", ed2.document, {
      seams,
      placements: [
        { panelId: front.panelId, translation: [0, 0.6, 0.2], yawRad: 0 },
        { panelId: backId, translation: [1.06, 0.6, -0.2], yawRad: Math.PI },
        { panelId: sleeve.panelId, translation: [-0.06, 0.95, 0.65], yawRad: Math.PI / 2 },
        { panelId: sleeveRId, translation: [0.52, 0.95, -0.95], yawRad: -Math.PI / 2 },
      ],
      avatar,
      materials: { cotton: { ...DEFAULT_MATERIAL } },
    });
    expect(validateGarmentProject(project).valid).toBe(true);

    // 14/15. assemble + simulate.
    const first = rebuildGarment(project);
    expect(first.assembled.components).toHaveLength(1);
    expect(first.assembled.weldPairs).toHaveLength(31);
    const solverA = new CpuSolver();
    const fitA = runFitting(first.assembled, first.fitting, solverA, avatar, {
      relaxationSteps: 2, simulationSteps: 3,
    });
    expect(fitA.ok).toBe(true);
    expect(fitA.hasNaNInf).toBe(false);
    const posA = Array.from(solverA.getPositions());

    // 16/17. return to 2D and modify the pattern (lengthen a little more).
    const edited = applyPatternEdit(project, (pattern) => {
      const pts = pattern.points.filter((p) => p.panelId === front.panelId && p.y > 0.3);
      let next = pattern;
      for (const pt of pts) next = movePoint(next, front.panelId, pt.id, [pt.x, pt.y + 0.02]);
      return next;
    });
    expect(edited.metadata.revision).toBe(2);
    expect(serializeGarmentProject(edited)).not.toBe(serializeGarmentProject(project));

    // 18/19. reassemble + simulate again without rebuilding the project.
    const second = rebuildGarment(edited);
    const solverB = new CpuSolver();
    const fitB = runFitting(second.assembled, second.fitting, solverB, avatar, {
      relaxationSteps: 2, simulationSteps: 3,
    });
    expect(fitB.ok).toBe(true);
    expect(Array.from(solverB.getPositions())).not.toEqual(posA);
  });
});
