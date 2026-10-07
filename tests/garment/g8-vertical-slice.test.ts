/**
 * G8E — commercial vertical-slice QA (adversarial verification).
 *
 * Proves the workflow:
 *   2D panels -> triangulation -> seams -> garment topology ->
 *   3D assembly -> avatar collision -> simulation -> editable result
 *
 * Each case records expected behavior inline. Severity of any failure here
 * is critical (pipeline correctness). Production code is NOT modified by
 * these tests; failures are reported with reproduction via the case name.
 */
import { describe, expect, it } from "vitest";
import {
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  movePoint,
  serializePatternDocument,
  deserializePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import type { Vec2 } from "../../src/pattern/pattern-geometry.js";
import { validateSeams, type Seam } from "../../src/garment/sewing.js";
import {
  assembleGarment,
  createFittingScene,
  runFitting,
  type PanelPlacement,
} from "../../src/garment/assembly.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import { addRectPanel, buildTshirtProject } from "../../src/garment/tshirt.js";
import {
  applyPatternEdit,
  deserializeGarmentProject,
  rebuildGarment,
  serializeGarmentProject,
} from "../../src/garment/project.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";

function rectDoc(id: string, w: number, h: number, origin: Vec2 = [0, 0]) {
  let document = createPatternDocument(id, id);
  const r = addRectPanel(document, "panel", origin, w, h);
  return { document: r.document, refs: r.refs };
}

function stableSeam(id: string, a: { panel: string; loop: string; segs: string[] }, b: { panel: string; loop: string; segs: string[] }, n = 5, revB = false): Seam {
  return {
    id,
    sideA: { panelId: a.panel, loopId: a.loop, segmentIds: a.segs, reversed: false },
    sideB: { panelId: b.panel, loopId: b.loop, segmentIds: b.segs, reversed: revB },
    stitchCount: n,
  };
}

const IDENTITY_PLACE = (panelId: string, x = 0, y = 1, z = 0.3, yaw = 0): PanelPlacement => ({
  panelId, translation: [x, y, z], yawRad: yaw,
});

describe("G8E vertical slice", () => {
  it("Case 1 — single closed panel assembles and simulates", () => {
    const { document, refs } = rectDoc("g8e-case1", 0.4, 0.3);
    const g = assembleGarment(document, [], [IDENTITY_PLACE(refs.panelId)]);
    expect(g.diagnostics).toEqual([]);
    expect(g.indices.length / 3).toBe(2);
    const fitting = createFittingScene(g, {});
    const solver = new CpuSolver();
    const result = runFitting(g, fitting, solver, null, { relaxationSteps: 2, simulationSteps: 2 });
    expect(result.ok).toBe(true);
    expect(result.hasNaNInf).toBe(false);
  });

  it("Case 2 — two rectangular panels joined by one seam", () => {
    let document = createPatternDocument("g8e-case2", "two rects");
    const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3); document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3); document = b.document;
    const seam = stableSeam("seam/ab",
      { panel: a.refs.panelId, loop: a.refs.loopId, segs: [a.refs.segmentIds[1]] },
      { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[3]] }, 6, true);
    expect(validateSeams(document, [seam]).valid).toBe(true);
    const g = assembleGarment(document, [seam], [
      IDENTITY_PLACE(a.refs.panelId, 0, 1, 0.2),
      IDENTITY_PLACE(b.refs.panelId, 0.4, 1, -0.2, Math.PI),
    ]);
    expect(g.diagnostics).toEqual([]);
    expect(g.weldPairs).toHaveLength(6);
    expect(g.components).toHaveLength(1);
    // Orientation preserved: first stitch of reversed side starts at the far end.
    expect(g.weldPairs[0].gapM).toBeLessThan(0.45);
  });

  it("Case 3 — front + back garment with three seams", () => {
    let document = createPatternDocument("g8e-case3", "front-back");
    const f = addRectPanel(document, "front", [0, 0], 0.46, 0.62); document = f.document;
    const b = addRectPanel(document, "back", [0, 0], 0.46, 0.62); document = b.document;
    const seams = [
      stableSeam("seam/shoulder",
        { panel: f.refs.panelId, loop: f.refs.loopId, segs: [f.refs.segmentIds[2]] },
        { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[2]] }, 7, true),
      stableSeam("seam/left",
        { panel: f.refs.panelId, loop: f.refs.loopId, segs: [f.refs.segmentIds[3]] },
        { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[3]] }, 7, true),
      stableSeam("seam/right",
        { panel: f.refs.panelId, loop: f.refs.loopId, segs: [f.refs.segmentIds[1]] },
        { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[1]] }, 7, true),
    ];
    const g = assembleGarment(document, seams, [
      IDENTITY_PLACE(f.refs.panelId, 0, 0.6, 0.2),
      IDENTITY_PLACE(b.refs.panelId, 0.46, 0.6, -0.2, Math.PI),
    ]);
    expect(g.diagnostics).toEqual([]);
    expect(g.weldPairs).toHaveLength(21);
    expect(g.components).toHaveLength(1);
  });

  it("Case 4 — shirt-like topology with multiple seams + avatar + simulation", () => {
    const { project } = buildTshirtProject();
    const { assembled, fitting } = rebuildGarment(project);
    // 7+7+7+5+5 stitch pairs across 5 seams, one connected component.
    expect(assembled.weldPairs).toHaveLength(31);
    expect(assembled.components).toHaveLength(1);
    expect(assembled.penetratingVertexCount).toBe(0);
    const solver = new CpuSolver();
    const result = runFitting(assembled, fitting, solver, project.avatar, {
      relaxationSteps: 2, simulationSteps: 3,
    });
    expect(result.hasNaNInf).toBe(false);
    expect(result.stepsTaken).toBe(5);
    expect(result.ok).toBe(true);
  });

  it("Case 5 — unequal-length seam pair stays deterministic", () => {
    let document = createPatternDocument("g8e-case5", "unequal");
    const a = addRectPanel(document, "A", [0, 0], 0.5, 0.3); document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.1, 0.3); document = b.document;
    const seam = stableSeam("seam/unequal",
      { panel: a.refs.panelId, loop: a.refs.loopId, segs: [a.refs.segmentIds[2]] },
      { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[2]] }, 5);
    const place = [
      IDENTITY_PLACE(a.refs.panelId, 0, 1, 0.2),
      IDENTITY_PLACE(b.refs.panelId, 0, 1, -0.2, Math.PI),
    ];
    const g1 = assembleGarment(document, [seam], place);
    const g2 = assembleGarment(document, [seam], place);
    expect(g1.diagnostics).toEqual([]);
    expect(g1.weldPairs).toEqual(g2.weldPairs);
    expect(g1.weldPairs).toHaveLength(5);
  });

  it("Case 6 — concave panel triangulates and assembles", () => {
    // L-shaped concave hexagon.
    let document = createPatternDocument("g8e-case6", "concave");
    const p = createPanel(document, "L", "cotton"); document = p.document;
    const loop = createBoundaryLoop(document, p.panelId, "outer"); document = loop.document;
    const corners: Vec2[] = [[0, 0], [0.4, 0], [0.4, 0.2], [0.2, 0.2], [0.2, 0.4], [0, 0.4]];
    const pids: string[] = [];
    for (const c of corners) {
      const r = createPoint(document, p.panelId, c); document = r.document; pids.push(r.pointId);
    }
    for (let i = 0; i < 6; i++) {
      const r = createBoundaryLine(document, p.panelId, loop.loopId, pids[i], pids[(i + 1) % 6]);
      document = r.document;
    }
    const g = assembleGarment(document, [], [IDENTITY_PLACE(p.panelId)]);
    expect(g.diagnostics).toEqual([]);
    expect(g.indices.length / 3).toBe(4); // hexagon -> 4 triangles
    expect(g.components).toHaveLength(1);
  });

  it("Case 7 — garment placed partially inside avatar is reported, not hidden", () => {
    const { document, refs } = rectDoc("g8e-case7", 0.4, 0.3);
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 1.0, 0.3] });
    // Shifted -0.1 in x: two corners (radial 0.3) stay outside, two (radial 0.1) go inside.
    const g = assembleGarment(document, [], [IDENTITY_PLACE(refs.panelId, -0.1, 1, 0.3)], { avatar });
    // Panel plane z=0.3 passes through the avatar core -> some verts inside.
    expect(g.penetratingVertexCount).toBeGreaterThan(0);
    expect(g.diagnostics.map((d) => d.code)).toContain("penetrating-placement");
    // Fitting still runs explicitly; caller decides. Must not throw or NaN.
    const fitting = createFittingScene(g, { avatar });
    const solver = new CpuSolver();
    const result = runFitting(g, fitting, solver, avatar, { relaxationSteps: 1, simulationSteps: 1 });
    expect(result.hasNaNInf).toBe(false);
    expect(result.stages.map((s) => s.stage)).toContain("collision-validation");
  });

  it("Case 8 — disconnected garment is detected", () => {
    let document = createPatternDocument("g8e-case8", "disconnected");
    const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3); document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3); document = b.document;
    const c = addRectPanel(document, "C", [0, 0], 0.4, 0.3); document = c.document;
    const seam = stableSeam("seam/ab",
      { panel: a.refs.panelId, loop: a.refs.loopId, segs: [a.refs.segmentIds[1]] },
      { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[3]] }, 3);
    const g = assembleGarment(document, [seam], [
      IDENTITY_PLACE(a.refs.panelId, 0, 1, 0.2),
      IDENTITY_PLACE(b.refs.panelId, 0.4, 1, -0.2, Math.PI),
      IDENTITY_PLACE(c.refs.panelId, 5, 5, 5),
    ]);
    expect(g.components).toHaveLength(2);
    expect(g.diagnostics.map((d) => d.code)).toContain("disconnected-component");
  });
});

describe("G8E adversarial cases", () => {
  it("empty panel (no boundary) yields a structured triangulation failure", () => {
    let document = createPatternDocument("g8e-adv-empty", "empty");
    const p = createPanel(document, "ghost", "cotton"); document = p.document;
    const g = assembleGarment(document, [], [IDENTITY_PLACE(p.panelId)]);
    expect(g.diagnostics.map((d) => d.code)).toContain("triangulation-failed");
    expect(g.panelRanges).toHaveLength(0);
  });

  it("zero-length segment document fails triangulation deterministically", () => {
    let document = createPatternDocument("g8e-adv-zero", "zero");
    const p = createPanel(document, "Z", "cotton"); document = p.document;
    const loop = createBoundaryLoop(document, p.panelId, "outer"); document = loop.document;
    const pids: string[] = [];
    for (const c of [[0, 0], [0, 0], [1, 0], [0, 1]] as Vec2[]) {
      const r = createPoint(document, p.panelId, c); document = r.document; pids.push(r.pointId);
    }
    for (let i = 0; i < 4; i++) {
      const r = createBoundaryLine(document, p.panelId, loop.loopId, pids[i], pids[(i + 1) % 4]);
      document = r.document;
    }
    const g1 = assembleGarment(document, [], [IDENTITY_PLACE(p.panelId)]);
    const g2 = assembleGarment(document, [], [IDENTITY_PLACE(p.panelId)]);
    expect(g1.diagnostics).toEqual(g2.diagnostics);
    expect(g1.diagnostics.length).toBeGreaterThan(0);
  });

  it("reversed seams resolve deterministically and preserve endpoints", () => {
    let document = createPatternDocument("g8e-adv-rev", "rev");
    const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3); document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3); document = b.document;
    const fwd = stableSeam("seam/fwd",
      { panel: a.refs.panelId, loop: a.refs.loopId, segs: [a.refs.segmentIds[0]] },
      { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[0]] }, 4, false);
    const rev = stableSeam("seam/rev",
      { panel: a.refs.panelId, loop: a.refs.loopId, segs: [a.refs.segmentIds[0]] },
      { panel: b.refs.panelId, loop: b.refs.loopId, segs: [b.refs.segmentIds[0]] }, 4, true);
    const place = [IDENTITY_PLACE(a.refs.panelId), IDENTITY_PLACE(b.refs.panelId, 0, 1, -0.3, Math.PI)];
    const gFwd = assembleGarment(document, [fwd], place);
    const gRev = assembleGarment(document, [rev], place);
    // Reversed correspondence swaps the B-side endpoints; A-side is unchanged.
    expect(gFwd.weldPairs.map((w) => w.vertexA)).toEqual(gRev.weldPairs.map((w) => w.vertexA));
    expect(gFwd.weldPairs.map((w) => w.vertexB)).toEqual([...gRev.weldPairs.map((w) => w.vertexB)].reverse());
    expect(assembleGarment(document, [rev], place).weldPairs).toEqual(gRev.weldPairs);
  });

  it("duplicate seams and invalid IDs are rejected with structured diagnostics", () => {
    const { document, refs } = rectDoc("g8e-adv-dup", 0.4, 0.3);
    const mk = (id: string): Seam => stableSeam(id,
      { panel: refs.panelId, loop: refs.loopId, segs: [refs.segmentIds[0]] },
      { panel: refs.panelId, loop: refs.loopId, segs: [refs.segmentIds[1]] }, 3);
    const dup = validateSeams(document, [mk("s1"), mk("s2")]);
    // Same two boundary paths under different ids: duplicates — invalid.
    expect(dup.valid).toBe(false);
    expect(dup.diagnostics.map((d) => d.code)).toContain("duplicate-seam");
    const distinct: Seam[] = [mk("s1"), stableSeam("s3",
      { panel: refs.panelId, loop: refs.loopId, segs: [refs.segmentIds[0]] },
      { panel: refs.panelId, loop: refs.loopId, segs: [refs.segmentIds[2]] }, 3)];
    expect(validateSeams(document, distinct).valid).toBe(true);
    const exact: Seam[] = [mk("s1"), { ...mk("s1"), id: "s1-clone" }];
    expect(validateSeams(document, exact).diagnostics.map((d) => d.code)).toContain("duplicate-seam");
    const badRef: Seam = stableSeam("bad",
      { panel: refs.panelId, loop: refs.loopId, segs: ["no-such-segment"] },
      { panel: refs.panelId, loop: refs.loopId, segs: [refs.segmentIds[1]] }, 3);
    const g = assembleGarment(document, [badRef], [IDENTITY_PLACE(refs.panelId)]);
    expect(g.diagnostics.map((d) => d.code)).toContain("failed-seam-reference");
  });

  it("extremely thin panels and extremely unequal seams resolve", () => {
    let document = createPatternDocument("g8e-adv-thin", "thin");
    const thin = addRectPanel(document, "thin", [0, 0], 0.5, 0.005); document = thin.document;
    const tiny = addRectPanel(document, "tiny", [0, 0], 0.05, 0.3); document = tiny.document;
    const seam = stableSeam("seam/thin",
      { panel: thin.refs.panelId, loop: thin.refs.loopId, segs: [thin.refs.segmentIds[2]] },
      { panel: tiny.refs.panelId, loop: tiny.refs.loopId, segs: [tiny.refs.segmentIds[2]] }, 5);
    const g = assembleGarment(document, [seam], [
      IDENTITY_PLACE(thin.refs.panelId),
      IDENTITY_PLACE(tiny.refs.panelId, 0, 1, -0.3, Math.PI),
    ]);
    expect(g.weldPairs).toHaveLength(5);
    for (const w of g.weldPairs) expect(Number.isFinite(w.gapM)).toBe(true);
  });

  it("near-coincident points do not corrupt assembly", () => {
    let document = createPatternDocument("g8e-adv-coinc", "coincident");
    const p = createPanel(document, "N", "cotton"); document = p.document;
    const loop = createBoundaryLoop(document, p.panelId, "outer"); document = loop.document;
    const corners: Vec2[] = [[0, 0], [0.4, 0], [0.4 + 1e-7, 0.3], [0, 0.3]];
    const pids: string[] = [];
    for (const c of corners) {
      const r = createPoint(document, p.panelId, c); document = r.document; pids.push(r.pointId);
    }
    for (let i = 0; i < 4; i++) {
      const r = createBoundaryLine(document, p.panelId, loop.loopId, pids[i], pids[(i + 1) % 4]);
      document = r.document;
    }
    const g = assembleGarment(document, [], [IDENTITY_PLACE(p.panelId)]);
    expect(g.indices.length).toBeGreaterThan(0);
    expect(g.diagnostics.filter((d) => d.code === "nan-inf-state")).toEqual([]);
  });

  it("repeated serialization/deserialization is byte-stable", () => {
    const { project } = buildTshirtProject();
    const s0 = serializeGarmentProject(project);
    const s1 = serializeGarmentProject(deserializeGarmentProject(s0));
    const s2 = serializeGarmentProject(deserializeGarmentProject(s1));
    expect(s1).toBe(s0);
    expect(s2).toBe(s0);
    const pat0 = serializePatternDocument(project.pattern);
    expect(serializePatternDocument(deserializePatternDocument(pat0))).toBe(pat0);
  });

  it("repeated assembly and repeated simulation restart are stable", () => {
    const { project } = buildTshirtProject();
    const first = rebuildGarment(project);
    const second = rebuildGarment(project);
    expect(Array.from(first.assembled.positions)).toEqual(Array.from(second.assembled.positions));
    expect(first.assembled.weldPairs).toEqual(second.assembled.weldPairs);
    const run = (): number[] => {
      const { assembled, fitting } = rebuildGarment(project);
      const solver = new CpuSolver();
      runFitting(assembled, fitting, solver, project.avatar, { relaxationSteps: 2, simulationSteps: 2 });
      return Array.from(solver.getPositions());
    };
    expect(run()).toEqual(run());
  });

  it("pattern edit preserves seam references and panel ordering", () => {
    const { project } = buildTshirtProject();
    const beforeIds = project.pattern.panels.map((p) => p.id);
    const frontPanelId = beforeIds[0];
    const target = project.pattern.points.find((p) => p.panelId === frontPanelId)!;
    const edited = applyPatternEdit(project, (pattern: PatternDocument) =>
      movePoint(pattern, frontPanelId, target.id, [target.x + 0.01, target.y]),
    );
    expect(edited.pattern.panels.map((p) => p.id)).toEqual(beforeIds);
    // All seam panel/loop references still resolve after the edit.
    const { assembled } = rebuildGarment(edited);
    expect(assembled.diagnostics.filter((d) => d.code === "failed-seam-reference")).toEqual([]);
    expect(assembled.weldPairs).toHaveLength(31);
  });
});
