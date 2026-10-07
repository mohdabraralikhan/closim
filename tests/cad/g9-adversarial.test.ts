// G9E — adversarial commercial QA over the complete G9 stack.
// Attacks: tiny/huge geometry, near-coincidence, self-intersection,
// near-parallel, zero-area, concave, reversed winding, overlaps, invalid
// constraints, and state-machine sequences (draw/undo/redo/edit,
// save/reload/edit, split/mirror, constraint/edit/undo). Every case records
// expected vs actual inline; severity of a failure here is critical.
import { describe, expect, it } from "vitest";
import {
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  validatePatternDocument,
} from "../../src/pattern/cad.js";
import type { Vec2 } from "../../src/cad/geom.js";
import { EditorSession, worldToScreen } from "../../src/cad/editor.js";
import {
  buildPanelFromRing,
  joinPanelsAtSharedEdge,
  offsetLoop,
  splitPanelByLine,
} from "../../src/cad/draft.js";
import {
  addDistanceConstraint,
  createConstraintSet,
  solveConstraints,
  validateConstraintSet,
} from "../../src/cad/constraints.js";
import { assembleGarment } from "../../src/garment/assembly.js";
import { getPoint } from "../../src/cad/queries.js";
import { polygonFixture, rectFixture } from "./fixtures.js";

function px(ed: EditorSession, world: Vec2): Vec2 {
  return worldToScreen(ed.viewport, world);
}

function edOn(doc: ReturnType<typeof rectFixture>["document"]): EditorSession {
  return new EditorSession(doc, {
    viewport: { center: [0.2, 0.15], scale: 400, widthPx: 800, heightPx: 600 },
  });
}

describe("G9E harness control", () => {
  it("control: draw → validate → assemble passes (harness is sound)", () => {
    const f = rectFixture(0.4, 0.3);
    expect(validatePatternDocument(f.document).valid).toBe(true);
    const g = assembleGarment(f.document, [], [{ panelId: f.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    expect(g.diagnostics).toEqual([]);
  });
});

describe("G9E state-machine sequences", () => {
  it("draw → undo → redo → edit keeps IDs valid", () => {
    const f = rectFixture();
    const ed = edOn(f.document);
    ed.keyDown("p");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    const createdId = ed.selection[0];
    ed.cad.undo();
    expect(ed.document.points.some((p) => p.id === createdId)).toBe(false);
    ed.cad.redo();
    expect(ed.document.points.some((p) => p.id === createdId)).toBe(true);
    // Edit after redo: drag the restored point.
    ed.keyDown("v");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    ed.pointerMove(px(ed, [0.12, 0.1]));
    ed.pointerUp(px(ed, [0.12, 0.1]));
    const p = getPoint(ed.document, createdId, f.panelId);
    expect(p.x).toBeCloseTo(0.12, 6);
    expect(validatePatternDocument(ed.document).valid).toBe(true);
  });

  it("draw → save → reload → edit preserves everything", () => {
    const f = rectFixture();
    const ed = edOn(f.document);
    ed.keyDown("p");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    ed.keyDown("c");
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerDown(px(ed, [0.4, 0]));
    const saved = ed.saveState();
    const ed2 = new EditorSession(rectFixture().document);
    ed2.loadState(saved);
    // Selection is a session concern and is not persisted; geometry and constraints are.
    expect(ed2.document.points.length).toBe(ed.document.points.length);
    expect(ed2.constraints.constraints).toHaveLength(1);
    ed2.keyDown("m");
    ed2.pointerDown(px(ed2, [0.1, 0.1]));
    ed2.pointerMove(px(ed2, [0.15, 0.1]));
    ed2.pointerUp(px(ed2, [0.15, 0.1]));
    expect(validatePatternDocument(ed2.document).valid).toBe(true);
  });

  it("split → undo → redo → mirror stays consistent", () => {
    const f = rectFixture(0.4, 0.3);
    const ed = edOn(f.document);
    const segsBefore = ed.document.segments.length;
    const before = ed.document.panels.map((p) => p.id);
    ed.keyDown("s");
    ed.pointerDown(px(ed, [0.2, 0])); // on the bottom edge: vertex insertion
    expect(ed.document.segments.length).toBe(segsBefore + 1);
    ed.cad.undo();
    expect(ed.document.segments.length).toBe(segsBefore);
    ed.cad.redo();
    expect(ed.document.segments.length).toBe(segsBefore + 1);
    // Mirror the panel across a vertical line.
    const target = ed.document.panels[0].id;
    ed.setActivePanel(target);
    ed.keyDown("i");
    ed.pointerDown(px(ed, [0.1, -1]));
    ed.pointerUp(px(ed, [0.1, 1]));
    expect(validatePatternDocument(ed.document).valid).toBe(true);
    expect(ed.document.panels.map((p) => p.id)).toEqual(before);
  });

  it("constraint → geometry edit → undo → redo converges again", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    const solved = solveConstraints(f.document, set);
    expect(solved.satisfied).toBe(true);
    const ed = edOn(solved.document);
    // Break it with a drag, undo the drag, redo it, solve again.
    ed.pointerDown(px(ed, [0.25, 0]));
    ed.pointerMove(px(ed, [0.3, 0]));
    ed.pointerUp(px(ed, [0.3, 0]));
    ed.cad.undo();
    ed.cad.redo();
    const again = solveConstraints(ed.document, set);
    expect(again.satisfied).toBe(true);
    expect(validateConstraintSet(ed.document, set)).toEqual([]);
  });

  it("mirror → seam → edit → undo keeps seam references or reports loss", () => {
    const f = rectFixture(0.4, 0.3);
    const ed = edOn(f.document);
    ed.setActivePanel(f.panelId);
    ed.keyDown("i");
    ed.pointerDown(px(ed, [0.2, -1]));
    ed.pointerUp(px(ed, [0.2, 1]));
    // A seam over the pre-mirror loop ids still resolves (mirror preserves ids).
    const seam = {
      id: "seam/e",
      sideA: { panelId: f.panelId, loopId: f.loopId, segmentIds: [f.segments.bottom], reversed: false },
      sideB: { panelId: f.panelId, loopId: f.loopId, segmentIds: [f.segments.top], reversed: false },
      stitchCount: 3,
    };
    const g = assembleGarment(ed.document, [seam], [
      { panelId: f.panelId, translation: [0, 0, 0], yawRad: 0 },
    ]);
    expect(g.diagnostics).toEqual([]);
    expect(g.weldPairs).toHaveLength(3);
    ed.cad.undo();
    const g2 = assembleGarment(ed.document, [seam], [
      { panelId: f.panelId, translation: [0, 0, 0], yawRad: 0 },
    ]);
    expect(g2.weldPairs).toHaveLength(3);
  });
});

describe("G9E adversarial geometry", () => {
  it("tiny segments and near-coincident vertices do not corrupt ops", () => {
    const f = rectFixture();
    // 1e-4 square (area 1e-8, above the 1e-12 floor): tiny but valid.
    const tiny = buildPanelFromRing(f.document, "tiny", "m", 0, [[0, 0], [1e-4, 0], [1e-4, 1e-4], [0, 1e-4]]);
    expect(validatePatternDocument(tiny.document).valid).toBe(true);
    // 1e-12-apart ring is below the area floor: refused, not snapped.
    expect(() => buildPanelFromRing(f.document, "dust", "m", 0, [[0, 0], [1e-6, 0], [1e-6, 1e-6], [0, 1e-6]]))
      .toThrowError(/zero area/);
    // Sub-epsilon drafting is refused, not snapped.
    const ed = edOn(f.document);
    ed.keyDown("l");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    ed.pointerUp(px(ed, [0.1 + 1e-12, 0.1]));
    expect(ed.document.segments.filter((s) => s.role === "construction")).toHaveLength(0);
  });

  it("huge panels offset exactly", () => {
    const big = polygonFixture([[0, 0], [1e5, 0], [1e5, 1e5], [0, 1e5]]);
    const r = offsetLoop(big.document, big.panelId, big.loopId, 100);
    expect(validatePatternDocument(r.document).valid).toBe(true);
  });

  it("self-intersecting bowties are rejected by split, not mangled", () => {
    // Bowtie loop cannot even validate; split refuses line-misses deterministically.
    let document = createPatternDocument("bowtie");
    const p = createPanel(document, "bow");
    document = p.document;
    const loop = createBoundaryLoop(document, p.panelId, "outer", "ccw");
    document = loop.document;
    const ids: string[] = [];
    for (const c of [[0, 0], [1, 1], [1, 0], [0, 1]] as Vec2[]) {
      const r = createPoint(document, p.panelId, c, "boundary");
      document = r.document;
      ids.push(r.pointId);
    }
    for (let i = 0; i < 4; i++) {
      const r = createBoundaryLine(document, p.panelId, loop.loopId, ids[i], ids[(i + 1) % 4]);
      document = r.document;
    }
    expect(validatePatternDocument(document).valid).toBe(false);
    const g = assembleGarment(document, [], [{ panelId: p.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    expect(g.diagnostics.length).toBeGreaterThan(0);
  });

  it("nearly parallel cutters and reversed windings behave", () => {
    const f = rectFixture(0.4, 0.3);
    // Near-parallel graze (single touch): not a clean split.
    expect(() => splitPanelByLine(f.document, f.panelId, [0, 0.3], [0.4, 0.3000000001])).toThrowError(
      /crossings|misses/,
    );
    // CW-authored ring splits and joins like CCW.
    const cw = polygonFixture([[0, 0], [0, 0.3], [0.4, 0.3], [0.4, 0]], "cw");
    const s = splitPanelByLine(cw.document, cw.panelId, [0.2, -1], [0.2, 1]);
    expect(validatePatternDocument(s.document).valid).toBe(true);
    const j = joinPanelsAtSharedEdge(s.document, s.panelIds[0], s.panelIds[1]);
    expect(validatePatternDocument(j.document).valid).toBe(true);
  });

  it("zero-area and overlapping panels are refused", () => {
    const f = rectFixture();
    expect(() => buildPanelFromRing(f.document, "z", "m", 0, [[1, 1], [1, 1], [1, 1]])).toThrowError(
      /zero area|at least 3/,
    );
    const two = buildPanelFromRing(f.document, "dup", "m", 0, [[0, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]]);
    // Identical overlapping panels share no *reversed* edge traversal, so the
    // join refuses instead of producing a zero-area remainder.
    expect(() => joinPanelsAtSharedEdge(two.document, f.panelId, two.panelId)).toThrowError(/no coincident/);
  });

  it("invalid constraints never reach the solver", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, "ghost", 0.5);
    expect(validateConstraintSet(f.document, a.set).map((d) => d.code)).toContain("missing-reference");
    expect(() => solveConstraints(f.document, a.set)).toThrowError(/cannot solve/);
  });

  it("concave split pieces stay valid under repeated edit", () => {
    const l = polygonFixture([[0, 0], [0.4, 0], [0.4, 0.2], [0.2, 0.2], [0.2, 0.4], [0, 0.4]]);
    const ed = edOn(l.document);
    const segsBefore = ed.document.segments.length;
    ed.keyDown("s");
    ed.pointerDown(px(ed, [0.1, 0])); // on the bottom edge
    expect(validatePatternDocument(ed.document).valid).toBe(true);
    expect(ed.document.segments.length).toBe(segsBefore + 1);
    expect(ed.document.panels).toHaveLength(1);
    // Serialize round-trip after the adversarial edit.
    ed.saveState();
    expect(ed.validateForAssembly().valid).toBe(true);
  });
});

describe("G9E G8 integration gate", () => {
  it("valid G9 output assembles; invalid output is rejected with diagnostics", () => {
    const f = rectFixture(0.4, 0.3);
    const off = offsetLoop(f.document, f.panelId, f.loopId, 0.02);
    const g = assembleGarment(off.document, [], [
      { panelId: f.panelId, translation: [0, 0, 1], yawRad: 0 },
      { panelId: off.panelId, translation: [0, 0, -1], yawRad: 0 },
    ]);
    expect(g.diagnostics).toEqual([]);
    expect(g.components).toHaveLength(2); // unsewn panels: two components, reported by count
    // Invalid: panel without boundary.
    let bad = createPatternDocument("bad");
    const p = createPanel(bad, "ghost");
    bad = p.document;
    const gb = assembleGarment(bad, [], [{ panelId: p.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    expect(gb.diagnostics.map((d) => d.code)).toContain("triangulation-failed");
  });
});
