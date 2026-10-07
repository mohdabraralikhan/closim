// G9D editor tests: viewport math, tools, selection, drag gestures,
// shortcuts, numeric input, cancel semantics, save/load, assembly gate.
import { describe, expect, it } from "vitest";
import {
  createViewport,
  EditorSession,
  fitToView,
  panByPixels,
  screenToWorld,
  worldToScreen,
  zoomAt,
} from "../../src/cad/editor.js";
import type { Vec2 } from "../../src/cad/geom.js";
import { getPoint } from "../../src/cad/queries.js";
import { validatePatternDocument } from "../../src/pattern/cad.js";
import { rectFixture } from "./fixtures.js";

function editor() {
  const f = rectFixture(0.4, 0.3);
  const ed = new EditorSession(f.document, {
    viewport: { center: [0.2, 0.15], scale: 400, widthPx: 800, heightPx: 600 },
  });
  return { ed, panelId: f.panelId, bl: f.points.bl, bottom: f.segments.bottom };
}

function px(ed: EditorSession, world: Vec2): Vec2 {
  return worldToScreen(ed.viewport, world);
}

describe("G9D viewport", () => {
  it("converts world<->screen and round-trips", () => {
    const v = createViewport(800, 600, 400);
    const s = worldToScreen(v, [0.2, 0.15]);
    expect(s[0]).toBeCloseTo(480, 9);
    expect(s[1]).toBeCloseTo(240, 9);
    const w = screenToWorld(v, s);
    expect(w[0]).toBeCloseTo(0.2, 12);
    expect(w[1]).toBeCloseTo(0.15, 12);
  });

  it("pans, zooms around the cursor, and fits bounds", () => {
    const v = createViewport(800, 600, 100);
    const panned = panByPixels(v, 100, 50);
    expect(panned.center).toEqual([-1, 0.5]);
    const anchor: Vec2 = [400, 300];
    const before = screenToWorld(v, anchor);
    const zoomed = zoomAt(v, anchor, 2);
    expect(screenToWorld(zoomed, anchor)[0]).toBeCloseTo(before[0], 12);
    expect(zoomed.scale).toBe(200);
    const fit = fitToView(v, [0, 0], [0.4, 0.3]);
    expect(fit.center).toEqual([0.2, 0.15]);
    expect(fit.scale).toBeCloseTo(Math.min(720 / 0.4, 520 / 0.3), 9);
    expect(() => zoomAt(v, anchor, 0)).toThrowError(/positive/);
  });
});

describe("G9D select and box selection", () => {
  it("clicks to select, shift-clicks to toggle, drags a box", () => {
    const { ed, bl } = editor();
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerUp(px(ed, [0, 0]));
    expect(ed.getStatus().selectionCount).toBe(1);
    expect(ed.selection).toEqual([bl]);
    // Shift-click the same point deselects.
    ed.pointerDown(px(ed, [0, 0]), { shift: true });
    ed.pointerUp(px(ed, [0, 0]), { shift: true });
    expect(ed.getStatus().selectionCount).toBe(0);
    // Box over the whole rect selects points + segments + panel.
    ed.pointerDown(px(ed, [-0.1, -0.1]));
    ed.pointerMove(px(ed, [0.5, 0.4]));
    ed.pointerUp(px(ed, [0.5, 0.4]));
    expect(ed.getStatus().selectionCount).toBeGreaterThan(4);
  });

  it("drags a selected point in one undoable gesture", () => {
    const { ed, bl, panelId } = editor();
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerUp(px(ed, [0, 0]));
    const depth0 = ed.cad.history.undoDepth;
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerMove(px(ed, [0.05, 0.05]));
    expect(ed.getStatus().gestureOpen).toBe(true);
    ed.pointerUp(px(ed, [0.05, 0.05]));
    const p = getPoint(ed.document, bl, panelId);
    expect(p.x).toBeCloseTo(0.05, 6);
    expect(p.y).toBeCloseTo(0.05, 6);
    expect(ed.cad.history.undoDepth).toBe(depth0 + 1); // coalesced
    ed.keyDown("z", { ctrl: true });
    const q = getPoint(ed.document, bl, panelId);
    expect([q.x, q.y]).toEqual([0, 0]);
  });
});

describe("G9D tools", () => {
  it("draws points and lines with snapping", () => {
    const { ed } = editor();
    ed.keyDown("p");
    expect(ed.tool).toBe("point");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    expect(ed.getStatus().selectionCount).toBe(1);
    ed.keyDown("l");
    ed.pointerDown(px(ed, [0, 0])); // snaps to bl vertex
    expect(ed.lastSnap?.kind).toBe("vertex");
    ed.pointerUp(px(ed, [0.3, 0.2]));
    const segs = ed.document.segments.filter((s) => s.role === "construction");
    expect(segs.length).toBeGreaterThanOrEqual(1);
  });

  it("moves points with the move tool and numeric input in mm", () => {
    const { ed, bl, panelId } = editor();
    ed.keyDown("m");
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerMove(px(ed, [0.02, 0]));
    ed.pointerUp(px(ed, [0.02, 0]));
    expect(getPoint(ed.document, bl, panelId).x).toBeCloseTo(0.02, 6);
    // Numeric: type dx,dy in mm and commit.
    ed.pointerDown(px(ed, [0.02, 0]));
    for (const ch of "10,5") ed.keyDown(ch);
    ed.keyDown("Enter");
    const p = getPoint(ed.document, bl, panelId);
    expect(p.x).toBeCloseTo(0.03, 6);
    expect(p.y).toBeCloseTo(0.005, 6);
  });

  it("rotates selected points around a picked pivot", () => {
    const { ed, bl, panelId } = editor();
    ed.pointerDown(px(ed, [0.4, 0])); // select br
    ed.pointerUp(px(ed, [0.4, 0]));
    ed.keyDown("r");
    ed.pointerDown(px(ed, [0, 0])); // pivot at bl
    ed.pointerMove(px(ed, [0, 0.4])); // 90° CCW
    ed.pointerUp(px(ed, [0, 0.4]));
    const p = getPoint(ed.document, ed.selection[0], panelId);
    expect(p.x).toBeCloseTo(0, 6);
    expect(p.y).toBeCloseTo(0.4, 6);
    void bl;
  });

  it("mirrors the active panel across a picked line", () => {
    const { ed, panelId } = editor();
    ed.keyDown("i");
    ed.pointerDown(px(ed, [0.2, -1]));
    ed.pointerUp(px(ed, [0.2, 1]));
    expect(validatePatternDocument(ed.document).valid).toBe(true);
    expect(ed.document.panels.find((p) => p.id === panelId)).toBeDefined();
  });

  it("offsets a loop into a new panel with numeric mm input", () => {
    const { ed } = editor();
    ed.keyDown("o");
    for (const ch of "50") ed.keyDown(ch);
    ed.keyDown("Enter");
    expect(ed.defaultOffsetM).toBeCloseTo(0.05, 12);
    ed.pointerDown(px(ed, [0.2, 0])); // bottom edge
    ed.pointerUp(px(ed, [0.2, 0]));
    expect(ed.document.panels).toHaveLength(2);
    expect(validatePatternDocument(ed.document).valid).toBe(true);
  });

  it("splits a segment on click and trims target-then-cutter", () => {
    const { ed, bottom } = editor();
    ed.keyDown("s");
    ed.pointerDown(px(ed, [0.2, 0]));
    expect(ed.document.segments.some((s) => s.id === bottom)).toBe(false); // split replaced it
    ed.keyDown("t");
    // Target: left edge; cutter: bottom edge (they meet at bl).
    ed.pointerDown(px(ed, [0, 0.2]));
    expect(ed.getStatus().hint).toBe("pick cutter");
    ed.pointerDown(px(ed, [0.2, 0]));
    expect(validatePatternDocument(ed.document).valid).toBe(true);
  });

  it("measures world-space distances and accumulates a chain", () => {
    const { ed } = editor();
    ed.keyDown("u");
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerMove(px(ed, [0.3, 0.4]));
    const st = ed.getStatus();
    expect(st.measurementM).toBeCloseTo(0.5, 6);
    expect(st.hint).toContain("500.0 mm");
    ed.pointerDown(px(ed, [0.3, 0.4]));
    expect(ed.getStatus().hint).toContain("chain");
  });

  it("adds a distance constraint by picking two points", () => {
    const { ed, bl } = editor();
    ed.keyDown("c");
    ed.pointerDown(px(ed, [0, 0]));
    expect(ed.getStatus().hint).toBe("pick second point");
    ed.pointerDown(px(ed, [0.4, 0]));
    expect(ed.constraints.constraints).toHaveLength(1);
    expect(ed.constraints.constraints[0].kind).toBe("distance");
    void bl;
  });
});

describe("G9D session behaviors", () => {
  it("pans with middle-drag, zooms with the wheel, never simulates", () => {
    const { ed } = editor();
    const c0 = ed.viewport.center;
    ed.pointerDown([400, 300], {}, "middle");
    ed.pointerMove([500, 300]);
    ed.pointerUp([500, 300]);
    expect(ed.viewport.center[0]).toBeLessThan(c0[0]);
    const s0 = ed.viewport.scale;
    ed.wheel([400, 300], -100);
    expect(ed.viewport.scale).toBeGreaterThan(s0);
    // No solver contact: document untouched by navigation.
    expect(ed.cad.history.undoDepth).toBe(0);
  });

  it("cancels in priority order and deletes selection", () => {
    const { ed, bl } = editor();
    ed.keyDown("l");
    ed.pointerDown(px(ed, [0, 0]));
    ed.keyDown("Escape");
    expect(ed.pending.anchorWorld).toBeUndefined();
    ed.keyDown("v"); // back to select: the stray line-tool anchor is gone
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerUp(px(ed, [0, 0]));
    expect(ed.selection).toEqual([bl]);
    ed.keyDown("Escape");
    expect(ed.selection).toEqual([]);
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerUp(px(ed, [0, 0]));
    ed.keyDown("Delete");
    expect(ed.document.points.some((p) => p.id === bl)).toBe(false);
    expect(ed.selection).toEqual([]);
  });

  it("right-click cancels and space enables pan", () => {
    const { ed } = editor();
    ed.keyDown("l");
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerDown(px(ed, [0, 0]), {}, "right");
    expect(ed.pending.anchorWorld).toBeUndefined();
    ed.keyDown(" ");
    ed.pointerDown([100, 100]);
    ed.pointerMove([200, 100]);
    ed.pointerUp([200, 100]);
    ed.keyUp(" ");
    expect(ed.cad.history.undoDepth).toBe(0);
  });

  it("saves and reloads pattern plus constraints with dirty tracking", () => {
    const { ed } = editor();
    expect(ed.getStatus().dirty).toBe(false);
    ed.keyDown("p");
    ed.pointerDown(px(ed, [0.1, 0.1]));
    expect(ed.getStatus().dirty).toBe(true);
    ed.keyDown("c");
    ed.pointerDown(px(ed, [0, 0]));
    ed.pointerDown(px(ed, [0.4, 0]));
    const saved = ed.saveState();
    const ed2 = new EditorSession(rectFixture().document);
    ed2.loadState(saved);
    expect(ed2.document.points.length).toBe(ed.document.points.length);
    expect(ed2.constraints.constraints).toHaveLength(1);
    expect(ed2.getStatus().dirty).toBe(false);
    expect(ed2.validateForAssembly().valid).toBe(true);
  });

  it("status always answers the UX questions", () => {
    const { ed } = editor();
    const st = ed.getStatus();
    expect(st.tool).toBe("select");
    expect(st.activePanelId).not.toBeNull();
    expect(st.valid).toBe(true);
    expect(st.diagnosticCount).toBe(0);
    ed.pointerMove(px(ed, [0.1, 0.1]));
    expect(ed.getStatus().cursorWorld).not.toBeNull();
  });
});
