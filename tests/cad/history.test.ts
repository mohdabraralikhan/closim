// G9A history: undo/redo over immutable PatternDocument operations,
// gesture batching (one undo entry per drag), no-op suppression,
// redo invalidation, limits, and CadSession run/undo/redo semantics.
import { describe, it, expect } from "vitest";
import {
  createPoint,
  movePanel,
  movePoint,
  serializePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import { CadSession, History } from "../../src/cad/history.js";
import { rectFixture } from "./fixtures.js";

function session(): { session: CadSession; panelId: string; pointId: string } {
  const f = rectFixture();
  const s = new CadSession(f.document);
  return { session: s, panelId: f.panelId, pointId: f.points.tr };
}

describe("History recording", () => {
  it("records effective operations and suppresses no-ops", () => {
    const f = rectFixture();
    const h = new History();
    const moved = movePanel(f.document, f.panelId, [1, 0]);
    expect(h.record("move", f.document, moved)).toBe(true);
    expect(h.undoDepth).toBe(1);

    // Same-position move produces a fresh clone with identical content.
    const same = movePoint(f.document, f.panelId, f.points.tr, [0.4, 0.3]);
    expect(h.record("move no-op", f.document, same)).toBe(false);
    expect(h.undoDepth).toBe(1);
  });

  it("undo/redo round-trip restores byte-identical serialization", () => {
    const f = rectFixture();
    const h = new History();
    const before = serializePatternDocument(f.document);
    const moved = movePanel(f.document, f.panelId, [0.25, -0.1]);
    h.record("translate", f.document, moved);
    const after = serializePatternDocument(moved);
    expect(after).not.toBe(before);

    const entry = h.takeUndo()!;
    expect(serializePatternDocument(entry.before)).toBe(before);
    const redone = h.takeRedo();
    expect(redone).not.toBeNull();
    expect(serializePatternDocument(redone!.after)).toBe(after);
    expect(h.undoDepth).toBe(1);
    expect(h.redoDepth).toBe(0);
  });

  it("new operations clear the redo stack", () => {
    const f = rectFixture();
    const h = new History();
    const a = movePanel(f.document, f.panelId, [1, 0]);
    h.record("a", f.document, a);
    h.takeUndo();
    expect(h.redoDepth).toBe(1);
    const b = movePanel(f.document, f.panelId, [2, 0]);
    h.record("b", f.document, b);
    expect(h.redoDepth).toBe(0);
  });

  it("limit drops the oldest entry", () => {
    const f = rectFixture();
    const h = new History(3);
    let doc: PatternDocument = f.document;
    for (let i = 1; i <= 5; i++) {
      const next = movePanel(doc, f.panelId, [i, 0]);
      h.record(`move ${i}`, doc, next);
      doc = next;
    }
    expect(h.undoDepth).toBe(3);
    expect(h.peekUndo()!.label).toBe("move 5");
    expect(h.takeUndo()).not.toBeNull(); // move 5
    expect(h.takeUndo()).not.toBeNull(); // move 4
    expect(h.takeUndo()).not.toBeNull(); // move 3
    expect(h.takeUndo()).toBeNull();
  });

  it("rejects invalid limits", () => {
    expect(() => new History(0)).toThrow();
    expect(() => new History(1.5)).toThrow();
  });
});

describe("gestures", () => {
  it("coalesce a drag into ONE undo entry", () => {
    const f = rectFixture();
    const h = new History();
    const start = f.document;
    h.beginGesture("drag point", start);
    let doc = start;
    for (let i = 1; i <= 10; i++) {
      const next = movePoint(doc, f.panelId, f.points.tr, [0.4 + i * 0.01, 0.3]);
      h.record("drag frame", doc, next);
      doc = next;
    }
    expect(h.endGesture()).toBe(true);
    expect(h.undoDepth).toBe(1);
    expect(h.peekUndo()!.label).toBe("drag point");
    // One undo jumps back to the drag start, not to the previous frame.
    expect(serializePatternDocument(h.takeUndo()!.before)).toBe(serializePatternDocument(start));
  });

  it("a gesture with only no-op frames pushes nothing", () => {
    const f = rectFixture();
    const h = new History();
    h.beginGesture("noop drag", f.document);
    h.record("noop frame", f.document, movePoint(f.document, f.panelId, f.points.tr, [0.4, 0.3]));
    expect(h.endGesture()).toBe(false);
    expect(h.undoDepth).toBe(0);
  });

  it("cancelGesture returns the start document and pushes nothing", () => {
    const f = rectFixture();
    const h = new History();
    h.beginGesture("drag", f.document);
    h.record("frame", f.document, movePanel(f.document, f.panelId, [5, 5]));
    const restored = h.cancelGesture();
    expect(restored).not.toBeNull();
    expect(serializePatternDocument(restored!)).toBe(serializePatternDocument(f.document));
    expect(h.undoDepth).toBe(0);
    expect(h.cancelGesture()).toBeNull(); // nothing open
  });

  it("nested gestures only push at the outermost end", () => {
    const f = rectFixture();
    const h = new History();
    h.beginGesture("outer", f.document);
    h.record("f1", f.document, movePanel(f.document, f.panelId, [1, 0]));
    h.beginGesture("inner", f.document);
    h.record("f2", f.document, movePanel(f.document, f.panelId, [2, 0]));
    expect(h.endGesture()).toBe(false); // closes inner
    expect(h.undoDepth).toBe(0);
    expect(h.endGesture()).toBe(true); // closes outer -> one entry
    expect(h.undoDepth).toBe(1);
    expect(h.peekUndo()!.label).toBe("outer");
  });

  it("undo/redo are blocked while a gesture is open", () => {
    const f = rectFixture();
    const h = new History();
    h.record("setup", f.document, movePanel(f.document, f.panelId, [1, 0]));
    h.beginGesture("open", f.document);
    expect(h.takeUndo()).toBeNull();
    expect(h.takeRedo()).toBeNull();
    h.cancelGesture();
    expect(h.takeUndo()).not.toBeNull();
  });

  it("snapshot reports depths and labels", () => {
    const f = rectFixture();
    const h = new History();
    expect(h.snapshot()).toEqual({
      undoDepth: 0, redoDepth: 0, undoLabel: null, redoLabel: null, gestureLabel: null,
    });
    h.record("step", f.document, movePanel(f.document, f.panelId, [1, 0]));
    const snap = h.snapshot();
    expect(snap.undoLabel).toBe("step");
    expect(snap.undoDepth).toBe(1);
    h.beginGesture("drag", f.document);
    expect(h.snapshot().gestureLabel).toBe("drag");
  });
});

describe("CadSession", () => {
  it("run applies ops returning documents and ops returning result objects", () => {
    const { session: s, panelId } = session();
    const docResult = s.run("translate", (d) => movePanel(d, panelId, [1, 0]));
    expect(docResult).toBe(s.document);
    expect(s.history.undoDepth).toBe(1);

    const created = s.run("create point", (d) => createPoint(d, panelId, [0.01, 0.01], "construction"));
    expect(created.pointId).toBeTruthy();
    expect(created.document).toBe(s.document);
    expect(s.document.points.some((p) => p.id === created.pointId)).toBe(true);
    expect(s.history.undoDepth).toBe(2);
  });

  it("undo/redo walk the document through history exactly", () => {
    const { session: s, panelId } = session();
    const original = serializePatternDocument(s.document);
    s.run("m1", (d) => movePanel(d, panelId, [1, 0]));
    s.run("m2", (d) => movePanel(d, panelId, [0, 1]));
    const after2 = serializePatternDocument(s.document);

    expect(s.undo()).not.toBeNull();
    expect(serializePatternDocument(s.document)).not.toBe(after2);
    expect(s.undo()).not.toBeNull();
    expect(serializePatternDocument(s.document)).toBe(original);
    expect(s.undo()).toBeNull(); // exhausted

    expect(s.redo()).not.toBeNull();
    expect(s.redo()).not.toBeNull();
    expect(serializePatternDocument(s.document)).toBe(after2);
    expect(s.redo()).toBeNull();
  });

  it("failed ops throw without recording or advancing the document", () => {
    const { session: s, panelId } = session();
    const before = s.document;
    expect(() =>
      s.run("bad", (d) => movePanel(d, "missing-panel", [1, 0])),
    ).toThrowError(/missing-reference/);
    expect(s.document).toBe(before);
    expect(s.history.undoDepth).toBe(0);
  });

  it("cancelGesture restores the gesture start document", () => {
    const { session: s, panelId } = session();
    const before = serializePatternDocument(s.document);
    s.history.beginGesture("drag", s.document);
    s.run("frame", (d) => movePanel(d, panelId, [0.1, 0]));
    s.run("frame", (d) => movePanel(d, panelId, [0.2, 0]));
    expect(s.cancelGesture()).toBe(true);
    expect(serializePatternDocument(s.document)).toBe(before);
    expect(s.history.undoDepth).toBe(0);
    expect(s.cancelGesture()).toBe(false); // no gesture open
  });

  it("reset replaces the document and clears history", () => {
    const { session: s, panelId } = session();
    s.run("move", (d) => movePanel(d, panelId, [1, 0]));
    const fresh = rectFixture().document;
    s.reset(fresh);
    expect(s.document).toBe(fresh);
    expect(s.history.undoDepth).toBe(0);
    expect(s.history.redoDepth).toBe(0);
    expect(s.undo()).toBeNull();
  });

  it("gesture through the session produces a single undo step", () => {
    const { session: s, panelId, pointId } = session();
    const before = serializePatternDocument(s.document);
    s.history.beginGesture("drag", s.document);
    for (let i = 1; i <= 8; i++) {
      s.run("frame", (d) => movePoint(d, panelId, pointId, [0.4 + i * 0.005, 0.3]));
    }
    expect(s.history.endGesture()).toBe(true);
    expect(s.history.undoDepth).toBe(1);
    s.undo();
    expect(serializePatternDocument(s.document)).toBe(before);
  });
});
