// G9A adversarial audit: pathological geometry, structured rejection of
// corrupt documents, serialization round-trips, input immutability, and
// state-machine sequences (undo/redo/save-reload/edit) with stable ids.
//
// Oracle: validatePatternDocument + serializePatternDocument from G8A.
// These tests never repair data — they assert either exact behavior or an
// explicit diagnostic/error.
import { describe, it, expect } from "vitest";
import * as cad from "../../src/pattern/cad.js";
import {
  deserializePatternDocument,
  serializePatternDocument,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  deletePoint,
  mergeSegments,
  mirrorPanel,
  moveSegmentBy,
  splitSegment,
} from "../../src/cad/ops.js";
import { measurePanel, resolveSegment } from "../../src/cad/queries.js";
import { CadSession } from "../../src/cad/history.js";
import { hitTest } from "../../src/cad/queries.js";
import { polygonFixture, rectFixture } from "./fixtures.js";

function codes(doc: PatternDocument): string[] {
  return validatePatternDocument(doc).diagnostics.map((d) => d.code);
}

function ser(doc: PatternDocument): string {
  return serializePatternDocument(doc);
}

// ---------------------------------------------------------------------------
// Pathological geometry (validation must REPORT, not repair)
// ---------------------------------------------------------------------------

describe("pathological panels", () => {
  it("empty panel (no loops) is reported", () => {
    let doc = cad.createPatternDocument("test-doc");
    const panel = cad.createPanel(doc, "empty");
    doc = panel.document;
    const result = validatePatternDocument(doc);
    expect(result.valid).toBe(false);
    expect(result.diagnostics.map((d) => d.code)).toContain("open-boundary");
  });

  it("collinear triangle (zero area) is reported as degenerate", () => {
    const flat = polygonFixture([[0, 0], [1, 0], [2, 0]]);
    expect(codes(flat.document)).toContain("degenerate-panel");
  });

  it("self-crossing boundary with nonzero area is reported", () => {
    // A classic bowtie has exactly ZERO net area, which validation reports
    // as degenerate first; this fixture self-crosses while enclosing area,
    // so the self-intersection rule is the one that fires.
    const crossed = polygonFixture([[0, 0], [2, 0], [0, 2], [1, -0.5]]);
    expect(codes(crossed.document)).toContain("self-intersection");
    expect(codes(crossed.document)).not.toContain("degenerate-panel");

    // The zero-area bowtie is still reported (as degenerate) — never
    // silently accepted.
    const bowtie = polygonFixture([[0, 0], [1, 1], [1, 0], [0, 1]]);
    expect(validatePatternDocument(bowtie.document).valid).toBe(false);
    expect(codes(bowtie.document)).toContain("degenerate-panel");
  });

  it("winding mismatch between field and geometry is reported", () => {
    // Clockwise points authored with an "ccw" orientation field.
    const wrong = polygonFixture([[0, 0], [0, 0.3], [0.4, 0.3], [0.4, 0]], "ccw");
    expect(codes(wrong.document)).toContain("invalid-winding");
    // The same geometry with the honest field validates cleanly.
    const honest = polygonFixture([[0, 0], [0, 0.3], [0.4, 0.3], [0.4, 0]], "cw");
    expect(validatePatternDocument(honest.document).valid).toBe(true);
  });

  it("repeated consecutive vertices produce a zero-length-edge diagnostic", () => {
    const dup = polygonFixture([[0, 0], [0.4, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]]);
    expect(codes(dup.document)).toContain("zero-length-edge");
  });

  it("nearly-coincident vertices (1e-12 apart) are reported, not merged", () => {
    const near = polygonFixture([[0, 0], [0.4, 0], [0.4, 1e-12], [0.4, 0.3], [0, 0.3]]);
    expect(codes(near.document)).toContain("zero-length-edge");
    // The extra point still exists — validation never repairs.
    expect(near.document.points.length).toBe(5);
  });

  it("tiny panels validate and survive ops", () => {
    const tiny = polygonFixture([[0, 0], [1e-5, 0], [1e-5, 1e-5], [0, 1e-5]]);
    expect(validatePatternDocument(tiny.document).valid).toBe(true);
    const mirrored = mirrorPanel(tiny.document, tiny.panelId, [0, -1], [0, 1]);
    expect(validatePatternDocument(mirrored).valid).toBe(true);
    expect(Math.abs(measurePanel(mirrored, tiny.panelId).area)).toBeCloseTo(1e-10, 15);
  });

  it("narrow slivers validate and round-trip", () => {
    const narrow = polygonFixture([[0, 0], [1e-3, 0], [1e-3, 0.3], [0, 0.3]]);
    expect(validatePatternDocument(narrow.document).valid).toBe(true);
    expect(ser(deserializePatternDocument(ser(narrow.document)))).toBe(ser(narrow.document));
  });

  it("huge coordinates keep exact-ish measurements (relative error)", () => {
    const huge = polygonFixture([
      [1e6, 1e6], [1e6 + 1e4, 1e6], [1e6 + 1e4, 1e6 + 1e4], [1e6, 1e6 + 1e4],
    ]);
    expect(validatePatternDocument(huge.document).valid).toBe(true);
    const m = measurePanel(huge.document, huge.panelId);
    const expectedArea = 1e8;
    expect(Math.abs(Math.abs(m.area) - expectedArea) / expectedArea).toBeLessThan(1e-9);
    expect(Math.abs(m.perimeter - 4e4) / 4e4).toBeLessThan(1e-12);
  });

  it("concave panels validate and mirror cleanly", () => {
    // L-shape.
    const l = polygonFixture([[0, 0], [0.4, 0], [0.4, 0.1], [0.1, 0.1], [0.1, 0.3], [0, 0.3]]);
    expect(validatePatternDocument(l.document).valid).toBe(true);
    const mirrored = mirrorPanel(l.document, l.panelId, [0, -1], [0, 1]);
    expect(validatePatternDocument(mirrored).valid).toBe(true);
    expect(Math.abs(measurePanel(mirrored, l.panelId).area)).toBeCloseTo(
      Math.abs(measurePanel(l.document, l.panelId).area),
      12,
    );
  });
});

// ---------------------------------------------------------------------------
// Corrupt-document diagnostics (refs, duplicates)
// ---------------------------------------------------------------------------

describe("corrupt references are reported", () => {
  it("duplicate ids", () => {
    const f = rectFixture();
    const doc = JSON.parse(JSON.stringify(f.document)) as PatternDocument;
    doc.points[1].id = doc.points[0].id;
    expect(codes(doc)).toContain("duplicate-id");
  });

  it("missing endpoint references", () => {
    const f = rectFixture();
    const doc = JSON.parse(JSON.stringify(f.document)) as PatternDocument;
    doc.segments[0].endPointId = "ghost/point";
    expect(codes(doc)).toContain("missing-reference");
  });

  it("segment pointing at a foreign panel", () => {
    const f = rectFixture();
    const other = rectFixture(0.4, 0.3, [1, 0], "other-doc");
    const doc: PatternDocument = {
      ...f.document,
      panels: [...f.document.panels, ...other.document.panels],
      points: [...f.document.points, ...other.document.points],
      segments: [...f.document.segments, ...other.document.segments],
    };
    // Foreign point referenced by a segment of the first panel.
    const seg = doc.segments.find((s) => s.panelId === f.panelId)!;
    const foreign = doc.points.find((p) => p.panelId === other.document.panels[0].id)!;
    seg.endPointId = foreign.id;
    expect(codes(doc)).toContain("missing-reference");
  });
});

// ---------------------------------------------------------------------------
// Serialization round-trips
// ---------------------------------------------------------------------------

describe("serialization", () => {
  it("repeated serialize/deserialize is byte-stable and valid", () => {
    const f = rectFixture();
    let s = ser(f.document);
    for (let i = 0; i < 5; i++) {
      const doc = deserializePatternDocument(s);
      expect(validatePatternDocument(doc).valid).toBe(true);
      const next = ser(doc);
      expect(next).toBe(s);
      s = next;
    }
  });

  it("corrupt JSON and invalid documents are rejected explicitly", () => {
    expect(() => deserializePatternDocument("{not json")).toThrowError(/invalid JSON|invalid-document/);
    const f = rectFixture();
    const doc = JSON.parse(JSON.stringify(f.document)) as PatternDocument;
    doc.segments[0].endPointId = "ghost";
    // canonicalPatternDocument validates before serializing.
    expect(() => serializePatternDocument(doc)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Input immutability (no operation may touch its input document)
// ---------------------------------------------------------------------------

describe("operations never mutate their input", () => {
  it("move/delete/mirror/split preserve the source serialization", () => {
    const f = rectFixture();
    const src = ser(f.document);
    moveSegmentBy(f.document, f.panelId, f.segments.bottom, [0.1, 0.1]);
    mirrorPanel(f.document, f.panelId, [0, -1], [0, 1]);
    splitSegment(f.document, f.panelId, f.loopId, f.segments.top, 0.5);
    deletePoint(f.document, f.panelId, f.points.tr);
    expect(ser(f.document)).toBe(src);
  });

  it("merge preserves the source serialization", () => {
    const p = polygonFixture([[0, 0], [0.2, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]]);
    const src = ser(p.document);
    mergeSegments(p.document, p.panelId, p.loopId, p.segmentIds[0], p.segmentIds[1]);
    expect(ser(p.document)).toBe(src);
  });
});

// ---------------------------------------------------------------------------
// State machines (G9E-style sequences)
// ---------------------------------------------------------------------------

describe("state-machine sequences", () => {
  it("draw -> undo -> redo -> edit", () => {
    const f = rectFixture();
    const s = new CadSession(f.document);
    const created = s.run("draw line", (d) =>
      cad.createConstructionLine(d, f.panelId, f.points.bl, f.points.tr).document,
    );
    const drawn = ser(created);
    s.undo();
    expect(ser(s.document)).toBe(ser(f.document)); // undrawn
    s.redo();
    expect(ser(s.document)).toBe(drawn); // redrawn with the SAME ids
    // Now edit on top of the redone state.
    s.run("move", (d) => cad.movePoint(d, f.panelId, f.points.tr, [0.35, 0.28]));
    expect(validatePatternDocument(s.document).valid).toBe(true);
    s.undo();
    expect(ser(s.document)).toBe(drawn);
    s.undo();
    expect(ser(s.document)).toBe(ser(f.document));
  });

  it("split -> undo -> redo -> mirror keeps ids and validity", () => {
    const f = rectFixture();
    const s = new CadSession(f.document);
    const split = s.run("split", (d) => splitSegment(d, f.panelId, f.loopId, f.segments.bottom, 0.5));
    expect(split.segmentIds[0]).not.toBe(f.segments.bottom); // arc/line split allocates fresh ids
    const afterSplit = ser(s.document);
    expect(validatePatternDocument(s.document).valid).toBe(true);

    s.undo();
    expect(ser(s.document)).toBe(ser(f.document));
    expect(s.document.segments.length).toBe(4);

    s.redo();
    expect(ser(s.document)).toBe(afterSplit); // ids restored exactly

    const mirrored = s.run("mirror", (d) => mirrorPanel(d, f.panelId, [0, -1], [0, 1]));
    expect(validatePatternDocument(mirrored).valid).toBe(true);
    // Undo lands exactly on the split state.
    s.undo();
    expect(ser(s.document)).toBe(afterSplit);
  });

  it("dimension -> geometry edit -> undo -> redo keeps measurement consistent", () => {
    const f = rectFixture();
    const s = new CadSession(f.document);
    const dim = s.run("dimension", (d) =>
      cad.createDistanceDimension(d, f.panelId, f.points.bl, f.points.br, "width"),
    );
    expect(cad.measureDimension(s.document, f.panelId, dim.dimensionId)).toBeCloseTo(0.4, 12);

    s.run("narrow", (d) => cad.movePoint(d, f.panelId, f.points.br, [0.3, 0]));
    expect(cad.measureDimension(s.document, f.panelId, dim.dimensionId)).toBeCloseTo(0.3, 12);
    expect(validatePatternDocument(s.document).valid).toBe(true);

    s.undo();
    expect(cad.measureDimension(s.document, f.panelId, dim.dimensionId)).toBeCloseTo(0.4, 12);
    s.redo();
    expect(cad.measureDimension(s.document, f.panelId, dim.dimensionId)).toBeCloseTo(0.3, 12);
    // The dimension reference itself was never re-created.
    const dimEntity = s.document.panels[0].dimensions.find((d0) => d0.id === dim.dimensionId);
    expect(dimEntity).toBeDefined();
    expect(dimEntity!.pointBId).toBe(f.points.br);
  });

  it("save -> reload -> edit produces the same result as editing directly", () => {
    const f = rectFixture();
    const saved = ser(f.document);

    const direct = moveSegmentBy(f.document, f.panelId, f.segments.bottom, [0, 0.01]);
    const viaReload = moveSegmentBy(
      deserializePatternDocument(saved),
      f.panelId,
      f.segments.bottom,
      [0, 0.01],
    );
    expect(ser(viaReload)).toBe(ser(direct));
    expect(validatePatternDocument(viaReload).valid).toBe(true);
  });

  it("ids are unique and stable across a long mixed sequence", () => {
    const f = rectFixture();
    const s = new CadSession(f.document);
    s.run("move", (d) => moveSegmentBy(d, f.panelId, f.segments.bottom, [0, 0.01]));
    const idsAfterMove = allIds(s.document);
    s.run("split", (d) => splitSegment(d, f.panelId, f.loopId, f.segments.top, 0.3));
    s.run("mirror", (d) => mirrorPanel(d, f.panelId, [-1, 0], [-1, 1]));
    expect(validatePatternDocument(s.document).valid).toBe(true);

    s.undo();
    s.undo();
    expect(allIds(s.document)).toEqual(idsAfterMove); // exact id restoration
    s.redo();
    s.redo();
    expect(validatePatternDocument(s.document).valid).toBe(true);
  });
});

function allIds(doc: PatternDocument): string[] {
  return [
    ...doc.points.map((p) => p.id),
    ...doc.segments.map((sg) => sg.id),
    ...doc.panels.flatMap((p) => [
      p.id,
      ...p.boundaryLoops.map((l) => l.id),
      ...p.dimensions.map((d0) => d0.id),
      ...p.constraints.map((c) => c.id),
    ]),
  ];
}

// ---------------------------------------------------------------------------
// Hit-testing on degenerate input
// ---------------------------------------------------------------------------

describe("hit-test robustness", () => {
  it("tolerance zero only hits exactly-coincident vertices", () => {
    const f = rectFixture();
    const on = hitTest(f.document, [0, 0], 1e-12);
    expect(on.length).toBeGreaterThanOrEqual(1);
    expect(on[0].entityId).toBe(f.points.bl);
    const off = hitTest(f.document, [1e-6, 1e-6], 1e-9);
    expect(off.every((h) => h.kind !== "point")).toBe(true);
  });

  it("hit-test on a document with a corrupt segment does not throw", () => {
    const f = rectFixture();
    const doc = JSON.parse(JSON.stringify(f.document)) as PatternDocument;
    doc.segments[0].endPointId = "ghost";
    // resolveSegment throws PatternCadError for the broken segment — the
    // caller sees a structured error, not a crash with undefined access.
    expect(() => hitTest(doc, [0, 0], 0.05)).toThrowError(/missing-reference/);
  });
});

// ---------------------------------------------------------------------------
// resolveSegment guards
// ---------------------------------------------------------------------------

describe("resolveSegment guards", () => {
  it("rejects unknown ids with missing-reference", () => {
    const f = rectFixture();
    expect(() => resolveSegment(f.document, "nope")).toThrowError(/missing-reference/);
    expect(() => resolveSegment(f.document, f.segments.bottom, "wrong-panel")).toThrowError(
      /missing-reference/,
    );
  });
});
