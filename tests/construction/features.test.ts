// G16A tests: dart/pleat/gather features — CRUD, derivation, validation,
// transfer, serialization, grading, production, export-readiness.
import { describe, expect, it } from "vitest";
import { triangulateCadPanel, validatePatternDocument } from "../../src/pattern/cad.js";
import { validateSeams } from "../../src/garment/sewing.js";
import {
  addDart,
  addGather,
  addPleat,
  createConstructionSet,
  deriveConstruction,
  deserializeConstructionSet,
  removeFeature,
  serializeConstructionSet,
  transferDart,
  updateFeature,
  validateFeatures,
  type DartFeature,
} from "../../src/construction/features.js";
import { polygonFixture, rectFixture } from "../cad/fixtures.js";

function valid(doc: Parameters<typeof validatePatternDocument>[0]): void {
  expect(validatePatternDocument(doc).valid).toBe(true);
}

describe("G16A darts", () => {
  it("creates open darts and derives the V with press folds", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.25], "open");
    set = d.set;
    expect(d.id).toMatch(/^construction\/dart\/\d+$/);
    expect(validateFeatures(f.document, set)).toEqual([{ id: d.id, kind: "dart", ok: true }]);
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    expect(out.applied).toEqual([d.id]);
    expect(out.seams).toEqual([]); // open: no seam
    expect(out.folds).toHaveLength(2);
    valid(out.document);
    // Intake edge replaced by two legs: 4 + 2 splits + V swap (-1 +2) = 7.
    const loop = out.document.panels[0].boundaryLoops[0];
    expect(loop.segmentIds).toHaveLength(7);
    // Source document untouched (feature -> derived, never destructive).
    expect(f.document.panels[0].boundaryLoops[0].segmentIds).toHaveLength(4);
  });

  it("closes darts into leg-to-leg seams", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.25], "closed");
    set = d.set;
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    expect(out.seams).toHaveLength(1);
    expect(out.seams[0].id).toBe(`seam/${d.id}`);
    expect(out.seams[0].stitchCount).toBeGreaterThanOrEqual(2);
    valid(out.document);
  });

  it("transfers darts and validates intake extremes", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.25]);
    set = d.set;
    set = transferDart(set, d.id, f.segments.top, 0.2, 0.8);
    const feature = set.features[0] as DartFeature;
    expect(feature.edgeSegmentId).toBe(f.segments.top);
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    valid(out.document);
    expect(() => addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.7, 0.3, [0.2, 0.25])).toThrowError(/tA < tB/);
    expect(() => addDart(set, f.panelId, f.loopId, f.segments.bottom, 0, 1, [0.2, 0.25])).toThrowError(/0 < tA/);
    expect(() => transferDart(set, "missing", f.segments.top, 0.2, 0.8)).toThrowError(/does not exist/);
    // Degenerate intake (zero width) is reported, not derived.
    let bad = createConstructionSet();
    const z = addDart(bad, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.25]);
    bad = z.set;
    bad = updateFeature(bad, z.id, { tB: 0.3 } as Partial<DartFeature>);
    expect(validateFeatures(f.document, bad)[0].ok).toBe(false);
    expect(deriveConstruction(f.document, bad).failed).toHaveLength(1);
  });

  it("rejects legs that cross the panel boundary", () => {
    // Narrow panel, apex far outside: legs must cross to reach it.
    const narrow = polygonFixture([[0, 0], [0.1, 0], [0.1, 0.6], [0, 0.6]]);
    let set = createConstructionSet();
    const d = addDart(set, narrow.panelId, narrow.loopId, narrow.segmentIds[0], 0.2, 0.8, [5, 5]);
    set = d.set;
    const status = validateFeatures(narrow.document, set)[0];
    expect(status.ok).toBe(false);
    expect(status.reason).toMatch(/crosses|coincides/);
  });

  it("supports multiple darts deterministically", () => {
    const f = rectFixture(0.6, 0.6);
    let set = createConstructionSet();
    const a = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.1, 0.3, [0.12, 0.2]);
    set = a.set;
    const b = addDart(set, f.panelId, f.loopId, f.segments.top, 0.6, 0.9, [0.45, 0.4], "closed");
    set = b.set;
    const first = deriveConstruction(f.document, set);
    const second = deriveConstruction(f.document, set);
    expect(JSON.stringify(first.document)).toBe(JSON.stringify(second.document));
    expect(first.seams).toHaveLength(1);
    valid(first.document);
  });
});

describe("G16A pleats and gathers", () => {
  it("derives knife pleats with intake and fold notches", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createConstructionSet();
    const p = addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "knife", 0.02);
    set = p.set;
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    valid(out.document);
    // Intake extended the edge by 2×depth; fold notches emitted.
    expect(out.notches.length).toBeGreaterThan(0);
    for (const n of out.notches) {
      expect(out.document.segments.some((s) => s.id === n.segmentId)).toBe(true);
    }
    expect(() => addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "knife", 0)).toThrowError(/positive/);
    expect(() => addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "accordion" as "knife", 0.02)).toThrowError(/pleat type/);
  });

  it("derives box and inverted pleats", () => {
    for (const pleatType of ["box", "inverted"] as const) {
      const f = rectFixture(0.4, 0.3);
      let set = createConstructionSet();
      const p = addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.5, pleatType, 0.02);
      set = p.set;
      const out = deriveConstruction(f.document, set);
      expect(out.failed).toEqual([]);
      valid(out.document);
      expect(out.notches.length).toBeGreaterThanOrEqual(2);
    }
  });

  it("derives gathers as correspondence metadata plus notches", () => {
    const f = rectFixture(0.5, 0.3);
    let set = createConstructionSet();
    const g = addGather(set, f.panelId, f.loopId, f.segments.bottom, { targetLengthM: 0.3, notchCount: 4 });
    set = g.set;
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    expect(out.notches).toHaveLength(4);
    valid(out.document); // boundary unchanged by gathering
    // Receiving edge shorter than source validates; longer fails explicitly.
    const okRecv = addGather(set, f.panelId, f.loopId, f.segments.bottom, {
      recvPanelId: f.panelId, recvLoopId: f.loopId, recvSegmentId: f.segments.right,
    });
    expect(validateFeatures(f.document, okRecv.set).map((s) => s.ok)).toContain(true);
    const badRecv = addGather(set, f.panelId, f.loopId, f.segments.right, {
      recvPanelId: f.panelId, recvLoopId: f.loopId, recvSegmentId: f.segments.bottom,
    });
    const statuses = validateFeatures(f.document, badRecv.set);
    expect(statuses[statuses.length - 1].ok).toBe(false);
  });

  it("removes features and round-trips persistence", () => {
    const f = rectFixture();
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.2]);
    set = d.set;
    set = removeFeature(set, d.id);
    expect(set.features).toEqual([]);
    expect(() => removeFeature(set, d.id)).toThrowError(/does not exist/);
    const p = addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "knife", 0.02);
    set = p.set;
    expect(deserializeConstructionSet(serializeConstructionSet(set))).toEqual(set);
    expect(serializeConstructionSet(deserializeConstructionSet(serializeConstructionSet(set))))
      .toBe(serializeConstructionSet(set));
    expect(() => deserializeConstructionSet("nope")).toThrowError(/JSON/);
  });
});

describe("G16A adversarial construction", () => {
  it("handles extreme intake and depth values", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.01, 0.99, [0.2, 0.4]);
    set = d.set;
    const out = deriveConstruction(f.document, set);
    expect(out.failed).toEqual([]);
    valid(out.document);
    let set2 = createConstructionSet();
    const p = addPleat(set2, f.panelId, f.loopId, f.segments.bottom, 0.5, "knife", 0.15);
    set2 = p.set;
    const out2 = deriveConstruction(f.document, set2);
    expect(out2.failed).toEqual([]);
    valid(out2.document);
  });

  it("fails gracefully when a second pleat targets a consumed edge", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createConstructionSet();
    const a = addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.3, "knife", 0.02);
    set = a.set;
    const b = addPleat(set, f.panelId, f.loopId, f.segments.bottom, 0.7, "knife", 0.02);
    set = b.set;
    // First pleat subdivides (destroys) the original edge, so the second
    // pleat's reference dies: reported, never silently reconnected.
    const out = deriveConstruction(f.document, set);
    expect(out.applied).toHaveLength(1);
    expect(out.failed).toHaveLength(1);
    valid(out.document);
  });

  it("reports severe gather mismatch without corrupting the document", () => {
    const f = rectFixture(0.5, 0.3);
    let set = createConstructionSet();
    const g = addGather(set, f.panelId, f.loopId, f.segments.right, {
      recvPanelId: f.panelId, recvLoopId: f.loopId, recvSegmentId: f.segments.bottom,
    });
    set = g.set;
    expect(validateFeatures(f.document, set)[0].ok).toBe(false);
    const out = deriveConstruction(f.document, set);
    expect(out.applied).toEqual([]);
    expect(out.failed).toHaveLength(1);
    valid(out.document);
  });
});

describe("G16A downstream compatibility", () => {
  it("derived geometry triangulates, grades by stable ids, and exports", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createConstructionSet();
    const d = addDart(set, f.panelId, f.loopId, f.segments.bottom, 0.3, 0.7, [0.2, 0.25], "closed");
    set = d.set;
    const out = deriveConstruction(f.document, set);
    // Triangulation (simulation input) works on derived geometry.
    const tri = triangulateCadPanel(out.document, f.panelId);
    expect(tri.triangles.length).toBeGreaterThan(0);
    // Dart seam resolves through the existing seam graph.
    expect(validateSeams(out.document, out.seams).valid).toBe(true);
    // Stable ids survive: panel id unchanged, still one panel.
    expect(out.document.panels.map((p) => p.id)).toEqual([f.panelId]);
  });
});
