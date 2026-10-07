// G11A tests: production entities, allowance derivation, validation, persistence.
import { describe, expect, it } from "vitest";
import { PatternCadError } from "../../src/pattern/cad.js";
import {
  addAllowance,
  addAnnotation,
  addCutLine,
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  allowanceBoundary,
  clearEdgeAllowance,
  createProductionSet,
  cutBoundary,
  deserializeProductionSet,
  grainlineVector,
  notchFrame,
  removeAllowance,
  removeNotch,
  serializeProductionSet,
  setEdgeAllowance,
  setPanelMeta,
  validateProductionSet,
} from "../../src/cad/production.js";
import { circleFixture, dShapeFixture, polygonFixture, rectFixture } from "./fixtures.js";

describe("G11A production entities", () => {
  it("builds allowances with per-edge overrides and stable ids", () => {
    const f = rectFixture();
    let set = createProductionSet();
    const a = addAllowance(set, f.panelId, f.loopId, 0.01);
    set = a.set;
    expect(a.id).toMatch(/^production\/allowance\/\d+$/);
    set = setEdgeAllowance(set, a.id, f.segments.bottom, 0.02);
    expect(set.allowances[0].perEdgeM[f.segments.bottom]).toBe(0.02);
    set = clearEdgeAllowance(set, a.id, f.segments.bottom);
    expect(set.allowances[0].perEdgeM).toEqual({});
    // Re-adding for the same loop replaces (one allowance per loop).
    const b = addAllowance(set, f.panelId, f.loopId, 0.015);
    expect(b.set.allowances).toHaveLength(1);
    expect(b.set.allowances[0].defaultM).toBe(0.015);
    set = removeAllowance(b.set, b.id);
    expect(set.allowances).toEqual([]);
    expect(() => removeAllowance(set, b.id)).toThrowError(PatternCadError);
    expect(() => addAllowance(set, f.panelId, f.loopId, -1)).toThrowError(PatternCadError);
    expect(() => setEdgeAllowance(set, "missing", f.segments.bottom, 0.01)).toThrowError(/does not exist/);
  });

  it("builds markings with validated inputs", () => {
    const f = rectFixture();
    let set = createProductionSet();
    const n = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "double", 0.005);
    set = n.set;
    const g = addGrainline(set, f.panelId, [0.2, 0.05], [0.2, 0.25]);
    set = g.set;
    const fo = addFoldLine(set, f.panelId, [0.1, 0.05], [0.1, 0.25], "valley", "center-fold");
    set = fo.set;
    const d = addDrillMark(set, f.panelId, [0.2, 0.15], "cross");
    set = d.set;
    const il = addInternalLine(set, f.panelId, [[0.1, 0.1], [0.2, 0.2]], "dart", "front dart");
    set = il.set;
    const c = addCutLine(set, f.panelId, f.loopId, "sewing");
    set = c.set;
    const an = addAnnotation(set, f.panelId, [0.2, 0.15], "match to back");
    set = an.set;
    const lb = addLabelRegion(set, f.panelId, [0.05, 0.05], [0.35, 0.12], { panel: "front" });
    set = lb.set;
    expect(validateProductionSet(f.document, set)).toEqual([]);
    set = removeNotch(set, n.id);
    expect(set.notches).toEqual([]);
    expect(() => addNotch(set, f.panelId, f.loopId, f.segments.bottom, 2, "single", 0.005)).toThrowError(/t must be/);
    expect(() => addGrainline(set, f.panelId, [0.1, 0.1], [0.1, 0.1])).toThrowError(/distinct/);
    expect(() => addFoldLine(set, f.panelId, [0, 0], [1, 1], "sideways" as "mountain", "x")).toThrowError(/direction/);
    expect(() => addDrillMark(set, f.panelId, [NaN, 0])).toThrowError(/finite/);
    expect(() => addInternalLine(set, f.panelId, [[0, 0]], "dart")).toThrowError(/at least 2/);
    expect(() => addLabelRegion(set, f.panelId, [1, 1], [0, 0], {})).toThrowError(/non-degenerate/);
    expect(() => setPanelMeta(set, { panelId: f.panelId, cutQuantity: 0 })).toThrowError(/positive integer/);
  });

  it("stores panel metadata with upsert semantics", () => {
    const f = rectFixture();
    let set = createProductionSet();
    set = setPanelMeta(set, { panelId: f.panelId, cutQuantity: 2, section: "body", notes: "self" });
    set = setPanelMeta(set, { panelId: f.panelId, cutQuantity: 4 });
    expect(set.panelMeta).toHaveLength(1);
    expect(set.panelMeta[0].cutQuantity).toBe(4);
    expect(validateProductionSet(f.document, set)).toEqual([]);
  });
});

describe("G11A allowance derivation", () => {
  it("offsets a rectangle uniformly with exact miters", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createProductionSet();
    set = addAllowance(set, f.panelId, f.loopId, 0.05).set;
    const out = allowanceBoundary(f.document, set, f.panelId, f.loopId);
    expect(out.issues).toEqual([]);
    expect(out.maxDeviationM).toBe(0);
    expect(out.ring).toHaveLength(4);
    const xs = out.ring.map((p) => p[0]).sort((a, b) => a - b);
    const ys = out.ring.map((p) => p[1]).sort((a, b) => a - b);
    expect(xs[0]).toBeCloseTo(-0.05, 12);
    expect(xs[3]).toBeCloseTo(0.45, 12);
    expect(ys[0]).toBeCloseTo(-0.05, 12);
    expect(ys[3]).toBeCloseTo(0.35, 12);
    // Deterministic: identical recomputation.
    expect(allowanceBoundary(f.document, set, f.panelId, f.loopId)).toEqual(out);
  });

  it("supports per-edge allowances", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createProductionSet();
    const a = addAllowance(set, f.panelId, f.loopId, 0.01);
    set = setEdgeAllowance(a.set, a.id, f.segments.bottom, 0.05);
    const out = allowanceBoundary(f.document, set, f.panelId, f.loopId);
    expect(out.issues).toEqual([]);
    const ys = out.ring.map((p) => p[1]).sort((a2, b2) => a2 - b2);
    expect(ys[0]).toBeCloseTo(-0.05, 9); // bottom edge pushed further
    const tops = out.ring.map((p) => p[1]).sort((a2, b2) => a2 - b2);
    expect(tops[3]).toBeCloseTo(0.31, 9); // top edge keeps default
  });

  it("offsets arc loops with bounded deviation", () => {
    const c = circleFixture(0.1);
    let set = createProductionSet();
    set = addAllowance(set, c.panelId, c.loopId, 0.01).set;
    const out = allowanceBoundary(c.document, set, c.panelId, c.loopId);
    expect(out.ring.length).toBeGreaterThan(8);
    expect(out.maxDeviationM).toBe(1e-4);
    // Mean radius grows by ~the allowance.
    const radii = out.ring.map((p) => Math.hypot(p[0], p[1]));
    const mean = radii.reduce((s, r) => s + r, 0) / radii.length;
    expect(mean).toBeCloseTo(0.11, 3);
    const d = dShapeFixture(1);
    let set2 = createProductionSet();
    set2 = addAllowance(set2, d.panelId, d.loopId, 0.02).set;
    const out2 = allowanceBoundary(d.document, set2, d.panelId, d.loopId);
    expect(out2.ring.length).toBeGreaterThan(4);
  });

  it("reports spikes on acute corners instead of hiding them", () => {
    const needle = polygonFixture([[0, 0], [1, 0], [0, 0.01]]);
    let set = createProductionSet();
    set = addAllowance(set, needle.panelId, needle.loopId, 0.05).set;
    const out = allowanceBoundary(needle.document, set, needle.panelId, needle.loopId);
    expect(out.issues.map((i) => i.code)).toContain("spike");
    expect(out.ring).toHaveLength(3); // geometry still returned for inspection
  });

  it("reports self-intersecting offsets on narrow notches", () => {
    // V bite whose walls cross when offset past half the mouth width.
    const v = polygonFixture([[0, 0], [2, 0], [2, 1], [1.2, 1], [1, 0.2], [0.8, 1], [0, 1]]);
    let set = createProductionSet();
    set = addAllowance(set, v.panelId, v.loopId, 0.3).set;
    const out = allowanceBoundary(v.document, set, v.panelId, v.loopId);
    expect(out.issues.map((i) => i.code)).toContain("self-intersection");
    expect(out.ring.length).toBeGreaterThan(0); // still returned for inspection
  });

  it("requires an allowance and valid loop", () => {
    const f = rectFixture();
    expect(() => allowanceBoundary(f.document, createProductionSet(), f.panelId, f.loopId)).toThrowError(/no allowance/);
  });
});

describe("G11A frames and cut resolution", () => {
  it("frames notches on lines and arcs", () => {
    const f = rectFixture(0.4, 0.3);
    const frame = notchFrame(f.document, {
      id: "n", panelId: f.panelId, loopId: f.loopId,
      segmentId: f.segments.bottom, t: 0.5, kind: "single", depthM: 0.005,
    });
    expect(frame.pos[0]).toBeCloseTo(0.2, 12);
    expect(frame.pos[1]).toBeCloseTo(0, 12);
    expect(frame.outward[1]).toBeCloseTo(-1, 12); // outward below the bottom edge
    const d = dShapeFixture(1);
    const arcFrame = notchFrame(d.document, {
      id: "n", panelId: d.panelId, loopId: d.loopId,
      segmentId: d.segments.arc, t: 0.5, kind: "single", depthM: 0.005,
    });
    expect(Math.hypot(arcFrame.pos[0], arcFrame.pos[1])).toBeCloseTo(1, 9);
  });

  it("resolves cut lines from sewing or allowance sources", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createProductionSet();
    const sew = addCutLine(set, f.panelId, f.loopId, "sewing");
    set = sew.set;
    const ring = cutBoundary(f.document, set, sew.id);
    expect(ring.source).toBe("sewing");
    expect(ring.ring).toHaveLength(4);
    const allow = addAllowance(set, f.panelId, f.loopId, 0.02);
    set = allow.set;
    const cut = addCutLine(set, f.panelId, f.loopId, "allowance");
    set = cut.set;
    const aring = cutBoundary(f.document, set, cut.id);
    expect(aring.source).toBe("allowance");
    expect(aring.ring).toHaveLength(4);
    expect(() => cutBoundary(f.document, set, "missing")).toThrowError(/does not exist/);
  });

  it("measures grainline vectors", () => {
    expect(grainlineVector({ id: "g", panelId: "p", from: [0, 0], to: [3, 4] })).toEqual({
      direction: [0.6, 0.8],
      lengthM: 5,
    });
  });
});

describe("G11A validation and persistence", () => {
  it("detects orphaned and out-of-panel production data", () => {
    const f = rectFixture();
    let set = createProductionSet();
    set = addNotch(set, f.panelId, f.loopId, "ghost-segment", 0.5, "single", 0.005).set;
    set = addDrillMark(set, f.panelId, [99, 99]).set;
    set = addCutLine(set, f.panelId, f.loopId, "allowance").set; // no allowance exists
    set = setPanelMeta(set, { panelId: "ghost-panel", cutQuantity: 2 });
    const codes = validateProductionSet(f.document, set).map((d) => d.code);
    expect(codes).toContain("missing-reference");
    expect(codes).toContain("outside-panel");
  });

  it("round-trips byte-identically", () => {
    const f = rectFixture();
    let set = createProductionSet();
    set = addAllowance(set, f.panelId, f.loopId, 0.01).set;
    set = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.25, "single", 0.004).set;
    set = addGrainline(set, f.panelId, [0.2, 0.05], [0.2, 0.25]).set;
    set = setPanelMeta(set, { panelId: f.panelId, cutQuantity: 2 });
    const s0 = serializeProductionSet(set);
    expect(serializeProductionSet(deserializeProductionSet(s0))).toBe(s0);
    expect(() => deserializeProductionSet("nope")).toThrowError(/JSON/);
    expect(() => deserializeProductionSet('{"version":9}')).toThrowError(/version/);
  });
});
