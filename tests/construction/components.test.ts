// G16 component tests: bands, pockets, buttons, registry, regen/invalidation.
import { describe, expect, it } from "vitest";
import { validatePatternDocument } from "../../src/pattern/cad.js";
import {
  addComponent,
  createComponentSet,
  deriveComponents,
  deserializeComponentSet,
  fingerprintDependencies,
  registeredComponentTypes,
  registerComponentDerivation,
  removeComponent,
  serializeComponentSet,
  updateComponentParams,
  validateComponents,
} from "../../src/construction/components.js";
import { validateSeams } from "../../src/garment/sewing.js";
import { rectFixture } from "../cad/fixtures.js";

function valid(doc: Parameters<typeof validatePatternDocument>[0]): void {
  expect(validatePatternDocument(doc).valid).toBe(true);
}

function collarOnBottom() {
  const f = rectFixture(0.4, 0.6);
  let set = createComponentSet();
  const c = addComponent(set, f.panelId, "collar-band", {
    edgeLoopId: f.loopId, edgeSegmentId: f.segments.top, heightM: 0.08, name: "collar",
  }, [f.segments.top, f.loopId, f.panelId]);
  set = c.set;
  return { f, set, id: c.id };
}

describe("G16 components", () => {
  it("derives collar bands with seams, notches, and grainlines", () => {
    const { f, set, id } = collarOnBottom();
    expect(registeredComponentTypes()).toContain("collar-band");
    const out = deriveComponents(f.document, set);
    expect(out.failed).toEqual([]);
    expect(out.applied).toEqual([id]);
    expect(out.document.panels).toHaveLength(2);
    expect(out.seams).toHaveLength(1);
    expect(validateSeams(out.document, out.seams).valid).toBe(true);
    expect(out.notches).toHaveLength(1);
    expect(out.grainlines).toHaveLength(1);
    valid(out.document);
    expect(out.set.components[0].status).toBe("current");
    expect(out.set.components[0].derived.panels).toHaveLength(1);
  });

  it("preserves identity when sources are unchanged", () => {
    const { f, set } = collarOnBottom();
    const first = deriveComponents(f.document, set);
    const panelId = first.set.components[0].derived.panels[0];
    const second = deriveComponents(f.document, first.set);
    expect(second.set.components[0].derived.panels).toEqual([panelId]);
    expect(second.document.panels).toHaveLength(2);
  });

  it("regenerates when sources change and reports seam invalidation", () => {
    const { f, set, id } = collarOnBottom();
    const first = deriveComponents(f.document, set);
    const bandId = first.set.components[0].derived.panels[0];
    // Lengthen the neckline edge: collar must regenerate longer.
    const moved = { ...f.document };
    const topSeg = f.document.segments.find((s) => s.id === f.segments.top)!;
    const endPt = moved.points.find((p) => p.id === topSeg.startPointId)!;
    endPt.x += 0.1;
    const statuses = validateComponents(moved, first.set);
    expect(statuses[0].status).toBe("stale");
    // A caller seam touching the old band is reported on regeneration.
    const seam = {
      id: "seam/caller", sideA: { panelId: bandId, loopId: "l", segmentIds: [] },
      sideB: { panelId: f.panelId, loopId: f.loopId, segmentIds: [] },
    };
    const second = deriveComponents(moved, first.set, [seam]);
    expect(second.invalidatedSeamIds).toEqual(["seam/caller"]);
    // Deterministic re-derivation preserves the panel id (identity kept);
    // the invalidation report above tells seam owners to re-resolve.
    expect(second.set.components[0].derived.panels[0]).toBe(bandId);
    // ...but the geometry follows the longer neckline.
    const widthAfter = Math.max(...second.document.points.filter((p) => p.panelId === bandId).map((p) => p.x));
    const widthBefore = Math.max(...first.document.points.filter((p) => p.panelId === bandId).map((p) => p.x));
    expect(widthAfter).toBeGreaterThan(widthBefore);
    valid(second.document);
    void id;
  });

  it("derives cuffs, pockets, buttons, and buttonholes", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createComponentSet();
    const cuff = addComponent(set, f.panelId, "cuff-band", {
      edgeLoopId: f.loopId, edgeSegmentId: f.segments.bottom, heightM: 0.06, name: "cuff",
    }, [f.segments.bottom]);
    set = cuff.set;
    const pocket = addComponent(set, f.panelId, "patch-pocket", {
      center: [0.2, 0.3], widthM: 0.12, heightM: 0.14, name: "pocket",
    }, [f.panelId]);
    set = pocket.set;
    const button = addComponent(set, f.panelId, "button", { pos: [0.2, 0.4], diameterM: 0.015 }, [f.panelId]);
    set = button.set;
    const hole = addComponent(set, f.panelId, "buttonhole", { pos: [0.25, 0.4], lengthM: 0.02, angleRad: 0 }, [f.panelId]);
    set = hole.set;
    const out = deriveComponents(f.document, set);
    expect(out.failed).toEqual([]);
    expect(out.document.panels).toHaveLength(3); // base + cuff + pocket
    expect(out.drills).toHaveLength(1);
    expect(out.internals.map((i) => i.kind)).toContain("pocket");
    expect(out.internals.map((i) => i.kind)).toContain("buttonhole");
    expect(validateSeams(out.document, out.seams).valid).toBe(true);
    valid(out.document);
  });

  it("validates params, params updates, removal, and unknown types", () => {
    const f = rectFixture();
    let set = createComponentSet();
    expect(() => addComponent(set, f.panelId, "nope", {}, [])).toThrowError(/no derivation/);
    const c = addComponent(set, f.panelId, "button", { pos: [0.1, 0.1], diameterM: 0.01 }, []);
    set = c.set;
    set = updateComponentParams(set, c.id, { diameterM: 0.02 });
    expect(set.components[0].status).toBe("stale");
    expect(() => updateComponentParams(set, "missing", {})).toThrowError(/does not exist/);
    set = removeComponent(set, c.id);
    expect(set.components).toEqual([]);
    expect(() => removeComponent(set, c.id)).toThrowError(/does not exist/);
    // Fingerprints change with dependency geometry.
    const fp1 = fingerprintDependencies(f.document, [f.segments.bottom]);
    const moved = { ...f.document, points: f.document.points.map((p) => ({ ...p })) };
    const bottomSeg = moved.segments.find((s) => s.id === f.segments.bottom)!;
    moved.points.find((p) => p.id === bottomSeg.startPointId)!.x += 0.05;
    expect(fingerprintDependencies(moved, [f.segments.bottom])).not.toBe(fp1);
    expect(fingerprintDependencies(f.document, [f.segments.bottom])).toBe(fp1);
  });

  it("extends through the registry without touching framework code", () => {
    const f = rectFixture();
    registerComponentDerivation("test-annotation-only", (doc, feature) => ({
      document: doc, panels: [], seams: [], folds: [], notches: [], drills: [],
      internals: [], annotations: [{ panelId: feature.panelId, pos: [0.1, 0.1], note: "custom" }],
      grainlines: [],
    }));
    expect(registeredComponentTypes()).toContain("test-annotation-only");
    expect(() => registerComponentDerivation("test-annotation-only", (doc) => ({
      document: doc, panels: [], seams: [], folds: [], notches: [], drills: [],
      internals: [], annotations: [], grainlines: [],
    }))).toThrowError(/already registered/);
    let set = createComponentSet();
    const c = addComponent(set, f.panelId, "test-annotation-only", {}, [f.panelId]);
    set = c.set;
    const out = deriveComponents(f.document, set);
    expect(out.annotations).toHaveLength(1);
    expect(out.applied).toEqual([c.id]);
  });

  it("round-trips component persistence", () => {
    const { set } = collarOnBottom();
    expect(deserializeComponentSet(serializeComponentSet(set))).toEqual(set);
    expect(serializeComponentSet(deserializeComponentSet(serializeComponentSet(set)))).toBe(serializeComponentSet(set));
    expect(() => deserializeComponentSet("nope")).toThrowError(/JSON/);
  });

  it("relocates pockets with stable identity", () => {
    const f = rectFixture(0.4, 0.6);
    let set = createComponentSet();
    const p = addComponent(set, f.panelId, "patch-pocket", {
      center: [0.1, 0.2], widthM: 0.1, heightM: 0.12, name: "pocket",
    }, [f.panelId]);
    set = p.set;
    const first = deriveComponents(f.document, set);
    const panelId = first.set.components[0].derived.panels[0];
    set = updateComponentParams(first.set, p.id, { center: [0.3, 0.4] });
    const second = deriveComponents(f.document, set);
    expect(second.failed).toEqual([]);
    expect(second.set.components[0].derived.panels[0]).toBe(panelId);
    const pts = second.document.points.filter((q) => q.panelId === panelId);
    expect(Math.min(...pts.map((q) => q.x))).toBeCloseTo(0.25, 9);
  });

  it("rejects invalid attachments and prunes deleted components", () => {
    const f = rectFixture();
    let set = createComponentSet();
    const c = addComponent(set, f.panelId, "button", { pos: [0.1, 0.1], diameterM: 0.01 }, ["ghost-id"]);
    set = c.set;
    expect(validateComponents(f.document, set)[0].status).toBe("invalid");
    expect(deriveComponents(f.document, set).failed).toHaveLength(1);
    // Deletion + re-derivation from base leaves no trace of the component.
    const g = rectFixture();
    let set2 = createComponentSet();
    const p = addComponent(set2, g.panelId, "patch-pocket", {
      center: [0.2, 0.15], widthM: 0.1, heightM: 0.12, name: "pocket",
    }, [g.panelId]);
    set2 = p.set;
    const withPocket = deriveComponents(g.document, set2);
    expect(withPocket.document.panels).toHaveLength(2);
    set2 = removeComponent(set2, p.id);
    const pruned = deriveComponents(g.document, set2);
    expect(pruned.document.panels).toHaveLength(1);
    expect(pruned.applied).toEqual([]);
    valid(pruned.document);
  });
});
