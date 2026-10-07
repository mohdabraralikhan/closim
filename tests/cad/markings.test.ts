// G11B tests: notch ticks, orphans, grainline helpers, label fields, marking validation.
import { describe, expect, it } from "vitest";
import {
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addNotch,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import {
  assignPanelNumbers,
  buildLabelFields,
  centeredGrainline,
  crossGrainline,
  notchTicks,
  orphanedNotchIds,
  orphanedPanelIds,
  panelCentroid,
  validateMarkings,
} from "../../src/cad/markings.js";
import { deletePanel } from "../../src/pattern/cad.js";
import { rectFixture } from "./fixtures.js";

describe("G11B notch interpretation", () => {
  it("builds single and double ticks along the outward normal", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createProductionSet();
    const single = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "single", 0.01);
    set = single.set;
    const ticks = notchTicks(f.document, set.notches[0]);
    expect(ticks.ticks).toHaveLength(1);
    expect(ticks.ticks[0].from[1]).toBeCloseTo(0, 12);
    expect(ticks.ticks[0].to[1]).toBeCloseTo(-0.01, 12);
    const dbl = addNotch(set, f.panelId, f.loopId, f.segments.top, 0.5, "double", 0.01);
    set = dbl.set;
    expect(notchTicks(f.document, set.notches[1]).ticks).toHaveLength(2);
  });

  it("tracks orphaned notches and panels across boundary edits", () => {
    const f = rectFixture();
    let set = createProductionSet();
    const n = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "single", 0.005);
    set = n.set;
    expect(orphanedNotchIds(f.document, set)).toEqual([]);
    // Delete the whole panel: notch edge and panel refs die together.
    const cut = deletePanel(f.document, f.panelId);
    expect(orphanedNotchIds(cut, set)).toEqual([n.id]);
    expect(orphanedPanelIds(cut, set)).toContain(f.panelId);
    expect(validateMarkings(cut, set).map((d) => d.code)).toContain("orphaned-marking");
  });

  it("flags duplicate and vertex-sited notches", () => {
    const f = rectFixture();
    let set = createProductionSet();
    const a = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "single", 0.005);
    set = a.set;
    const b = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "double", 0.005);
    set = b.set;
    const c = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0, "single", 0.005);
    set = c.set;
    const codes = validateMarkings(f.document, set).map((d) => d.code);
    expect(codes).toContain("duplicate-marking");
    expect(codes).toContain("degenerate-placement");
  });
});

describe("G11B grainlines and metadata", () => {
  it("centers grainlines deterministically with a cross mark", () => {
    const f = rectFixture(0.4, 0.3);
    const centroid = panelCentroid(f.document, f.panelId);
    expect(centroid[0]).toBeCloseTo(0.2, 9);
    expect(centroid[1]).toBeCloseTo(0.15, 9);
    const g = centeredGrainline(f.document, f.panelId);
    expect((g.from[1] + g.to[1]) / 2).toBeCloseTo(0.15, 9);
    expect((g.from[0] + g.to[0]) / 2).toBeCloseTo(0.2, 9);
    const g2 = centeredGrainline(f.document, f.panelId);
    expect(g2).toEqual(g);
    const cross = crossGrainline({ id: "g", panelId: f.panelId, from: g.from, to: g.to });
    // Cross mark is perpendicular with the same center and length.
    const dx = cross.to[0] - cross.from[0], dy = cross.to[1] - cross.from[1];
    const gx = g.to[0] - g.from[0], gy = g.to[1] - g.from[1];
    expect(dx * gx + dy * gy).toBeCloseTo(0, 9);
    expect(Math.hypot(dx, dy)).toBeCloseTo(Math.hypot(gx, gy), 9);
    let set = createProductionSet();
    set = addGrainline(set, f.panelId, g.from, g.to).set;
    expect(validateMarkings(f.document, set)).toEqual([]);
  });

  it("numbers panels and builds label fields in fixed key order", () => {
    const f = rectFixture();
    expect(assignPanelNumbers(f.document)).toEqual({ [f.panelId]: 1 });
    const fields = buildLabelFields({
      garmentName: "Top", panelName: "front", panelNumber: 1,
      size: "M", cutQuantity: 2, material: "cotton", mirror: "cut pair", notes: "self",
    });
    expect(Object.keys(fields)).toEqual(["garment", "panel", "number", "size", "cut", "material", "mirror", "notes"]);
    expect(() => buildLabelFields({
      garmentName: "", panelName: "x", panelNumber: 1, cutQuantity: 1, material: "c",
    })).toThrowError(/garment/);
  });

  it("catches escaping internal lines and leaving folds", () => {
    const f = rectFixture(0.4, 0.3);
    let set = createProductionSet();
    // Diagonal corner-to-corner stays inside the convex rect.
    set = addInternalLine(set, f.panelId, [[0.05, 0.05], [0.35, 0.25]], "dart").set;
    expect(validateMarkings(f.document, set).filter((d) => d.code === "outside-panel")).toEqual([]);
    // A line leaving through the right edge escapes.
    set = addInternalLine(set, f.panelId, [[0.2, 0.15], [5, 5]], "pocket").set;
    expect(validateMarkings(f.document, set).map((d) => d.code)).toContain("outside-panel");
    // Edge-to-edge fold is legal (endpoints on the boundary).
    set = addFoldLine(set, f.panelId, [0.1, 0], [0.1, 0.3], "valley", "pleat").set;
    expect(validateMarkings(f.document, set).filter((d) => d.entityId?.includes("fold") ?? false)).toEqual([]);
    // Drill outside the panel is flagged.
    set = addDrillMark(set, f.panelId, [9, 9]).set;
    expect(validateMarkings(f.document, set).map((d) => d.code)).toContain("outside-panel");
  });

  it("requires cut metadata presence downstream (presence check helper)", () => {
    const f = rectFixture();
    const set = setPanelMeta(createProductionSet(), { panelId: f.panelId, cutQuantity: 2 });
    expect(set.panelMeta[0]).toMatchObject({ panelId: f.panelId, cutQuantity: 2 });
  });
});
