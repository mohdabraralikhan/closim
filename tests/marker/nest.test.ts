// G14B/C tests: rules, orientation, engine validity, determinism, failures.
import { describe, expect, it } from "vitest";
import {
  addCutItem,
  createCutPlan,
  createFabric,
  createMarker,
  defaultNestingConstraint,
  expandCutPlan,
  type CutItem,
  type MarkerPiece,
} from "../../src/marker/model.js";import {
  checkNestingFeasibility,
  fabricRulesFrom,
  grainAligned,
  orientPiece,
  pieceGrainDeg,
  resolvePieceRules,
  type RuleProblem,
} from "../../src/marker/fabric.js";
import {
  auditPlacements,
  nestPieces,
  polygonGap,
  polygonsOverlap,
} from "../../src/marker/nest.js";
import {
  addCoreGradingPoints,
  addCoreRules,
  addCoreSizes,
  buildGradingFixture,
} from "../grading/fixtures.js";

function rectPiece(instanceId: string, w: number, h: number, cutItemId = "item/1"): MarkerPiece {
  return {
    instanceId, cutItemId, panelId: "panel/1", sizeId: "size/m", mirrored: false,
    polygon: [[0, 0], [w, 0], [w, h], [0, h]],
    areaM2: w * h, grainRad: Math.PI / 2,
  };
}

function item(id: string, mirror: CutItem["mirror"] = "allowed", rotationsDeg?: number[]): CutItem {
  return { id, panelId: "panel/1", sizeId: "size/m", quantity: 1, mirror, ...(rotationsDeg ? { rotationsDeg } : {}) };
}

const fabric = () => createFabric({ id: "f", name: "F", widthM: 1.0, lengthM: 5, materialType: "c" });

function rules(constraint = defaultNestingConstraint(), opts: Parameters<typeof fabricRulesFrom>[2] = {}) {
  return fabricRulesFrom(fabric(), constraint, opts);
}

describe("G14C rules", () => {
  it("computes usable width and validates selvedge", () => {
    const r = fabricRulesFrom(fabric(), defaultNestingConstraint(), { selvedgeM: 0.05 });
    expect(r.usableWidthM).toBeCloseTo(0.9, 12);
    expect(() => fabricRulesFrom(fabric(), defaultNestingConstraint(), { selvedgeM: 1 })).toThrowError(/usable width/);
  });

  it("resolves precedence: item rotations beat fabric defaults", () => {
    const r = rules();
    const pr = resolvePieceRules(r, item("i", "allowed", [0]), "inst");
    expect(pr.rotationsDeg).toEqual([0]);
    const pr2 = resolvePieceRules(r, item("i2"), "inst2");
    expect(pr2.rotationsDeg).toEqual([0, 180]);
    expect(pr2.mirrorVariants).toEqual([false, true]); // as-authored default first
  });

  it("never infers mirroring from rotation", () => {
    const r = rules();
    expect(resolvePieceRules(r, item("i", "forbidden"), "x").mirrorVariants).toEqual([false]);
    expect(resolvePieceRules(r, item("i", "required"), "x").mirrorVariants).toEqual([true]);
  });

  it("restricts directional fabric to the 0/180 class", () => {
    const r = rules(defaultNestingConstraint(), { directional: true });
    const problems: RuleProblem[] = [];
    const pr = resolvePieceRules({ ...r, defaultRotationsDeg: [0, 90, 180, 270] }, item("i"), "x", problems);
    expect(pr.rotationsDeg).toEqual([0, 180]);
    const problems2: RuleProblem[] = [];
    resolvePieceRules({ ...r, defaultRotationsDeg: [90] }, item("i"), "x", problems2);
    expect(problems2.map((p) => p.code)).toContain("incompatible-rotation");
  });

  it("checks grain alignment modulo 180", () => {
    expect(grainAligned(0, 180, 1)).toBe(true);
    expect(grainAligned(0, 90, 5)).toBe(false);
    expect(pieceGrainDeg(Math.PI / 2, 90, false)).toBeCloseTo(180, 9);
    expect(pieceGrainDeg(Math.PI / 2, 0, true)).toBeCloseTo(90, 9);
  });

  it("detects too-wide pieces before nesting", () => {
    const r = rules();
    const pieces = [rectPiece("wide#1", 2.0, 0.1)];
    const problems = checkNestingFeasibility(pieces, r, [item("item/1")], () => Math.PI / 2, 0.01);
    expect(problems.map((p) => p.code)).toContain("too-wide");
  });
});

describe("G14B predicates", () => {
  it("detects overlap but allows touching", () => {
    const a = [[0, 0], [1, 0], [1, 1], [0, 1]] as Array<[number, number]>;
    const b = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]] as Array<[number, number]>;
    const c = [[1, 0], [2, 0], [2, 1], [1, 1]] as Array<[number, number]>;
    expect(polygonsOverlap(a, b)).toBe(true);
    expect(polygonsOverlap(a, c)).toBe(false); // shared edge only
    expect(polygonGap(a, c)).toBeCloseTo(0, 12);
    expect(polygonGap(a, [[2, 0], [3, 0], [3, 1], [2, 1]] as Array<[number, number]>)).toBeCloseTo(1, 12);
  });

  it("orients pieces deterministically", () => {
    const p = rectPiece("r#1", 0.4, 0.2);
    const r = rules();
    const pr = resolvePieceRules(r, item("item/1"), "r#1");
    const os = orientPiece(p, pr, p.grainRad);
    // 2 rotations x 2 mirror variants, rotation order first.
    expect(os.map((o) => [o.rotationDeg, o.mirrored])).toEqual([
      [0, false], [0, true], [180, false], [180, true],
    ]);
    expect(orientPiece(p, pr, p.grainRad)).toEqual(os);
  });
});

describe("G14B engine", () => {
  function nest(rects: Array<[string, number, number]>, widthM = 1.0, seed = 0, strategy?: "area" | "perimeter" | "width") {
    const pieces = rects.map(([id, w, h]) => rectPiece(id, w, h));
    const items = [item("item/1")];
    return nestPieces({
      pieces, items,
      rules: fabricRulesFrom(createFabric({ id: "f", name: "F", widthM, lengthM: 5, materialType: "c" }), defaultNestingConstraint()),
      constraint: defaultNestingConstraint(),
      grainRadOf: () => Math.PI / 2,
      options: { seed, ...(strategy ? { strategy } : {}) },
    });
  }

  it("packs rectangles bottom-left with exact utilization", () => {
    const r = nest([["a#1", 0.4, 0.3], ["b#1", 0.2, 0.2]]);
    expect(r.unplaced).toEqual([]);
    expect(r.placements).toHaveLength(2);
    const a = r.placed.find((p) => p.instanceId === "a#1")!;
    expect([a.minX, a.minY]).toEqual([0.01, 0.01]); // margin origin
    expect(r.markerLengthM).toBeCloseTo(0.31, 9);
    expect(r.utilization).toBeCloseTo(0.16 / (1.0 * 0.31), 9);
    // Deterministic rerun.
    expect(nest([["a#1", 0.4, 0.3], ["b#1", 0.2, 0.2]]).placements).toEqual(r.placements);
  });

  it("packs concave pieces without overlap", () => {
    const ell: MarkerPiece = {
      instanceId: "l#1", cutItemId: "item/1", panelId: "panel/1", sizeId: "size/m", mirrored: false,
      polygon: [[0, 0], [0.4, 0], [0.4, 0.2], [0.2, 0.2], [0.2, 0.4], [0, 0.4]],
      areaM2: 0.12, grainRad: Math.PI / 2,
    };
    const r = nestPieces({
      pieces: [ell, rectPiece("r#1", 0.2, 0.2)],
      items: [item("item/1")],
      rules: rules(),
      constraint: defaultNestingConstraint(),
      grainRadOf: () => Math.PI / 2,
      options: {},
    });
    expect(r.unplaced).toEqual([]);
    const marker = createMarker("m", "M", fabric(), "cut/1", defaultNestingConstraint(), [ell, rectPiece("r#1", 0.2, 0.2)]);
    expect(auditPlacements({ ...marker, placements: r.placements }, [ell, rectPiece("r#1", 0.2, 0.2)], rules(), () => Math.PI / 2)).toEqual([]);
  });

  it("respects spacing between pieces", () => {
    const pieces = [rectPiece("a#1", 0.2, 0.1), rectPiece("b#1", 0.2, 0.1)];
    const constraint = defaultNestingConstraint({ spacingM: 0.05 });
    const r = nestPieces({ pieces, items: [item("item/1")], rules: rules(constraint), constraint, grainRadOf: () => 0, options: {} });
    expect(r.unplaced).toEqual([]);
    const [pa, pb] = [r.placed[0].polygon, r.placed[1].polygon];
    expect(polygonGap(pa, pb)).toBeGreaterThanOrEqual(0.05 - 1e-9);
    // Impossible spacing + capped length fails structurally (open length
    // alone would stack forever, so the cap forces the verdict).
    const tight = nestPieces({
      pieces, items: [item("item/1")],
      rules: fabricRulesFrom(createFabric({ id: "f", name: "F", widthM: 0.5, lengthM: 5, materialType: "c" }), defaultNestingConstraint()),
      constraint: defaultNestingConstraint({ spacingM: 0.5 }),
      grainRadOf: () => 0, options: { lengthLimitM: 0.2 },
    });
    expect(tight.unplaced.length).toBeGreaterThan(0);
    expect(tight.unplaced[0].reason).toBe("over-length");
  });

  it("reports too-wide pieces and honours strategies and seeds", () => {
    const narrow = nest([["a#1", 0.4, 0.3]], 0.1);
    expect(narrow.placements).toEqual([]);
    expect(narrow.unplaced[0].reason).toBe("incompatible-rules");
    const mixed = [["a#1", 0.5, 0.1], ["b#1", 0.2, 0.4], ["c#1", 0.3, 0.3]] as Array<[string, number, number]>;
    const byArea = nest(mixed, 1.0, 0, "area");
    const byWidth = nest(mixed, 1.0, 0, "width");
    for (const r of [byArea, byWidth]) expect(r.unplaced).toEqual([]);
    expect(byArea.strategy).toBe("area");
    expect(byWidth.strategy).toBe("width");
    expect(nest(mixed, 1.0, 7, "area").placements).toEqual(nest(mixed, 1.0, 7, "area").placements);
  });

  it("enforces grain restrictions", () => {
    const pieces = [rectPiece("a#1", 0.4, 0.2)];
    const constraint = defaultNestingConstraint({ grainToleranceDeg: 5 });
    // Piece grain is +X (0°); fabric axis is +Y (90°): misaligned everywhere.
    const r = nestPieces({
      pieces, items: [item("item/1")], rules: rules(constraint),
      constraint, grainRadOf: () => 0, options: {},
    });
    expect(r.placements).toEqual([]);
    expect(r.unplaced).toHaveLength(1);
  });

  it("nests the graded two-panel garment across sizes", () => {
    const g = buildGradingFixture();
    let doc = addCoreSizes(g.doc);
    doc = addCoreGradingPoints(doc, g.ids);
    doc = addCoreRules(doc);
    let plan = createCutPlan("cut/g", "Graded", doc.id);
    const fa = addCutItem(plan, { panelId: g.ids.frontPanel, sizeId: "size/m", quantity: 1, mirror: "allowed" });
    plan = fa.plan;
    const fb = addCutItem(plan, { panelId: g.ids.backPanel, sizeId: "size/s", quantity: 1, mirror: "allowed" });
    plan = fb.plan;
    const pieces = expandCutPlan(doc, plan);
    expect(pieces).toHaveLength(2);
    const grainRadOf = (panelId: string): number =>
      doc.master.document.panels.find((p) => p.id === panelId)!.grainAngleRad;
    const wide = createFabric({ id: "f", name: "F", widthM: 1.5, lengthM: 5, materialType: "c" });
    const r = nestPieces({
      pieces, items: plan.items, rules: fabricRulesFrom(wide, defaultNestingConstraint()),
      constraint: defaultNestingConstraint(), grainRadOf, options: {},
    });
    expect(r.unplaced).toEqual([]);
    expect(r.utilization).toBeGreaterThan(0);
    expect(r.utilization).toBeLessThanOrEqual(1);
    // Source pattern untouched by nesting preparation.
    expect(doc.master.document.panels).toHaveLength(2);
  });
});
