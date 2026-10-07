// G14D/E tests: metrics, multi-seed optimization, cost, workspace, export.
import { describe, expect, it } from "vitest";
import {
  addCutItem,
  createCutPlan,
  createFabric,
  createMarker,
  defaultNestingConstraint,
  type CutItem,
  type MarkerPiece,
} from "../../src/marker/model.js";
import { fabricRulesFrom } from "../../src/marker/fabric.js";
import { auditPlacements, nestPieces } from "../../src/marker/nest.js";
import {
  benchmarkNesting,
  compareResults,
  estimateCost,
  measureResult,
  optimizeMarker,
  totalPatternArea,
} from "../../src/marker/optimize.js";
import { MarkerWorkspace } from "../../src/marker/workspace.js";
import {
  exportMarkerDXF,
  exportMarkerJSON,
  exportMarkerPackage,
  importMarkerJSON,
} from "../../src/marker/export.js";

function rectPiece(instanceId: string, w: number, h: number, cutItemId = "item/1"): MarkerPiece {
  return {
    instanceId, cutItemId, panelId: "panel/1", sizeId: "size/m", mirrored: false,
    polygon: [[0, 0], [w, 0], [w, h], [0, h]],
    areaM2: w * h, grainRad: Math.PI / 2,
  };
}

const fabric = () => createFabric({ id: "f", name: "F", widthM: 1.0, lengthM: 5, materialType: "c" });
const items: CutItem[] = [{ id: "item/1", panelId: "panel/1", sizeId: "size/m", quantity: 1, mirror: "allowed" }];

function input(pieces: MarkerPiece[], seed = 0) {
  const rules = fabricRulesFrom(fabric(), defaultNestingConstraint());
  return {
    pieces, items, rules,
    constraint: defaultNestingConstraint(),
    grainRadOf: () => Math.PI / 2,
    options: { seed },
  };
}

describe("G14D metrics and comparison", () => {
  it("computes exact utilization against hand figures", () => {
    const r = nestPieces(input([rectPiece("a#1", 0.4, 0.3), rectPiece("b#1", 0.2, 0.2)]));
    const m = measureResult(r, 0.16);
    expect(m.markerLengthM).toBeCloseTo(0.31, 9);
    expect(m.fabricAreaM2).toBeCloseTo(0.31, 9);
    expect(m.patternAreaM2).toBeCloseTo(0.16, 12);
    expect(m.utilization).toBeCloseTo(0.16 / 0.31, 9);
    expect(m.wasteM2).toBeCloseTo(0.31 - 0.16, 9);
    expect(m.wastePct).toBeCloseTo(1 - 0.16 / 0.31, 9);
    expect(m.placedCount).toBe(2);
    expect(m.unplacedCount).toBe(0);
    expect(totalPatternArea(input([rectPiece("a#1", 0.4, 0.3)]))).toBeCloseTo(0.12, 12);
  });

  it("prefers fully-placed results, then the objective", () => {
    const full = nestPieces(input([rectPiece("a#1", 0.4, 0.3)]));
    const partial = { ...nestPieces(input([rectPiece("a#1", 0.4, 0.3)], 1)), placements: [], placed: [] as never[], unplaced: [{ instanceId: "a#1", reason: "no-position" as const, detail: "x" }] };
    const mf = measureResult(full, 0.12);
    const mp = measureResult(partial, 0.12);
    expect(compareResults({ result: full, metrics: mf }, { result: partial, metrics: mp })).toBeLessThan(0);
    expect(compareResults({ result: partial, metrics: mp }, { result: full, metrics: mf })).toBeGreaterThan(0);
  });

  it("estimates cost only from explicit prices", () => {
    const r = nestPieces(input([rectPiece("a#1", 0.4, 0.3)]));
    const m = measureResult(r, 0.12);
    const c = estimateCost(m, { fabricPricePerM: 10, laborPerMarker: 5, wastePricePerM2: 2 });
    expect(c.fabricCost).toBeCloseTo(m.markerLengthM * 10, 9);
    expect(c.totalCost).toBeCloseTo(c.fabricCost + c.wasteCost + 5, 9);
    expect(c.fabricConsumedM).toBe(m.markerLengthM);
    expect(() => estimateCost(m, { fabricPricePerM: -1, laborPerMarker: 0, wastePricePerM2: 0 })).toThrowError(/finite/);
  });
});

describe("G14D multi-seed optimization", () => {
  const mixed = [
    rectPiece("a#1", 0.5, 0.1), rectPiece("b#1", 0.2, 0.4),
    rectPiece("c#1", 0.3, 0.3), rectPiece("d#1", 0.15, 0.15),
  ];

  it("runs seeds deterministically and keeps the best", () => {
    const once = optimizeMarker(input(mixed), { seeds: [0, 1, 2], strategies: ["area", "width"], objective: "min-length" });
    const twice = optimizeMarker(input(mixed), { seeds: [0, 1, 2], strategies: ["area", "width"], objective: "min-length" });
    expect(once.runs).toHaveLength(6);
    expect(once.best.result.placements).toEqual(twice.best.result.placements);
    for (const run of once.runs) expect(run.result.unplaced).toEqual([]);
    // Best is at least as short as every run.
    for (const run of once.runs) {
      expect(once.best.metrics.markerLengthM).toBeLessThanOrEqual(run.metrics.markerLengthM + 1e-12);
    }
  });

  it("supports weighted objectives and cancellation", () => {
    const weighted = optimizeMarker(input(mixed), {
      seeds: [0, 1], objective: { kind: "weighted", lengthWeight: 0.5 },
    });
    expect(weighted.best).toBeDefined();
    expect(() => optimizeMarker(input(mixed), { seeds: [0], objective: { kind: "weighted", lengthWeight: 2 } })).toThrowError(/weight/);
    let calls = 0;
    const cancelled = optimizeMarker(input(mixed), {
      seeds: [0, 1, 2, 3],
      shouldCancel: () => ++calls > 1,
    });
    expect(cancelled.cancelled).toBe(true);
    expect(cancelled.runs.length).toBeLessThan(4);
    expect(cancelled.runs.length).toBeGreaterThan(0);
  });

  it("benchmarks record time and statistics", () => {
    const record = benchmarkNesting("mixed", input(mixed), 3);
    expect(record.name).toBe("mixed");
    expect(record.elapsedMs).toBeGreaterThanOrEqual(0);
    expect(record.utilization).toBeGreaterThan(0);
    expect(record.placedCount).toBe(4);
    expect(record.iterations).toBeGreaterThan(0);
  });
});

describe("G14E workspace", () => {
  function openWorkspace() {
    const pieces = [rectPiece("a#1", 0.4, 0.3), rectPiece("b#1", 0.2, 0.2)];
    const marker = createMarker("m", "M", fabric(), "cut/1", defaultNestingConstraint(), pieces);
    const rules = fabricRulesFrom(fabric(), defaultNestingConstraint());
    return new MarkerWorkspace(marker, pieces, items, rules, () => Math.PI / 2);
  }

  it("optimizes, previews, and reports metrics", () => {
    const ws = openWorkspace();
    const report = ws.optimize([0, 1], "min-length");
    expect(report.best.result.unplaced).toEqual([]);
    const preview = ws.preview();
    expect(preview.placedIds).toHaveLength(2);
    expect(preview.unplacedIds).toEqual([]);
    expect(preview.metrics.utilization).toBeCloseTo(0.16 / preview.metrics.fabricAreaM2, 9);
    expect(preview.revision).toBe(2);
    expect(ws.audit()).toEqual([]);
  });

  it("rejects invalid manual moves without corrupting state", () => {
    const ws = openWorkspace();
    ws.optimize([0]);
    const before = ws.save();
    const bad = ws.movePlacement("b#1", 0.05, 0.05); // onto piece A
    expect(bad.ok).toBe(false);
    expect(bad.reason).toMatch(/overlap|spacing/);
    expect(ws.save()).toBe(before); // unchanged
    const good = ws.movePlacement("b#1", 0.7, 0.01);
    expect(good.ok).toBe(true);
    expect(ws.audit()).toEqual([]);
  });

  it("gates rotation to the allowed set", () => {
    const ws = openWorkspace();
    ws.optimize([0]);
    expect(ws.rotatePlacement("b#1", 90).ok).toBe(false); // not in [0, 180]
    expect(ws.rotatePlacement("ghost", 0).ok).toBe(false);
    const ok = ws.rotatePlacement("b#1", 180);
    expect(ok.ok).toBe(true);
    expect(ws.audit()).toEqual([]);
  });

  it("supports undo/redo, restore-auto, selection, and save/load", () => {
    const ws = openWorkspace();
    ws.optimize([0]);
    ws.select(["b#1", "ghost"]);
    expect(ws.selection).toEqual(["b#1"]);
    expect(ws.inspect("b#1")?.placement).not.toBeNull();
    expect(ws.inspect("ghost")).toBeNull();
    ws.movePlacement("b#1", 0.7, 0.01);
    expect(ws.undo()).toBe(true);
    expect(ws.redo()).toBe(true);
    expect(ws.restoreAuto("b#1")).toBe(true);
    expect(ws.preview().unplacedIds).toEqual(["b#1"]);
    expect(ws.restoreAuto("b#1")).toBe(false);
    const saved = ws.save();
    ws.optimize([1]);
    ws.load(saved);
    expect(ws.preview().unplacedIds).toEqual(["b#1"]);
    expect(() => ws.load('{"format":"x"}')).toThrowError(/marker package|invalid marker|valid JSON/);
  });

  it("re-audits when nesting settings change", () => {
    const ws = openWorkspace();
    ws.optimize([0]);
    expect(ws.audit()).toEqual([]);
    // Widening spacing past the actual gaps flags violations without moving pieces.
    const problems = ws.setConstraint({ spacingM: 5 });
    expect(problems.length).toBeGreaterThan(0);
    expect(problems[0].code).toMatch(/spacing-violation|overlap|outside-fabric/);
    expect(() => ws.setConstraint({ spacingM: -1 })).toThrowError(/spacing/);
  });
});

describe("G14E marker export", () => {
  function placed() {
    const pieces = [rectPiece("a#1", 0.4, 0.3), rectPiece("b#1", 0.2, 0.2)];
    const marker = createMarker("m", "M", fabric(), "cut/1", defaultNestingConstraint(), pieces);
    const rules = fabricRulesFrom(fabric(), defaultNestingConstraint());
    const ws = new MarkerWorkspace(marker, pieces, items, rules, () => Math.PI / 2);
    ws.optimize([0]);
    return { ws, pieces };
  }

  it("writes JSON packages and DXF with layers", () => {
    const { ws, pieces } = placed();
    const preview = ws.preview();
    const plan = { id: "cut/1", name: "C", gradingId: "g", items } as never;
    const json = exportMarkerJSON(ws.marker, plan, preview.metrics);
    const back = importMarkerJSON(json);
    expect(back.marker.id).toBe("m");
    expect(back.metrics.utilization).toBeCloseTo(preview.metrics.utilization, 12);
    expect(exportMarkerJSON(ws.marker, plan, preview.metrics)).toBe(json); // deterministic
    const dxf = exportMarkerDXF(ws.marker, pieces, preview.metrics);
    expect(dxf.dxf).toContain("MARKER_BOUND");
    expect(dxf.dxf).toContain("PIECES");
    expect(dxf.dxf).toContain("$INSUNITS");
    expect(dxf.entityCount).toBeGreaterThan(8);
    expect(dxf.warnings.some((w) => w.includes("subset"))).toBe(true);
    expect(() => importMarkerJSON("nope")).toThrowError(/parse/);
  });

  it("refuses unplaced quantities and invalid markers", () => {
    const { ws, pieces } = placed();
    const plan = { id: "cut/1", name: "C", gradingId: "g", items } as never;
    ws.restoreAuto("b#1");
    const preview = ws.preview();
    expect(() => exportMarkerDXF(ws.marker, pieces, preview.metrics)).toThrowError(/unplaced/);
    const rules = fabricRulesFrom(fabric(), defaultNestingConstraint());
    expect(() => exportMarkerPackage(ws.marker, pieces, plan, preview.metrics, rules, () => 0)).toThrowError(/unplaced/);
  });
});
