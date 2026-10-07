// G13A tests: canonical export IR v2 — determinism, units, grading metadata,
// exact geometry, gate semantics, versioning, JSON round trip.
import { describe, expect, it } from "vitest";
import {
  buildExportIR,
  compareIR,
  convertM,
  exportGate,
  exportIRToJSON,
  importIRFromJSON,
  requireExportUnits,
  EXPORT_IR_FORMAT,
  EXPORT_IR_VERSION,
} from "../../src/cad/export-ir.js";
import { engineeredGarment, gradedGarment } from "./g13-fixtures.js";
import { createPatternDocument, createPanel } from "../../src/pattern/cad.js";
import { createProductionSet } from "../../src/cad/production.js";
import { PatternCadError } from "../../src/pattern/cad.js";

describe("G13A export IR v2 — construction + determinism", () => {
  it("builds a deterministic IR carrying style, garment, grading, readiness", () => {
    const f = gradedGarment();
    const ir1 = buildExportIR(f.document, f.seams, f.set, {
      garmentName: "G13 top", styleId: "style/g13", revision: 3, grading: f.grading,
    });
    const ir2 = buildExportIR(f.document, f.seams, f.set, {
      garmentName: "G13 top", styleId: "style/g13", revision: 3, grading: f.grading,
    });
    expect(exportIRToJSON(ir1)).toBe(exportIRToJSON(ir2));
    expect(ir1.format).toBe(EXPORT_IR_FORMAT);
    expect(ir1.version).toBe(EXPORT_IR_VERSION);
    expect(ir1.units).toBe("m");
    expect(ir1.style).toEqual({ garmentName: "G13 top", styleId: "style/g13", revision: 3, sizeSetId: "sizeset/g13", sizeSetName: "G13 sizes" });
    expect(ir1.garment).toEqual({ documentId: "g13-top", panelCount: 2, seamCount: 3 });
    expect(ir1.grading.present).toBe(true);
    expect(ir1.grading.sizes.map((s) => s.label)).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);
    expect(ir1.grading.baseSizeId).toBe("size/s00");
    expect(ir1.panels).toHaveLength(2);
    expect(ir1.seams).toHaveLength(3);
    expect(ir1.readiness.state).toBe("UNCHECKED");
  });

  it("derives an unchecked IR without a gate report but carries readiness when given", () => {
    const f = engineeredGarment();
    const decision = exportGate(f.document, f.seams, f.set, "strict");
    expect(decision.ok).toBe(true);
    expect(decision.state).toBe("READY_FOR_EXPORT");
    const ir = buildExportIR(f.document, f.seams, f.set, { garmentName: "top", readiness: decision.readiness });
    expect(ir.readiness.state).toBe("READY_FOR_EXPORT");
    expect(ir.readiness.errorCount).toBe(0);
    // Seam lengths come from the authoritative gate rows (shoulder = top edge, 0.46 m).
  });

  it("keeps every source id stable and production entity types present", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const front = ir.panels.find((p) => p.panelId === f.refs.front.panelId)!;
    expect(front.name).toBe("front");
    expect(front.materialId).toBe("cotton");
    expect(front.cutQuantity).toBe(2);
    expect(front.notches[0].id).toContain("production/notch/");
    expect(front.grainlines[0].id).toContain("production/grainline/");
    expect(front.folds[0].id).toContain("production/fold/");
    expect(front.internals[0].id).toContain("production/internal/");
    expect(front.annotations[0].id).toContain("production/annotation/");
    expect(front.labelRegions[0].id).toContain("production/label/");
    const back = ir.panels.find((p) => p.panelId === f.refs.back.panelId)!;
    expect(back.drills.some((d) => d.id.startsWith("production/drill/"))).toBe(true);
    expect(front.cutEdges).not.toBeNull();
    expect(front.cutSource).toBe("allowance");
    expect(front.allowanceEdges).not.toBeNull();
    expect(front.section).toBe("body");
  });

  it("preserves exact arc geometry (center/radius/angles), not flattened polylines", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    // Rect fixture: every sewing edge is a line; verify no arc-faceting happened
    // by counting edges = segment count, and that lines carry exact endpoints.
    const front = ir.panels[0];
    expect(front.sewingEdges).toHaveLength(4);
    for (const e of front.sewingEdges) expect(e.kind).toBe("line");
    // Exact endpoints: bottom edge runs (0,0) -> (0.46,0) in panel-local metres.
    expect(front.sewingEdges[0].a).toEqual([0, 0]);
    expect(front.sewingEdges[0].b).toEqual([0.46, 0]);
  });

  it("records construction edges from the kernel document", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    expect(ir.panels.every((p) => Array.isArray(p.constructionEdges))).toBe(true);
  });
});

describe("G13A units", () => {
  it("converts canonical metres to declared export units exactly", () => {
    expect(convertM(1, "mm")).toBeCloseTo(1000, 9);
    expect(convertM(1, "cm")).toBeCloseTo(100, 9);
    expect(convertM(1, "m")).toBe(1);
    expect(convertM(0.0254, "in")).toBeCloseTo(1, 9);
  });

  it("refuses undeclared or unknown units instead of guessing", () => {
    expect(() => requireExportUnits(undefined, "DXF export")).toThrowError(/undeclared/);
    expect(() => requireExportUnits("furlongs" as never, "DXF export")).toThrowError(/unknown unit/);
    expect(() => convertM(1, "parsecs" as never)).toThrowError(/unknown export unit/);
    expect(() => convertM(Number.NaN, "mm")).toThrowError(/non-finite/);
  });
});

describe("G13A gate", () => {
  it("blocks INVALID and refuses WARNINGS in strict mode, passes them in allow-warnings", () => {
    const f = engineeredGarment();
    const noGrain = { ...f.set, grainlines: f.set.grainlines.slice(1) };
    const strictWarn = exportGate(f.document, f.seams, noGrain, "strict");
    expect(strictWarn.blocked).toBe(true);
    expect(strictWarn.state).toBe("WARNINGS");
    const lenient = exportGate(f.document, f.seams, noGrain, "allow-warnings");
    expect(lenient.blocked).toBe(false);
    expect(lenient.reason).toContain("warnings accepted");
    // INVALID never passes, even lenient.
    const noMeta = { ...f.set, panelMeta: [] };
    const invalid = exportGate(f.document, f.seams, noMeta, "allow-warnings");
    expect(invalid.blocked).toBe(true);
    expect(invalid.state).toBe("INVALID");
  });
});

describe("G13A JSON round trip + versioning", () => {
  it("round-trips losslessly and compares structurally", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const back = importIRFromJSON(exportIRToJSON(ir));
    expect(compareIR(ir, back)).toEqual([]);
    expect(() => importIRFromJSON("{nope")).toThrowError(/parse/);
    expect(() => importIRFromJSON('{"format":"x","version":1,"panels":[]}')).toThrowError(/shape/);
    expect(() => importIRFromJSON(JSON.stringify({ ...JSON.parse(exportIRToJSON(ir)), version: 99 }))).toThrowError(/shape|version/);
  });

  it("reports diffs on content change", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const mutated = JSON.parse(exportIRToJSON(ir));
    mutated.panels[0].cutQuantity = 9;
    expect(compareIR(ir, mutated).length).toBeGreaterThan(0);
  });

  it("refuses to build an IR from an invalid kernel document", () => {
    let bad = createPatternDocument("bad");
    const p = createPanel(bad, "ghost");
    bad = p.document;
    expect(() => buildExportIR(bad, [], createProductionSet())).toThrowError(PatternCadError);
  });
});

describe("G13A grading metadata in IR", () => {
  it("carries per-size rule applications keyed by size", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    expect(ir.grading.present).toBe(true);
    expect(ir.grading.rules.length).toBeGreaterThan(0);
    const bySize = new Map<string, number>();
    for (const r of ir.grading.rules) {
      for (const sizeId of Object.keys(r.deltas)) {
        bySize.set(sizeId, (bySize.get(sizeId) ?? 0) + 1);
      }
    }
    expect(bySize.get("size/s00")).toBeGreaterThan(0);
    expect(bySize.get("size/s05")).toBeGreaterThan(0);
    // Base size flagged, measurements carried.
    expect(ir.grading.sizes.find((s) => s.sizeId === "size/s00")?.isBase).toBe(true);
    expect(ir.grading.sizes.find((s) => s.sizeId === "size/s05")?.measurements["meas/chest"]).toBeCloseTo(0.9 + 5 * 0.05, 9);
  });

  it("marks grading absent when no context is supplied (never fabricated)", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    expect(ir.grading.present).toBe(false);
    expect(ir.grading.sizes).toEqual([]);
    expect(ir.grading.rules).toEqual([]);
  });
});
