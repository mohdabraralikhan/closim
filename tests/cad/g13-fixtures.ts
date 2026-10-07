// Shared G13 fixtures: a production-engineered garment (READY gate) plus a
// deterministic grading context (XS–XXL) built through the G12 grading APIs.
import {
  addAllowance,
  addCutLine,
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  addAnnotation,
  createProductionSet,
  setPanelMeta,
  type ProductionSet,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import type { Seam } from "../../src/garment/sewing.js";
import {
  addRule,
  addGradingPoint,
  createGradingPoint,
  createRule,
  createSize,
  addSize,
  createGradingDocument,
  createMasterPattern,
  createRuleTable,
  createSizeSet,
  createMeasurementDefinition,
  addMeasurementDefinition,
  deriveSize,
  type GradingDocument,
} from "../../src/grading/index.js";
import {
  createPatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import type { ExportGradingContext } from "../../src/cad/export-ir.js";

export interface EngineeredFixture {
  document: PatternDocument;
  seams: Seam[];
  set: ProductionSet;
  refs: { front: { panelId: string; loopId: string; segmentIds: string[] }; back: { panelId: string; loopId: string; segmentIds: string[] } };
}

/**
 * Two-panel top engineered to pass the strict gate: allowances, grainlines,
 * notches, cut lines, quantities, fold, drill, dart, annotation, label.
 */
export function engineeredGarment(): EngineeredFixture {
  let document = createPatternDocument("g13-top", "G13 top");
  const front = addRectPanel(document, "front", [0, 0], 0.46, 0.62);
  document = front.document;
  const back = addRectPanel(document, "back", [0.6, 0], 0.46, 0.62);
  document = back.document;
  const seg = (r: { panelId: string; loopId: string; segmentIds: string[] }, k: number): string => r.segmentIds[k];
  const side = (r: { panelId: string; loopId: string; segmentIds: string[] }, k: number, reversed = false) => ({
    panelId: r.panelId, loopId: r.loopId, segmentIds: [seg(r, k)], reversed,
  });
  const seams: Seam[] = [
    { id: "seam/shoulder", sideA: side(front.refs, 2), sideB: side(back.refs, 2, true), stitchCount: 7 },
    { id: "seam/side-left", sideA: side(front.refs, 3), sideB: side(back.refs, 3, true), stitchCount: 7 },
    { id: "seam/side-right", sideA: side(front.refs, 1), sideB: side(back.refs, 1, true), stitchCount: 7 },
  ];
  let set = createProductionSet();
  for (const r of [front.refs, back.refs]) {
    set = addAllowance(set, r.panelId, r.loopId, 0.01).set;
    const g = centeredGrainline(document, r.panelId);
    set = addGrainline(set, r.panelId, g.from, g.to).set;
    set = addNotch(set, r.panelId, r.loopId, r.segmentIds[0], 0.5, "single", 0.005).set;
    set = addCutLine(set, r.panelId, r.loopId, "allowance").set;
    set = setPanelMeta(set, { panelId: r.panelId, cutQuantity: 2, section: "body" });
  }
  set = addFoldLine(set, front.refs.panelId, [0.1, 0.1], [0.1, 0.5], "valley", "pleat").set;
  set = addDrillMark(set, back.refs.panelId, [0.8, 0.31], "circle").set;
  set = addInternalLine(set, front.refs.panelId, [[0.15, 0.1], [0.25, 0.2]], "dart", "front dart").set;
  set = addAnnotation(set, front.refs.panelId, [0.3, 0.4], "match point").set;
  set = addLabelRegion(set, front.refs.panelId, [0.05, 0.05], [0.4, 0.12], { garment: "G13 top", panel: "front" }).set;
  return {
    document,
    seams,
    set,
    refs: {
      front: { panelId: front.refs.panelId, loopId: front.refs.loopId, segmentIds: front.refs.segmentIds },
      back: { panelId: back.refs.panelId, loopId: back.refs.loopId, segmentIds: back.refs.segmentIds },
    },
  };
}

export interface GradedFixture extends EngineeredFixture {
  grading: ExportGradingContext;
  gradingDoc: GradingDocument;
}

/**
 * Graded context: 6 sizes (XS S M L XL XXL) via per-size rules on the two
 * boundary corners of each panel. Seam-symmetric: both panels' shared edges
 * receive equal deltas so the strict gate's seam-length check still passes.
 */
export function gradedGarment(sizeCount = 6): GradedFixture {
  const base = engineeredGarment();
  let document = base.document;
  const master = createMasterPattern("master/g13", "G13 master", document);
  let doc = createGradingDocument({
    id: "grading/g13",
    name: "G13 grading",
    master,
    sizeSet: createSizeSet("sizeset/g13", "G13 sizes"),
    ruleTable: createRuleTable("ruletable/g13", "G13 rules"),
    seams: base.seams,
  });
  doc = addMeasurementDefinition(doc, createMeasurementDefinition({ id: "meas/chest", name: "chest circumference", ordering: "increasing" }));
  const sizes = Array.from({ length: sizeCount }, (_, i) => `s${String(i).padStart(2, "0")}`);
  const labelPool = ["XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"];
  const labels = sizes.map((_, i) => labelPool[i]);
  sizes.forEach((s, i) => {
    doc = addSize(doc, createSize({
      id: `size/${s}`, label: labels[i], displayName: labels[i],
      measurements: [{ measurementId: "meas/chest", valueM: 0.9 + i * 0.05, unit: "m" as const }],
    }));
  });
  // One grading point per panel: the top-left vertex of each panel's box.
  const step = 0.004; // 4 mm per size step, both panels identically => seams stay matched.
  let idx = 0;
  for (const panelId of [base.refs.front.panelId, base.refs.back.panelId]) {
    const panelPts = document.points.filter((p) => p.panelId === panelId && p.y > 0.6 - 1e-9);
    const minX = Math.min(...panelPts.map((p) => p.x));
    const pt = panelPts.find((p) => Math.abs(p.x - minX) < 1e-12)!;
    const gpId = `gp/shoulder-${idx === 0 ? "front" : "back"}`;
    doc = addGradingPoint(doc, createGradingPoint(gpId, { kind: "point", panelId, pointId: pt.id }));
    const deltas: Record<string, [number, number]> = {};
    sizes.forEach((s, i2) => { deltas[`size/${s}`] = [-(step * i2), 0]; });
    doc = addRule(doc, createRule(`rule/shoulder-${idx === 0 ? "front" : "back"}`, gpId, "per-size", deltas));
    idx++;
  }
  // Derive every size and package the audit surface for the export layer.
  const ruleApplications: ExportGradingContext["ruleApplications"] = {};
  const sizeContext = doc.sizeSet.sizes.map((s, i) => ({
    sizeId: s.id, label: s.label, isBase: i === 0,
    measurements: Object.fromEntries(s.measurements.map((m) => [m.measurementId, m.valueM])),
  }));
  for (const size of doc.sizeSet.sizes) {
    const { report } = deriveSize(doc, size.id);
    ruleApplications[size.id] = report.applied.map((a) => ({
      ruleId: a.ruleId,
      gradingPointId: a.gradingPointId,
      mode: a.mode,
      delta: [a.delta[0], a.delta[1]] as [number, number],
      masterPosition: [a.masterPosition[0], a.masterPosition[1]] as [number, number],
      gradedPosition: [a.gradedPosition[0], a.gradedPosition[1]] as [number, number],
    }));
  }
  void document;
  return {
    ...base,
    gradingDoc: doc,
    grading: {
      sizes: sizeContext,
      baseSizeId: doc.sizeSet.baseSizeId,
      sizeSetId: doc.sizeSet.id,
      sizeSetName: doc.sizeSet.name,
      ruleApplications,
    },
  };
}
