// Shared G12A grading fixtures: two seamed rectangle panels plus an
// arc-edged panel for curved-boundary cases. All entity references are the
// stable CAD IDs returned by the CAD creation ops.
import {
  createBoundaryArc,
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import type { Seam } from "../../src/garment/sewing.js";
import {
  addRule,
  addSize,
  addGradingPoint,
  createGradingDocument,
  createGradingPoint,
  createMasterPattern,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  type GradingDocument,
} from "../../src/grading/index.js";

export const SEAM_ID = "seam/side";

export interface TwoPanelFixture {
  document: PatternDocument;
  ids: {
    frontPanel: string;
    backPanel: string;
    frontLoop: string;
    backLoop: string;
    points: Record<"A" | "B" | "C" | "D" | "E" | "F" | "G" | "H", string>;
    segments: Record<"ab" | "bc" | "cd" | "da" | "ef" | "fg" | "gh" | "he", string>;
  };
}

/**
 * Front panel: A(0,0) B(0.4,0) C(0.4,0.6) D(0,0.6), ccw.
 * Back panel:  E(0.4,0) F(0.8,0) G(0.8,0.6) H(0.4,0.6), ccw.
 * Seam joins front B→C to back H→E.
 */
export function buildTwoPanelDocument(): TwoPanelFixture {
  let document = createPatternDocument("doc/g12fixture", "Grading fixture");
  const points: TwoPanelFixture["ids"]["points"] = {} as never;
  const segments: TwoPanelFixture["ids"]["segments"] = {} as never;

  const addPoints = (panelId: string, defs: Array<["A" | "B" | "C" | "D" | "E" | "F" | "G" | "H", [number, number]]>): void => {
    for (const [key, position] of defs) {
      const point = createPoint(document, panelId, position);
      document = point.document;
      points[key] = point.pointId;
    }
  };
  const addLines = (panelId: string, loopId: string, defs: Array<["ab" | "bc" | "cd" | "da" | "ef" | "fg" | "gh" | "he", "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H", "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H"]>): void => {
    for (const [key, start, end] of defs) {
      const segment = createBoundaryLine(document, panelId, loopId, points[start], points[end]);
      document = segment.document;
      segments[key] = segment.segmentId;
    }
  };

  let r = createPanel(document, "Front");
  document = r.document;
  const frontPanel = r.panelId;
  const frontLoop = createBoundaryLoop(document, frontPanel, "outer");
  document = frontLoop.document;
  addPoints(frontPanel, [["A", [0, 0]], ["B", [0.4, 0]], ["C", [0.4, 0.6]], ["D", [0, 0.6]]]);
  addLines(frontPanel, frontLoop.loopId, [["ab", "A", "B"], ["bc", "B", "C"], ["cd", "C", "D"], ["da", "D", "A"]]);

  r = createPanel(document, "Back");
  document = r.document;
  const backPanel = r.panelId;
  const backLoop = createBoundaryLoop(document, backPanel, "outer");
  document = backLoop.document;
  addPoints(backPanel, [["E", [0.4, 0]], ["F", [0.8, 0]], ["G", [0.8, 0.6]], ["H", [0.4, 0.6]]]);
  addLines(backPanel, backLoop.loopId, [["ef", "E", "F"], ["fg", "F", "G"], ["gh", "G", "H"], ["he", "H", "E"]]);

  return { document, ids: { frontPanel, backPanel, frontLoop: frontLoop.loopId, backLoop: backLoop.loopId, points, segments } };
}

export function buildSideSeam(fixture: TwoPanelFixture): Seam {
  return {
    id: SEAM_ID,
    sideA: { panelId: fixture.ids.frontPanel, loopId: fixture.ids.frontLoop, segmentIds: [fixture.ids.segments.bc], reversed: false },
    sideB: { panelId: fixture.ids.backPanel, loopId: fixture.ids.backLoop, segmentIds: [fixture.ids.segments.he], reversed: true },
    stitchCount: 3,
  };
}

/** Arc panel: A(0,0) B(0.4,0) arc→C(0.4,0.8) D(0,0.8), ccw; arc center O(0.4,0.4). */
export function buildArcPanelDocument(): {
  document: PatternDocument;
  ids: { panel: string; loop: string; points: Record<"A" | "B" | "C" | "D" | "O", string>; segments: Record<"ab" | "arc" | "cd" | "da", string> };
} {
  let document = createPatternDocument("doc/g12arc", "Arc fixture");
  const points = {} as Record<"A" | "B" | "C" | "D" | "O", string>;
  const segments = {} as Record<"ab" | "arc" | "cd" | "da", string>;
  const panelResult = createPanel(document, "Arc panel");
  document = panelResult.document;
  const panel = panelResult.panelId;
  const loopResult = createBoundaryLoop(document, panel, "outer");
  document = loopResult.document;
  const loop = loopResult.loopId;
  const pointDefs: Array<["A" | "B" | "C" | "D" | "O", [number, number]]> = [
    ["A", [0, 0]], ["B", [0.4, 0]], ["O", [0.4, 0.4]], ["C", [0.4, 0.8]], ["D", [0, 0.8]],
  ];
  for (const [key, position] of pointDefs) {
    const point = createPoint(document, panel, position, key === "O" ? "construction" : "boundary");
    document = point.document;
    points[key] = point.pointId;
  }
  const ab = createBoundaryLine(document, panel, loop, points.A, points.B);
  document = ab.document;
  segments.ab = ab.segmentId;
  const arc = createBoundaryArc(document, panel, loop, points.B, points.C, points.O, Math.PI);
  document = arc.document;
  segments.arc = arc.segmentId;
  const cd = createBoundaryLine(document, panel, loop, points.C, points.D);
  document = cd.document;
  segments.cd = cd.segmentId;
  const da = createBoundaryLine(document, panel, loop, points.D, points.A);
  document = da.document;
  segments.da = da.segmentId;
  return { document, ids: { panel, loop, points, segments } };
}

export interface GradingFixture extends TwoPanelFixture {
  doc: GradingDocument;
  seam: Seam;
}

/** Empty grading document (no sizes, no grading points) over the two-panel master. */
export function buildGradingFixture(): GradingFixture {
  const fixture = buildTwoPanelDocument();
  const seam = buildSideSeam(fixture);
  const master = createMasterPattern("master/g12", "G12 test master", fixture.document);
  const doc = createGradingDocument({
    id: "grading/g12",
    name: "G12 grading",
    master,
    sizeSet: createSizeSet("sizeset/core", "Core sizes"),
    ruleTable: createRuleTable("ruletable/core", "Core rules"),
    seams: [seam],
  });
  return { ...fixture, doc, seam };
}

/** Adds active sizes S, M, L (in order) — S becomes the base size. */
export function addCoreSizes(doc: GradingDocument): GradingDocument {
  let next = addSize(doc, createSize({ id: "size/s", label: "S", displayName: "Small" }));
  next = addSize(next, createSize({ id: "size/m", label: "M", displayName: "Medium" }));
  next = addSize(next, createSize({ id: "size/l", label: "L", displayName: "Large" }));
  return next;
}

/** Standard grading points: neck (point), hem (corner), underarm (edge midpoint), centre-back (seam endpoint). */
export function addCoreGradingPoints(doc: GradingDocument, ids: TwoPanelFixture["ids"]): GradingDocument {
  let next = doc;
  next = addGradingPoint(next, createGradingPoint("gp/neck", { kind: "point", panelId: ids.frontPanel, pointId: ids.points.C }));
  next = addGradingPoint(next, createGradingPoint("gp/hem-side", { kind: "corner", panelId: ids.frontPanel, segmentIdA: ids.segments.ab, segmentIdB: ids.segments.bc }));
  next = addGradingPoint(next, createGradingPoint("gp/underarm", { kind: "edge-relative", panelId: ids.frontPanel, segmentId: ids.segments.cd, t: 0.5 }));
  next = addGradingPoint(next, createGradingPoint("gp/cb", { kind: "seam-point", seamId: SEAM_ID, side: "b", segmentId: ids.segments.he, endpoint: "start" }));
  return next;
}

export function addCoreRules(doc: GradingDocument): GradingDocument {
  let next = doc;
  next = addRule(next, createRule("rule/neck", "gp/neck", "per-size", { "size/s": [0, -0.01], "size/m": [0, -0.02], "size/l": [0, -0.03] }));
  next = addRule(next, createRule("rule/hem", "gp/hem-side", "transition", { "size/s": [0.01, 0], "size/m": [0.01, 0], "size/l": [0.01, 0] }));
  next = addRule(next, createRule("rule/cb", "gp/cb", "per-size", { "size/s": [-0.005, 0], "size/m": [-0.01, 0], "size/l": [-0.015, 0] }));
  next = addRule(next, createRule("rule/underarm", "gp/underarm", "transition", { "size/s": [0, 0.005], "size/m": [0, 0.005], "size/l": [0, 0.005] }));
  return next;
}
