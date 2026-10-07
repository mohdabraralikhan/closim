// G14 marker export: a manufacturing artifact separate from pattern export.
//
// Pattern Export (G11E) carries boundaries, allowances, and markings for
// making the garment. Marker Export carries fabric, placements, and
// utilization for cutting it. Different artifacts, different files, one
// shared R12 idiom.

import { PatternCadError } from "../pattern/cad.js";
import type { Vec2 } from "../cad/geom.js";
import { auditPlacements, placedPolygon } from "./nest.js";
import {
  serializeMarker,
  type CutPlan,
  type Marker,
  type MarkerPiece,
} from "./model.js";
import type { FabricRules } from "./fabric.js";
import type { MarkerMetrics } from "./optimize.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export interface MarkerPackage {
  format: "closim-marker-package";
  version: 1;
  marker: Marker;
  cutPlan: CutPlan;
  metrics: MarkerMetrics;
}

export function exportMarkerJSON(marker: Marker, cutPlan: CutPlan, metrics: MarkerMetrics): string {
  const pkg: MarkerPackage = { format: "closim-marker-package", version: 1, marker: clone(marker), cutPlan: clone(cutPlan), metrics: clone(metrics) };
  return canonicalJson(pkg);
}

export function importMarkerJSON(json: string): MarkerPackage {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new PatternCadError("invalid-document", "marker package does not parse");
  }
  const pkg = parsed as MarkerPackage;
  if (!pkg || pkg.format !== "closim-marker-package" || pkg.version !== 1 || !pkg.marker || !pkg.cutPlan || !pkg.metrics) {
    throw new PatternCadError("invalid-document", "marker package has the wrong shape or version");
  }
  return clone(pkg);
}

// ---------------------------------------------------------------------------
// Marker DXF (R12 LINE/TEXT, millimetres, own compact writer)
// ---------------------------------------------------------------------------

const LAYERS = ["MARKER_BOUND", "PIECES", "LABELS"] as const;

function fmtMM(metres: number): string {
  return (metres * 1000).toFixed(4);
}

export interface MarkerDXF {
  dxf: string;
  warnings: string[];
  entityCount: number;
}

export function exportMarkerDXF(
  marker: Marker,
  pieces: MarkerPiece[],
  metrics: MarkerMetrics,
  opts: { facetedArcs?: boolean } = {},
): MarkerDXF {
  // Gate: geometric audit clean AND every piece instance placed (quantities match).
  const byPiece = new Map(pieces.map((p) => [p.instanceId, p]));
  const placed = new Set(marker.placements.map((p) => p.instanceId));
  const missing = pieces.filter((p) => !placed.has(p.instanceId)).map((p) => p.instanceId);
  if (missing.length > 0) {
    throw new PatternCadError(
      "invalid-document",
      `marker export refused: ${missing.length} piece(s) unplaced (${missing.slice(0, 3).join(", ")})`,
      missing[0],
    );
  }
  // Geometric audit lives in exportMarkerPackage (needs fabric rules);
  // this writer assumes a gated marker.
  const lines: string[] = [];
  let entityCount = 0;
  const line = (x1: number, y1: number, x2: number, y2: number, layer: string): void => {
    lines.push(
      "0", "LINE", "8", layer,
      "10", fmtMM(x1), "20", fmtMM(y1), "30", "0.0000",
      "11", fmtMM(x2), "21", fmtMM(y2), "31", "0.0000",
    );
    entityCount++;
  };
  const text = (x: number, y: number, heightM: number, value: string, layer: string): void => {
    lines.push(
      "0", "TEXT", "8", layer,
      "10", fmtMM(x), "20", fmtMM(y), "30", "0.0000",
      "40", fmtMM(heightM), "1", value.replace(/[\r\n]+/g, " "),
    );
    entityCount++;
  };
  const W = marker.fabric.usableWidthM;
  const L = metrics.markerLengthM;
  // Usable-width boundary.
  line(0, 0, W, 0, "MARKER_BOUND");
  line(W, 0, W, L, "MARKER_BOUND");
  line(W, L, 0, L, "MARKER_BOUND");
  line(0, L, 0, 0, "MARKER_BOUND");
  for (const placement of marker.placements) {
    const piece = byPiece.get(placement.instanceId)!;
    const poly = placedPolygon(piece, placement);
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      line(a[0], a[1], b[0], b[1], "PIECES");
    }
    const label: Vec2 = [placement.x, placement.y];
    text(label[0], label[1], 0.005, `${piece.panelId} ${piece.sizeId}${placement.mirrored ? " (m)" : ""}`, "LABELS");
  }
  text(0, L + 0.02, 0.005, `utilization ${(metrics.utilization * 100).toFixed(1)}%`, "LABELS");
  const header = [
    "0", "SECTION", "2", "HEADER",
    "9", "$INSUNITS", "70", "4",
    "0", "ENDSEC",
    "0", "SECTION", "2", "TABLES",
    "0", "TABLE", "2", "LAYER", "70", String(LAYERS.length),
  ];
  for (const layer of LAYERS) {
    header.push("0", "LAYER", "2", layer, "70", "0", "62", "7", "6", "CONTINUOUS");
  }
  header.push("0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES");
  const warnings = [
    "DXF subset: R12 LINE/TEXT only; no SPLINE, HATCH, MTEXT, or paper-space layout",
    "marker DXF carries placements only; pattern geometry lives in the pattern export",
  ];
  if (opts.facetedArcs) warnings.push("piece boundaries faceted to sampling tolerance in DXF output");
  return { dxf: [...header, ...lines, "0", "ENDSEC", "0", "EOF"].join("\n") + "\n", warnings, entityCount };
}

export function exportMarkerPackage(
  marker: Marker,
  pieces: MarkerPiece[],
  cutPlan: CutPlan,
  metrics: MarkerMetrics,
  rules: FabricRules,
  grainRadOf: (panelId: string) => number,
  opts: { facetedArcs?: boolean } = {},
): { json: string; dxf: MarkerDXF } {
  const problems = auditPlacements(marker, pieces, rules, grainRadOf);
  if (problems.length > 0) {
    throw new PatternCadError(
      "invalid-document",
      `marker export refused: ${problems[0].code}: ${problems[0].message}`,
      problems[0].entityId,
    );
  }
  const placed = new Set(marker.placements.map((p) => p.instanceId));
  if (pieces.some((p) => !placed.has(p.instanceId))) {
    throw new PatternCadError("invalid-document", "marker export refused: unplaced pieces remain");
  }
  void serializeMarker;
  return {
    json: exportMarkerJSON(marker, cutPlan, metrics),
    dxf: exportMarkerDXF(marker, pieces, metrics, opts),
  };
}
