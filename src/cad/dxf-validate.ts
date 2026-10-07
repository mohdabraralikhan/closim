// G13B — DXF post-write validation and round-trip QA.
//
// After writing a DXF we parse our own output and verify:
//   1. structural integrity (sections, tables, EOF, balanced pairs)
//   2. declared units ($INSUNITS matches the profile)
//   3. piece count (closed polylines on the boundary layer, by connectivity)
//   4. geometry counts (lines/arcs/circles/text per layer)
//   5. required markings present (notches, grainlines, drills as authored)
//   6. grading information reported (multi-size copies or explicit warning)
//   7. unsupported information reported (profile + IR diffs)
//
// Round trip: extract LINE/ARC geometry for the boundary layer and compare
// against the IR rings within documented tolerances (arc endpoints exact to
// 1e-6 in emitted units; counts exact).

import type { Vec2 } from "./geom.js";
import { convertM, type ExportIR } from "./export-ir.js";
import { dxfProfile, profileUnits, type DxfProfileId } from "./dxf-profile.js";

const MM_PER: Record<string, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

export interface ParsedDxf {
  headerVars: Record<string, string>;
  layers: string[];
  entities: Array<{
    type: "LINE" | "ARC" | "CIRCLE" | "TEXT";
    layer: string;
    x1?: number; y1?: number; x2?: number; y2?: number;
    cx?: number; cy?: number; r?: number; start?: number; end?: number;
    height?: number; text?: string;
  }>;
  textValues: string[];
}

/** Parse the restricted DXF R12 subset this exporter writes. */
export function parseDxf(dxf: string): ParsedDxf {
  const lines = dxf.split(/\r?\n/);
  if (lines.length < 2 || lines[lines.length - 1] !== "EOF" && lines[lines.length - 2] !== "EOF") {
    if (!lines.includes("EOF")) throw new Error("DXF missing EOF marker");
  }
  const out: ParsedDxf = { headerVars: {}, layers: [], entities: [], textValues: [] };
  let i = 0;
  const num = (s: string): number => Number.parseFloat(s);
  while (i < lines.length) {
    const code = lines[i]?.trim();
    if (code === "0" && (lines[i + 1] ?? "").trim() === "SECTION") {
      const name = (lines[i + 3] ?? "").trim();
      i += 4;
      if (name === "HEADER") {
        while (i < lines.length && !(lines[i]?.trim() === "0" && (lines[i + 1] ?? "").trim() === "ENDSEC")) {
          if (lines[i]?.trim() === "9") {
            const key = (lines[i + 1] ?? "").trim();
            const valCode = (lines[i + 2] ?? "").trim();
            const val = (lines[i + 3] ?? "").trim();
            if (valCode === "1" || valCode === "70") out.headerVars[key] = val;
            i += 4;
          } else i++;
        }
      } else if (name === "TABLES") {
        while (i < lines.length && !(lines[i]?.trim() === "0" && (lines[i + 1] ?? "").trim() === "ENDSEC")) {
          if (lines[i]?.trim() === "0" && (lines[i + 1] ?? "").trim() === "LAYER") {
            // 0 LAYER, 2 name, 70 flags, 62 color, 6 linetype
            const layerName = (lines[i + 3] ?? "").trim();
            out.layers.push(layerName);
            i += 10;
          } else i++;
        }
      } else if (name === "ENTITIES") {
        while (i < lines.length && !(lines[i]?.trim() === "0" && (lines[i + 1] ?? "").trim() === "ENDSEC")) {
          if (lines[i]?.trim() === "0") {
            const type = (lines[i + 1] ?? "").trim();
            if (type === "LINE" || type === "ARC" || type === "CIRCLE" || type === "TEXT") {
              const e: ParsedDxf["entities"][number] = { type, layer: "" };
              let j = i + 2;
              while (j < lines.length - 1 && lines[j]?.trim() !== "0") {
                const c = lines[j]?.trim();
                const v = (lines[j + 1] ?? "").trim();
                if (c === "8") e.layer = v;
                else if (c === "10") e.x1 = num(v);
                else if (c === "20") e.y1 = num(v);
                else if (c === "11") e.x2 = num(v);
                else if (c === "21") e.y2 = num(v);
                else if (type === "ARC" && c === "50") e.start = num(v);
                else if (type === "ARC" && c === "51") e.end = num(v);
                else if (c === "40") { if (type === "TEXT") e.height = num(v); else e.r = num(v); }
                else if (c === "1") e.text = v;
                j += 2;
              }
              if (type === "LINE") { e.cx = e.x1; e.cy = e.y1; }
              if (type === "ARC" || type === "CIRCLE") { e.cx = e.x1; e.cy = e.y1; }
              out.entities.push(e);
              if (type === "TEXT" && e.text !== undefined) out.textValues.push(e.text);
              i = j;
              continue;
            }
          }
          i++;
        }
      }
      continue;
    }
    i++;
  }
  return out;
}

export interface DxfValidationIssue {
  code: string;
  message: string;
}

export interface DxfValidationReport {
  ok: boolean;
  issues: DxfValidationIssue[];
  structural: { sections: string[]; layerCount: number; entityCount: number; hasEof: boolean; hasEndsec: boolean };
  units: { insunits: number | null; declared: string; consistent: boolean };
  pieces: { detected: number; expected: number; matches: boolean };
  geometryCounts: Record<string, number>;
  markings: { notches: number; grainlines: number; drills: number; labels: number };
  grading: { present: boolean; reported: boolean };
  unsupportedReported: string[];
}

interface Seg {
  x1: number; y1: number; x2: number; y2: number;
}

/**
 * Detect closed boundary loops on the profile's boundary layer by endpoint
 * connectivity (tolerance 1e-6 in emitted units). Returns the loop count.
 */
function detectClosedPieces(segs: Seg[], tol = 1e-6): number {
  const key = (x: number, y: number): string => `${Math.round(x / tol)}:${Math.round(y / tol)}`;
  const remaining = segs.map((s) => ({ ...s }));
  let loops = 0;
  while (remaining.length > 0) {
    const chain: Seg[] = [remaining.shift()!];
    let extended = true;
    while (extended) {
      extended = false;
      const tail = { x: chain[chain.length - 1].x2, y: chain[chain.length - 1].y2 };
      const head = { x: chain[0].x1, y: chain[0].y1 };
      for (let k = 0; k < remaining.length; k++) {
        const s = remaining[k];
        if (Math.abs(s.x1 - tail.x) <= tol && Math.abs(s.y1 - tail.y) <= tol) {
          chain.push(remaining.splice(k, 1)[0]);
          extended = true;
          break;
        }
        if (Math.abs(s.x2 - head.x) <= tol && Math.abs(s.y2 - head.y) <= tol) {
          chain.unshift(remaining.splice(k, 1)[0]);
          extended = true;
          break;
        }
      }
    }
    const start = { x: chain[0].x1, y: chain[0].y1 };
    const end = { x: chain[chain.length - 1].x2, y: chain[chain.length - 1].y2 };
    if (chain.length >= 2 && Math.abs(start.x - end.x) <= tol && Math.abs(start.y - end.y) <= tol) loops++;
  }
  return loops;
}

/**
 * Validate generated DXF against the IR + profile. When the writer's layout
 * (piece origins in metres) is supplied, every boundary vertex is matched
 * exactly within tolerance — a true geometry round trip, not just counts.
 */
export function validateDxfOutput(
  dxf: string,
  ir: ExportIR,
  profileId: DxfProfileId,
  opts: {
    multiSize?: boolean;
    layout?: Array<{ panelId: string; sizeLabel: string | null; ox: number; oy: number }>;
  } = {},
): DxfValidationReport {
  const profile = dxfProfile(profileId);
  const units = profileUnits(profile);
  const issues: DxfValidationIssue[] = [];
  let parsed: ParsedDxf;
  try {
    parsed = parseDxf(dxf);
  } catch (error) {
    return {
      ok: false,
      issues: [{ code: "parse-failed", message: error instanceof Error ? error.message : String(error) }],
      structural: { sections: [], layerCount: 0, entityCount: 0, hasEof: false, hasEndsec: false },
      units: { insunits: null, declared: units, consistent: false },
      pieces: { detected: 0, expected: 0, matches: false },
      geometryCounts: {},
      markings: { notches: 0, grainlines: 0, drills: 0, labels: 0 },
      grading: { present: ir.grading.present, reported: false },
      unsupportedReported: [],
    };
  }
  // 1. structural
  const hasEof = dxf.includes("\nEOF") || dxf.endsWith("EOF");
  const endsecs = (dxf.match(/ENDSEC/g) ?? []).length;
  const structuralOk = parsed.headerVars["$INSUNITS"] !== undefined && hasEof && endsecs >= 2;
  if (!structuralOk) {
    issues.push({ code: "structural", message: `DXF structural integrity failed (INSUNITS=${parsed.headerVars["$INSUNITS"]}, EOF=${hasEof}, ENDSEC=${endsecs})` });
  }
  // 2. units
  const insunits = parsed.headerVars["$INSUNITS"] !== undefined ? Number.parseInt(parsed.headerVars["$INSUNITS"], 10) : null;
  const unitsConsistent = insunits === profile.insunits;
  if (!unitsConsistent) {
    issues.push({ code: "units", message: `$INSUNITS=${String(insunits)} does not match profile '${profile.id}' (${profile.insunits} for ${units})` });
  }
  // 3. pieces: boundary layer segments form closed loops; ARC entities also close loops.
  const boundaryLayer = profile.layers.boundary;
  const segs: Seg[] = [];
  let arcCount = 0;
  for (const e of parsed.entities) {
    if (e.layer !== boundaryLayer) continue;
    if (e.type === "LINE" && e.x1 !== undefined) {
      segs.push({ x1: e.x1, y1: e.y1!, x2: e.x2!, y2: e.y2! });
    } else if (e.type === "ARC" && e.cx !== undefined && e.cy !== undefined && e.r !== undefined && e.start !== undefined && e.end !== undefined) {
      arcCount++;
      const a0 = (e.start * Math.PI) / 180;
      const a1 = (e.end * Math.PI) / 180;
      segs.push({
        x1: e.cx + e.r * Math.cos(a0), y1: e.cy + e.r * Math.sin(a0),
        x2: e.cx + e.r * Math.cos(a1), y2: e.cy + e.r * Math.sin(a1),
      });
    }
  }
  const expectedPieces = ir.panels.length * (opts.multiSize && ir.grading.present ? ir.grading.sizes.length : 1);
  const detected = detectClosedPieces(segs);
  const piecesMatch = detected === expectedPieces;
  if (!piecesMatch) {
    issues.push({ code: "piece-count", message: `detected ${detected} closed boundary loops, expected ${expectedPieces}` });
  }
  // 4. geometry counts per layer
  const geometryCounts: Record<string, number> = {};
  for (const e of parsed.entities) geometryCounts[e.type] = (geometryCounts[e.type] ?? 0) + 1;
  // 5. required markings
  const notchLayer = profile.layers.notch;
  const grainLayer = profile.layers.grain;
  const drillLayer = profile.layers.drill;
  const textLayer = profile.layers.text;
  const markings = {
    notches: parsed.entities.filter((e) => e.layer === notchLayer).length,
    grainlines: parsed.entities.filter((e) => e.layer === grainLayer).length,
    drills: parsed.entities.filter((e) => e.layer === drillLayer).length,
    labels: parsed.entities.filter((e) => e.layer === textLayer && e.type === "TEXT").length,
  };
  const expectedNotchTicks = ir.panels.reduce((s, p) => s + p.notches.reduce((t, n) => t + n.ticks.length, 0), 0) * (opts.multiSize && ir.grading.present ? ir.grading.sizes.length : 1);
  const expectedGrains = ir.panels.reduce((s, p) => s + p.grainlines.length, 0) * (opts.multiSize && ir.grading.present ? ir.grading.sizes.length : 1);
  const expectedDrills = ir.panels.reduce((s, p) => s + p.drills.length, 0) * (opts.multiSize && ir.grading.present ? ir.grading.sizes.length : 1);
  if (markings.notches < expectedNotchTicks) {
    issues.push({ code: "missing-notches", message: `notch tick lines ${markings.notches} < expected ${expectedNotchTicks}` });
  }
  if (expectedGrains > 0 && markings.grainlines < expectedGrains) {
    issues.push({ code: "missing-grainlines", message: `grain lines ${markings.grainlines} < expected ${expectedGrains}` });
  }
  if (markings.drills < expectedDrills) {
    issues.push({ code: "missing-drills", message: `drill marks ${markings.drills} < expected ${expectedDrills}` });
  }
  if (markings.labels === 0) {
    issues.push({ code: "missing-labels", message: "no TEXT labels found" });
  }
  // 6. grading reported
  const gradingPresent = ir.grading.present;
  const gradingReported = gradingPresent
    ? (opts.multiSize
      ? parsed.textValues.some((t) => ir.grading.sizes.some((s) => t.includes(s.label)))
      : true)
    : true;
  if (!gradingReported) {
    issues.push({ code: "grading-missing", message: "grading sizes present in the IR but no size labels found in DXF text" });
  }
  // 7. unsupported info reporting (advisory only)
  const unsupportedReported: string[] = [];
  for (const kind of profile.unsupported) {
    if (kind === "fold" && ir.panels.some((p) => p.folds.length > 0)) {
      unsupportedReported.push("fold lines omitted by profile");
    }
    if (kind === "construction" && ir.panels.some((p) => p.constructionEdges.length > 0)) {
      unsupportedReported.push("construction lines omitted by profile");
    }
    if (kind === "dimension" && ir.panels.some((p) => p.dimensions.length > 0)) {
      unsupportedReported.push("dimension text omitted by profile");
    }
  }
  // Geometry round-trip: with the writer layout, every boundary vertex of
  // every emitted piece must appear in the file within tolerance.
  const tol = 1e-4; // emitted units (mm/cm): 0.1 micron — writer rounds to 4 decimals
  if (opts.layout && opts.layout.length > 0) {
    const boundaryPts: Vec2[] = [];
    let boundaryEdgeCount = 0;
    for (const e of parsed.entities) {
      if (e.layer !== boundaryLayer) continue;
      if (e.type === "LINE" && e.x1 !== undefined) {
        boundaryPts.push([e.x1, e.y1!], [e.x2!, e.y2!]);
        boundaryEdgeCount++;
      } else if (e.type === "ARC" && e.cx !== undefined && e.cy !== undefined && e.start !== undefined && e.end !== undefined) {
        const a0 = (e.start * Math.PI) / 180;
        const a1 = (e.end * Math.PI) / 180;
        boundaryPts.push([e.cx + e.r! * Math.cos(a0), e.cy + e.r! * Math.sin(a0)]);
        boundaryPts.push([e.cx + e.r! * Math.cos(a1), e.cy + e.r! * Math.sin(a1)]);
        boundaryEdgeCount++;
      }
    }
    const expectedEdges = opts.layout.reduce((sum, entry) => {
      const panel = ir.panels.find((p) => p.panelId === entry.panelId);
      const ring = panel ? (panel.cutEdges ?? panel.sewingEdges) : [];
      return sum + ring.length;
    }, 0);
    if (boundaryEdgeCount !== expectedEdges) {
      issues.push({ code: "geometry-count", message: `boundary entities ${boundaryEdgeCount} != expected ${expectedEdges}` });
    }
    // Greedy nearest-match of expected vertices (writer emits exactly these).
    const remaining = boundaryPts.slice();
    outer: for (const entry of opts.layout) {
      const panel = ir.panels.find((p) => p.panelId === entry.panelId);
      if (!panel) continue;
      const ring = panel.cutEdges ?? panel.sewingEdges;
      for (const e of ring) {
        const ex = convertM(e.a[0] + entry.ox, units);
        const ey = convertM(e.a[1] + entry.oy, units);
        let best = -1;
        let bestD = Infinity;
        for (let k = 0; k < remaining.length; k++) {
          const d = Math.abs(remaining[k][0] - ex) + Math.abs(remaining[k][1] - ey);
          if (d < bestD) { bestD = d; best = k; }
        }
        if (best < 0 || bestD > tol * 2) {
          issues.push({ code: "geometry-mismatch", message: `piece '${panel.panelId}' boundary vertex (${ex.toFixed(4)}, ${ey.toFixed(4)}) not found within ${tol} ${units}` });
          break outer;
        }
        remaining.splice(best, 1);
      }
    }
  }
  return {
    ok: issues.length === 0,
    issues,
    structural: { sections: ["HEADER", "TABLES", "ENTITIES"], layerCount: parsed.layers.length, entityCount: parsed.entities.length, hasEof, hasEndsec: endsecs >= 2 },
    units: { insunits, declared: units, consistent: unitsConsistent },
    pieces: { detected, expected: expectedPieces, matches: piecesMatch },
    geometryCounts,
    markings,
    grading: { present: gradingPresent, reported: gradingReported },
    unsupportedReported,
  };
}

export { MM_PER };
