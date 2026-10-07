// G13B — profile-driven apparel DXF writer.
//
// Consumes the G13A export IR; emits DXF R12 text through a declared profile.
// Geometry: true LINE/ARC/CIRCLE entities (arcs never flattened), one layer
// per semantic entity per the profile, millimetres (or the profile's units).
//
// Layout: pieces are laid out left-to-right with a fixed gap so receiving CAD
// sees non-overlapping pieces. Multi-size output replicates every piece once
// per size in row bands, each labelled `name (size)`. Grading data is never
// silently dropped: when the profile cannot carry it, warnings say so.
//
// Every export ends with the profile's compliance note as a warning — the
// file is generated DXF, not a certified AAMA/ASTM interchange.

import { PatternCadError } from "../pattern/cad.js";
import type { Vec2 } from "./geom.js";
import {
  convertM,
  requireExportUnits,
  type ExportIR,
  type ExportPanelIR,
  type ExportUnits,
} from "./export-ir.js";
import { dxfProfile, profileUnits, type DxfProfile, type DxfProfileId, type DxfEntityKind } from "./dxf-profile.js";

const MM_PER: Record<ExportUnits, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

export interface DxfWriteOptions {
  profile: DxfProfileId;
  /** Override the profile's unit assumption (must match a declared unit). */
  units?: ExportUnits;
  /** Gap between pieces in canonical metres. Default 0.1. */
  gapM?: number;
  /** Emit one copy per size (requires ir.grading.present). Default single-size. */
  multiSize?: boolean;
  /** Include cross-grain marks (default true). */
  crossGrains?: boolean;
}

export interface DxfExportResult {
  dxf: string;
  warnings: string[];
  entityCount: number;
  pieceCount: number;
  profile: DxfProfileId;
  units: ExportUnits;
  /** Layout origin per emitted piece (metres) — feeds round-trip validation. */
  layout: Array<{ panelId: string; sizeLabel: string | null; ox: number; oy: number }>;
}

function dxfEscape(text: string): string {
  return text.replace(/[\r\n]+/g, " ");
}

function sanitizeLabel(text: string): string {
  // DXF R12 TEXT is ASCII-hostile in practice; keep printable ASCII, mark the drop.
  return text.replace(/[^\x20-\x7E]/g, "?");
}

class DxfBuilder {
  readonly lines: string[] = [];
  entityCount = 0;

  line(x1: number, y1: number, x2: number, y2: number, layer: string, color?: number): void {
    this.lines.push(
      "0", "LINE", "8", layer,
      "10", x1.toFixed(4), "20", y1.toFixed(4), "30", "0.0000",
      "11", x2.toFixed(4), "21", y2.toFixed(4), "31", "0.0000",
    );
    if (color !== undefined) this.lines.push("62", String(color));
    this.entityCount++;
  }

  arc(x: number, y: number, r: number, startDeg: number, endDeg: number, layer: string, color?: number): void {
    this.lines.push(
      "0", "ARC", "8", layer,
      "10", x.toFixed(4), "20", y.toFixed(4), "30", "0.0000",
      "40", r.toFixed(4), "50", startDeg.toFixed(6), "51", endDeg.toFixed(6),
    );
    if (color !== undefined) this.lines.push("62", String(color));
    this.entityCount++;
  }

  circle(x: number, y: number, r: number, layer: string, color?: number): void {
    this.lines.push(
      "0", "CIRCLE", "8", layer,
      "10", x.toFixed(4), "20", y.toFixed(4), "30", "0.0000", "40", r.toFixed(4),
    );
    if (color !== undefined) this.lines.push("62", String(color));
    this.entityCount++;
  }

  text(x: number, y: number, height: number, value: string, layer: string, color?: number): void {
    this.lines.push(
      "0", "TEXT", "8", layer,
      "10", x.toFixed(4), "20", y.toFixed(4), "30", "0.0000",
      "40", height.toFixed(4), "1", dxfEscape(value),
    );
    if (color !== undefined) this.lines.push("62", String(color));
    this.entityCount++;
  }
}

function bboxOfPanel(panel: ExportPanelIR): { min: Vec2; max: Vec2 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const scan = (ring: readonly { a: Vec2; b: Vec2; kind: string; center?: Vec2; radiusM?: number }[]): void => {
    for (const e of ring) {
      for (const p of [e.a, e.b]) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      if (e.kind === "arc" && e.center && e.radiusM !== undefined) {
        const c = e.center, r = e.radiusM;
        if (c[0] - r < minX) minX = c[0] - r;
        if (c[0] + r > maxX) maxX = c[0] + r;
        if (c[1] - r < minY) minY = c[1] - r;
        if (c[1] + r > maxY) maxY = c[1] + r;
      }
    }
  };
  scan(panel.sewingEdges);
  if (panel.allowanceEdges) scan(panel.allowanceEdges);
  if (panel.cutEdges) scan(panel.cutEdges);
  for (const h of panel.holes) scan(h);
  for (const g of panel.grainlines) scan([{ kind: "line", a: g.from, b: g.to }]);
  if (!Number.isFinite(minX)) return { min: [0, 0], max: [0, 0] };
  return { min: [minX, minY], max: [maxX, maxY] };
}

/**
 * Emit DXF from the IR through a profile. Deterministic: identical IR +
 * identical options produce identical bytes.
 */
export function exportIRToDXFProfile(ir: ExportIR, opts: DxfWriteOptions): DxfExportResult {
  const profile = dxfProfile(opts.profile);
  if (!ir || !Array.isArray(ir.panels) || ir.panels.length === 0) {
    throw new PatternCadError("invalid-document", "DXF export needs at least one panel");
  }
  const units = opts.units ?? profileUnits(profile);
  requireExportUnits(units, `DXF profile '${profile.id}'`);
  const gapM = opts.gapM ?? 0.1;
  const multi = opts.multiSize === true;
  const warnings: string[] = [];
  const layerOf = (kind: DxfEntityKind): string => profile.layers[kind];
  const scale = (metres: number): number => metres * 1000 / MM_PER[units];
  const fmt = scale;

  const sizes = multi
    ? (ir.grading.present
      ? ir.grading.sizes
      : (() => {
        throw new PatternCadError("invalid-transform", "multiSize export requested but the IR carries no grading sizes");
      })())
    : null;
  if (multi && profile.grading.multiSize === false) {
    throw new PatternCadError("invalid-transform", `profile '${profile.id}' does not support multi-size DXF`);
  }
  if (profile.grading.mode === "none" && ir.grading.present) {
    warnings.push(`profile '${profile.id}' cannot carry grading data; export a separate grade-rule file`);
  }

  const sizeLabels = sizes ? sizes.map((s) => s.label) : [null];
  const b = new DxfBuilder();
  const supported = (kind: DxfEntityKind): boolean => !profile.unsupported.includes(kind);
  const layout: DxfExportResult["layout"] = [];

  // Size bands: each size gets its own horizontal band offset in y so copies
  // never overlap; within a band pieces run left-to-right from x = 0.
  const bandMinY = ir.panels.length > 0 ? Math.min(...ir.panels.map((p) => bboxOfPanel(p).min[1])) : 0;
  const bandHeightM = ir.panels.length > 0
    ? Math.max(...ir.panels.map((p) => bboxOfPanel(p).max[1])) - bandMinY
    : 0;

  sizeLabels.forEach((sizeLabel, sizeIndex) => {
    const suffix = sizeLabel ? ` (${sanitizeLabel(sizeLabel)})` : "";
    const oy = sizeIndex * (bandHeightM + gapM) - bandMinY;
    let cursorX = 0;
    for (const panel of ir.panels) {
      const bb = bboxOfPanel(panel);
      const ox = cursorX - bb.min[0];
      layout.push({ panelId: panel.panelId, sizeLabel, ox, oy });
      const px = (x: number): number => scale(x + ox);
      const py = (y: number): number => scale(y + oy);
      const cutRing = panel.cutEdges ?? panel.sewingEdges;
      const emitRing = (ring: readonly { a: Vec2; b: Vec2; kind: string; center?: Vec2; radiusM?: number; a0Rad?: number; sweepRad?: number }[], layer: string, color?: number): void => {
        for (const e of ring) {
          if (e.kind === "arc" && e.center && e.radiusM !== undefined && e.a0Rad !== undefined && e.sweepRad !== undefined) {
            if (profile.geometry.arcs) {
              // DXF ARC angles are degrees, CCW-positive, and the entity always
              // sweeps CCW from start to end angle.
              let start = (e.a0Rad * 180) / Math.PI;
              let sweep = (e.sweepRad * 180) / Math.PI;
              if (sweep < 0) {
                start += sweep;
                sweep = -sweep;
              }
              b.arc(px(e.center[0]), py(e.center[1]), scale(e.radiusM), start, start + sweep, layer, color);
            } else {
              b.line(px(e.a[0]), py(e.a[1]), px(e.b[0]), py(e.b[1]), layer, color);
            }
          } else {
            b.line(px(e.a[0]), py(e.a[1]), px(e.b[0]), py(e.b[1]), layer, color);
          }
        }
      };
      const layerCut = layerOf("boundary");
      emitRing(cutRing, layerCut, profile.layerColors[layerCut]);
      if (panel.cutEdges && panel.cutSource === "allowance" && supported("sewing")) {
        emitRing(panel.sewingEdges, layerOf("sewing"), profile.layerColors[layerOf("sewing")]);
      }
      if (panel.allowanceEdges && !(panel.cutEdges && panel.cutSource === "allowance") && supported("allowance")) {
        emitRing(panel.allowanceEdges, layerOf("allowance"), profile.layerColors[layerOf("allowance")]);
      }
      for (const hole of panel.holes) emitRing(hole, layerOf("hole"), profile.layerColors[layerOf("hole")]);
      if (supported("notch")) {
        for (const notch of panel.notches) {
          for (const tick of notch.ticks) {
            b.line(px(tick.from[0]), py(tick.from[1]), px(tick.to[0]), py(tick.to[1]), layerOf("notch"), profile.layerColors[layerOf("notch")]);
          }
          b.text(px(notch.pos[0]), py(notch.pos[1]), scale(0.003), sanitizeLabel(`notch:${notch.id}${suffix}`), layerOf("text"));
        }
      }
      if (supported("grain")) {
        for (const grain of panel.grainlines) {
          b.line(px(grain.from[0]), py(grain.from[1]), px(grain.to[0]), py(grain.to[1]), layerOf("grain"), profile.layerColors[layerOf("grain")]);
        }
        if (opts.crossGrains !== false) {
          for (const cg of panel.crossGrains) {
            b.line(px(cg.from[0]), py(cg.from[1]), px(cg.to[0]), py(cg.to[1]), layerOf("grain"));
          }
        }
      }
      if (supported("fold")) {
        for (const fold of panel.folds) {
          b.line(px(fold.a[0]), py(fold.a[1]), px(fold.b[0]), py(fold.b[1]), layerOf("fold"), profile.layerColors[layerOf("fold")]);
          b.text(px((fold.a[0] + fold.b[0]) / 2), py((fold.a[1] + fold.b[1]) / 2), scale(0.003), sanitizeLabel(`${fold.direction} ${fold.foldType}${suffix}`), layerOf("text"));
        }
      } else if (panel.folds.length > 0) {
        warnings.push(`profile '${profile.id}' does not support fold lines; ${panel.folds.length} fold(s) omitted from piece '${panel.name}'`);
      }
      if (supported("drill")) {
        for (const drill of panel.drills) {
          if (drill.mark === "circle") {
            b.circle(px(drill.pos[0]), py(drill.pos[1]), scale(drill.radiusM), layerOf("drill"), profile.layerColors[layerOf("drill")]);
          } else if (drill.mark === "cross") {
            const r = drill.radiusM;
            b.line(px(drill.pos[0] - r), py(drill.pos[1]), px(drill.pos[0] + r), py(drill.pos[1]), layerOf("drill"));
            b.line(px(drill.pos[0]), py(drill.pos[1] - r), px(drill.pos[0]), py(drill.pos[1] + r), layerOf("drill"));
          } else {
            b.circle(px(drill.pos[0]), py(drill.pos[1]), scale(0.001), layerOf("drill"));
          }
        }
      }
      if (supported("internal")) {
        for (const internal of panel.internals) {
          for (const e of internal.edges) {
            b.line(px(e.a[0]), py(e.a[1]), px(e.b[0]), py(e.b[1]), layerOf("internal"), profile.layerColors[layerOf("internal")]);
          }
          if (internal.label) {
            b.text(px(internal.edges[0].a[0]), py(internal.edges[0].a[1]), scale(0.003), sanitizeLabel(internal.label), layerOf("text"));
          }
        }
      }
      for (const ce of panel.constructionEdges) {
        if (supported("construction")) {
          b.line(px(ce.a[0]), py(ce.a[1]), px(ce.b[0]), py(ce.b[1]), layerOf("construction"));
        }
      }
      for (const ann of panel.annotations) {
        b.text(px(ann.pos[0]), py(ann.pos[1]), scale(0.0025), sanitizeLabel(ann.note), layerOf("text"));
      }
      for (const region of panel.labelRegions) {
        const corners: Vec2[] = [region.min, [region.max[0], region.min[1]], region.max, [region.min[0], region.max[1]]];
        for (let i = 0; i < corners.length; i++) {
          const a = corners[i], c = corners[(i + 1) % corners.length];
          b.line(px(a[0]), py(a[1]), px(c[0]), py(c[1]), layerOf("text"));
        }
        Object.entries(region.fields).forEach(([k, v], i) => {
          b.text(px(region.min[0]), py(region.max[1] - 0.005 * (i + 1)), scale(0.0035), sanitizeLabel(`${k}: ${v}${suffix}`), layerOf("text"));
        });
      }
      if (supported("dimension")) {
        for (const dim of panel.dimensions) {
          const label = `${(dim.lengthM * 1000 / MM_PER[units]).toFixed(1)} ${units}`;
          b.text(px(dim.mid[0]), py(dim.mid[1]), scale(0.0025), sanitizeLabel(label), layerOf("dimension"));
        }
      }
      b.text(px(bb.min[0]), py(bb.max[1] + 0.01), scale(0.004), sanitizeLabel(`${panel.name}${suffix} x${panel.cutQuantity} [${ir.style.garmentName} rev${ir.style.revision}]`), layerOf("text"));
      if (panel.allowanceIssues.length > 0) {
        warnings.push(`piece '${panel.name}' allowance issues: ${panel.allowanceIssues.map((i) => i.code).join(", ")}`);
      }
      cursorX += bb.max[0] - bb.min[0] + gapM;
    }
  });

  // Grading notice: never silent.
  if (ir.grading.present) {
    if (multi) {
      warnings.push(`multi-size DXF: sizes ${(sizes ?? []).map((s) => s.label).join(", ")} as per-size piece copies`);
      warnings.push("per-size deltas are NOT embedded as grade-rule tables; export the separate grade-rule file for factory CAD");
    } else {
      warnings.push(`single-size DXF: ${ir.grading.sizes.length} size(s) present in the IR but only the base geometry emitted; grade data lives in the separate grade-rule export`);
    }
  }
  if (profile.unsupported.length > 0) {
    warnings.push(`profile '${profile.id}' unsupported entities: ${profile.unsupported.join(", ")}`);
  }
  warnings.push(profile.complianceNote);

  const header = [
    "0", "SECTION", "2", "HEADER",
    "9", "$ACADVER", "1", "AC1009",
    "9", "$INSUNITS", "70", String(profile.insunits),
    "0", "ENDSEC",
    "0", "SECTION", "2", "TABLES",
    "0", "TABLE", "2", "LAYER", "70", String(Object.keys(profile.layers).length),
  ];
  const seen = new Set<string>();
  for (const layerName of Object.values(profile.layers)) {
    if (seen.has(layerName)) continue;
    seen.add(layerName);
    header.push("0", "LAYER", "2", layerName, "70", "0", "62", String(profile.layerColors[layerName] ?? 7), "6", "CONTINUOUS");
  }
  header.push("0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES");
  const footer = ["0", "ENDSEC", "0", "EOF"];
  return {
    dxf: [...header, ...b.lines, ...footer].join("\n") + "\n",
    warnings,
    entityCount: b.entityCount,
    pieceCount: ir.panels.length * (sizes ? sizes.length : 1),
    profile: profile.id,
    units,
    layout,
  };
}
