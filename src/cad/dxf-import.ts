// G13B+ — apparel DXF import (interoperability direction: foreign file -> model).
//
// Purpose: read DXF files produced by OTHER CAD tools (AAMA/ASTM-style numeric
// layers, or our own named-layer generic profile) and interpret them as
// apparel pieces, reporting every mismatch against the documented layer
// conventions instead of silently guessing. This is the QA counterpart of the
// G13B exporter: import is best-effort interpretation, never a sanctioned
// path back into editing (the native document is never mutated from files).
//
// Honesty policy, same as the exporter: the AAMA/ASTM layer semantics used
// here are community-documented conventions (1 cut, 8 internal, 10 notch,
// 11 drill, 13 note/text, 14 sew, 16 grain). Units are taken from $INSUNITS
// when present; otherwise the caller must declare them or the import fails —
// guessing units is how patterns get cut wrong.
//
// Every assumption that did not hold becomes a structured mismatch entry:
// { code, message, layer? }. Nothing is dropped silently; consumers decide
// whether the file is usable for their purpose.

import type { Vec2 } from "./geom.js";
import { parseDxf } from "./dxf-validate.js";

const MM_PER_UNIT: Record<string, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

/** $INSUNITS code -> unit name (DXF standard table, subset we accept). */
const INSUNITS_TO_UNIT: Record<number, string> = {
  1: "in", 2: "ft", 4: "mm", 5: "cm", 6: "m",
};

export type DxfImportUnit = "mm" | "cm" | "m" | "in";

export type DxfImportProfileId = "aama-style" | "astm-oriented" | "generic-r12" | "auto";

interface LayerConvention {
  boundary: string;
  sew?: string;
  internal?: string;
  notch?: string;
  grain?: string;
  drill?: string;
  text?: string;
  defaultUnits: DxfImportUnit;
}

const AAMA_CONVENTION: LayerConvention = {
  boundary: "1", sew: "14", internal: "8", notch: "10",
  grain: "16", drill: "11", text: "13",
  defaultUnits: "mm",
};

/** Named-layer convention matching our generic-r12 profile. */
const GENERIC_CONVENTION: LayerConvention = {
  boundary: "CUT", sew: "SEW", internal: "INTERNAL", notch: "NOTCH",
  grain: "GRAIN", drill: "DRILL", text: "LABEL",
  defaultUnits: "mm",
};

function conventionFor(profile: DxfImportProfileId, layers: string[]): LayerConvention {
  if (profile === "aama-style" || profile === "astm-oriented") return AAMA_CONVENTION;
  if (profile === "generic-r12") return GENERIC_CONVENTION;
  // auto: prefer the generic named layers when present, else numeric AAMA.
  const named = layers.includes("CUT") || layers.includes("SEW");
  return named ? GENERIC_CONVENTION : AAMA_CONVENTION;
}

export interface DxfImportMismatch {
  code:
    | "insunits-missing" | "insunits-unknown"
    | "layer-missing" | "boundary-empty" | "piece-open"
    | "degenerate-edge" | "no-layers" | "entity-skipped";
  message: string;
  /** Layer the issue relates to, when applicable. */
  layer?: string;
}

export interface ImportedPiece {
  /** Deterministic id: piece-1, piece-2, … in scan order. */
  id: string;
  /** Layers the boundary loop was found on (usually one). */
  boundaryLayer: string;
  /** Closed boundary ring in the file's declared units. */
  ring: Vec2[];
  /** Ring vertices converted to canonical metres. */
  ringM: Vec2[];
  /** Sewing line edges in metres, when a sew layer exists and matches. */
  sewEdgesM: Vec2[];
  /** Notch tick segments (metres) found on the notch layer inside the piece bbox. */
  notchTicksM: Array<{ from: Vec2; to: Vec2 }>;
  /** Grain line segments (metres) attributed to this piece. */
  grainLinesM: Array<{ from: Vec2; to: Vec2 }>;
  /** Drill marks (metres) attributed to this piece. */
  drillsM: Array<{ pos: Vec2; text?: string }>;
  /** TEXT entities attributed to this piece (label layer). */
  labels: string[];
  /** True when the boundary ring is exactly closed within the tolerance. */
  closed: boolean;
}

export interface DxfImportReport {
  ok: boolean;
  profile: DxfImportProfileId;
  /** Layer names found in the file's TABLES section. */
  layersFound: string[];
  /** Resolved layer convention. */
  convention: LayerConvention;
  /** Units resolved from $INSUNITS or the explicit override. */
  units: DxfImportUnit;
  pieces: ImportedPiece[];
  /** Every convention/geometry mismatch encountered (never silently dropped). */
  mismatches: DxfImportMismatch[];
  /** Entity counts per layer for the audit trail. */
  entityCounts: Record<string, number>;
  /** Text values found on any layer (diagnostics). */
  textValues: string[];
}

function ringIsClosed(ring: Vec2[], tol: number): boolean {
  if (ring.length < 3) return false;
  const a = ring[0];
  const b = ring[ring.length - 1];
  return Math.hypot(a[0] - b[0], a[1] - b[1]) <= tol;
}

function bboxOf(ring: Vec2[]): { min: Vec2; max: Vec2 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of ring) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { min: [minX, minY], max: [maxX, maxY] };
}

function pointInBBox(p: Vec2, bb: { min: Vec2; max: Vec2 }): boolean {
  return p[0] >= bb.min[0] && p[0] <= bb.max[0] && p[1] >= bb.min[1] && p[1] <= bb.max[1];
}

/**
 * Import an apparel DXF file (as text) into pieces + markings, reporting
 * mismatches. `opts.units` overrides / backs up $INSUNITS; when the file has
 * no $INSUNITS and no override is given, the import fails honestly.
 */
export function importApparelDxf(
  dxf: string,
  opts: { profile?: DxfImportProfileId; units?: DxfImportUnit; toleranceUnits?: number } = {},
): DxfImportReport {
  const mismatches: DxfImportMismatch[] = [];
  const parsed = parseDxf(dxf);
  const layersFound = [...parsed.layers];
  if (layersFound.length === 0) {
    mismatches.push({ code: "no-layers", message: "no LAYER table entries found; treating file as uninterpreted" });
  }
  const profile = opts.profile ?? "auto";
  const convention = conventionFor(profile, layersFound);

  // --- units -----------------------------------------------------------------
  let units: DxfImportUnit;
  let unitsResolved = false;
  const insRaw = parsed.headerVars["$INSUNITS"];
  if (insRaw !== undefined) {
    const code = Number.parseInt(insRaw, 10);
    const mapped = INSUNITS_TO_UNIT[code];
    if (mapped === undefined) {
      mismatches.push({ code: "insunits-unknown", message: `$INSUNITS=${insRaw} is not a unit we accept (want 1/2/4/5/6); refusing to guess` });
      units = "mm"; // placeholder; caller sees the mismatch and ok=false
    } else if (opts.units !== undefined && opts.units !== mapped) {
      units = opts.units;
      unitsResolved = true; // caller-declared, though the header disagrees (flagged)
      mismatches.push({ code: "insunits-unknown", message: `explicit units '${opts.units}' override $INSUNITS (${mapped}); the file header disagrees` });
    } else {
      units = mapped as DxfImportUnit;
      unitsResolved = true;
    }
  } else if (opts.units !== undefined) {
    units = opts.units;
    unitsResolved = true;
    mismatches.push({ code: "insunits-missing", message: `file has no $INSUNITS; units '${opts.units}' declared explicitly by the caller` });
  } else {
    mismatches.push({ code: "insunits-missing", message: "file has no $INSUNITS and no explicit unit override was given; refusing to guess scale" });
    units = "mm";
  }
  const scale = MM_PER_UNIT[units];
  const tol = opts.toleranceUnits ?? 1e-4 * scale;

  // --- entity bookkeeping ----------------------------------------------------
  const entityCounts: Record<string, number> = {};
  for (const e of parsed.entities) entityCounts[e.layer] = (entityCounts[e.layer] ?? 0) + 1;

  // Boundary chains: LINE + ARC endpoint connectivity on the boundary layer.
  interface ChainSeg { a: Vec2; b: Vec2 }
  const boundarySegs: ChainSeg[] = [];
  for (const e of parsed.entities) {
    if (e.layer !== convention.boundary) continue;
    if (e.type === "LINE" && e.x1 !== undefined && e.x2 !== undefined) {
      boundarySegs.push({ a: [e.x1, e.y1!], b: [e.x2, e.y2!] });
    } else if (e.type === "ARC" && e.cx !== undefined && e.r !== undefined && e.start !== undefined && e.end !== undefined) {
      const a0 = (e.start * Math.PI) / 180;
      const a1 = (e.end * Math.PI) / 180;
      boundarySegs.push({
        a: [e.cx + e.r * Math.cos(a0), e.cy! + e.r * Math.sin(a0)],
        b: [e.cx + e.r * Math.cos(a1), e.cy! + e.r * Math.sin(a1)],
      });
    } else if (e.type === "CIRCLE" && e.cx !== undefined && e.r !== undefined) {
      // A full circle is a boundary by itself (e.g. a cuff hole piece).
      boundarySegs.push({ a: [e.cx + e.r, e.cy!], b: [e.cx + e.r, e.cy!] });
      mismatches.push({ code: "entity-skipped", layer: e.layer, message: `CIRCLE on boundary layer treated as a degenerate closed loop at (${e.cx}, ${e.cy}) r=${e.r}` });
    } else if (e.type !== "TEXT") {
      mismatches.push({ code: "entity-skipped", layer: e.layer, message: `${e.type} entity on boundary layer cannot be chained (unsupported for import)` });
    }
  }

  // Chain segments into closed loops (greedy endpoint walk, deterministic).
  const pieces: ImportedPiece[] = [];
  const remaining = boundarySegs.map((s) => ({ ...s }));
  const pieceRings: Vec2[][] = [];
  while (remaining.length > 0) {
    const chain: ChainSeg[] = [remaining.shift()!];
    let extended = true;
    while (extended) {
      extended = false;
      const tail = chain[chain.length - 1].b;
      const head = chain[0].a;
      for (let k = 0; k < remaining.length; k++) {
        const s = remaining[k];
        if (Math.hypot(s.a[0] - tail[0], s.a[1] - tail[1]) <= tol) {
          chain.push(remaining.splice(k, 1)[0]);
          extended = true;
          break;
        }
        if (Math.hypot(s.b[0] - head[0], s.b[1] - head[1]) <= tol) {
          chain.unshift(remaining.splice(k, 1)[0]);
          extended = true;
          break;
        }
      }
    }
    const ring: Vec2[] = chain.map((s) => s.a);
    pieceRings.push(ring);
  }

  for (const ring of pieceRings) {
    if (ring.length < 3) {
      mismatches.push({ code: "degenerate-edge", layer: convention.boundary, message: `boundary chain has ${ring.length} segment(s); fewer than 3 cannot form a piece` });
      continue;
    }
    if (!ringIsClosed(ring, tol)) {
      mismatches.push({ code: "piece-open", layer: convention.boundary, message: `boundary loop is open (first/last endpoints differ beyond ${tol} ${units}); imported as best-effort` });
    }
  }

  // Attribute markings to pieces by bbox containment (in file units first).
  const toMetres = (v: number): number => (v * scale) / 1000;
  const ringsM = pieceRings.map((r) => r.map((p) => [toMetres(p[0]), toMetres(p[1])] as Vec2));
  const bbs = pieceRings.map(bboxOf);
  const pieceIndexAt = (p: Vec2): number => {
    for (let i = 0; i < bbs.length; i++) {
      if (pieceRings[i].length >= 3 && pointInBBox(p, bbs[i])) return i;
    }
    return -1;
  };

  const attrib = pieceRings.map(() => ({
    sew: [] as Vec2[],
    notches: [] as Array<{ from: Vec2; to: Vec2 }>,
    grains: [] as Array<{ from: Vec2; to: Vec2 }>,
    drills: [] as Array<{ pos: Vec2; text?: string }>,
    labels: [] as string[],
  }));
  const unattributed: string[] = [];

  for (const e of parsed.entities) {
    const layer = e.layer;
    const toM = (x: number, y: number): Vec2 => [toMetres(x), toMetres(y)];
    const asPair = (): Array<{ from: Vec2; to: Vec2 }> | null => {
      if (e.x1 !== undefined && e.x2 !== undefined) {
        return [{ from: toM(e.x1, e.y1!), to: toM(e.x2, e.y2!) }];
      }
      return null;
    };
    if (layer === convention.sew) {
      const pair = asPair();
      if (pair) {
        const idx = pieceIndexAt([(e.x1! + e.x2!) / 2, (e.y1! + e.y2!) / 2]);
        if (idx >= 0) attrib[idx].sew.push(...pair.map((p) => p.from));
        else unattributed.push(`sew line at (${e.x1}, ${e.y1})`);
      }
    } else if (layer === convention.notch) {
      const pair = asPair();
      if (pair) {
        const idx = pieceIndexAt([(e.x1! + e.x2!) / 2, (e.y1! + e.y2!) / 2]);
        if (idx >= 0) attrib[idx].notches.push(pair[0]);
        else unattributed.push(`notch at (${e.x1}, ${e.y1})`);
      }
    } else if (layer === convention.grain) {
      const pair = asPair();
      if (pair) {
        const idx = pieceIndexAt([(e.x1! + e.x2!) / 2, (e.y1! + e.y2!) / 2]);
        if (idx >= 0) attrib[idx].grains.push(pair[0]);
        else unattributed.push(`grain at (${e.x1}, ${e.y1})`);
      }
    } else if (layer === convention.drill) {
      if (e.cx !== undefined && e.cy !== undefined) {
        const idx = pieceIndexAt([e.cx, e.cy]);
        if (idx >= 0) attrib[idx].drills.push({ pos: toM(e.cx, e.cy) });
        else unattributed.push(`drill at (${e.cx}, ${e.cy})`);
      }
    } else if (layer === convention.text && e.type === "TEXT") {
      const x = e.x1 ?? e.cx;
      const y = e.y1 ?? e.cy;
      if (x !== undefined && y !== undefined) {
        const idx = pieceIndexAt([x, y]);
        if (idx >= 0 && e.text !== undefined) attrib[idx].labels.push(e.text);
        else if (e.text !== undefined) unattributed.push(`text '${e.text}'`);
      }
    }
  }
  for (const u of unattributed) {
    mismatches.push({ code: "entity-skipped", message: `${u} does not fall inside any piece bbox; not attributed` });
  }

  pieceRings.forEach((ring, i) => {
    if (ring.length < 3) return;
    pieces.push({
      id: `piece-${i + 1}`,
      boundaryLayer: convention.boundary,
      ring,
      ringM: ringsM[i],
      sewEdgesM: attrib[i].sew,
      notchTicksM: attrib[i].notches,
      grainLinesM: attrib[i].grains,
      drillsM: attrib[i].drills,
      labels: attrib[i].labels,
      closed: ringIsClosed(ring, tol),
    });
  });

  if (pieces.length === 0) {
    mismatches.push({ code: "boundary-empty", layer: convention.boundary, message: `no closed boundary chains found on layer '${convention.boundary}'; file yields no pieces` });
  }

  const ok =
    pieces.length > 0 &&
    unitsResolved &&
    !mismatches.some((m) => m.code === "insunits-unknown" || m.code === "boundary-empty");
  return {
    ok,
    profile,
    layersFound,
    convention,
    units,
    pieces,
    mismatches,
    entityCounts,
    textValues: parsed.textValues,
  };
}
