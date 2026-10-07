// G13B — DXF apparel export profiles.
//
// A profile is a *declared contract*: which entities the exporter may emit,
// which layers it uses, its unit assumption, and — critically — what it does
// NOT support. Adapters consult the profile; nothing apparel-specific leaks
// into the CAD writer itself.
//
// Honesty policy: profiles describe conventions documented by the community
// and by public summaries of AAMA/ASTM DXF practice. This repository does not
// redistribute the standards, and no profile claims certified compliance:
// `certified` stays false and every export repeats the caveat. Consumers who
// need certified files must validate against the purchased standard text.
//
// Layer conventions (AAMA-style numeric layers, widely documented):
//   1  cut/boundary outline        8  internal lines
//   9  (reserved; unused here)     10 notches
//   11 drill marks                 13 annotation text/notes
//   14 sewing line                 16 grain line
// The generic profile uses the same semantic mapping with named layers so
// plain CAD tools get a readable drawing.

export type DxfProfileId = "generic-r12" | "aama-style" | "astm-oriented";

export type DxfEntityKind =
  | "boundary" | "sewing" | "allowance" | "notch" | "grain" | "fold"
  | "drill" | "internal" | "text" | "dimension" | "hole" | "construction";

/** What the exporter writes into a DXF file for one profile. */
export interface DxfProfile {
  id: DxfProfileId;
  /** Human-readable label (used in warnings + manifests). */
  label: string;
  /** Numeric DXF layers (AAMA/ASTM style) or named layers (generic). */
  layers: Record<DxfEntityKind, string>;
  /** ACI colours per layer, deterministic order. */
  layerColors: Record<string, number>;
  /** Units the profile's numeric coordinates represent. */
  units: ExportUnitsDxf;
  /** $INSUNITS header value consistent with `units`. */
  insunits: number;
  /** Entities this profile will NOT emit; adapter turns them into warnings. */
  unsupported: DxfEntityKind[];
  /** How grading data is carried. */
  grading: {
    /** "sizes-as-text" = one TEXT per size under each piece; "separate-file" = don't embed; "none". */
    mode: "sizes-as-text" | "separate-file" | "none";
    /** Multi-size DXF: every piece emitted once per size, suffixed labels. */
    multiSize: boolean;
  };
  /** Flattening behaviour the adapter must declare. */
  geometry: {
    /** Emit true ARC entities (true for all current profiles). */
    arcs: boolean;
    /** Text handling: TEXT entity per label field (no MTEXT). */
    textEntity: "TEXT";
  };
  /** Free-text note repeated in every export's warnings. */
  complianceNote: string;
  /** Always false: no certified-standards claims in this repo. */
  certified: false;
}

type ExportUnitsDxf = "mm" | "cm" | "m" | "in";

const GENERIC_NOTE =
  "generic DXF R12 subset (LINE/ARC/CIRCLE/TEXT); apparel semantics mapped to named layers; " +
  "not certified against AAMA or ASTM D6673 — validate against the applicable standard text before factory use";

const AAMA_NOTE =
  "AAMA-style numeric layers per community-documented practice (1 cut, 8 internal, 10 notch, " +
  "11 drill, 13 note, 14 sew, 16 grain); this file is NOT certified AAMA output — verify in the receiving system";

const ASTM_NOTE =
  "ASTM-D6673-oriented DXF: 2D sewn-pattern piece exchange with grade data as size text. " +
  "ASTM D6673 covers pattern-piece and grade-rule exchange and explicitly does NOT cover " +
  "numerical cutter instructions or complete marker laying/spreading data; this file is neither, " +
  "and is not certified ASTM output";

const SHARED_LAYERS: Record<DxfEntityKind, string> = {
  boundary: "1",
  sewing: "14",
  allowance: "8",
  notch: "10",
  grain: "16",
  fold: "8",
  drill: "11",
  internal: "8",
  text: "13",
  dimension: "13",
  hole: "1",
  construction: "8",
};

/** All three profiles share the semantic layer mapping; only names differ. */
const NAMED_LAYERS: Record<DxfEntityKind, string> = {
  boundary: "CUT",
  sewing: "SEW",
  allowance: "ALLOWANCE",
  notch: "NOTCH",
  grain: "GRAIN",
  fold: "FOLD",
  drill: "DRILL",
  internal: "INTERNAL",
  text: "LABEL",
  dimension: "DIM",
  hole: "CUT",
  construction: "CONSTRUCTION",
};

function numericProfileLayers(): Record<DxfEntityKind, string> {
  return { ...SHARED_LAYERS };
}

function layerColorsFor(names: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  names.forEach((n, i) => { out[n] = [7, 1, 2, 3, 4, 5, 6][i % 7]; });
  return out;
}

export const DXF_PROFILES: Record<DxfProfileId, DxfProfile> = {
  "generic-r12": {
    id: "generic-r12",
    label: "Generic DXF R12 (named layers)",
    layers: NAMED_LAYERS,
    layerColors: layerColorsFor(Object.values(NAMED_LAYERS)),
    units: "mm",
    insunits: 4,
    unsupported: ["dimension"],
    grading: { mode: "sizes-as-text", multiSize: true },
    geometry: { arcs: true, textEntity: "TEXT" },
    complianceNote: GENERIC_NOTE,
    certified: false,
  },
  "aama-style": {
    id: "aama-style",
    label: "AAMA-style numeric layers",
    layers: numericProfileLayers(),
    layerColors: layerColorsFor(Object.values(SHARED_LAYERS)),
    units: "mm",
    insunits: 4,
    unsupported: ["fold", "construction"],
    grading: { mode: "sizes-as-text", multiSize: true },
    geometry: { arcs: true, textEntity: "TEXT" },
    complianceNote: AAMA_NOTE,
    certified: false,
  },
  "astm-oriented": {
    id: "astm-oriented",
    label: "ASTM D6673-oriented (pattern pieces + grade text)",
    layers: numericProfileLayers(),
    layerColors: layerColorsFor(Object.values(SHARED_LAYERS)),
    units: "cm",
    insunits: 5,
    grading: { mode: "sizes-as-text", multiSize: true },
    unsupported: ["fold", "construction", "dimension"],
    geometry: { arcs: true, textEntity: "TEXT" },
    complianceNote: ASTM_NOTE,
    certified: false,
  },
};

/** Resolve a profile by id (throws on unknown — never fall back silently). */
export function dxfProfile(id: DxfProfileId): DxfProfile {
  const p = DXF_PROFILES[id];
  if (!p) throw new Error(`unknown DXF profile '${id}'`);
  return p;
}

/** Units declared by a profile, as an export-units key. */
export function profileUnits(p: DxfProfile): "mm" | "cm" {
  return p.units === "cm" ? "cm" : "mm";
}
