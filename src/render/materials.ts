// G15B — visual fabric/material system.
//
// A VisualMaterial describes APPEARANCE ONLY: base color, roughness,
// metallic, opacity, texture transforms, weave/print descriptors. It never
// carries solver physics — visual roughness is not friction, visual
// stiffness does not exist here. The optional physicalRef names the solver
// material id this visual belongs with; the mapping is by reference, and a
// visual edit never mutates cloth physics.
//
// Textures are descriptors ({kind, ref}), not pixels: loading happens in the
// browser adapter. Missing textures resolve to a deterministic fallback.

import { PatternCadError } from "../pattern/cad.js";

export const MATERIAL_SCHEMA_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type RGB = [number, number, number];

export type TextureKind = "none" | "procedural" | "file" | "data-url";

export interface TextureRef {
  kind: TextureKind;
  /** Procedural name ("checker", "twill", "satin", "plain"), path, or data URL. */
  ref: string;
  /** Preferred resolution for file/data-url textures (0 = native). */
  resolutionPx?: number;
}

export interface VisualMaterial {
  schemaVersion: typeof MATERIAL_SCHEMA_VERSION;
  id: string;
  name: string;
  version: number;
  baseColor: RGB;
  roughness: number;
  metallic: number;
  opacity: number;
  /** Normal/bump strength multiplier (0 = off). */
  normalScale: number;
  /** Texture tiling in repeats per metre. */
  textureScale: number;
  /** Texture rotation in degrees. */
  textureRotationDeg: number;
  /** Weave/print appearance tag (free-form, e.g. "twill-2/1"). */
  weave: string;
  /** Edge (hem/cut) tint, or null for base color. */
  edgeColor: RGB | null;
  colorMap: TextureRef;
  normalMap: TextureRef;
  roughnessMap: TextureRef;
  /** Solver material id this visual belongs with (reference only). */
  physicalRef?: string;
}

function finiteUnit(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function finiteRGB(value: unknown): value is RGB {
  return Array.isArray(value) && value.length === 3 && value.every(finiteUnit);
}

export function createMaterial(partial: {
  id: string;
  name: string;
  baseColor: RGB;
  roughness?: number;
  metallic?: number;
  opacity?: number;
  normalScale?: number;
  textureScale?: number;
  textureRotationDeg?: number;
  weave?: string;
  edgeColor?: RGB | null;
  colorMap?: TextureRef;
  normalMap?: TextureRef;
  roughnessMap?: TextureRef;
  physicalRef?: string;
}): VisualMaterial {
  const { id, name, baseColor } = partial;
  if (!id || !name) throw new PatternCadError("invalid-document", "material needs id and name");
  if (!finiteRGB(baseColor)) throw new PatternCadError("invalid-transform", `material '${id}' base color invalid`);
  const num = (v: number | undefined, fallback: number, label: string): number => {
    const value = v ?? fallback;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new PatternCadError("invalid-transform", `material '${id}' ${label} invalid`);
    }
    return value;
  };
  const roughness = num(partial.roughness, 0.85, "roughness");
  const metallic = num(partial.metallic, 0, "metallic");
  if (roughness > 1 || metallic > 1) {
    throw new PatternCadError("invalid-transform", `material '${id}' roughness/metallic must be <= 1`);
  }
  const opacity = num(partial.opacity, 1, "opacity");
  if (opacity > 1) throw new PatternCadError("invalid-transform", `material '${id}' opacity must be <= 1`);
  const texture = (t: TextureRef | undefined): TextureRef => {
    const tex = t ?? { kind: "none", ref: "" };
    if (tex.kind !== "none" && tex.kind !== "procedural" && tex.kind !== "file" && tex.kind !== "data-url") {
      throw new PatternCadError("invalid-transform", `material '${id}' texture kind invalid`);
    }
    if (tex.kind !== "none" && !tex.ref) {
      throw new PatternCadError("invalid-transform", `material '${id}' texture needs a ref`);
    }
    if (tex.resolutionPx !== undefined && (!Number.isInteger(tex.resolutionPx) || tex.resolutionPx < 0)) {
      throw new PatternCadError("invalid-transform", `material '${id}' texture resolution invalid`);
    }
    return clone(tex);
  };
  if (partial.edgeColor !== undefined && partial.edgeColor !== null && !finiteRGB(partial.edgeColor)) {
    throw new PatternCadError("invalid-transform", `material '${id}' edge color invalid`);
  }
  return {
    schemaVersion: MATERIAL_SCHEMA_VERSION,
    id,
    name,
    version: 1,
    baseColor: [...baseColor],
    roughness,
    metallic,
    opacity,
    normalScale: num(partial.normalScale, 0, "normalScale"),
    textureScale: num(partial.textureScale, 4, "textureScale"),
    textureRotationDeg: partial.textureRotationDeg ?? 0,
    weave: partial.weave ?? "plain",
    edgeColor: partial.edgeColor === undefined ? null : partial.edgeColor ? [...partial.edgeColor] : null,
    colorMap: texture(partial.colorMap),
    normalMap: texture(partial.normalMap),
    roughnessMap: texture(partial.roughnessMap),
    ...(partial.physicalRef ? { physicalRef: partial.physicalRef } : {}),
  };
}

export function validateMaterial(material: VisualMaterial): string[] {
  const errors: string[] = [];
  if (!material || typeof material !== "object" || material.schemaVersion !== MATERIAL_SCHEMA_VERSION) {
    return ["material shape or schema version is invalid"];
  }
  try {
    createMaterial(material);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  return errors;
}

// ---------------------------------------------------------------------------
// Fabric presets (appearance only — no physical-accuracy claims)
// ---------------------------------------------------------------------------

export interface FabricPreset extends Pick<VisualMaterial,
  "baseColor" | "roughness" | "metallic" | "normalScale" | "textureScale" | "weave"> {
  description: string;
}

export const FABRIC_PRESETS: Record<string, FabricPreset> = {
  cotton: {
    description: "matte plain-weave cotton appearance",
    baseColor: [0.92, 0.9, 0.86], roughness: 0.9, metallic: 0,
    normalScale: 0.4, textureScale: 6, weave: "plain",
  },
  denim: {
    description: "twill denim appearance with visible diagonal",
    baseColor: [0.16, 0.25, 0.45], roughness: 0.85, metallic: 0,
    normalScale: 0.8, textureScale: 5, weave: "twill-2/1",
  },
  silk: {
    description: "low-roughness satin sheen appearance",
    baseColor: [0.95, 0.9, 0.85], roughness: 0.35, metallic: 0,
    normalScale: 0.15, textureScale: 8, weave: "satin",
  },
  wool: {
    description: "soft high-roughness wool appearance",
    baseColor: [0.45, 0.42, 0.4], roughness: 0.95, metallic: 0,
    normalScale: 0.6, textureScale: 4, weave: "plain-brushed",
  },
  linen: {
    description: "slubbed linen appearance",
    baseColor: [0.85, 0.8, 0.7], roughness: 0.88, metallic: 0,
    normalScale: 0.7, textureScale: 3, weave: "plain-slub",
  },
  synthetic: {
    description: "smooth synthetic appearance with slight sheen",
    baseColor: [0.7, 0.72, 0.75], roughness: 0.5, metallic: 0,
    normalScale: 0.1, textureScale: 10, weave: "plain-filament",
  },
};

export function materialFromPreset(
  id: string, name: string, preset: keyof typeof FABRIC_PRESETS,
  overrides: Partial<VisualMaterial> = {},
): VisualMaterial {
  const base = FABRIC_PRESETS[preset];
  if (!base) throw new PatternCadError("invalid-document", `unknown fabric preset '${preset}'`);
  const { description: _description, ...look } = base;
  void _description;
  return createMaterial({ id, name, ...look, ...overrides });
}

// ---------------------------------------------------------------------------
// Material library (assignment + texture cache + fallback)
// ---------------------------------------------------------------------------

export const FALLBACK_MATERIAL_ID = "material/fallback-grey";

export function fallbackMaterial(): VisualMaterial {
  return createMaterial({
    id: FALLBACK_MATERIAL_ID, name: "Fallback grey",
    baseColor: [0.6, 0.6, 0.6], roughness: 0.9, metallic: 0, weave: "plain",
  });
}

export interface MaterialLibrary {
  materials: VisualMaterial[];
  /** Panel id -> material id (garment assignment). */
  assignment: Record<string, string>;
  /** Texture id -> load state (adapter-reported). */
  textures: Record<string, { status: "empty" | "loading" | "ready" | "failed"; resolutionPx: number }>;
}

export function createLibrary(): MaterialLibrary {
  return { materials: [fallbackMaterial()], assignment: {}, textures: {} };
}

export function addMaterial(library: MaterialLibrary, material: VisualMaterial): MaterialLibrary {
  const errors = validateMaterial(material);
  if (errors.length > 0) throw new PatternCadError("invalid-document", `invalid material: ${errors[0]}`);
  if (library.materials.some((m) => m.id === material.id)) {
    throw new PatternCadError("duplicate-id", `material '${material.id}' already in library`, material.id);
  }
  const next = clone(library);
  next.materials.push(clone(material));
  for (const tex of [material.colorMap, material.normalMap, material.roughnessMap]) {
    if (tex.kind !== "none" && next.textures[tex.ref] === undefined) {
      next.textures[tex.ref] = { status: "empty", resolutionPx: tex.resolutionPx ?? 0 };
    }
  }
  return next;
}

export function replaceMaterial(library: MaterialLibrary, material: VisualMaterial): MaterialLibrary {
  const errors = validateMaterial(material);
  if (errors.length > 0) throw new PatternCadError("invalid-document", `invalid material: ${errors[0]}`);
  const index = library.materials.findIndex((m) => m.id === material.id);
  if (index < 0) throw new PatternCadError("missing-reference", `material '${material.id}' not in library`, material.id);
  const next = clone(library);
  next.materials[index] = { ...clone(material), version: library.materials[index].version + 1 };
  return next;
}

export function assignMaterial(library: MaterialLibrary, panelId: string, materialId: string): MaterialLibrary {
  if (!library.materials.some((m) => m.id === materialId)) {
    throw new PatternCadError("missing-reference", `material '${materialId}' not in library`, materialId);
  }
  const next = clone(library);
  next.assignment[panelId] = materialId;
  return next;
}

/** Resolve with fallback: unknown ids yield the fallback material (and the miss is reported). */
export function resolveMaterial(library: MaterialLibrary, materialId: string): { material: VisualMaterial; missed: boolean } {
  const found = library.materials.find((m) => m.id === materialId);
  if (found) return { material: found, missed: false };
  return { material: library.materials.find((m) => m.id === FALLBACK_MATERIAL_ID)!, missed: true };
}

export function markTexture(
  library: MaterialLibrary, ref: string, status: "loading" | "ready" | "failed", resolutionPx = 0,
): MaterialLibrary {
  const next = clone(library);
  next.textures[ref] = { status, resolutionPx };
  return next;
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

export function serializeLibrary(library: MaterialLibrary): string {
  return canonicalJson(library);
}

export function deserializeLibrary(serialized: string): MaterialLibrary {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized material library is not valid JSON");
  }
  const library = parsed as MaterialLibrary;
  if (!library || !Array.isArray(library.materials) || typeof library.assignment !== "object") {
    throw new PatternCadError("invalid-document", "material library shape is invalid");
  }
  for (const material of library.materials) {
    const errors = validateMaterial(material);
    if (errors.length > 0) throw new PatternCadError("invalid-document", `invalid material: ${errors[0]}`);
  }
  return clone(library);
}
