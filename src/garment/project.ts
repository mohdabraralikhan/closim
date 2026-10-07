/**
 * G8D — versioned native garment project (file format).
 *
 * A GarmentProject bundles everything a commercial garment needs in one
 * deterministic document:
 *
 *   project metadata | PatternDocument (G8A) | Seam[] (G8B) |
 *   material table | avatar reference | simulation config | schema version
 *
 * Determinism: canonical JSON (sorted keys, -0 normalized); entity IDs and
 * panel ordering come from the embedded PatternDocument unchanged; repeated
 * serialize/deserialize round-trips are byte-identical. No timestamps, no
 * randomness, no absolute paths.
 */

import {
  validatePatternDocument,
  type PatternDocument,
} from "../pattern/cad.js";
import { validateSeams, type Seam } from "./sewing.js";
import { validateAvatarSpec, type AvatarSpec } from "./avatar.js";
import {
  assembleGarment,
  createFittingScene,
  type AssembleOptions,
  type AssembledGarment,
  type FittingScene,
  type PanelPlacement,
} from "./assembly.js";
import { DEFAULT_MATERIAL, type ClothMaterial } from "../physics/types.js";
import { DEFAULT_CONTACT_PARAMS, type ContactParams } from "../collision/types.js";

export const GARMENT_PROJECT_SCHEMA_VERSION = 1;

export interface GarmentProjectMetadata {
  name: string;
  /** Free-form description. */
  description?: string;
  /** Author label (not an identity system). */
  author?: string;
  /** Monotonic revision counter owned by the editor. */
  revision: number;
}

export interface SimulationConfig {
  dt: number;
  relaxationSteps: number;
  simulationSteps: number;
  gravity: [number, number, number];
  arealDensityKgM2: number;
  contact: ContactParams;
  /** World-space floor height, or null for no floor. */
  floorY: number | null;
  /**
   * Interior cloth mesh density target (metres, null = boundary
   * triangulation untouched). Set for garments meant to drape.
   */
  meshMaxEdgeM: number | null;
}

export interface GarmentProject {
  schemaVersion: typeof GARMENT_PROJECT_SCHEMA_VERSION;
  id: string;
  metadata: GarmentProjectMetadata;
  pattern: PatternDocument;
  seams: Seam[];
  /** Material id -> physical parameters. Must cover every panel materialId. */
  materials: Record<string, ClothMaterial>;
  /** Default 3D placement per panel (panel order = placements order). */
  placements: PanelPlacement[];
  avatar: AvatarSpec | null;
  simulation: SimulationConfig;
}

export type ProjectDiagnosticCode =
  | "invalid-project"
  | "unsupported-schema"
  | "missing-material"
  | "invalid-material"
  | "invalid-simulation-config"
  | "invalid-placement"
  | "invalid-avatar"
  | "pattern-error"
  | "seam-error";

export interface ProjectDiagnostic {
  code: ProjectDiagnosticCode;
  message: string;
  entityId?: string;
}

export interface ProjectValidation {
  valid: boolean;
  diagnostics: ProjectDiagnostic[];
}

export function defaultSimulationConfig(): SimulationConfig {
  return {
    dt: 1 / 60,
    relaxationSteps: 5,
    simulationSteps: 10,
    gravity: [0, -9.81, 0],
    arealDensityKgM2: 0.15,
    contact: { ...DEFAULT_CONTACT_PARAMS },
    floorY: null,
    meshMaxEdgeM: null,
  };
}

export function createGarmentProject(
  id: string,
  name: string,
  pattern: PatternDocument,
  opts: {
    description?: string;
    author?: string;
    seams?: Seam[];
    materials?: Record<string, ClothMaterial>;
    placements?: PanelPlacement[];
    avatar?: AvatarSpec | null;
    simulation?: Partial<SimulationConfig>;
  } = {},
): GarmentProject {
  if (typeof id !== "string" || id.trim().length === 0) {
    throw new Error("garment project: id must be a non-empty string");
  }
  return {
    schemaVersion: GARMENT_PROJECT_SCHEMA_VERSION,
    id,
    metadata: {
      name,
      ...(opts.description ? { description: opts.description } : {}),
      ...(opts.author ? { author: opts.author } : {}),
      revision: 1,
    },
    pattern: JSON.parse(JSON.stringify(pattern)) as PatternDocument,
    seams: JSON.parse(JSON.stringify(opts.seams ?? [])) as Seam[],
    materials: JSON.parse(JSON.stringify(opts.materials ?? { "default-material": { ...DEFAULT_MATERIAL } })) as Record<string, ClothMaterial>,
    placements: JSON.parse(JSON.stringify(opts.placements ?? [])) as PanelPlacement[],
    avatar: opts.avatar ? JSON.parse(JSON.stringify(opts.avatar)) as AvatarSpec : null,
    simulation: { ...defaultSimulationConfig(), ...(opts.simulation ?? {}) },
  };
}

function finiteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function validateGarmentProject(project: GarmentProject): ProjectValidation {
  const diagnostics: ProjectDiagnostic[] = [];
  const push = (code: ProjectDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  if (!project || typeof project !== "object") {
    return { valid: false, diagnostics: [{ code: "invalid-project", message: "project must be an object" }] };
  }
  if (project.schemaVersion !== GARMENT_PROJECT_SCHEMA_VERSION) {
    push("unsupported-schema", `schemaVersion ${String((project as { schemaVersion?: unknown }).schemaVersion)} is unsupported; expected ${GARMENT_PROJECT_SCHEMA_VERSION}`);
    return { valid: false, diagnostics };
  }
  if (typeof project.id !== "string" || project.id.trim() === "") {
    push("invalid-project", "project id must be a non-empty string");
  }
  const meta = project.metadata;
  if (!meta || typeof meta.name !== "string" || meta.name.length === 0) {
    push("invalid-project", "metadata.name must be a non-empty string");
  }
  if (!Number.isInteger(meta?.revision) || (meta?.revision ?? 0) < 1) {
    push("invalid-project", "metadata.revision must be a positive integer");
  }

  // Pattern (G8A owns the detailed diagnostics; summarize here).
  try {
    const result = validatePatternDocument(project.pattern);
    if (!result.valid) {
      for (const d of result.diagnostics) {
        push("pattern-error", `${d.code}: ${d.message}`, d.entityId ?? d.panelId);
      }
    }
  } catch (error) {
    push("pattern-error", error instanceof Error ? error.message : String(error));
  }

  // Seams (G8B owns detail).
  try {
    const result = validateSeams(project.pattern, project.seams ?? []);
    if (!result.valid) {
      for (const d of result.diagnostics) push("seam-error", `${d.seamId}: ${d.code} — ${d.message}`, d.seamId);
    }
  } catch (error) {
    push("seam-error", error instanceof Error ? error.message : String(error));
  }

  // Materials: every panel materialId must resolve; params must be finite/sane.
  const materials = project.materials ?? {};
  const panelMaterialIds = new Set((project.pattern?.panels ?? []).map((p) => p.materialId));
  for (const mid of panelMaterialIds) {
    if (!materials[mid]) push("missing-material", `panel material '${mid}' has no entry in materials`, mid);
  }
  for (const [mid, mat] of Object.entries(materials)) {
    const fields: Array<[string, number | undefined, (v: number) => boolean]> = [
      ["arealDensityKgM2", mat?.arealDensityKgM2, (v) => v > 0],
      ["thickness", mat?.thickness, (v) => v > 0],
      ["stretchWarp", mat?.stretchWarp, (v) => v > 0],
      ["stretchWeft", mat?.stretchWeft, (v) => v > 0],
      ["shear", mat?.shear, (v) => v > 0],
      ["bendWarp", mat?.bendWarp, (v) => v >= 0],
      ["bendWeft", mat?.bendWeft, (v) => v >= 0],
      ["damping", mat?.damping, (v) => v >= 0 && v <= 0.1],
    ];
    for (const [field, value, ok] of fields) {
      if (!finiteNumber(value) || !ok(value as number)) {
        push("invalid-material", `material '${mid}.${field}' is invalid (${String(value)})`, mid);
      }
    }
    if (mat && !finiteNumber(mat.stretchCoupling)) {
      push("invalid-material", `material '${mid}.stretchCoupling' must be finite`, mid);
    }
  }

  // Placements: finite, unique, reference known panels.
  const panelIds = new Set((project.pattern?.panels ?? []).map((p) => p.id));
  const seenPlacements = new Set<string>();
  for (const p of project.placements ?? []) {
    if (!panelIds.has(p.panelId)) {
      push("invalid-placement", `placement references unknown panel '${p.panelId}'`, p.panelId);
    }
    if (seenPlacements.has(p.panelId)) {
      push("invalid-placement", `duplicate placement for panel '${p.panelId}'`, p.panelId);
    }
    seenPlacements.add(p.panelId);
    if (!Array.isArray(p.translation) || p.translation.length !== 3 || !p.translation.every(finiteNumber) || !finiteNumber(p.yawRad)) {
      push("invalid-placement", `placement for panel '${p.panelId}' is non-finite`, p.panelId);
    }
    const wrap = (p as { wrap?: unknown }).wrap;
    if (wrap !== undefined && wrap !== null) {
      const w = wrap as { center?: unknown; radiusM?: unknown; facingRad?: unknown; refLx?: unknown; mirror?: unknown };
      const centerOk = Array.isArray(w.center) && w.center.length === 2 && w.center.every(finiteNumber);
      if (!centerOk || !finiteNumber(w.radiusM) || !(w.radiusM as number > 0) ||
        !finiteNumber(w.facingRad) || !finiteNumber(w.refLx) ||
        (w.mirror !== undefined && w.mirror !== null && typeof w.mirror !== "boolean")) {
        push("invalid-placement", `placement wrap for panel '${p.panelId}' must have finite center/facing/refLx, positive radiusM, boolean mirror`, p.panelId);
      }
    }
  }

  // Avatar.
  if (project.avatar !== null && project.avatar !== undefined) {
    try {
      validateAvatarSpec(project.avatar);
    } catch (error) {
      push("invalid-avatar", error instanceof Error ? error.message : String(error));
    }
  }

  // Simulation config.
  const sim = project.simulation;
  if (!sim || !finiteNumber(sim.dt) || !(sim.dt > 0) || !Number.isInteger(sim.relaxationSteps) || sim.relaxationSteps < 0 ||
    !Number.isInteger(sim.simulationSteps) || sim.simulationSteps < 0 ||
    !Array.isArray(sim.gravity) || sim.gravity.length !== 3 || !sim.gravity.every(finiteNumber) ||
    !finiteNumber(sim.arealDensityKgM2) || !(sim.arealDensityKgM2 > 0)) {
    push("invalid-simulation-config", "simulation dt/steps/gravity/arealDensity are invalid");
  } else {
    const c = sim.contact;
    if (!c || !finiteNumber(c.dHatM) || !(c.dHatM > 0) || !finiteNumber(c.kappaJ) || !(c.kappaJ > 0) ||
      !finiteNumber(c.dMinM) || !(c.dMinM > 0) || !(c.dMinM < c.dHatM) ||
      !finiteNumber(c.frictionMu) || c.frictionMu < 0 || !finiteNumber(c.frictionEpsM) || !(c.frictionEpsM > 0)) {
      push("invalid-simulation-config", "simulation contact parameters are invalid");
    }
    if (sim.floorY !== null && sim.floorY !== undefined && !finiteNumber(sim.floorY)) {
      push("invalid-simulation-config", "simulation floorY must be finite or null");
    }
    if (sim.meshMaxEdgeM !== null && sim.meshMaxEdgeM !== undefined &&
      (!finiteNumber(sim.meshMaxEdgeM) || !(sim.meshMaxEdgeM > 0))) {
      push("invalid-simulation-config", "simulation meshMaxEdgeM must be a positive length or null");
    }
  }

  const valid = diagnostics.length === 0;
  return { valid, diagnostics };
}

// --- canonical JSON (sorted keys, -0 normalized) ----------------------------

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export function serializeGarmentProject(project: GarmentProject): string {
  const validation = validateGarmentProject(project);
  if (!validation.valid) {
    throw new Error(`cannot serialize invalid garment project:\n${validation.diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  return canonicalJson(project);
}

export function deserializeGarmentProject(serialized: string): GarmentProject {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("garment project: serialized project is not valid JSON");
  }
  const validation = validateGarmentProject(parsed as GarmentProject);
  if (!validation.valid) {
    throw new Error(`invalid garment project:\n${validation.diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  return JSON.parse(JSON.stringify(parsed)) as GarmentProject;
}

// --- rebuild: 2D edit -> 3D result without reconstructing the project -------

export interface RebuildResult {
  assembled: AssembledGarment;
  fitting: FittingScene;
}

/**
 * Rebuild simulation input from the current project state. A 2D pattern edit
 * (mutating `project.pattern` / bumping `metadata.revision`) flows through
 * this single call — the caller never rebuilds the project manually.
 */
export function rebuildGarment(
  project: GarmentProject,
  assembleOpts: AssembleOptions = {},
): RebuildResult {
  const validation = validateGarmentProject(project);
  if (!validation.valid) {
    throw new Error(`cannot rebuild invalid garment project:\n${validation.diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  const material = project.materials[project.pattern.panels[0]?.materialId] ?? { ...DEFAULT_MATERIAL };
  // Solver-facing paths (placement diagnostics, contact) use the decimated
  // proxy when the avatar carries one; rendering keeps the full mesh.
  const simAvatar = project.avatar?.collision ?? project.avatar ?? null;
  const assembled = assembleGarment(project.pattern, project.seams, project.placements, {
    ...assembleOpts,
    avatar: assembleOpts.avatar !== undefined ? assembleOpts.avatar : simAvatar,
    interiorMaxEdgeM: assembleOpts.interiorMaxEdgeM !== undefined
      ? assembleOpts.interiorMaxEdgeM
      : (project.simulation.meshMaxEdgeM ?? undefined),
  });
  const fitting = createFittingScene(assembled, {
    material,
    arealDensityKgM2: project.simulation.arealDensityKgM2,
    gravity: project.simulation.gravity,
    contact: project.simulation.contact,
    avatar: simAvatar ? [simAvatar] : null,
    floorY: project.simulation.floorY,
  });
  return { assembled, fitting };
}

/** Apply a pattern edit and bump the revision (the G8 edit loop primitive). */
export function applyPatternEdit(
  project: GarmentProject,
  edit: (pattern: PatternDocument) => PatternDocument,
): GarmentProject {
  const next: GarmentProject = {
    ...JSON.parse(JSON.stringify(project)) as GarmentProject,
    metadata: { ...project.metadata, revision: project.metadata.revision + 1 },
  };
  next.pattern = edit(next.pattern);
  const validation = validateGarmentProject(next);
  if (!validation.valid) {
    throw new Error(`pattern edit produced an invalid project:\n${validation.diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  return next;
}
