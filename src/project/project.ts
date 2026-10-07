// G17A — commercial project model, operations, and migrations.
//
// Layers (strict):
//   SOURCE   — AppProject: garments, construction, grading, production,
//              markers, materials, scenes, project settings.
//   DERIVED  — simulation snapshots, render caches, nesting results, export
//              files, thumbnails. Rebuildable; never authoritative.
//   USER     — preferences/session (owned by the settings track, not here).
//
// Timestamps are wall-clock metadata (ISO strings), excluded from
// content-fingerprint comparisons. Geometry determinism is unaffected:
// same garment content always serializes identically.

import { PatternCadError } from "../pattern/cad.js";
import {
  validateGarmentProject,
  type GarmentProject,
} from "../garment/project.js";
import type { ConstructionSet } from "../construction/features.js";
import type { ComponentSet } from "../construction/components.js";
import type { GradingDocument } from "../grading/types.js";
import type { ProductionSet } from "../cad/production.js";
import { validateMarker, type Marker } from "../marker/model.js";
import type { MaterialLibrary } from "../render/materials.js";
import type { RenderPackage } from "../render/package.js";
import {
  validateProductionSpecification,
  type ProductionSpecification,
} from "../production/specification-model.js";
import { validateBOM, type BOM } from "../production/bom.js";
import { validateProductionRun, type ProductionRun } from "../production/run.js";
import { validateRevisionLedger, type RevisionLedger } from "../production/revision.js";

export const PROJECT_SCHEMA_VERSION = 1;
export const APP_VERSION = "0.1.0";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export interface ProjectExportRef {
  id: string;
  kind: string;
  path: string;
  createdAt: string;
}

export interface ProjectGarmentEntry {
  id: string;
  name: string;
  garment: GarmentProject;
  construction?: ConstructionSet;
  components?: ComponentSet;
  grading?: GradingDocument;
  production?: ProductionSet;
  productionSpecification?: ProductionSpecification;
  bom?: BOM;
  productionRun?: ProductionRun;
  revisionLedger?: RevisionLedger;
  marker?: Marker;
  markers?: Marker[];
  materials?: MaterialLibrary;
  presentation?: RenderPackage;
  /** Export artifacts derived from this garment (payloads live under exports/). */
  exports?: ProjectExportRef[];
  /** Project-level settings that travel with the garment (units default etc.). */
  settings?: Record<string, string>;
}

export interface ProjectMetadata {
  createdAt: string;
  updatedAt: string;
  applicationVersion: string;
}

export interface AppProject {
  schemaVersion: typeof PROJECT_SCHEMA_VERSION;
  id: string;
  name: string;
  metadata: ProjectMetadata;
  garments: ProjectGarmentEntry[];
  /** Opaque project settings (key/value strings). */
  settings: Record<string, string>;
}

export function createProject(id: string, name: string, now?: string): AppProject {
  if (!id || !name) throw new PatternCadError("invalid-document", "project needs id and name");
  const timestamp = now ?? new Date().toISOString();
  return {
    schemaVersion: PROJECT_SCHEMA_VERSION,
    id,
    name,
    metadata: { createdAt: timestamp, updatedAt: timestamp, applicationVersion: APP_VERSION },
    garments: [],
    settings: {},
  };
}

function touch(project: AppProject, now?: string): void {
  project.metadata.updatedAt = now ?? new Date().toISOString();
}

export function addGarment(
  project: AppProject, entry: ProjectGarmentEntry, now?: string,
): AppProject {
  const next = clone(project);
  if (next.garments.some((g) => g.id === entry.id)) {
    throw new PatternCadError("duplicate-id", `garment '${entry.id}' already in project`, entry.id);
  }
  if (!entry.id || !entry.name || !entry.garment) {
    throw new PatternCadError("invalid-document", "garment entry needs id, name, and garment");
  }
  const validation = validateGarmentProject(entry.garment);
  if (!validation.valid) {
    throw new PatternCadError(
      "invalid-document",
      `garment '${entry.id}' is invalid: ${validation.diagnostics[0].code}: ${validation.diagnostics[0].message}`,
      entry.id,
    );
  }
  next.garments.push(clone(entry));
  touch(next, now);
  return next;
}

export function removeGarment(project: AppProject, garmentId: string, now?: string): AppProject {
  const next = clone(project);
  if (!next.garments.some((g) => g.id === garmentId)) {
    throw new PatternCadError("missing-reference", `garment '${garmentId}' not in project`, garmentId);
  }
  next.garments = next.garments.filter((g) => g.id !== garmentId);
  touch(next, now);
  return next;
}

export function renameProject(project: AppProject, name: string, now?: string): AppProject {
  if (!name) throw new PatternCadError("invalid-document", "project name must be non-empty");
  const next = clone(project);
  next.name = name;
  touch(next, now);
  return next;
}

/** Deep duplicate with a fresh project id (garment ids preserved: same content). */
export function duplicateProject(project: AppProject, newId: string, now?: string): AppProject {
  if (!newId) throw new PatternCadError("invalid-document", "duplicate needs a new project id");
  if (newId === project.id) {
    throw new PatternCadError("duplicate-id", "duplicate project id must differ from the source");
  }
  const next = clone(project);
  next.id = newId;
  const timestamp = now ?? new Date().toISOString();
  next.metadata = { createdAt: timestamp, updatedAt: timestamp, applicationVersion: APP_VERSION };
  return next;
}

export function setProjectSetting(project: AppProject, key: string, value: string, now?: string): AppProject {
  if (!key) throw new PatternCadError("invalid-document", "setting key must be non-empty");
  const next = clone(project);
  next.settings[key] = value;
  touch(next, now);
  return next;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type ProjectDiagnosticCode =
  | "invalid-document"
  | "unsupported-schema"
  | "duplicate-id"
  | "missing-reference"
  | "invalid-garment"
  | "invalid-production-specification"
  | "invalid-production-data";

export interface ProjectDiagnostic {
  code: ProjectDiagnosticCode;
  message: string;
  entityId?: string;
}

export function validateProject(project: AppProject): ProjectDiagnostic[] {
  const diagnostics: ProjectDiagnostic[] = [];
  const fail = (code: ProjectDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  if (!project || typeof project !== "object" || !Array.isArray(project.garments)) {
    return [{ code: "invalid-document", message: "project shape is invalid" }];
  }
  if (project.schemaVersion !== PROJECT_SCHEMA_VERSION) {
    fail("unsupported-schema", `schema version ${String(project.schemaVersion)} unsupported (current ${PROJECT_SCHEMA_VERSION})`);
    return diagnostics;
  }
  if (!project.id || !project.name) fail("invalid-document", "project needs id and name");
  const seen = new Set<string>();
  for (const entry of project.garments) {
    if (!entry.id || seen.has(entry.id)) {
      fail("duplicate-id", `duplicate or missing garment id '${String(entry?.id)}'`, entry?.id);
      continue;
    }
    seen.add(entry.id);
    if (!entry.garment) {
      fail("missing-reference", `garment entry '${entry.id}' has no garment document`, entry.id);
      continue;
    }
    const result = validateGarmentProject(entry.garment);
    if (!result.valid) {
      fail("invalid-garment", `garment '${entry.id}': ${result.diagnostics[0].code}: ${result.diagnostics[0].message}`, entry.id);
    }
    if (entry.productionSpecification) {
      const errors = validateProductionSpecification(entry.productionSpecification);
      if (errors.length) {
        fail("invalid-production-specification", `garment '${entry.id}': ${errors[0]}`, entry.id);
      } else if (entry.productionSpecification.garmentId !== entry.garment.id) {
        fail("invalid-production-specification", `garment '${entry.id}' production specification references '${entry.productionSpecification.garmentId}'`, entry.id);
      } else if (entry.grading) {
        const active = new Set(entry.grading.sizeSet.sizes.filter((size) => size.active).map((size) => size.id));
        const missing = entry.productionSpecification.sizeRange.find((sizeId) => !active.has(sizeId));
        if (missing) {
          fail("invalid-production-specification", `garment '${entry.id}' specification size '${missing}' is missing or inactive`, entry.id);
        }
      }
    }
    if (entry.bom) {
      const issues = validateBOM(entry.bom, { markers: [...(entry.markers ?? []), ...(entry.marker ? [entry.marker] : [])] });
      if (issues.length) fail("invalid-production-data", `garment '${entry.id}' BOM: ${issues[0].message}`, entry.id);
      else if (entry.bom.garmentId !== entry.garment.id) {
        fail("invalid-production-data", `garment '${entry.id}' BOM references '${entry.bom.garmentId}'`, entry.id);
      } else if (entry.productionSpecification && entry.bom.revision !== entry.productionSpecification.revision) {
        fail("invalid-production-data", `garment '${entry.id}' BOM revision differs from production specification`, entry.id);
      }
    }
    const allMarkers = [...(entry.markers ?? []), ...(entry.marker ? [entry.marker] : [])];
    const markerIds = allMarkers.map((marker) => marker.id);
    if (new Set(markerIds).size !== markerIds.length) {
      fail("invalid-production-data", `garment '${entry.id}' has duplicate marker ids`, entry.id);
    }
    for (const marker of allMarkers) {
      const markerIssues = validateMarker(marker);
      if (markerIssues.length) {
        fail("invalid-production-data", `garment '${entry.id}' marker '${marker.id}': ${markerIssues[0].message}`, entry.id);
      }
    }
    if (entry.productionRun) {
      const errors = validateProductionRun(entry.productionRun);
      if (errors.length) fail("invalid-production-data", `garment '${entry.id}' run: ${errors[0]}`, entry.id);
      else if (entry.productionRun.garmentId !== entry.garment.id) {
        fail("invalid-production-data", `garment '${entry.id}' run references '${entry.productionRun.garmentId}'`, entry.id);
      } else if (!entry.grading || entry.productionRun.gradingId !== entry.grading.id) {
        fail("invalid-production-data", `garment '${entry.id}' production run has no matching grading document`, entry.id);
      } else if (entry.productionSpecification &&
        (entry.productionRun.revision !== entry.productionSpecification.revision ||
          entry.productionRun.styleNumber !== entry.productionSpecification.styleNumber)) {
        fail("invalid-production-data", `garment '${entry.id}' run style/revision differs from production specification`, entry.id);
      }
    }
    if (entry.revisionLedger) {
      const errors = validateRevisionLedger(entry.revisionLedger);
      if (errors.length) fail("invalid-production-data", `garment '${entry.id}' revision ledger: ${errors[0]}`, entry.id);
      else if (entry.revisionLedger.garmentId !== entry.garment.id) {
        fail("invalid-production-data", `garment '${entry.id}' revision ledger references '${entry.revisionLedger.garmentId}'`, entry.id);
      }
    }
  }
  return diagnostics;
}

// ---------------------------------------------------------------------------
// Content fingerprint (timestamps excluded): same content <=> same fingerprint
// ---------------------------------------------------------------------------

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

function fnv1a(text: string): string {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Content fingerprint: identity (project id) and volatile metadata
 * (timestamps, app version) are excluded, so equal garment content always
 * fingerprints equal — across duplicates, save/load cycles, and migrations
 * that preserve content.
 */
export function fingerprintProject(project: AppProject): string {
  const stripped = clone(project);
  stripped.id = "";
  stripped.metadata = { createdAt: "", updatedAt: "", applicationVersion: "" };
  return `fp/${fnv1a(canonicalJson(stripped))}`;
}

export function serializeProject(project: AppProject): string {
  return canonicalJson(project);
}

export function deserializeProject(serialized: string): AppProject {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized project is not valid JSON");
  }
  const project = parsed as AppProject;
  const diagnostics = validateProject(project);
  if (diagnostics.length > 0) {
    throw new PatternCadError("invalid-document", `invalid project: ${diagnostics[0].code}: ${diagnostics[0].message}`);
  }
  return clone(project);
}

// ---------------------------------------------------------------------------
// Migrations (framework, not one-off conversions)
// ---------------------------------------------------------------------------

export interface Migration {
  from: number;
  to: number;
  migrate: (document: unknown) => unknown;
}

export class ProjectMigrator {
  private readonly migrations = new Map<number, Migration>();

  register(migration: Migration): void {
    if (!Number.isInteger(migration.from) || !Number.isInteger(migration.to) || migration.to !== migration.from + 1) {
      throw new PatternCadError("invalid-document", "migrations must step exactly one schema version");
    }
    if (this.migrations.has(migration.from)) {
      throw new PatternCadError("duplicate-id", `migration from v${migration.from} already registered`);
    }
    this.migrations.set(migration.from, migration);
  }

  /** Migrate any older document to the current schema (unknown versions throw). */
  migrateToLatest(document: unknown): AppProject {
    let current = clone(document) as { schemaVersion?: unknown };
    if (!current || typeof current !== "object" || !Number.isInteger(current.schemaVersion)) {
      throw new PatternCadError("invalid-document", "migrating document has no integer schemaVersion");
    }
    let version = current.schemaVersion as number;
    if (version > PROJECT_SCHEMA_VERSION) {
      throw new PatternCadError("invalid-document", `document schema v${version} is newer than this application (v${PROJECT_SCHEMA_VERSION})`);
    }
    while (version < PROJECT_SCHEMA_VERSION) {
      const migration = this.migrations.get(version);
      if (!migration) {
        throw new PatternCadError("invalid-document", `no migration registered from schema v${version}`);
      }
      current = migration.migrate(current) as { schemaVersion?: unknown };
      version++;
      if (current.schemaVersion !== version) {
        throw new PatternCadError("invalid-document", `migration to v${version} did not declare its version`);
      }
    }
    return deserializeProject(JSON.stringify(current));
  }
}
