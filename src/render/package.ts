// G15 render package: the "renders/" side of a garment package.
//
// A RenderPackage binds a presentation scene, its materials, and its capture
// plans to a garment reference. It carries no pattern geometry and no pixels:
// captures execute in the browser adapter (app/render-g15.ts) from these
// exact plans, so previews stay reproducible without coupling render assets
// to production pattern geometry.

import { PatternCadError } from "../pattern/cad.js";
import type { VisualMaterial } from "./materials.js";
import { validateMaterial } from "./materials.js";
import { deserializeScene, type CapturePlan, type PresentationScene } from "./presentation.js";

export interface RenderPackage {
  format: "closim-render-package";
  version: 1;
  id: string;
  scene: PresentationScene;
  materials: VisualMaterial[];
  captures: CapturePlan[];
  source: { garmentId: string; simEpoch: number };
}

export function createRenderPackage(partial: {
  id: string;
  scene: PresentationScene;
  materials: VisualMaterial[];
  captures: CapturePlan[];
  simEpoch?: number;
}): RenderPackage {
  if (!partial.id) throw new PatternCadError("invalid-document", "render package needs an id");
  if (partial.scene.garmentId === undefined) {
    throw new PatternCadError("invalid-document", "render package scene has no garment reference");
  }
  for (const material of partial.materials) {
    const errors = validateMaterial(material);
    if (errors.length > 0) throw new PatternCadError("invalid-document", `invalid material: ${errors[0]}`);
  }
  for (const capture of partial.captures) {
    if (capture.sceneId !== partial.scene.id) {
      throw new PatternCadError("invalid-document", `capture '${capture.id}' belongs to another scene`);
    }
  }
  const ids = new Set(partial.captures.map((c) => c.id));
  if (ids.size !== partial.captures.length) {
    throw new PatternCadError("duplicate-id", "duplicate capture ids in render package");
  }
  return {
    format: "closim-render-package",
    version: 1,
    id: partial.id,
    scene: JSON.parse(JSON.stringify(partial.scene)) as PresentationScene,
    materials: JSON.parse(JSON.stringify(partial.materials)) as VisualMaterial[],
    captures: JSON.parse(JSON.stringify(partial.captures)) as CapturePlan[],
    source: { garmentId: partial.scene.garmentId, simEpoch: partial.simEpoch ?? 0 },
  };
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

export function serializeRenderPackage(pkg: RenderPackage): string {
  return canonicalJson(pkg);
}

export function deserializeRenderPackage(serialized: string): RenderPackage {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized render package is not valid JSON");
  }
  const pkg = parsed as RenderPackage;
  if (!pkg || pkg.format !== "closim-render-package" || pkg.version !== 1 || !pkg.scene || !Array.isArray(pkg.materials)) {
    throw new PatternCadError("invalid-document", "render package shape or version is invalid");
  }
  deserializeScene(JSON.stringify(pkg.scene)); // structural gate
  return JSON.parse(JSON.stringify(pkg)) as RenderPackage;
}
