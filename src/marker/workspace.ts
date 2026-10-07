// G14E — marker workspace (headless designer session).
//
// Inspect and control the marker without touching internal APIs: inputs,
// optimize/cancel, manual move/rotate gated by revalidation (invalid manual
// edits are rejected and the saved state never goes invalid), restore-auto,
// selection + metadata inspection, undo/redo, save/load.
//
// Nesting time is the engine/optimizer's business; this layer only records
// elapsed milliseconds per action for the performance split.

import { PatternCadError } from "../pattern/cad.js";
import type { GradingDocument } from "../grading/types.js";
import { auditPlacements, placedPolygon, type NestResult } from "./nest.js";
import {
  createMarker,
  deserializeMarker,
  expandCutPlan,
  serializeMarker,
  validateMarker,
  validateNestingConstraint,
  type CutPlan,
  type Fabric,
  type Marker,
  type MarkerDiagnostic,
  type MarkerPiece,
  type NestingConstraint,
} from "./model.js";
import { fabricRulesFrom, resolvePieceRules, type FabricRules } from "./fabric.js";
import {
  measureResult,
  optimizeMarker,
  type MarkerMetrics,
  type Objective,
  type OptimizeReport,
  type WeightedObjective,
} from "./optimize.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Rebuild a marker from manufacturing inputs (quantities/fabric change path). */
export function rebuildMarker(
  grading: GradingDocument,
  plan: CutPlan,
  fabric: Fabric,
  constraint: NestingConstraint,
  name: string,
  id: string,
): { marker: Marker; pieces: MarkerPiece[] } {
  const pieces = expandCutPlan(grading, plan);
  const marker = createMarker(id, name, fabric, plan.id, constraint, pieces);
  return { marker, pieces };
}

export interface ManualEditResult {
  ok: boolean;
  reason?: string;
  elapsedMs: number;
}

export interface MarkerPreview {
  placements: Marker["placements"];
  placedIds: string[];
  unplacedIds: string[];
  metrics: MarkerMetrics;
  revision: number;
}

const HISTORY_LIMIT = 50;

export class MarkerWorkspace {
  marker: Marker;
  readonly pieces: MarkerPiece[];
  readonly items: CutPlan["items"];
  readonly rules: FabricRules;
  readonly grainRadOf: (panelId: string) => number;
  selection: string[] = [];
  lastResult: NestResult | null = null;
  lastOptimizeMs = 0;
  private undoStack: Marker[] = [];
  private redoStack: Marker[] = [];
  private cancelFlag = false;

  constructor(
    marker: Marker,
    pieces: MarkerPiece[],
    items: CutPlan["items"],
    rules: FabricRules,
    grainRadOf: (panelId: string) => number,
  ) {
    const diagnostics = validateMarker(marker);
    if (diagnostics.length > 0) {
      throw new PatternCadError("invalid-document", `cannot open invalid marker: ${diagnostics[0].message}`);
    }
    this.marker = clone(marker);
    this.pieces = clone(pieces);
    this.items = clone(items);
    this.rules = clone(rules);
    this.grainRadOf = grainRadOf;
  }

  static open(
    grading: GradingDocument,
    plan: CutPlan,
    fabric: Fabric,
    constraint: NestingConstraint,
    name: string,
    id: string,
    grainRadOf: (panelId: string) => number,
  ): MarkerWorkspace {
    const { marker, pieces } = rebuildMarker(grading, plan, fabric, constraint, name, id);
    const rules = fabricRulesFrom(fabric, constraint);
    return new MarkerWorkspace(marker, pieces, plan.items, rules, grainRadOf);
  }

  private commit(label: string): void {
    void label;
    this.undoStack.push(clone(this.marker));
    if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
    this.redoStack = [];
    this.marker.revision++;
  }

  undo(): boolean {
    const prev = this.undoStack.pop();
    if (!prev) return false;
    this.redoStack.push(clone(this.marker));
    this.marker = prev;
    this.pruneSelection();
    return true;
  }

  redo(): boolean {
    const next = this.redoStack.pop();
    if (!next) return false;
    this.undoStack.push(clone(this.marker));
    this.marker = next;
    this.pruneSelection();
    return true;
  }

  private pruneSelection(): void {
    const alive = new Set(this.pieces.map((p) => p.instanceId));
    this.selection = this.selection.filter((id) => alive.has(id));
  }

  select(instanceIds: string[]): void {
    const alive = new Set(this.pieces.map((p) => p.instanceId));
    this.selection = [...new Set(instanceIds)].filter((id) => alive.has(id));
  }

  inspect(instanceId: string): (MarkerPiece & { placement: Marker["placements"][number] | null }) | null {
    const piece = this.pieces.find((p) => p.instanceId === instanceId);
    if (!piece) return null;
    return { ...clone(piece), placement: clone(this.marker.placements.find((q) => q.instanceId === instanceId) ?? null) };
  }

  cancelOptimize(): void {
    this.cancelFlag = true;
  }

  optimize(
    seeds: number[] = [0],
    objective: Objective | WeightedObjective = "min-length",
    strategies: NestResult["strategy"][] = ["area"],
  ): OptimizeReport {
    const started = Date.now();
    this.cancelFlag = false;
    const report = optimizeMarker(
      {
        pieces: this.pieces,
        items: this.items,
        rules: this.rules,
        constraint: this.marker.constraint,
        grainRadOf: this.grainRadOf,
      },
      { seeds, strategies, objective, shouldCancel: () => this.cancelFlag },
    );
    this.lastOptimizeMs = Date.now() - started;
    this.commit("optimize");
    this.marker.placements = clone(report.best.result.placements);
    this.lastResult = report.best.result;
    return report;
  }

  /** Manual move: rejected (state unchanged) unless the result audits clean. */
  movePlacement(instanceId: string, x: number, y: number): ManualEditResult {
    const started = Date.now();
    const index = this.marker.placements.findIndex((p) => p.instanceId === instanceId);
    if (index < 0) return { ok: false, reason: `piece '${instanceId}' is not placed`, elapsedMs: Date.now() - started };
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return { ok: false, reason: "target coordinates must be finite", elapsedMs: Date.now() - started };
    }
    const trial = clone(this.marker);
    trial.placements[index] = { ...trial.placements[index], x, y, manual: true };
    const problems = auditPlacements(trial, this.pieces, this.rules, this.grainRadOf);
    if (problems.length > 0) {
      return { ok: false, reason: problems[0].message, elapsedMs: Date.now() - started };
    }
    this.commit("move");
    this.marker.placements = trial.placements;
    return { ok: true, elapsedMs: Date.now() - started };
  }

  /** Manual rotate: rotation must belong to the item's allowed set. */
  rotatePlacement(instanceId: string, rotationDeg: number): ManualEditResult {
    const started = Date.now();
    const index = this.marker.placements.findIndex((p) => p.instanceId === instanceId);
    if (index < 0) return { ok: false, reason: `piece '${instanceId}' is not placed`, elapsedMs: Date.now() - started };
    const piece = this.pieces.find((p) => p.instanceId === instanceId)!;
    const item = this.items.find((i) => i.id === piece.cutItemId)!;
    const pr = resolvePieceRules(this.rules, item, instanceId, []);
    const norm = ((rotationDeg % 360) + 360) % 360;
    if (!pr.rotationsDeg.some((r) => Math.abs((((r % 360) + 360) % 360) - norm) <= 1e-9)) {
      return { ok: false, reason: `rotation ${rotationDeg}° is not allowed for '${instanceId}'`, elapsedMs: Date.now() - started };
    }
    const trial = clone(this.marker);
    trial.placements[index] = { ...trial.placements[index], rotationDeg: norm, manual: true };
    const problems = auditPlacements(trial, this.pieces, this.rules, this.grainRadOf);
    if (problems.length > 0) {
      return { ok: false, reason: problems[0].message, elapsedMs: Date.now() - started };
    }
    this.commit("rotate");
    this.marker.placements = trial.placements;
    return { ok: true, elapsedMs: Date.now() - started };
  }

  /** Drop a manual placement back to unplaced (re-nest via optimize). */
  restoreAuto(instanceId: string): boolean {
    if (!this.marker.placements.some((p) => p.instanceId === instanceId)) return false;
    this.commit("restore-auto");
    this.marker.placements = this.marker.placements.filter((p) => p.instanceId !== instanceId);
    return true;
  }

  /**
   * Replace nesting settings. Existing placements are kept as-is; the
   * returned audit tells whether they still hold (re-run optimize to fix).
   */
  setConstraint(patch: Partial<NestingConstraint>): MarkerDiagnostic[] {
    const merged = { ...clone(this.marker.constraint), ...clone(patch) };
    validateNestingConstraint(merged);
    this.commit("settings");
    this.marker.constraint = merged;
    return auditPlacements(this.marker, this.pieces, this.rules, this.grainRadOf);
  }

  preview(): MarkerPreview {
    const placedIds = this.marker.placements.map((p) => p.instanceId);
    const placedSet = new Set(placedIds);
    const metrics = measureResult(
      {
        placements: this.marker.placements,
        placed: [],
        unplaced: [],
        markerLengthM: this.marker.placements.length > 0
          ? Math.max(...this.marker.placements.map((p) => {
            const piece = this.pieces.find((q) => q.instanceId === p.instanceId)!;
            return placedPolygon(piece, p).reduce((m, v) => Math.max(m, v[1]), -Infinity);
          }))
          : 0,
        markerWidthM: this.rules.usableWidthM,
        utilization: 0,
        iterations: 0,
        seed: this.lastResult?.seed ?? 0,
        strategy: this.lastResult?.strategy ?? "area",
      },
      this.pieces.filter((p) => placedSet.has(p.instanceId)).reduce((s, p) => s + p.areaM2, 0),
    );
    return {
      placements: clone(this.marker.placements),
      placedIds,
      unplacedIds: this.pieces.map((p) => p.instanceId).filter((id) => !placedSet.has(id)),
      metrics,
      revision: this.marker.revision,
    };
  }

  save(): string {
    return serializeMarker(this.marker);
  }

  load(serialized: string): void {
    const marker = deserializeMarker(serialized);
    // Piece set must match (placements reference instances by id).
    const mine = new Set(this.pieces.map((p) => p.instanceId));
    for (const placement of marker.placements) {
      if (!mine.has(placement.instanceId)) {
        throw new PatternCadError("missing-reference", `saved marker references unknown piece '${placement.instanceId}'`);
      }
    }
    this.commit("load");
    this.marker = marker;
    this.pruneSelection();
  }

  /** Live geometric audit of current placements (empty = valid marker). */
  audit(): ReturnType<typeof auditPlacements> {
    return auditPlacements(this.marker, this.pieces, this.rules, this.grainRadOf);
  }
}
