// G10B interactive garment manipulation.
// Layering (per the G10 manipulation model):
//   persistent garment transform  -> PanelPlacement edits (assembly state, never CAD geometry)
//   temporary manipulation state  -> pins + drag sessions (through the solver constraint interface)
//   simulation state              -> owned by the solver, only consumed, never written directly

import type { ClothSolver } from "../backend/solver.js";
import type { PanelPlacement } from "../garment/assembly.js";
import type { AvatarSpec } from "../garment/avatar.js";
import type { GarmentProject } from "../garment/project.js";
import { validateGarmentProject } from "../garment/project.js";
import { type Vec3, WorkspaceError } from "./types.js";
import type { GarmentWorkspace } from "./viewport.js";

export function translatePlacement(placement: PanelPlacement, delta: Vec3): PanelPlacement {
  return {
    panelId: placement.panelId,
    translation: [
      placement.translation[0] + delta[0],
      placement.translation[1] + delta[1],
      placement.translation[2] + delta[2],
    ],
    yawRad: placement.yawRad,
  };
}

export function yawPlacement(placement: PanelPlacement, yawDeltaRad: number): PanelPlacement {
  return {
    panelId: placement.panelId,
    translation: [...placement.translation] as Vec3,
    yawRad: placement.yawRad + yawDeltaRad,
  };
}

/** Translate every panel placement (moves the whole garment). */
export function translateGarment(placements: readonly PanelPlacement[], delta: Vec3): PanelPlacement[] {
  return placements.map((p) => translatePlacement(p, delta));
}

/** Rotate every panel placement around a world pivot, Y axis only (placement yaw model). */
export function rotateGarmentAroundPivot(placements: readonly PanelPlacement[], yawDeltaRad: number, pivot: Vec3): PanelPlacement[] {
  const c = Math.cos(yawDeltaRad);
  const s = Math.sin(yawDeltaRad);
  return placements.map((p) => {
    const dx = p.translation[0] - pivot[0];
    const dz = p.translation[2] - pivot[2];
    return {
      panelId: p.panelId,
      translation: [
        pivot[0] + dx * c + dz * s,
        p.translation[1],
        pivot[2] - dx * s + dz * c,
      ],
      yawRad: p.yawRad + yawDeltaRad,
    };
  });
}

/** Deep-clone via JSON like the project kernel does. */
export function clonePlacements(placements: readonly PanelPlacement[]): PanelPlacement[] {
  return JSON.parse(JSON.stringify(placements)) as PanelPlacement[];
}

export function defaultPlacements(project: GarmentProject): PanelPlacement[] {
  return clonePlacements(project.placements);
}

/**
 * Immutable placement edit: clone the project, swap placements, bump revision,
 * validate. Throws WorkspaceError("invalid-transform") on invalid result.
 */
export function withPlacements(
  project: GarmentProject,
  placements: PanelPlacement[],
): GarmentProject {
  const next = JSON.parse(JSON.stringify(project)) as GarmentProject;
  next.placements = clonePlacements(placements);
  next.metadata.revision = project.metadata.revision + 1;
  const validation = validateGarmentProject(next);
  if (!validation.valid) {
    const detail = validation.diagnostics.map((d) => `${d.code}: ${d.message}`).join("; ");
    throw new WorkspaceError("invalid-transform", `placement edit produced invalid project: ${detail}`);
  }
  return next;
}

/** Move a single panel's placement by a world delta. */
export function movePanelPlacement(project: GarmentProject, panelId: string, delta: Vec3): GarmentProject {
  const placements = clonePlacements(project.placements);
  const placement = placements.find((p) => p.panelId === panelId);
  if (!placement) throw new WorkspaceError("invalid-entity", `no placement for panel ${panelId}`);
  const index = placements.indexOf(placement);
  placements[index] = translatePlacement(placement, delta);
  return withPlacements(project, placements);
}

/** Rotate a single panel's placement yaw around its own placement origin. */
export function rotatePanelPlacement(project: GarmentProject, panelId: string, yawDeltaRad: number): GarmentProject {
  const placements = clonePlacements(project.placements);
  const placement = placements.find((p) => p.panelId === panelId);
  if (!placement) throw new WorkspaceError("invalid-entity", `no placement for panel ${panelId}`);
  const index = placements.indexOf(placement);
  placements[index] = yawPlacement(placement, yawDeltaRad);
  return withPlacements(project, placements);
}

/** Transform the whole garment (translate + yaw around the garment bounds center). */
export function transformGarmentPlacements(
  project: GarmentProject,
  delta: Vec3,
  yawDeltaRad: number,
  pivot: Vec3,
): GarmentProject {
  let placements = translateGarment(project.placements, delta);
  if (yawDeltaRad !== 0) placements = rotateGarmentAroundPivot(placements, yawDeltaRad, pivot);
  return withPlacements(project, placements);
}

/** Center the garment XZ bounds on the avatar XZ bounds (keeps current height). */
export function repositionAroundAvatar(
  project: GarmentProject,
  garmentBounds: { min: Vec3; max: Vec3 },
  avatar: AvatarSpec,
): GarmentProject {
  const ap = Float32Array.from(avatar.positions);
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < ap.length; i += 3) {
    if (ap[i] < minX) minX = ap[i];
    if (ap[i] > maxX) maxX = ap[i];
    if (ap[i + 2] < minZ) minZ = ap[i + 2];
    if (ap[i + 2] > maxZ) maxZ = ap[i + 2];
  }
  const targetX = (minX + maxX) / 2;
  const targetZ = (minZ + maxZ) / 2;
  const deltaX = targetX - (garmentBounds.min[0] + garmentBounds.max[0]) / 2;
  const deltaZ = targetZ - (garmentBounds.min[2] + garmentBounds.max[2]) / 2;
  return transformGarmentPlacements(project, [deltaX, 0, deltaZ], 0, [0, 0, 0]);
}

// ---------------------------------------------------------------------------
// Pins: temporary constraints through the existing ClothSolver pin interface.
// ---------------------------------------------------------------------------

export interface PinRecord {
  /** Stable pin ID ("pin/1", "pin/2", ...). */
  id: string;
  /** Vertex index in the assembled mesh — valid only for one garment epoch. */
  vertexId: number;
  /** World-space pin target. */
  target: Vec3;
  /** Garment epoch this pin was created against. */
  garmentEpoch: number;
}

export interface PinInvalidatedEvent {
  removed: PinRecord[];
  reason: "garment-replaced" | "workspace-reset";
}

/**
 * Owns temporary pins and mirrors them into the solver's existing
 * pinVertex/unpinVertex constraint interface. Never touches solver state
 * by any other route.
 */
export class PinManager {
  private pins = new Map<string, PinRecord>();
  private nextIndex = 1;
  private solver: ClothSolver | null = null;
  private workspace: GarmentWorkspace | null = null;
  private unsubscribe: (() => void) | null = null;
  private lastInvalidation: PinInvalidatedEvent | null = null;

  attach(workspace: GarmentWorkspace, solver: ClothSolver): void {
    this.workspace = workspace;
    this.solver = solver;
    if (this.unsubscribe) this.unsubscribe();
    this.unsubscribe = workspace.onGarmentReplaced((event) => {
      this.invalidate(event.reason === "project-load" ? "workspace-reset" : "garment-replaced");
    });
  }

  get invalidation(): PinInvalidatedEvent | null {
    return this.lastInvalidation;
  }

  private invalidate(reason: PinInvalidatedEvent["reason"]): void {
    if (this.pins.size === 0) {
      this.lastInvalidation = { removed: [], reason };
      return;
    }
    const removed = [...this.pins.values()];
    this.pins.clear();
    this.lastInvalidation = { removed, reason };
  }

  list(): PinRecord[] {
    return [...this.pins.values()];
  }

  get(id: string): PinRecord | null {
    return this.pins.get(id) ?? null;
  }

  has(id: string): boolean {
    return this.pins.has(id);
  }

  /** Pin a vertex at its current render position (or an explicit target). */
  create(vertexId: number, target?: Vec3): PinRecord {
    const workspace = this.requireWorkspace();
    const n = workspace.vertexCount;
    if (!Number.isInteger(vertexId) || vertexId < 0 || vertexId >= n) {
      throw new WorkspaceError("invalid-pin", `vertex ${vertexId} out of range (0..${n - 1})`);
    }
    const resolved: Vec3 = target ?? [
      workspace.renderPositions[vertexId * 3],
      workspace.renderPositions[vertexId * 3 + 1],
      workspace.renderPositions[vertexId * 3 + 2],
    ];
    if (!Number.isFinite(resolved[0]) || !Number.isFinite(resolved[1]) || !Number.isFinite(resolved[2])) {
      throw new WorkspaceError("invalid-pin", `pin target is not finite`);
    }
    const record: PinRecord = {
      id: `pin/${this.nextIndex++}`,
      vertexId,
      target: [...resolved] as Vec3,
      garmentEpoch: workspace.garmentEpoch,
    };
    this.pins.set(record.id, record);
    this.requireSolver().pinVertex(vertexId, resolved);
    return record;
  }

  remove(id: string): boolean {
    const record = this.pins.get(id);
    if (!record) return false;
    this.pins.delete(id);
    this.requireSolver().unpinVertex(record.vertexId);
    return true;
  }

  /** Move a pin target (drag). Re-pins through the solver (targets overwrite). */
  updateTarget(id: string, target: Vec3): PinRecord {
    const record = this.pins.get(id);
    if (!record) throw new WorkspaceError("invalid-pin", `unknown pin ${id}`);
    if (!Number.isFinite(target[0]) || !Number.isFinite(target[1]) || !Number.isFinite(target[2])) {
      throw new WorkspaceError("invalid-pin", "pin target is not finite");
    }
    record.target = [...target] as Vec3;
    this.requireSolver().pinVertex(record.vertexId, target);
    return record;
  }

  clear(): void {
    const solver = this.solver;
    for (const record of this.pins.values()) solver?.unpinVertex(record.vertexId);
    this.pins.clear();
  }

  serialize(): string {
    return JSON.stringify({ version: 1, pins: this.list() });
  }

  /** Restore pins for the CURRENT garment epoch only; stale entries are dropped. */
  restore(serialized: string): PinRecord[] {
    const workspace = this.requireWorkspace();
    const parsed = JSON.parse(serialized) as { version: number; pins: PinRecord[] };
    if (parsed.version !== 1 || !Array.isArray(parsed.pins)) {
      throw new WorkspaceError("invalid-pin", "malformed pin serialization");
    }
    this.clear();
    const restored: PinRecord[] = [];
    for (const raw of parsed.pins) {
      if (raw.garmentEpoch !== workspace.garmentEpoch) continue;
      if (raw.vertexId < 0 || raw.vertexId >= workspace.vertexCount) continue;
      const record = this.create(raw.vertexId, raw.target);
      restored.push(record);
    }
    return restored;
  }

  private requireWorkspace(): GarmentWorkspace {
    if (!this.workspace) throw new WorkspaceError("invalid-state", "PinManager not attached to a workspace");
    return this.workspace;
  }

  private requireSolver(): ClothSolver {
    if (!this.solver) throw new WorkspaceError("invalid-state", "PinManager not attached to a solver");
    return this.solver;
  }
}

// ---------------------------------------------------------------------------
// Drag: a controlled interaction state implemented as a temporary pin.
// ---------------------------------------------------------------------------

export interface DragState {
  pin: PinRecord;
  /** target = dragPoint + grabOffset keeps the grab offset stable. */
  grabOffset: Vec3;
  garmentEpoch: number;
}

export class DragController {
  private pins: PinManager;
  private workspace: GarmentWorkspace;
  private active_: DragState | null = null;

  constructor(pins: PinManager, workspace: GarmentWorkspace) {
    this.pins = pins;
    this.workspace = workspace;
    workspace.onGarmentReplaced(() => {
      if (this.active_) this.active_ = null;
    });
  }

  get isActive(): boolean {
    return this.active_ !== null;
  }

  get state(): DragState | null {
    return this.active_;
  }

  begin(vertexId: number, grabPoint: Vec3): DragState {
    if (this.active_) throw new WorkspaceError("invalid-state", "a drag is already active");
    const pin = this.pins.create(vertexId);
    this.active_ = {
      pin,
      grabOffset: [
        pin.target[0] - grabPoint[0],
        pin.target[1] - grabPoint[1],
        pin.target[2] - grabPoint[2],
      ],
      garmentEpoch: this.workspace.garmentEpoch,
    };
    return this.active_;
  }

  moveTo(target: Vec3): void {
    const drag = this.active_;
    if (!drag) throw new WorkspaceError("invalid-state", "no active drag");
    if (drag.garmentEpoch !== this.workspace.garmentEpoch) {
      this.active_ = null;
      throw new WorkspaceError("stale-reference", "drag invalidated by garment rebuild");
    }
    const resolved: Vec3 = [
      target[0] + drag.grabOffset[0],
      target[1] + drag.grabOffset[1],
      target[2] + drag.grabOffset[2],
    ];
    this.pins.updateTarget(drag.pin.id, resolved);
  }

  /** End the drag; optionally keep the pin (pin-on-drop). */
  end(keepPin = false): PinRecord | null {
    const drag = this.active_;
    if (!drag) return null;
    this.active_ = null;
    if (!keepPin) this.pins.remove(drag.pin.id);
    return keepPin ? this.pins.get(drag.pin.id) : null;
  }

  cancel(): void {
    this.end(false);
  }
}
