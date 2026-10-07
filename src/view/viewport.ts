// G10A workspace hub: the single source of render state for the 3D viewport.
// Ownership model:
//   - 2D pattern document  (authoritative design, owned by the project)
//   - assembled garment    (derived, replaced atomically via setGarment)
//   - simulation state     (owned by the solver, consumed via publishSimPositions)
// The camera never triggers a solve; rendering only ever reads renderPositions.

import type { AssembledGarment } from "../garment/assembly.js";
import type { AvatarSpec } from "../garment/avatar.js";
import type { GarmentProject } from "../garment/project.js";
import { OrbitCamera } from "./camera.js";
import {
  type SelectionState,
  clearSelection,
  createSelection,
  pruneSelection,
  type PrunedSelection,
} from "./selection.js";
import {
  type Bounds,
  type VisibilityFlags,
  computeBounds,
  defaultVisibility,
  WorkspaceError,
} from "./types.js";

export type GarmentReplaceReason = "initial-load" | "rebuild" | "project-load";

export interface GarmentReplacedEvent {
  reason: GarmentReplaceReason;
  garmentEpoch: number;
  prunedSelection: PrunedSelection;
}

export class GarmentWorkspace {
  readonly camera = new OrbitCamera({}, () => {
    this.cameraDirty_ = true;
  });
  readonly selection: SelectionState = createSelection();
  visibility: VisibilityFlags = defaultVisibility();

  garmentEpoch = 0;
  simEpoch = 0;

  private project_: GarmentProject | null = null;
  private assembled_: AssembledGarment | null = null;
  private restPositions_ = new Float32Array(0);
  private renderPositions_ = new Float32Array(0);
  private viewDirty_ = true;
  private cameraDirty_ = true;
  private firstLoadFramed = false;

  private garmentListeners: Array<(event: GarmentReplacedEvent) => void> = [];

  get project(): GarmentProject | null {
    return this.project_;
  }

  get assembled(): AssembledGarment | null {
    return this.assembled_;
  }

  /** Latest state to render: simulated positions when published, else placed rest state. */
  get renderPositions(): Float32Array {
    return this.renderPositions_;
  }

  get restPositions(): Float32Array {
    return this.restPositions_;
  }

  get vertexCount(): number {
    return this.renderPositions_.length / 3;
  }

  get viewDirty(): boolean {
    return this.viewDirty_;
  }

  get cameraDirty(): boolean {
    return this.cameraDirty_;
  }

  clearViewDirty(): void {
    this.viewDirty_ = false;
  }

  clearCameraDirty(): void {
    this.cameraDirty_ = false;
  }

  onGarmentReplaced(listener: (event: GarmentReplacedEvent) => void): () => void {
    this.garmentListeners.push(listener);
    return () => {
      this.garmentListeners = this.garmentListeners.filter((l) => l !== listener);
    };
  }

  markCameraDirty(): void {
    this.cameraDirty_ = true;
  }

  /**
   * Atomically replace the derived garment. All interaction state that
   * references mesh indices (selection regions, pins, drag sessions) is
   * invalidated; stable pattern IDs are pruned, never silently kept.
   */
  setGarment(project: GarmentProject, assembled: AssembledGarment, reason: GarmentReplaceReason): void {
    const n = assembled.positions.length;
    if (n % 3 !== 0) throw new WorkspaceError("invalid-state", "assembled positions length is not a multiple of 3");
    for (let i = 0; i < assembled.positions.length; i++) {
      if (!Number.isFinite(assembled.positions[i])) {
        throw new WorkspaceError("invalid-state", `non-finite assembled position component ${i}`);
      }
    }
    this.project_ = project;
    this.assembled_ = assembled;
    this.restPositions_ = assembled.positions.slice();
    this.renderPositions_ = assembled.positions.slice();
    this.garmentEpoch++;
    this.simEpoch = 0;
    this.viewDirty_ = true;
    const pruned = pruneSelection(this.selection, assembled);
    const event: GarmentReplacedEvent = { reason, garmentEpoch: this.garmentEpoch, prunedSelection: pruned };
    for (const listener of this.garmentListeners) listener(event);
    if (!this.firstLoadFramed && reason === "initial-load") {
      this.firstLoadFramed = true;
    }
  }

  /**
   * The only sanctioned path from solver state into the viewport.
   * Copies Float64 solver positions into the Float32 render buffer.
   */
  publishSimPositions(source: Float64Array): void {
    const assembled = this.assembled_;
    if (!assembled) throw new WorkspaceError("no-garment", "cannot publish simulation positions without a garment");
    if (source.length !== this.renderPositions_.length) {
      throw new WorkspaceError("invalid-state", `position length mismatch: solver ${source.length} vs garment ${this.renderPositions_.length}`);
    }
    for (let i = 0; i < source.length; i++) this.renderPositions_[i] = source[i];
    this.simEpoch++;
    this.viewDirty_ = true;
  }

  /** Reset render positions back to the placed rest state (simulation discarded). */
  resetToRestPositions(): void {
    if (!this.assembled_) throw new WorkspaceError("no-garment", "no garment loaded");
    this.renderPositions_.set(this.restPositions_);
    this.simEpoch = 0;
    this.viewDirty_ = true;
  }

  currentBounds(): Bounds {
    if (!this.assembled_) throw new WorkspaceError("no-garment", "no garment loaded");
    return computeBounds(this.renderPositions_);
  }

  frameGarment(aspect: number, margin?: number): void {
    const b = this.currentBounds();
    this.camera.frame(b.min, b.max, aspect, margin);
    this.cameraDirty_ = true;
  }

  frameAvatar(avatar: AvatarSpec, aspect: number, margin?: number): void {
    const positions = Float32Array.from(avatar.positions);
    const b = computeBounds(positions);
    this.camera.frame(b.min, b.max, aspect, margin);
    this.cameraDirty_ = true;
  }

  frameSelection(aspect: number, margin?: number): boolean {
    const assembled = this.assembled_;
    const sel = this.selection;
    if (!assembled) return false;
    let bounds: Bounds | null = null;
    if (sel.garment) {
      bounds = this.currentBounds();
    } else if (sel.avatar && this.project_?.avatar) {
      this.frameAvatar(this.project_.avatar, aspect, margin);
      return true;
    } else if (sel.region) {
      bounds = computeBounds(this.renderPositions_, sel.region.vertexStart, sel.region.vertexCount);
    } else if (sel.panelIds.length > 0) {
      bounds = this.boundsOfPanels(sel.panelIds);
    } else if (sel.seamIds.length > 0) {
      bounds = this.boundsOfSeams(sel.seamIds);
    }
    if (!bounds) return false;
    this.camera.frame(bounds.min, bounds.max, aspect, margin);
    this.cameraDirty_ = true;
    return true;
  }

  boundsOfPanels(panelIds: string[]): Bounds | null {
    const assembled = this.assembled_;
    if (!assembled) return null;
    let bounds: Bounds | null = null;
    for (const panelId of panelIds) {
      const range = assembled.panelRanges.find((r) => r.panelId === panelId);
      if (!range) continue;
      const b = computeBounds(this.renderPositions_, range.vertexStart, range.vertexCount);
      bounds = bounds ? mergeBounds(bounds, b) : b;
    }
    return bounds;
  }

  boundsOfSeams(seamIds: string[]): Bounds | null {
    const assembled = this.assembled_;
    if (!assembled) return null;
    const wanted = new Set(seamIds);
    let bounds: Bounds | null = null;
    const includeVertex = (v: number) => {
      const b: Bounds = {
        min: [this.renderPositions_[v * 3], this.renderPositions_[v * 3 + 1], this.renderPositions_[v * 3 + 2]],
        max: [...this.renderPositions_.slice(v * 3, v * 3 + 3)] as [number, number, number],
      };
      bounds = bounds ? mergeBounds(bounds, b) : b;
    };
    for (const weld of assembled.weldPairs) {
      if (!wanted.has(weld.seamId)) continue;
      includeVertex(weld.vertexA);
      includeVertex(weld.vertexB);
    }
    return bounds;
  }

  vertexPanelId(vertex: number): string | null {
    const assembled = this.assembled_;
    if (!assembled || vertex < 0 || vertex >= this.vertexCount) return null;
    for (const r of assembled.panelRanges) {
      if (vertex >= r.vertexStart && vertex < r.vertexStart + r.vertexCount) return r.panelId;
    }
    return null;
  }

  requireAssembled(): AssembledGarment {
    if (!this.assembled_) throw new WorkspaceError("no-garment", "no garment loaded");
    return this.assembled_;
  }

  /** Drop all selection state (used on project close). */
  reset(): void {
    clearSelection(this.selection);
    this.visibility = defaultVisibility();
    this.project_ = null;
    this.assembled_ = null;
    this.restPositions_ = new Float32Array(0);
    this.renderPositions_ = new Float32Array(0);
    this.garmentEpoch++;
    this.simEpoch = 0;
    this.viewDirty_ = true;
    for (const listener of this.garmentListeners) {
      listener({ reason: "project-load", garmentEpoch: this.garmentEpoch, prunedSelection: { panels: [], seams: [], region: false } });
    }
  }
}

export function mergeBounds(a: Bounds, b: Bounds): Bounds {
  return {
    min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
    max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
  };
}
