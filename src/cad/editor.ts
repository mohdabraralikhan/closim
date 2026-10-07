// G9D — 2D pattern editor interaction layer (headless).
//
// The repository has no frontend framework, so G9D is a UI-free workspace
// model: viewport math (pan/zoom/fit), a pointer/keyboard state machine over
// the G9A session (tools, drag gestures, box selection, snapping, numeric
// input, shortcuts, undo/redo), and a status readout answering: what is
// selected, what tool is active, cursor coordinates, current measurement,
// geometry validity, and dirty state.
//
// A future renderer binds screen pixels to these methods without changing
// any geometry semantics. Editing never touches simulation: every mutation
// goes through CadSession ops on the PatternDocument.

import {
  PatternCadError,
  createPoint,
  deserializePatternDocument,
  localToGlobal,
  movePoint,
  serializePatternDocument,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
  type PatternPanel,
  type PatternValidationResult,
} from "../pattern/cad.js";
import {
  addToSelection,
  clearSelection,
  isSelected,
  pruneSelection,
  removeFromSelection,
  setSelection,
  toggleSelection,
  type Selection,
} from "./selection.js";
import { CadSession } from "./history.js";
import {
  boxSelect,
  getPanel,
  getPoint,
  getSegment,
  globalPointToPanelLocal,
  hitTest,
  nearestHit,
  pointInPanelGlobal,
  resolveSegment,
  segmentParam,
  type Hit,
} from "./queries.js";
import { snapPosition, type SnapKind, type SnapResult } from "./snap.js";
import { deletePoint, mirrorPanel, splitSegment, trimSegment } from "./ops.js";
import {
  addDistanceConstraint,
  createConstraintSet,
  deserializeConstrainedDocument,
  measurePointDistance,
  serializeConstrainedDocument,
  type ConstraintSet,
} from "./constraints.js";
import { draftLine, offsetLoop } from "./draft.js";
import { dist, rotateAround, type Vec2 } from "./geom.js";

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

export interface Viewport {
  /** World-space center (metres, y-up). */
  center: Vec2;
  /** Pixels per metre. */
  scale: number;
  widthPx: number;
  heightPx: number;
}

export function createViewport(widthPx = 800, heightPx = 600, scale = 400): Viewport {
  if (!(widthPx > 0) || !(heightPx > 0) || !(scale > 0)) {
    throw new PatternCadError("invalid-transform", "viewport dimensions and scale must be positive");
  }
  return { center: [0, 0], scale, widthPx, heightPx };
}

/** World (y-up metres) -> screen (y-down pixels). */
export function worldToScreen(v: Viewport, world: Vec2): Vec2 {
  return [
    (world[0] - v.center[0]) * v.scale + v.widthPx / 2,
    v.heightPx / 2 - (world[1] - v.center[1]) * v.scale,
  ];
}

/** Screen (y-down pixels) -> world (y-up metres). */
export function screenToWorld(v: Viewport, screen: Vec2): Vec2 {
  return [
    v.center[0] + (screen[0] - v.widthPx / 2) / v.scale,
    v.center[1] - (screen[1] - v.heightPx / 2) / v.scale,
  ];
}

/** Pan so content follows the cursor by (dxPx, dyPx). */
export function panByPixels(v: Viewport, dxPx: number, dyPx: number): Viewport {
  return {
    ...v,
    center: [v.center[0] - dxPx / v.scale, v.center[1] + dyPx / v.scale],
  };
}

/** Zoom around a screen anchor (factor > 1 zooms in). */
export function zoomAt(v: Viewport, anchorPx: Vec2, factor: number): Viewport {
  if (!Number.isFinite(factor) || !(factor > 0)) {
    throw new PatternCadError("invalid-transform", "zoom factor must be positive and finite");
  }
  const world = screenToWorld(v, anchorPx);
  const scale = v.scale * factor;
  return {
    ...v,
    scale,
    center: [
      world[0] - (anchorPx[0] - v.widthPx / 2) / scale,
      world[1] + (anchorPx[1] - v.heightPx / 2) / scale,
    ],
  };
}

/** Fit a world bounding box with a pixel margin. */
export function fitToView(v: Viewport, min: Vec2, max: Vec2, marginPx = 40): Viewport {
  const dx = max[0] - min[0], dy = max[1] - min[1];
  if (!(dx > 0) || !(dy > 0)) return { ...v, center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2] };
  const scale = Math.min((v.widthPx - 2 * marginPx) / dx, (v.heightPx - 2 * marginPx) / dy);
  if (!(scale > 0) || !Number.isFinite(scale)) return v;
  return { ...v, scale, center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2] };
}

// ---------------------------------------------------------------------------
// Tools & session
// ---------------------------------------------------------------------------

export type ToolId =
  | "select" | "point" | "line" | "move" | "rotate" | "mirror"
  | "offset" | "trim" | "split" | "measure" | "constraint";

const SHORTCUTS: Record<string, ToolId> = {
  v: "select", p: "point", l: "line", m: "move", r: "rotate", i: "mirror",
  o: "offset", t: "trim", s: "split", u: "measure", c: "constraint",
};

export interface EditorStatus {
  tool: ToolId;
  selectionCount: number;
  cursorWorld: Vec2 | null;
  snapKind: SnapKind | null;
  /** Human-readable live hint: rubber length, drag delta, box size, measure. */
  hint: string | null;
  measurementM: number | null;
  valid: boolean;
  diagnosticCount: number;
  dirty: boolean;
  gestureOpen: boolean;
  numericBuffer: string;
  activePanelId: EntityId | null;
}

interface Pending {
  anchorWorld?: Vec2;
  pivotWorld?: Vec2;
  startWorld?: Vec2;
  rotateStarts?: Map<EntityId, Vec2>;
  targetSegmentId?: EntityId;
  targetPanelId?: EntityId;
  measurePoints?: Vec2[];
  constraintPointIds?: EntityId[];
  constraintPanelId?: EntityId;
  constraintTargetM?: number;
  boxStartWorld?: Vec2;
  boxShift?: boolean;
  panStartScreen?: Vec2;
  panStartCenter?: Vec2;
  grabbedPointId?: EntityId;
  grabbedPanelId?: EntityId;
  panning?: boolean;
  boxing?: boolean;
}

export interface EditorOptions {
  viewport?: Viewport;
  /** Snap capture radius in screen pixels. Default 8. */
  snapPx?: number;
  /** Grid step in metres; undefined disables grid snapping. Default off. */
  gridM?: number;
  /** Default offset distance in metres. Default 0.01. */
  defaultOffsetM?: number;
}

const fmtM = (v: number): string => `${(v * 1000).toFixed(1)} mm`;

export class EditorSession {
  readonly cad: CadSession;
  constraints: ConstraintSet;
  selection: Selection = [];
  viewport: Viewport;
  tool: ToolId = "select";
  activePanelId: EntityId | null = null;
  snapPx: number;
  gridM: number | undefined;
  defaultOffsetM: number;
  cursorWorld: Vec2 | null = null;
  lastSnap: SnapResult | null = null;
  numericBuffer = "";
  pending: Pending = {};
  private spaceHeld = false;

  constructor(doc: PatternDocument, opts: EditorOptions = {}) {
    this.cad = new CadSession(doc);
    this.constraints = createConstraintSet();
    this.viewport = opts.viewport ?? createViewport();
    this.snapPx = opts.snapPx ?? 8;
    this.gridM = opts.gridM;
    this.defaultOffsetM = opts.defaultOffsetM ?? 0.01;
    if (doc.panels.length > 0) this.activePanelId = doc.panels[0].id;
  }

  get document(): PatternDocument {
    return this.cad.document;
  }

  // -- helpers -------------------------------------------------------------

  private tolM(): number {
    return this.snapPx / this.viewport.scale;
  }

  private snapWorld(raw: Vec2): SnapResult {
    const r = snapPosition(this.document, raw, {
      toleranceM: this.tolM(),
      ...(this.gridM !== undefined ? { gridM: this.gridM } : {}),
    });
    this.lastSnap = r;
    return r;
  }

  /** First panel (store order) containing a world position, else null. */
  panelAt(world: Vec2): EntityId | null {
    for (const panel of this.document.panels) {
      try {
        if (pointInPanelGlobal(this.document, panel, world)) return panel.id;
      } catch {
        continue;
      }
    }
    return null;
  }

  /** Resolve which panel a world-space creation belongs to. */
  private panelForCreation(world: Vec2): EntityId {
    const id = this.panelAt(world) ?? this.activePanelId ?? this.document.panels[0]?.id;
    if (!id) throw new PatternCadError("missing-reference", "no panel to draw into");
    return id;
  }

  private localOf(panelId: EntityId, world: Vec2): Vec2 {
    return globalPointToPanelLocal(this.document, panelId, world);
  }

  setTool(tool: ToolId): void {
    this.tool = tool;
    this.pending = {};
    this.numericBuffer = "";
  }

  setActivePanel(panelId: EntityId | null): void {
    if (panelId !== null) getPanel(this.document, panelId);
    this.activePanelId = panelId;
  }

  // -- pointer -------------------------------------------------------------

  pointerDown(screen: Vec2, mods: { shift?: boolean } = {}, button: "left" | "middle" | "right" = "left"): void {
    if (button === "right") {
      this.cancel();
      return;
    }
    if (button === "middle" || this.spaceHeld) {
      this.pending = { panning: true, panStartScreen: [...screen], panStartCenter: [...this.viewport.center] };
      return;
    }
    const world = screenToWorld(this.viewport, screen);
    this.cursorWorld = world;
    switch (this.tool) {
      case "select": this.downSelect(world, mods.shift ?? false); break;
      case "point": this.downPoint(world); break;
      case "line": this.pending = { anchorWorld: this.snapWorld(world).pos }; break;
      case "move": this.downGrab(world, "move"); break;
      case "rotate": this.downRotate(world); break;
      case "mirror": this.pending = { anchorWorld: this.snapWorld(world).pos }; break;
      case "offset": this.downOffset(world); break;
      case "trim": this.downTrim(world); break;
      case "split": this.downSplit(world); break;
      case "measure": this.downMeasure(world); break;
      case "constraint": this.downConstraint(world); break;
    }
  }

  pointerMove(screen: Vec2): void {
    const world = screenToWorld(this.viewport, screen);
    this.cursorWorld = world;
    const p = this.pending;
    if (p.panning && p.panStartScreen && p.panStartCenter) {
      this.viewport = {
        ...this.viewport,
        center: [
          p.panStartCenter[0] - (screen[0] - p.panStartScreen[0]) / this.viewport.scale,
          p.panStartCenter[1] + (screen[1] - p.panStartScreen[1]) / this.viewport.scale,
        ],
      };
      return;
    }
    if (p.boxing && p.boxStartWorld) return; // box preview derives in status from cursor
    if (p.grabbedPointId && p.grabbedPanelId && p.startWorld && this.tool !== "rotate") {
      if (!this.cad.inGesture) this.cad.beginGesture("drag point");
      const snapped = this.snapWorld(world).pos;
      const local = this.localOf(p.grabbedPanelId, snapped);
      this.cad.run("drag", (d) => movePoint(d, p.grabbedPanelId!, p.grabbedPointId!, local));
      return;
    }
    if (this.tool === "rotate" && p.grabbedPanelId && p.pivotWorld && p.startWorld && p.rotateStarts) {
      const a0 = Math.atan2(p.startWorld[1] - p.pivotWorld[1], p.startWorld[0] - p.pivotWorld[0]);
      const a1 = Math.atan2(world[1] - p.pivotWorld[1], world[0] - p.pivotWorld[0]);
      this.rotateSelectionTo(p.grabbedPanelId, p.pivotWorld, p.rotateStarts, a1 - a0);
    }
  }

  pointerUp(screen: Vec2, mods: { shift?: boolean } = {}): void {
    const world = screenToWorld(this.viewport, screen);
    this.cursorWorld = world;
    const p = this.pending;
    if (p.panning) {
      this.pending = {};
      return;
    }
    switch (this.tool) {
      case "select": this.upSelect(world, mods.shift ?? false); break;
      case "line": this.upLine(world); break;
      case "move":
      case "rotate":
        if (this.cad.inGesture) this.cad.endGesture();
        this.pending = {};
        break;
      case "mirror": this.upMirror(world); break;
      default: break;
    }
  }

  wheel(screen: Vec2, deltaY: number): void {
    this.viewport = zoomAt(this.viewport, screen, Math.exp(-deltaY * 0.001));
  }

  keyDown(key: string, mods: { ctrl?: boolean; shift?: boolean } = {}): boolean {
    if (key === " ") {
      this.spaceHeld = true;
      return true;
    }
    if (mods.ctrl && (key === "z" || key === "Z")) {
      if (mods.shift) this.cad.redo();
      else this.cad.undo();
      this.afterMutation();
      return true;
    }
    if (mods.ctrl && key === "y") {
      this.cad.redo();
      this.afterMutation();
      return true;
    }
    if (key === "Escape") {
      this.cancel();
      return true;
    }
    if (key === "Enter") {
      this.commitNumeric();
      return true;
    }
    if ((key === "Delete" || key === "Backspace") && this.tool === "select") {
      this.deleteSelection();
      return true;
    }
    const tool = SHORTCUTS[key.toLowerCase()];
    if (tool && !mods.ctrl) {
      this.setTool(tool);
      return true;
    }
    if (/^[0-9.,\-+]$/.test(key) && this.tool !== "select") {
      this.numericBuffer += key;
      return true;
    }
    return false;
  }

  keyUp(key: string): void {
    if (key === " ") this.spaceHeld = false;
  }

  /** Escape / right-click: numeric -> pending -> gesture -> selection. */
  cancel(): void {
    if (this.numericBuffer) {
      this.numericBuffer = "";
      return;
    }
    if (this.pending.measurePoints?.length) {
      this.pending = {};
      return;
    }
    if (this.pending.anchorWorld || this.pending.targetSegmentId || this.pending.constraintPointIds?.length) {
      this.pending = {};
      return;
    }
    if (this.cad.inGesture) {
      this.cad.cancelGesture();
      this.pending = {};
      return;
    }
    if (this.pending.boxing || this.pending.panning) {
      this.pending = {};
      return;
    }
    this.selection = [];
  }

  /** Apply the numeric buffer to the current tool context. */
  commitNumeric(): void {
    const raw = this.numericBuffer.trim();
    this.numericBuffer = "";
    if (!raw) return;
    if (this.tool === "offset") {
      const v = Number(raw);
      if (Number.isFinite(v) && v !== 0) this.defaultOffsetM = Math.abs(v) / 1000; // typed in mm
      return;
    }
    if (this.tool === "constraint") {
      const v = Number(raw);
      if (Number.isFinite(v) && v > 0) this.pending.constraintTargetM = v / 1000;
      return;
    }
    if (this.tool === "move" && this.pending.grabbedPointId && this.pending.grabbedPanelId) {
      const parts = raw.split(",").map(Number);
      if (parts.every(Number.isFinite)) {
        const dx = parts[0] / 1000, dy = (parts[1] ?? 0) / 1000; // typed in mm
        const panelId = this.pending.grabbedPanelId, pointId = this.pending.grabbedPointId;
        const pt = getPoint(this.document, pointId, panelId);
        this.cad.run("numeric move", (d) => movePoint(d, panelId, pointId, [pt.x + dx, pt.y + dy]));
        this.afterMutation();
      }
      return;
    }
    if (this.tool === "rotate" && this.pending.grabbedPanelId && this.pending.pivotWorld && this.pending.rotateStarts) {
      const deg = Number(raw);
      if (Number.isFinite(deg)) {
        this.rotateSelectionTo(
          this.pending.grabbedPanelId, this.pending.pivotWorld,
          this.pending.rotateStarts, (deg * Math.PI) / 180,
        );
        if (this.cad.inGesture) this.cad.endGesture();
        this.pending = {};
        this.afterMutation();
      }
    }
  }

  // -- tool implementations --------------------------------------------------

  private afterMutation(): void {
    this.selection = pruneSelection(this.document, this.selection);
    if (this.activePanelId && !this.document.panels.some((p) => p.id === this.activePanelId)) {
      this.activePanelId = this.document.panels[0]?.id ?? null;
    }
  }

  private downSelect(world: Vec2, shift: boolean): void {
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit) {
      this.pending = { boxing: true, boxStartWorld: world, boxShift: shift } as Pending;
      return;
    }
    if (hit.kind === "point") {
      if (!shift && !isSelected(this.selection, hit.entityId)) this.selection = setSelection([hit.entityId]);
      else if (shift) this.selection = toggleSelection(this.selection, [hit.entityId]);
      // Prepare a potential point drag (gesture opens past the movement threshold).
      this.pending = { grabbedPointId: hit.entityId, grabbedPanelId: hit.panelId, startWorld: world };
    } else {
      this.selection = shift ? toggleSelection(this.selection, [hit.entityId]) : setSelection([hit.entityId]);
      if (hit.kind === "panel") this.activePanelId = hit.entityId;
      this.pending = {};
    }
  }

  private upSelect(world: Vec2, shift: boolean): void {
    const p = this.pending;
    if (p.boxing && p.boxStartWorld) {
      const ids = boxSelect(this.document, p.boxStartWorld, world, { mode: "overlap" });
      this.selection = shift ? addToSelection(this.selection, ids) : setSelection(ids);
      this.pending = {};
      return;
    }
    if (p.grabbedPointId && p.startWorld) {
      // Click without drag on an already-selected point with no shift: isolate it.
      if (dist(p.startWorld, world) * this.viewport.scale < 3 && !shift) {
        this.selection = setSelection([p.grabbedPointId]);
      }
      if (this.cad.inGesture) this.cad.endGesture();
      this.pending = {};
    }
  }

  private downPoint(world: Vec2): void {
    const snapped = this.snapWorld(world).pos;
    const panelId = this.panelForCreation(snapped);
    const local = this.localOf(panelId, snapped);
    const created = this.cad.run("add point", (d) => createPoint(d, panelId, local, "construction"));
    this.selection = setSelection([created.pointId]);
    this.activePanelId = panelId;
    this.afterMutation();
  }

  private upLine(world: Vec2): void {
    const anchor = this.pending.anchorWorld;
    this.pending = {};
    if (!anchor) return;
    const endSnap = this.snapWorld(world).pos;
    if (dist(anchor, endSnap) <= 1e-9) return; // click without drag: keep anchor? No — anchor-only is nothing.
    const panelId = this.panelForCreation(anchor);
    if ((this.panelAt(endSnap) ?? panelId) !== panelId && this.panelAt(endSnap) !== null) {
      return; // endpoints in different panels: refuse rather than mis-assign
    }
    const a = this.localOf(panelId, anchor);
    const b = this.localOf(panelId, endSnap);
    try {
      const r = this.cad.run("draft line", (d) => draftLine(d, panelId, a, b));
      this.selection = setSelection([r.segmentId]);
      this.activePanelId = panelId;
    } catch {
      return;
    }
    this.afterMutation();
  }

  private downGrab(world: Vec2, label: string): void {
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit || hit.kind !== "point") {
      this.pending = {};
      return;
    }
    this.selection = setSelection([hit.entityId]);
    this.cad.beginGesture(label === "move" ? "drag point" : label);
    this.pending = { grabbedPointId: hit.entityId, grabbedPanelId: hit.panelId, startWorld: world };
  }

  private downRotate(world: Vec2): void {
    const pointIds = this.selection.filter((id) =>
      this.document.points.some((q) => q.id === id),
    );
    if (pointIds.length === 0) {
      this.pending = {};
      return;
    }
    const panels = new Set(pointIds.map((id) => this.document.points.find((q) => q.id === id)!.panelId));
    if (panels.size !== 1) {
      this.pending = {};
      return;
    }
    const panelId = [...panels][0];
    const pivot = this.snapWorld(world).pos;
    // Store start positions for absolute (non-accumulating) rotation.
    const starts = new Map<EntityId, Vec2>();
    for (const id of pointIds) {
      const pt = getPoint(this.document, id, panelId);
      starts.set(id, this.worldOf(panelId, [pt.x, pt.y]));
    }
    this.cad.beginGesture("rotate");
    this.pending = { grabbedPanelId: panelId, pivotWorld: pivot, startWorld: world, rotateStarts: starts };
  }

  private worldOf(panelId: EntityId, local: Vec2): Vec2 {
    return localToGlobal(getPanel(this.document, panelId), local);
  }

  private rotateSelectionTo(panelId: EntityId, pivotWorld: Vec2, starts: Map<EntityId, Vec2>, delta: number): void {
    const pivotLocal = this.localOf(panelId, pivotWorld);
    this.cad.run("rotate", (d) => {
      let next = d;
      for (const [id, w0] of starts as unknown as Map<EntityId, Vec2>) {
        const startLocal = this.localOf(panelId, w0);
        const r = rotateAround(startLocal, pivotLocal, delta);
        next = movePoint(next, panelId, id, r);
      }
      return next;
    });
  }

  private downOffset(world: Vec2): void {
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit || hit.kind !== "segment") return;
    const panel = getPanel(this.document, hit.panelId);
    const loop = panel.boundaryLoops.find((l) => l.segmentIds.includes(hit.entityId));
    if (!loop || loop.role !== "outer") return;
    try {
      const r = this.cad.run("offset loop", (d) => offsetLoop(d, hit.panelId, loop.id, this.defaultOffsetM));
      this.selection = setSelection([r.panelId]);
      this.activePanelId = r.panelId;
    } catch {
      return;
    }
    this.afterMutation();
  }

  private downTrim(world: Vec2): void {
    const pendingTarget = this.pending.targetSegmentId;
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit || hit.kind !== "segment") {
      this.pending = {};
      return;
    }
    if (!pendingTarget) {
      this.pending = { targetSegmentId: hit.entityId, targetPanelId: hit.panelId };
      return;
    }
    if (hit.entityId === pendingTarget) {
      this.pending = {};
      return;
    }
    const panelId = hit.panelId;
    const targetPanel = this.pending.targetPanelId;
    this.pending = {};
    if (targetPanel !== panelId) return; // cross-panel trim is out of scope
    try {
      this.cad.run("trim", (d) =>
        trimSegment(d, panelId, pendingTarget, hit.entityId, this.localOf(panelId, world)),
      );
    } catch {
      return;
    }
    this.afterMutation();
  }

  private downSplit(world: Vec2): void {
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit || hit.kind !== "segment") return;
    const panel = getPanel(this.document, hit.panelId);
    const loop = panel.boundaryLoops.find((l) => l.segmentIds.includes(hit.entityId));
    if (!loop) return;
    const local = this.localOf(hit.panelId, world);
    const r = resolveSegment(this.document, hit.entityId, hit.panelId);
    const t = segmentParam(r, local);
    if (!(t > 0.05) || !(t < 0.95)) return; // too close to an endpoint: refuse
    try {
      const out = this.cad.run("split", (d) => splitSegment(d, hit.panelId, loop.id, hit.entityId, t));
      this.selection = setSelection([out.pointId]);
    } catch {
      return;
    }
    this.afterMutation();
  }

  private downMeasure(world: Vec2): void {
    const snapped = this.snapWorld(world).pos;
    const pts = [...(this.pending.measurePoints ?? []), snapped];
    this.pending = { ...this.pending, measurePoints: pts };
  }

  private downConstraint(world: Vec2): void {
    const hit = nearestHit(this.document, world, this.tolM());
    if (!hit || hit.kind !== "point") return;
    const ids = [...(this.pending.constraintPointIds ?? []), hit.entityId];
    const panelId = this.pending.constraintPanelId ?? hit.panelId;
    if (hit.panelId !== panelId) {
      this.pending = {};
      return;
    }
    if (ids.length < 2) {
      this.pending = { ...this.pending, constraintPointIds: ids, constraintPanelId: panelId };
      return;
    }
    const targetM = this.pending.constraintTargetM
      ?? measurePointDistance(this.document, panelId, ids[0], ids[1]);
    const added = addDistanceConstraint(this.constraints, panelId, ids[0], ids[1], targetM);
    this.constraints = added.set;
    this.selection = setSelection([added.id]);
    this.pending = {};
  }

  private upMirror(world: Vec2): void {
    const anchor = this.pending.anchorWorld;
    this.pending = {};
    if (!anchor || dist(anchor, world) <= 1e-9) return;
    const panels = new Set(
      this.selection
        .map((id) => this.document.points.find((q) => q.id === id)?.panelId)
        .filter((x): x is EntityId => !!x),
    );
    const targets = panels.size > 0 ? [...panels] : this.activePanelId ? [this.activePanelId] : [];
    if (targets.length === 0) return;
    try {
      this.cad.run("mirror", (d) => {
        let next = d;
        for (const pid of targets) next = mirrorPanel(next, pid, anchor, world);
        return next;
      });
    } catch {
      return;
    }
    this.afterMutation();
  }

  private deleteSelection(): void {
    const pointIds = this.selection.filter((id) =>
      this.document.points.some((q) => q.id === id),
    );
    if (pointIds.length === 0) {
      this.selection = [];
      return;
    }
    try {
      this.cad.run("delete", (d) => {
        let next = d;
        for (const id of pointIds) {
          const pt = next.points.find((q) => q.id === id);
          if (!pt) continue;
          try {
            next = deletePoint(next, pt.panelId, id);
          } catch {
            continue; // guarded points (arc centers, referenced) stay; rest still deletes
          }
        }
        return next;
      });
    } catch {
      return;
    }
    this.selection = [];
    this.afterMutation();
  }

  // -- persistence & status --------------------------------------------------

  /** Serialize workspace (pattern envelope + constraints). Dirty flag clears on load only. */
  saveState(): string {
    return serializeConstrainedDocument(serializePatternDocument(this.document), this.constraints);
  }

  loadState(serialized: string): void {
    const env = deserializeConstrainedDocument(serialized);
    const doc = deserializePatternDocument(env.documentJson);
    this.cad.reset(doc);
    this.constraints = env.set;
    this.selection = [];
    this.pending = {};
    this.activePanelId = doc.panels[0]?.id ?? null;
  }

  get dirty(): boolean {
    return this.cad.history.undoDepth > 0;
  }

  getStatus(): EditorStatus {
    const validation = validatePatternDocument(this.document);
    const pts = this.pending.measurePoints ?? [];
    let measurementM: number | null = null;
    let hint: string | null = null;
    if (this.tool === "measure" && pts.length >= 1 && this.cursorWorld) {
      const last = pts[pts.length - 1];
      measurementM = dist(last, this.cursorWorld);
      hint = pts.length >= 2
        ? `chain ${fmtM(chainLength(pts))} + ${fmtM(measurementM)}`
        : `dist ${fmtM(measurementM)}`;
    } else if (this.tool === "measure" && pts.length >= 2) {
      measurementM = chainLength(pts);
      hint = `chain ${fmtM(measurementM)}`;
    }
    if (this.tool === "line" && this.pending.anchorWorld && this.cursorWorld) {
      const d = dist(this.pending.anchorWorld, this.cursorWorld);
      hint = `len ${fmtM(d)}`;
      measurementM = d;
    }
    if (this.tool === "trim" && this.pending.targetSegmentId) hint = "pick cutter";
    if (this.tool === "constraint" && (this.pending.constraintPointIds?.length ?? 0) === 1) {
      hint = "pick second point";
    }
    if (this.tool === "mirror" && this.pending.anchorWorld) hint = "pick mirror line end";
    if (this.cad.inGesture) hint = `${hint ?? "drag"} …`;
    return {
      tool: this.tool,
      selectionCount: this.selection.length,
      cursorWorld: this.cursorWorld ? [...this.cursorWorld] : null,
      snapKind: this.lastSnap?.kind ?? null,
      hint,
      measurementM,
      valid: validation.valid,
      diagnosticCount: validation.diagnostics.length,
      dirty: this.dirty,
      gestureOpen: this.cad.inGesture,
      numericBuffer: this.numericBuffer,
      activePanelId: this.activePanelId,
    };
  }

  /** Gate before G8 assembly: kernel validation result (invalid patterns rejected upstream). */
  validateForAssembly(): PatternValidationResult {
    return validatePatternDocument(this.document);
  }
}

function chainLength(pts: Vec2[]): number {
  let total = 0;
  for (let i = 1; i < pts.length; i++) total += dist(pts[i - 1], pts[i]);
  return total;
}
