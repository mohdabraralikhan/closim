/** Deterministic seam construction over stable G8A pattern entity IDs. */
import {
  localToGlobal,
  type BoundaryLoop,
  type PatternDocument,
  type PatternPanel,
  type PatternSegment,
} from "../pattern/cad.js";
import type { Vec2 } from "../pattern/pattern-geometry.js";

export interface SeamSide {
  panelId: string;
  loopId: string;
  /** Ordered segment IDs; these are stable CAD references, not array indices. */
  segmentIds: string[];
  /** Reverse parameter direction on this side for sewing correspondence. */
  reversed: boolean;
}

export interface Seam {
  id: string;
  sideA: SeamSide;
  sideB: SeamSide;
  stitchCount: number;
  groupId?: string;
  metadata?: Record<string, string>;
}

export interface StitchPair {
  seamId: string;
  index: number;
  /** Normalized arc-length parameters in each seam side's authored direction. */
  tA: number;
  tB: number;
  pointA: Vec2;
  pointB: Vec2;
}

export interface SeamDiagnostic {
  code: "duplicate-id" | "duplicate-seam" | "missing-panel" | "missing-loop" |
    "missing-segment" | "segment-not-in-loop" | "invalid-chain" |
    "zero-length-seam" | "invalid-stitch-count" | "same-side";
  seamId: string;
  message: string;
}

export interface SeamValidation {
  valid: boolean;
  diagnostics: SeamDiagnostic[];
}

export interface AssemblyGraph {
  panelIds: string[];
  seams: Array<{ seamId: string; panelA: string; panelB: string; stitchPairs: StitchPair[] }>;
}

interface SampledPath {
  points: Vec2[];
  cumulative: number[];
  length: number;
}

const ARC_SEGMENTS = 128;
const EPS = 1e-12;

function panelById(document: PatternDocument, id: string): PatternPanel | undefined {
  return document.panels.find((panel) => panel.id === id);
}

function pointById(document: PatternDocument, id: string) {
  return document.points.find((point) => point.id === id);
}

function loopById(panel: PatternPanel, id: string): BoundaryLoop | undefined {
  return panel.boundaryLoops.find((loop) => loop.id === id);
}

function segmentPoint(document: PatternDocument, panel: PatternPanel, segment: PatternSegment, t: number): Vec2 {
  const a = pointById(document, segment.startPointId)!;
  const b = pointById(document, segment.endPointId)!;
  let local: Vec2;
  if (segment.kind === "line") {
    local = [a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t];
  } else {
    const center = pointById(document, segment.centerPointId)!;
    const start = Math.atan2(a.y - center.y, a.x - center.x);
    const radius = Math.hypot(a.x - center.x, a.y - center.y);
    const angle = start + segment.sweepRad * t;
    local = [center.x + radius * Math.cos(angle), center.y + radius * Math.sin(angle)];
  }
  return localToGlobal(panel, local);
}

function sampleSide(document: PatternDocument, side: SeamSide): SampledPath | null {
  const panel = panelById(document, side.panelId);
  if (!panel) return null;
  const points: Vec2[] = [];
  for (const segmentId of side.segmentIds) {
    const segment = document.segments.find((item) => item.id === segmentId);
    if (!segment) return null;
    const divisions = segment.kind === "arc" ? ARC_SEGMENTS : 1;
    for (let i = 0; i <= divisions; i++) {
      if (points.length > 0 && i === 0) continue;
      points.push(segmentPoint(document, panel, segment, i / divisions));
    }
  }
  const cumulative = [0];
  for (let i = 1; i < points.length; i++) {
    cumulative.push(cumulative[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]));
  }
  const length = cumulative.at(-1) ?? 0;
  if (side.reversed) {
    points.reverse();
    cumulative.length = 0;
    cumulative.push(0);
    for (let i = 1; i < points.length; i++) {
      cumulative.push(cumulative[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]));
    }
  }
  return { points, cumulative, length };
}

function atFraction(path: SampledPath, t: number): Vec2 {
  if (path.points.length === 1) return [...path.points[0]];
  const target = Math.max(0, Math.min(1, t)) * path.length;
  let lo = 0;
  let hi = path.cumulative.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >>> 1;
    if (path.cumulative[mid] <= target) lo = mid;
    else hi = mid;
  }
  const span = path.cumulative[hi] - path.cumulative[lo];
  const f = span > 0 ? (target - path.cumulative[lo]) / span : 0;
  return [
    path.points[lo][0] + (path.points[hi][0] - path.points[lo][0]) * f,
    path.points[lo][1] + (path.points[hi][1] - path.points[lo][1]) * f,
  ];
}

function pathKey(side: SeamSide): string {
  return `${side.panelId}/${side.loopId}/${side.segmentIds.join(",")}/${side.reversed ? "r" : "f"}`;
}

function hasValidChain(document: PatternDocument, side: SeamSide, loop: BoundaryLoop): boolean {
  if (side.segmentIds.length === 0 || new Set(side.segmentIds).size !== side.segmentIds.length) return false;
  const ids = loop.segmentIds;
  const indices = side.segmentIds.map((id) => ids.indexOf(id));
  if (indices.some((i) => i < 0)) return false;
  // IDs always follow the authored boundary order. `reversed` changes the
  // correspondence traversal only; it does not rewrite CAD references.
  return indices.every((idx, i) => i === 0 || idx === (indices[i - 1] + 1) % ids.length);
}

export function validateSeams(document: PatternDocument, seams: readonly Seam[]): SeamValidation {
  const diagnostics: SeamDiagnostic[] = [];
  const ids = new Set<string>();
  const pairs = new Set<string>();
  for (const seam of seams) {
    const error = (code: SeamDiagnostic["code"], message: string) => diagnostics.push({ code, seamId: seam.id, message });
    if (ids.has(seam.id)) error("duplicate-id", `duplicate seam id ${seam.id}`);
    ids.add(seam.id);
    if (!Number.isInteger(seam.stitchCount) || seam.stitchCount < 2) error("invalid-stitch-count", "stitchCount must be an integer >= 2");
    if (seam.sideA.panelId === seam.sideB.panelId && seam.sideA.loopId === seam.sideB.loopId && pathKey(seam.sideA) === pathKey(seam.sideB)) {
      error("same-side", "a seam cannot join the same boundary path to itself");
    }
    const pairKey = [pathKey(seam.sideA), pathKey(seam.sideB)].sort().join("|<->|");
    if (pairs.has(pairKey)) error("duplicate-seam", "the same two boundary paths are already joined");
    pairs.add(pairKey);
    for (const side of [seam.sideA, seam.sideB]) {
      const panel = panelById(document, side.panelId);
      if (!panel) { error("missing-panel", `panel ${side.panelId} does not exist`); continue; }
      const loop = loopById(panel, side.loopId);
      if (!loop) { error("missing-loop", `loop ${side.loopId} does not exist on panel ${side.panelId}`); continue; }
      for (const segmentId of side.segmentIds) {
        const segment = document.segments.find((item) => item.id === segmentId);
        if (!segment) error("missing-segment", `segment ${segmentId} does not exist`);
        else if (segment.panelId !== side.panelId || !loop.segmentIds.includes(segmentId)) error("segment-not-in-loop", `segment ${segmentId} is not in referenced boundary loop`);
      }
      if (!hasValidChain(document, side, loop)) error("invalid-chain", `segments on panel ${side.panelId} are not a contiguous ordered boundary chain`);
      const path = sampleSide(document, side);
      if (path && (!Number.isFinite(path.length) || path.length <= EPS)) error("zero-length-seam", `boundary path on panel ${side.panelId} has zero length`);
    }
  }
  return { valid: diagnostics.length === 0, diagnostics };
}

export function resolveStitchPairs(document: PatternDocument, seam: Seam): StitchPair[] {
  const valid = validateSeams(document, [seam]);
  if (!valid.valid) throw new Error(`cannot resolve invalid seam ${seam.id}: ${valid.diagnostics.map((d) => d.code).join(", ")}`);
  const pathA = sampleSide(document, seam.sideA)!;
  const pathB = sampleSide(document, seam.sideB)!;
  const output: StitchPair[] = [];
  for (let index = 0; index < seam.stitchCount; index++) {
    const t = index / (seam.stitchCount - 1);
    output.push({ seamId: seam.id, index, tA: t, tB: t, pointA: atFraction(pathA, t), pointB: atFraction(pathB, t) });
  }
  return output;
}

export function buildAssemblyGraph(document: PatternDocument, seams: readonly Seam[]): AssemblyGraph {
  const validation = validateSeams(document, seams);
  if (!validation.valid) throw new Error(`cannot build assembly graph: ${validation.diagnostics.map((d) => `${d.seamId}:${d.code}`).join(", ")}`);
  return {
    panelIds: document.panels.map((panel) => panel.id),
    seams: seams.map((seam) => ({ seamId: seam.id, panelA: seam.sideA.panelId, panelB: seam.sideB.panelId, stitchPairs: resolveStitchPairs(document, seam) })),
  };
}
