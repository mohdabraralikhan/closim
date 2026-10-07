// G14C — fabric rules and orientation system.
//
// Real-world placement rules, explicit and validated before nesting:
//
//   Fabric defaults -> piece overrides (item wins, deterministically)
//   directional fabric restricts rotation classes (never inferred mirroring)
//
// Grain runs along fabric +Y (marker length) unless grainAxisDeg says
// otherwise. Grain has no polarity: alignment is modulo 180°.

import {
  PatternCadError,
  type EntityId,
} from "../pattern/cad.js";
import {
  defaultNestingConstraint,
  type CutItem,
  type Fabric,
  type MarkerPiece,
  type NestingConstraint,
} from "./model.js";

export interface FabricRules {
  fabricId: string;
  /** Usable width minus selvedge/exclusion (the engine's hard boundary). */
  usableWidthM: number;
  /** Selvedge trimmed off each side (informational, already excluded above). */
  selvedgeM: number;
  /** Grain axis in degrees from +X (default 90 = along marker length). */
  grainAxisDeg: number;
  /** Directional fabric (nap, pile, one-way print): rotations limited to the 0°/180° class. */
  directional: boolean;
  /** Print repeat along Y in metres (engine snaps origins to multiples; absent = off). */
  repeatM?: number;
  defaultRotationsDeg: number[];
  defaultMirror: "mirrored" | "as-authored";
  grainToleranceDeg: number;
}

export interface PieceRules {
  instanceId: string;
  rotationsDeg: number[];
  /** Mirror variants to try, in trial order. */
  mirrorVariants: boolean[];
  grainToleranceDeg: number;
}

export interface RuleProblem {
  code: "incompatible-rotation" | "empty-rotations" | "too-wide" | "invalid-repeat";
  message: string;
  instanceId?: string;
}

const FULL_CIRCLE = 360;

function normDeg(deg: number): number {
  let d = deg % FULL_CIRCLE;
  if (d < 0) d += FULL_CIRCLE;
  return d;
}

/** True when two angles are congruent modulo 180° within tolerance. */
export function grainAligned(grainDeg: number, axisDeg: number, toleranceDeg: number): boolean {
  if (!(toleranceDeg >= 0)) return false;
  let diff = Math.abs(normDeg(grainDeg) - normDeg(axisDeg)) % 180;
  if (diff > 90) diff = 180 - diff;
  return diff <= toleranceDeg + 1e-9;
}

/** Grain direction of a piece instance under rotation/mirror (degrees from +X). */
export function pieceGrainDeg(grainRad: number, rotationDeg: number, mirrored: boolean): number {
  const base = (grainRad * 180) / Math.PI;
  const afterMirror = mirrored ? 180 - base : base;
  return normDeg(afterMirror + rotationDeg);
}

export function fabricRulesFrom(
  fabric: Fabric,
  constraint: NestingConstraint = defaultNestingConstraint(),
  opts: {
    selvedgeM?: number;
    grainAxisDeg?: number;
    directional?: boolean;
    repeatM?: number;
  } = {},
): FabricRules {
  const selvedgeM = opts.selvedgeM ?? 0;
  if (!(selvedgeM >= 0) || !Number.isFinite(selvedgeM)) {
    throw new PatternCadError("invalid-transform", "selvedge must be finite and >= 0", fabric.id);
  }
  const usableWidthM = fabric.usableWidthM - 2 * selvedgeM;
  if (!(usableWidthM > 0)) {
    throw new PatternCadError("invalid-transform", "selvedge leaves no usable width", fabric.id);
  }
  const grainAxisDeg = opts.grainAxisDeg ?? 90;
  if (!Number.isFinite(grainAxisDeg)) {
    throw new PatternCadError("invalid-transform", "grain axis must be finite", fabric.id);
  }
  if (opts.repeatM !== undefined && (!(opts.repeatM > 0) || !Number.isFinite(opts.repeatM))) {
    throw new PatternCadError("invalid-transform", "repeat must be positive", fabric.id);
  }
  return {
    fabricId: fabric.id,
    usableWidthM,
    selvedgeM,
    grainAxisDeg: normDeg(grainAxisDeg),
    directional: opts.directional ?? constraint.directionalFabric,
    ...(opts.repeatM !== undefined ? { repeatM: opts.repeatM } : {}),
    defaultRotationsDeg: [...constraint.rotationsDeg],
    defaultMirror: constraint.mirrorDefault,
    grainToleranceDeg: constraint.grainToleranceDeg,
  };
}

/**
 * Resolve effective rules for one cut item. Precedence (deterministic):
 * explicit item rotations > fabric defaults; item mirror policy decides the
 * variant list (never inferred from rotation).
 */
export function resolvePieceRules(
  rules: FabricRules,
  item: CutItem,
  instanceId: string,
  problems: RuleProblem[] = [],
): PieceRules {
  const rotationsDeg = [...(item.rotationsDeg ?? rules.defaultRotationsDeg)];
  if (rotationsDeg.length === 0) {
    problems.push({ code: "empty-rotations", message: `no rotations for '${instanceId}'`, instanceId });
  }
  const filtered = rules.directional
    ? rotationsDeg.filter((r) => {
      const n = normDeg(r);
      return Math.min(n, FULL_CIRCLE - n) <= 1e-9 || Math.abs(n - 180) <= 1e-9;
    })
    : rotationsDeg;
  if (filtered.length === 0 && rotationsDeg.length > 0) {
    problems.push({
      code: "incompatible-rotation",
      message: `directional fabric forbids all requested rotations for '${instanceId}'`,
      instanceId,
    });
  }
  const mirrorVariants =
    item.mirror === "required" ? [true]
    : item.mirror === "forbidden" ? [false]
    : rules.defaultMirror === "mirrored" ? [true, false] : [false, true];
  return { instanceId, rotationsDeg: filtered, mirrorVariants, grainToleranceDeg: rules.grainToleranceDeg };
}

/** Rotate a polygon about its centroid (deterministic, no mutation). */
export function transformPolygon(
  polygon: Array<[number, number]>,
  rotationDeg: number,
  mirrored: boolean,
): Array<[number, number]> {
  let pts = mirrored ? polygon.map((p) => [-p[0], p[1]] as [number, number]) : polygon.map((p) => [...p] as [number, number]);
  if (normDeg(rotationDeg) === 0) return pts;
  let cx = 0, cy = 0;
  for (const p of pts) {
    cx += p[0];
    cy += p[1];
  }
  cx /= pts.length;
  cy /= pts.length;
  const rad = (rotationDeg * Math.PI) / 180;
  const c = Math.cos(rad), s = Math.sin(rad);
  pts = pts.map((p) => [cx + c * (p[0] - cx) - s * (p[1] - cy), cy + s * (p[0] - cx) + c * (p[1] - cy)] as [number, number]);
  return pts;
}

export interface OrientedPiece {
  instanceId: string;
  panelId: EntityId;
  sizeId: EntityId;
  rotationDeg: number;
  mirrored: boolean;
  polygon: Array<[number, number]>;
  grainDeg: number;
  areaM2: number;
}

/** All candidate orientations for a piece under its rules (deterministic order). */
export function orientPiece(
  piece: MarkerPiece,
  rules: PieceRules,
  grainRad: number,
): OrientedPiece[] {
  const out: OrientedPiece[] = [];
  for (const rotationDeg of rules.rotationsDeg) {
    for (const mirrored of rules.mirrorVariants) {
      // As-authored instance geometry is already mirrored for mirror-required
      // items; the variant flag then selects the stored polygon directly.
      const polygon = transformPolygon(
        piece.polygon as Array<[number, number]>,
        rotationDeg,
        mirrored && !piece.mirrored,
      );
      out.push({
        instanceId: piece.instanceId,
        panelId: piece.panelId,
        sizeId: piece.sizeId,
        rotationDeg: normDeg(rotationDeg),
        mirrored: piece.mirrored || mirrored,
        polygon,
        grainDeg: pieceGrainDeg(grainRad, rotationDeg, mirrored !== piece.mirrored),
        areaM2: piece.areaM2,
      });
    }
  }
  return out;
}

/**
 * Pre-nesting feasibility: piece fits the usable width in at least one
 * orientation, rotation sets are non-empty, repeat is sane. Returns problems
 * (empty = clear to nest); the engine refuses to run on problems.
 */
export function checkNestingFeasibility(
  pieces: MarkerPiece[],
  rules: FabricRules,
  items: CutItem[],
  grainRadOf: (panelId: string) => number,
  marginM: number,
): RuleProblem[] {
  const problems: RuleProblem[] = [];
  const byItem = new Map(items.map((i) => [i.id, i]));
  for (const piece of pieces) {
    const item = byItem.get(piece.cutItemId);
    if (!item) {
      problems.push({ code: "incompatible-rotation", message: `piece '${piece.instanceId}' has no cut item`, instanceId: piece.instanceId });
      continue;
    }
    const pr = resolvePieceRules(rules, item, piece.instanceId, problems);
    const oriented = orientPiece(piece, pr, grainRadOf(piece.panelId));
    let fits = false;
    for (const o of oriented) {
      let minX = Infinity, maxX = -Infinity;
      for (const p of o.polygon) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
      }
      if (maxX - minX <= rules.usableWidthM - 2 * marginM + 1e-9) {
        fits = true;
        break;
      }
    }
    if (oriented.length > 0 && !fits) {
      problems.push({
        code: "too-wide",
        message: `piece '${piece.instanceId}' fits no orientation within ${rules.usableWidthM} m`,
        instanceId: piece.instanceId,
      });
    }
  }
  return problems;
}

export type { EntityId };
