/**
 * G8 final integration — deterministic basic T-shirt / simple top.
 *
 * Panels: front, back, left sleeve, right sleeve (rectangles in pattern
 * space). Seams: shoulder, left side, right side, two sleeve attachments.
 * Avatar: analytic capsule torso. Simulation: existing CPU solver defaults.
 *
 * Everything is deterministic: fixed ids ("tshirt/..." document id yields
 * stable entity ids), fixed order, fixed placements, fixed stitch counts.
 */

import {
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  insertBoundaryPoint,
  type PatternDocument,
} from "../pattern/cad.js";
import type { Seam, SeamSide } from "./sewing.js";
import { makeCapsuleAvatar, type AvatarSpec } from "./avatar.js";
import {
  createGarmentProject,
  type GarmentProject,
} from "./project.js";
import type { PanelPlacement } from "./assembly.js";
import type { Vec2 } from "../pattern/pattern-geometry.js";

export interface RectRefs {
  panelId: string;
  loopId: string;
  pointIds: string[];
  segmentIds: string[];
}

export function addRectPanel(
  document: PatternDocument,
  name: string,
  origin: Vec2,
  width: number,
  height: number,
  materialId = "cotton",
): { document: PatternDocument; refs: RectRefs } {
  let doc = document;
  const p = createPanel(doc, name, materialId);
  doc = p.document;
  const loop = createBoundaryLoop(doc, p.panelId, "outer");
  doc = loop.document;
  const corners: Vec2[] = [
    [origin[0], origin[1]],
    [origin[0] + width, origin[1]],
    [origin[0] + width, origin[1] + height],
    [origin[0], origin[1] + height],
  ];
  const pointIds: string[] = [];
  for (const c of corners) {
    const r = createPoint(doc, p.panelId, c);
    doc = r.document;
    pointIds.push(r.pointId);
  }
  const segmentIds: string[] = [];
  for (let i = 0; i < 4; i++) {
    const r = createBoundaryLine(doc, p.panelId, loop.loopId, pointIds[i], pointIds[(i + 1) % 4]);
    doc = r.document;
    segmentIds.push(r.segmentId);
  }
  return { document: doc, refs: { panelId: p.panelId, loopId: loop.loopId, pointIds, segmentIds } };
}

const side = (panelId: string, loopId: string, segmentIds: string[], reversed = false): SeamSide => ({
  panelId, loopId, segmentIds, reversed,
});

export interface TshirtBundle {
  project: GarmentProject;
  refs: { front: RectRefs; back: RectRefs; sleeveL: RectRefs; sleeveR: RectRefs };
  avatar: AvatarSpec;
}

/**
 * Build the complete deterministic T-shirt project.
 * Segment convention per rect: [0]=bottom, [1]=right, [2]=top, [3]=left.
 *
 * The top edge is split into right-shoulder / neck / left-shoulder pieces
 * and each side edge into upper (armhole) / lower (side-seam) pieces, so
 * seams join matching regions instead of spanning unrelated edges.
 */
export function buildTshirtProject(): TshirtBundle {
  let document = createPatternDocument("tshirt", "Basic T-shirt");
  const front = addRectPanel(document, "front", [0, 0], 0.46, 0.62);
  document = front.document;
  const back = addRectPanel(document, "back", [0, 0], 0.46, 0.62);
  document = back.document;
  const sleeveL = addRectPanel(document, "sleeve-left", [0, 0], 0.22, 0.14);
  document = sleeveL.document;
  const sleeveR = addRectPanel(document, "sleeve-right", [0, 0], 0.22, 0.14);
  document = sleeveR.document;

  // Split top edges (seg[2], stored p2 -> p3 i.e. right -> left) at the
  // neck boundaries x=0.31 then x=0.15. Pieces: topR = right shoulder
  // (lx .46 -> .31), neck (.31 -> .15), topL = left shoulder (.15 -> 0).
  // Split side edges at ly=0.37 into upper (armhole) / lower (side seam):
  // seg[3] is stored top -> bottom so upper comes first; seg[1] is stored
  // bottom -> top so lower comes first.
  interface ShirtPieces {
    topR: string; topL: string;
    leftUU: string; leftMid: string; leftLower: string;
    rightUU: string; rightMid: string; rightLower: string;
  }
  const splitShirtPanel = (
    doc: PatternDocument, panelId: string, loopId: string, segIds: string[], w: number,
  ): { document: PatternDocument; pieces: ShirtPieces } => {
    let d = doc;
    let r = insertBoundaryPoint(d, panelId, loopId, segIds[2], [0.31, 0.62]);
    d = r.document;
    const topRightFull = r.segmentIds[0];
    r = insertBoundaryPoint(d, panelId, loopId, r.segmentIds[1], [0.15, 0.62]);
    d = r.document;
    // Keep only the outer tip of each shoulder piece sewn (realistic ~4-5 cm
    // shoulder seams); the rest of the top edge stays a clean neckline.
    // topL is stored lx 0.15 -> 0: keep the second piece (0.04 -> 0).
    r = insertBoundaryPoint(d, panelId, loopId, r.segmentIds[1], [0.04, 0.62]);
    d = r.document;
    const topL = r.segmentIds[1];
    // topRightFull is stored lx 0.46 -> 0.31: keep the first piece (0.46 -> 0.42).
    r = insertBoundaryPoint(d, panelId, loopId, topRightFull, [0.42, 0.62]);
    d = r.document;
    const topR = r.segmentIds[0];
    // Left edge (seg[3], stored top -> bottom): split at 0.37, then the
    // upper remainder at 0.50 -> uu (0.62 -> 0.50), mid (0.50 -> 0.37),
    // lower (0.37 -> 0).
    r = insertBoundaryPoint(d, panelId, loopId, segIds[3], [0, 0.37]);
    d = r.document;
    const leftLower = r.segmentIds[1];
    r = insertBoundaryPoint(d, panelId, loopId, r.segmentIds[0], [0, 0.50]);
    d = r.document;
    const leftUU = r.segmentIds[0];
    const leftMid = r.segmentIds[1];
    // Right edge (seg[1], stored bottom -> top): lower, mid, uu in order.
    r = insertBoundaryPoint(d, panelId, loopId, segIds[1], [w, 0.37]);
    d = r.document;
    const rightLower = r.segmentIds[0];
    r = insertBoundaryPoint(d, panelId, loopId, r.segmentIds[1], [w, 0.50]);
    d = r.document;
    const rightMid = r.segmentIds[0];
    const rightUU = r.segmentIds[1];
    return {
      document: d,
      pieces: { topR, topL, leftUU, leftMid, leftLower, rightUU, rightMid, rightLower },
    };
  };
  const frontSplit = splitShirtPanel(document, front.refs.panelId, front.refs.loopId, front.refs.segmentIds, 0.46);
  document = frontSplit.document;
  const backSplit = splitShirtPanel(document, back.refs.panelId, back.refs.loopId, back.refs.segmentIds, 0.46);
  document = backSplit.document;
  const F = frontSplit.pieces;
  const B = backSplit.pieces;

  const seams: Seam[] = [
    {
      id: "seam/shoulder-left",
      sideA: side(front.refs.panelId, front.refs.loopId, [F.topL]),
      // Mirrored back wrap: both left-shoulder pieces run the same way, so
      // equal fractions meet tip to tip.
      sideB: side(back.refs.panelId, back.refs.loopId, [B.topL], false),
      stitchCount: 3, groupId: "shoulders",
    },
    {
      id: "seam/shoulder-right",
      sideA: side(front.refs.panelId, front.refs.loopId, [F.topR]),
      sideB: side(back.refs.panelId, back.refs.loopId, [B.topR], false),
      stitchCount: 3, groupId: "shoulders",
    },
    {
      id: "seam/side-left",
      // Continuous top -> bottom traversal: mid then lower (both stored
      // top -> bottom on seg[3]-derived pieces).
      sideA: side(front.refs.panelId, front.refs.loopId, [F.leftMid, F.leftLower]),
      sideB: side(back.refs.panelId, back.refs.loopId, [B.leftMid, B.leftLower], false),
      stitchCount: 9, groupId: "sides",
    },
    {
      id: "seam/side-right",
      // seg[1]-derived pieces are stored bottom -> top: lower then mid.
      sideA: side(front.refs.panelId, front.refs.loopId, [F.rightLower, F.rightMid]),
      sideB: side(back.refs.panelId, back.refs.loopId, [B.rightLower, B.rightMid], false),
      stitchCount: 9, groupId: "sides",
    },
    {
      id: "seam/sleeve-left",
      sideA: side(front.refs.panelId, front.refs.loopId, [F.leftUU]),
      sideB: side(sleeveL.refs.panelId, sleeveL.refs.loopId, [sleeveL.refs.segmentIds[2]]),
      stitchCount: 5, groupId: "sleeves",
    },
    {
      id: "seam/sleeve-right",
      sideA: side(front.refs.panelId, front.refs.loopId, [F.rightUU]),
      sideB: side(sleeveR.refs.panelId, sleeveR.refs.loopId, [sleeveR.refs.segmentIds[2]], true),
      stitchCount: 5, groupId: "sleeves",
    },
  ];

  // Pre-draped placements: front/back are isometrically wrapped around the
  // torso capsule (R matches the avatar surface, so the solver starts in
  // resting contact instead of free-falling flat panels); sleeves cap the
  // torso sides just outside the surface. translation[1] sets base height.
  const torso: [number, number] = [0.23, 0];
  const placements: PanelPlacement[] = [
    {
      panelId: front.refs.panelId, translation: [0.23, 0.6, 0], yawRad: 0,
      wrap: { center: torso, radiusM: 0.15, facingRad: 0, refLx: 0.23 },
    },
    {
      panelId: back.refs.panelId, translation: [0.23, 0.6, 0], yawRad: Math.PI,
      // Mirrored so the back's left edge lands on the same body side as the
      // front's left edge (side seams meet; shoulder tips meet with the
      // direct correspondence below).
      wrap: { center: torso, radiusM: 0.15, facingRad: Math.PI, refLx: 0.23, mirror: true },
    },
    {
      panelId: sleeveL.refs.panelId, translation: [0.23, 1.08, 0], yawRad: Math.PI / 2,
      wrap: { center: torso, radiusM: 0.155, facingRad: -Math.PI / 2, refLx: 0.11 },
    },
    {
      panelId: sleeveR.refs.panelId, translation: [0.23, 1.08, 0], yawRad: -Math.PI / 2,
      wrap: { center: torso, radiusM: 0.155, facingRad: Math.PI / 2, refLx: 0.11 },
    },
  ];

  const avatar = makeCapsuleAvatar({
    id: "avatar/torso", bodyPart: "torso",
    radiusM: 0.15, cylinderLengthM: 0.5, center: [0.23, 0.95, 0],
  });

  const project = createGarmentProject("garment/tshirt", "Basic T-shirt", document, {
    description: "G8 vertical-slice demonstration garment",
    author: "closim",
    seams,
    placements,
    avatar,
    simulation: {
      // Interior cloth resolution: boundary-only rectangles (2 tris each)
      // cannot drape; 2.5 cm edges drape on the torso capsule.
      meshMaxEdgeM: 0.025,
    },
    materials: {
      cotton: {
        arealDensityKgM2: 0.15, thickness: 0.001,
        stretchWarp: 20000, stretchWeft: 20000, stretchCoupling: 0,
        shear: 5000, bendWarp: 1e-5, bendWeft: 1e-5, damping: 0.001,
      },
    },
  });
  // Panels were authored with materialId "cotton" — align project table.
  return { project, refs: { front: front.refs, back: back.refs, sleeveL: sleeveL.refs, sleeveR: sleeveR.refs }, avatar };
}
