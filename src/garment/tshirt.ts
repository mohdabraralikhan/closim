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
 */
export function buildTshirtProject(): TshirtBundle {
  let document = createPatternDocument("tshirt", "Basic T-shirt");
  const front = addRectPanel(document, "front", [0, 0], 0.46, 0.62);
  document = front.document;
  const back = addRectPanel(document, "back", [0, 0], 0.46, 0.62);
  document = back.document;
  const sleeveL = addRectPanel(document, "sleeve-left", [0, 0], 0.3, 0.25);
  document = sleeveL.document;
  const sleeveR = addRectPanel(document, "sleeve-right", [0, 0], 0.3, 0.25);
  document = sleeveR.document;

  const seams: Seam[] = [
    {
      id: "seam/shoulder",
      sideA: side(front.refs.panelId, front.refs.loopId, [front.refs.segmentIds[2]]),
      sideB: side(back.refs.panelId, back.refs.loopId, [back.refs.segmentIds[2]], true),
      stitchCount: 7, groupId: "shoulders",
    },
    {
      id: "seam/side-left",
      sideA: side(front.refs.panelId, front.refs.loopId, [front.refs.segmentIds[3]]),
      sideB: side(back.refs.panelId, back.refs.loopId, [back.refs.segmentIds[3]], true),
      stitchCount: 7, groupId: "sides",
    },
    {
      id: "seam/side-right",
      sideA: side(front.refs.panelId, front.refs.loopId, [front.refs.segmentIds[1]]),
      sideB: side(back.refs.panelId, back.refs.loopId, [back.refs.segmentIds[1]], true),
      stitchCount: 7, groupId: "sides",
    },
    {
      id: "seam/sleeve-left",
      sideA: side(front.refs.panelId, front.refs.loopId, [front.refs.segmentIds[3]]),
      sideB: side(sleeveL.refs.panelId, sleeveL.refs.loopId, [sleeveL.refs.segmentIds[2]]),
      stitchCount: 5, groupId: "sleeves",
    },
    {
      id: "seam/sleeve-right",
      sideA: side(front.refs.panelId, front.refs.loopId, [front.refs.segmentIds[1]]),
      sideB: side(sleeveR.refs.panelId, sleeveR.refs.loopId, [sleeveR.refs.segmentIds[2]], true),
      stitchCount: 5, groupId: "sleeves",
    },
  ];

  const placements: PanelPlacement[] = [
    { panelId: front.refs.panelId, translation: [0, 0.6, 0.2], yawRad: 0 },
    { panelId: back.refs.panelId, translation: [0.46, 0.6, -0.2], yawRad: Math.PI },
    { panelId: sleeveL.refs.panelId, translation: [-0.06, 0.95, 0.15], yawRad: Math.PI / 2 },
    { panelId: sleeveR.refs.panelId, translation: [0.52, 0.95, 0.15], yawRad: -Math.PI / 2 },
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
