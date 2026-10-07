// G14 fixtures: piece input sets for marker building, plus a skip-gate
// document with a 45-degree grainline for the misalignment warning test.
import {
  createProductionSet,
  addGrainline,
  type ProductionSet,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import {
  createPatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import type { Seam } from "../../src/garment/sewing.js";
import type { MarkerPieceInput } from "../../src/cad/marker.js";
import { engineeredGarment } from "./g13-fixtures.js";

/**
 * Pieces for the engineered two-panel top: front and back, 1 copy each.
 * The fixture grainlines are vertical (90 deg); the marker warp is +X, so
 * every constrained placement exercises a 90-degree rotation.
 * Panel IDs are kernel-allocated (`{docId}/panel/{n}`), so callers pass the
 * fixture's actual refs.
 */
export function markerPieces(frontPanelId: string, backPanelId: string): MarkerPieceInput[] {
  return [
    { panelId: frontPanelId },
    { panelId: backPanelId },
  ];
}

/**
 * A minimal single-panel document with an arbitrary grainline direction,
 * used with skipGate to test grain-constraint warnings deterministically.
 */
export interface SkewFixture {
  document: PatternDocument;
  seams: Seam[];
  set: ProductionSet;
  panelId: string;
}

export function skewedGrainFixture(): SkewFixture {
  let document = createPatternDocument("g14-skew", "G14 skew panel");
  const panel = addRectPanel(document, "skew", [0, 0], 0.4, 0.5);
  document = panel.document;
  let set = createProductionSet();
  const g = centeredGrainline(document, panel.refs.panelId, Math.PI / 4); // 45 deg
  set = addGrainline(set, panel.refs.panelId, g.from, g.to).set;
  return { document, seams: [], set, panelId: panel.refs.panelId };
}

export { engineeredGarment };
