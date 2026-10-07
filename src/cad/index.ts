// G9A — public surface of the professional 2D pattern CAD layer.
//
// The document model, validation, serialization and triangulation bridge
// remain owned by src/pattern/cad.ts (G8A). This module adds the G9A
// editing layer on top of it: geometry predicates, document queries,
// undo/redo session, selection, snapping, and advanced editing operations.
//
//   PatternDocument (G8A) ──► cad/queries  ──► geometry reads
//                       ──► cad/ops       ──► advanced edits
//                       ──► cad/history   ──► undo/redo + gestures
//                       ──► cad/selection ──► selection sets
//                       ──► cad/snap      ──► explicit snapping
//
// G13 export layer: the G11E v1 IR (export.js) is retained for
// compatibility; the canonical G13 path is export-ir.js (IR v2) plus its
// adapters. The two IRs are intentionally separate modules.

export * from "./geom.js";
export * from "./queries.js";
export * from "./history.js";
export * from "./selection.js";
export * from "./snap.js";
export * from "./ops.js";
export * from "./draft.js";
export * from "./constraints.js";
export * from "./editor.js";
export * from "./production.js";
export * from "./markings.js";
export * from "./readiness.js";
export * from "./techsheet.js";
export * from "./export.js";
export {
  // Canonical G13 IR: re-exported selectively to avoid clashing with the
  // legacy v1 names in export.js (both are public but distinct modules).
  EXPORT_IR_FORMAT,
  compareIR as compareExportIR2,
  buildExportIR as buildExportIR2,
  exportIRToJSON as exportIR2ToJSON,
  importIRFromJSON as importExportIR2FromJSON,
  convertM,
  requireExportUnits,
  formatInUnits,
  exportGate,
  type ExportIR,
  type ExportPanelIR,
  type ExportSeamIR,
  type ExportNotchIR,
  type ExportSizeIR,
  type ExportGradingIR,
  type ExportStyleIR,
  type ExportGradingContext,
  type ExportGateDecision,
  type ExportGateMode,
  type ExportUnits,
  type ExportEdge,
  type ExportRing,
} from "./export-ir.js";
export * from "./dxf-profile.js";
export * from "./dxf-export.js";
export * from "./dxf-validate.js";
export * from "./svg-export.js";
export * from "./pdf-export.js";
export * from "./grade-rules-export.js";
export * from "./export-package.js";
export * from "./zip.js";
export * from "./dxf-import.js";
export * from "./marker.js";
export * from "./marker-svg.js";
