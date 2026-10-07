// G13 — separate grade-rule export.
//
// Many apparel workflows carry grade rules separately from the DXF (per the
// G13 brief and community practice around ASTM D6673's grade-table scope).
// This module emits the canonical machine-readable grade artifact from the
// same ExportGradingContext the IR used, so DXF + rules can never disagree.
//
// Two deterministic artifacts:
//   1. canonical JSON grade table (full fidelity, stable for diffing)
//   2. CSV size chart (measurements per size; spreadsheet-friendly)
//
// No timestamps, no paths, sorted keys — reproducible byte-for-byte.

import { canonicalJson, type ExportGradingContext } from "./export-ir.js";

export interface GradeRulesJSON {
  format: "closim-grade-rules";
  version: 1;
  baseSizeId: string;
  sizeSetId?: string;
  sizeSetName?: string;
  sizes: Array<{ sizeId: string; label: string; isBase: boolean; measurements: Record<string, number> }>;
  rules: Array<{
    ruleId: string;
    gradingPointId: string;
    mode: "per-size" | "transition";
    /** sizeId -> [dx, dy] in the referenced panel's local pattern coordinates (metres). */
    deltas: Record<string, [number, number]>;
  }>;
}

export interface GradeRulesResult {
  json: string;
  csv: string;
  sizeCount: number;
  ruleCount: number;
  warnings: string[];
}

/** Build the standalone grade-rule artifacts from the grading context. */
export function exportGradeRules(
  grading: ExportGradingContext,
  opts: { garmentName: string; styleId: string } ,
): GradeRulesResult {
  const warnings: string[] = [];
  const rules: GradeRulesJSON["rules"] = [];
  const seenPoint = new Set<string>();
  for (const [sizeId, apps] of Object.entries(grading.ruleApplications)) {
    for (const app of apps) {
      const key = app.gradingPointId;
      let entry = rules.find((r) => r.ruleId === app.ruleId && r.gradingPointId === key);
      if (!entry) {
        entry = {
          ruleId: app.ruleId,
          gradingPointId: key,
          mode: app.mode,
          deltas: {},
        };
        rules.push(entry);
      }
      entry.deltas[sizeId] = app.delta;
      seenPoint.add(key);
    }
  }
  // Deterministic order: by gradingPointId then ruleId.
  rules.sort((a, b) => a.gradingPointId.localeCompare(b.gradingPointId) || a.ruleId.localeCompare(b.ruleId));
  if (rules.length === 0) warnings.push("no grade rules present; the grade file carries size definitions only");

  const payload: GradeRulesJSON = {
    format: "closim-grade-rules",
    version: 1,
    baseSizeId: grading.baseSizeId,
    ...(grading.sizeSetId !== undefined ? { sizeSetId: grading.sizeSetId } : {}),
    ...(grading.sizeSetName !== undefined ? { sizeSetName: grading.sizeSetName } : {}),
    sizes: grading.sizes.map((s) => ({
      sizeId: s.sizeId, label: s.label, isBase: s.isBase, measurements: { ...s.measurements },
    })),
    rules: rules.map((r) => ({
      ruleId: r.ruleId,
      gradingPointId: r.gradingPointId,
      mode: r.mode,
      deltas: Object.fromEntries(Object.entries(r.deltas).map(([k, v]) => [k, [v[0], v[1]] as [number, number]])),
    })),
  };
  void opts;
  const csvLines: string[] = [];
  const measurementKeys = [...new Set(grading.sizes.flatMap((s) => Object.keys(s.measurements)))].sort();
  csvLines.push(["sizeId", "label", "isBase", ...measurementKeys].join(","));
  for (const s of grading.sizes) {
    csvLines.push([
      s.sizeId, s.label, s.isBase ? "true" : "false",
      ...measurementKeys.map((k) => (s.measurements[k] !== undefined ? String(s.measurements[k]) : "")),
    ].join(","));
  }
  return {
    json: canonicalJson(payload),
    csv: csvLines.join("\n") + "\n",
    sizeCount: grading.sizes.length,
    ruleCount: rules.length,
    warnings,
  };
}
