// Deterministic (canonical) JSON serialization for grading entities.
// Mirrors the pattern CAD canonicalizer: sorted object keys, normalized -0,
// shortest round-trip numbers. No timestamps, no randomness, no map ordering.
import type { GradingDocument } from "./types.js";
import { GradingError } from "./types.js";
import { validateGradingDocument } from "./validate.js";

export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

/** Deterministically serialize any grading entity or value. */
export function serializeGradingValue(value: unknown): string {
  return canonicalJson(value);
}

export function serializeGradingDocument(document: GradingDocument): string {
  const diagnostics = validateGradingDocument(document);
  if (diagnostics.length > 0) {
    throw new GradingError("invalid-document", `cannot serialize invalid grading document:\n${diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  return canonicalJson(document);
}

export function parseGradingDocument(serialized: string): GradingDocument {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new GradingError("invalid-document", "serialized grading document is not valid JSON");
  }
  const candidate = parsed as GradingDocument;
  if (!candidate || typeof candidate !== "object" || candidate.schemaVersion !== 1) {
    throw new GradingError("invalid-document", "grading document shape or schema version is invalid");
  }
  const diagnostics = validateGradingDocument(candidate);
  if (diagnostics.length > 0) {
    throw new GradingError("invalid-document", `cannot parse invalid grading document:\n${diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
  }
  return JSON.parse(JSON.stringify(candidate)) as GradingDocument;
}
