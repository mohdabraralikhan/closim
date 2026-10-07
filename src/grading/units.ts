// G12C — length units for body measurements. Values are stored canonically
// in metres everywhere in the grading model; units record how a value was
// entered/displayed and convert at the boundary.
import { GradingError } from "./types.js";

export const INCH_TO_M = 0.0254;

export const LENGTH_UNITS = ["m", "cm", "mm", "in"] as const;
export type LengthUnit = (typeof LENGTH_UNITS)[number];

const TO_METRES: Record<LengthUnit, number> = {
  m: 1,
  cm: 0.01,
  mm: 0.001,
  in: INCH_TO_M,
};

export function isLengthUnit(value: unknown): value is LengthUnit {
  return typeof value === "string" && (LENGTH_UNITS as readonly string[]).includes(value);
}

function factor(unit: LengthUnit): number {
  if (!isLengthUnit(unit)) {
    throw new GradingError("invalid-unit", `unknown length unit '${String(unit)}' (expected one of: ${LENGTH_UNITS.join(", ")})`);
  }
  return TO_METRES[unit];
}

/** Canonical form: metres. */
export function toMetres(value: number, unit: LengthUnit): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new GradingError("invalid-argument", `measurement value must be finite, got ${String(value)}`);
  }
  return value * factor(unit);
}

export function fromMetres(valueM: number, unit: LengthUnit): number {
  if (typeof valueM !== "number" || !Number.isFinite(valueM)) {
    throw new GradingError("invalid-argument", `measurement value must be finite, got ${String(valueM)}`);
  }
  return valueM / factor(unit);
}

export function convertLength(value: number, from: LengthUnit, to: LengthUnit): number {
  return fromMetres(toMetres(value, from), to);
}
