// G14D — nesting optimization, utilization metrics, and cost analysis.
//
// Definitions (explicit, SI):
//   markerLengthM  = max placed top edge (fabric consumed along Y)
//   fabricAreaM2   = usableWidthM × markerLengthM
//   patternAreaM2  = Σ placed piece areas (mirrored copies count fully)
//   wasteM2        = fabricAreaM2 − patternAreaM2
//   utilization    = patternAreaM2 / fabricAreaM2 (0 when length is 0)
//   wastePct       = 1 − utilization
//
// Multi-seed search keeps full validity: comparison prefers fully-placed
// results first, then the configured objective. No prices are hardcoded.

import { PatternCadError } from "../pattern/cad.js";
import { nestPieces, type NestInput, type NestResult } from "./nest.js";

export interface MarkerMetrics {
  markerLengthM: number;
  markerWidthM: number;
  fabricAreaM2: number;
  patternAreaM2: number;
  wasteM2: number;
  utilization: number;
  wastePct: number;
  placedCount: number;
  unplacedCount: number;
}

export function measureResult(result: NestResult, patternAreaM2: number): MarkerMetrics {
  if (!(patternAreaM2 >= 0) || !Number.isFinite(patternAreaM2)) {
    throw new PatternCadError("invalid-transform", "pattern area must be finite and >= 0");
  }
  const fabricAreaM2 = result.markerWidthM * result.markerLengthM;
  const wasteM2 = Math.max(0, fabricAreaM2 - patternAreaM2);
  const utilization = fabricAreaM2 > 0 ? patternAreaM2 / fabricAreaM2 : 0;
  return {
    markerLengthM: result.markerLengthM,
    markerWidthM: result.markerWidthM,
    fabricAreaM2,
    patternAreaM2,
    wasteM2,
    utilization,
    wastePct: 1 - utilization,
    placedCount: result.placements.length,
    unplacedCount: result.unplaced.length,
  };
}

export type Objective = "min-length" | "max-utilization" | "min-waste";

export interface WeightedObjective {
  kind: "weighted";
  /** Weight on normalized length (0..1); remainder falls on waste. */
  lengthWeight: number;
}

/**
 * Documented comparison: fully-placed results always beat partial ones;
 * then by objective (min length, max utilization, min waste, or the
 * explicit weighted blend of normalized length + waste fraction).
 * Ties break by lower seed, then strategy name — fully deterministic.
 */
export function compareResults(
  a: { result: NestResult; metrics: MarkerMetrics },
  b: { result: NestResult; metrics: MarkerMetrics },
  objective: Objective | WeightedObjective = "min-length",
): number {
  const aFull = a.result.unplaced.length === 0;
  const bFull = b.result.unplaced.length === 0;
  if (aFull !== bFull) return aFull ? -1 : 1;
  if (typeof objective === "object") {
    if (!(objective.lengthWeight >= 0) || !(objective.lengthWeight <= 1)) {
      throw new PatternCadError("invalid-transform", "length weight must be in [0,1]");
    }
    const norm = (m: MarkerMetrics): number => {
      const length = m.markerLengthM;
      return objective.lengthWeight * length + (1 - objective.lengthWeight) * m.wastePct * length;
    };
    const d = norm(a.metrics) - norm(b.metrics);
    if (d !== 0) return d;
  } else if (objective === "min-length" || objective === "min-waste") {
    const d = a.metrics.markerLengthM - b.metrics.markerLengthM;
    if (d !== 0) return d;
  } else if (objective === "max-utilization") {
    const d = b.metrics.utilization - a.metrics.utilization;
    if (d !== 0) return d;
  } else {
    throw new PatternCadError("invalid-transform", `unknown objective '${String(objective)}'`);
  }
  if (a.result.seed !== b.result.seed) return a.result.seed - b.result.seed;
  return a.result.strategy < b.result.strategy ? -1 : 1;
}

export interface OptimizeRun {
  seed: number;
  strategy: NestResult["strategy"];
  result: NestResult;
  metrics: MarkerMetrics;
  elapsedMs: number;
}

export interface OptimizeReport {
  best: OptimizeRun;
  runs: OptimizeRun[];
  cancelled: boolean;
  objective: Objective | WeightedObjective;
}

export interface OptimizeOptions {
  seeds?: number[];
  strategies?: NestResult["strategy"][];
  objective?: Objective | WeightedObjective;
  /** Checked between runs; effective for multi-run optimization. */
  shouldCancel?: () => boolean;
}

/** Total pattern area over piece instances (mirrored copies included). */
export function totalPatternArea(input: NestInput): number {
  return input.pieces.reduce((s, p) => s + p.areaM2, 0);
}

/**
 * Deterministic multi-seed search. Every run is validated by construction
 * (the engine only emits valid placements); the best run wins by
 * compareResults. Cancellation stops after the current run and keeps the
 * best result so far.
 */
export function optimizeMarker(input: NestInput, opts: OptimizeOptions = {}): OptimizeReport {
  const seeds = opts.seeds ?? [0];
  const strategies = opts.strategies ?? (["area"] as NestResult["strategy"][]);
  const objective = opts.objective ?? "min-length";
  if (seeds.length === 0) throw new PatternCadError("invalid-transform", "at least one seed is required");
  if (typeof objective === "object" && (!(objective.lengthWeight >= 0) || !(objective.lengthWeight <= 1))) {
    throw new PatternCadError("invalid-transform", "length weight must be in [0,1]");
  }
  if (typeof objective === "string" && objective !== "min-length" && objective !== "max-utilization" && objective !== "min-waste") {
    throw new PatternCadError("invalid-transform", `unknown objective '${objective}'`);
  }
  const area = totalPatternArea(input);
  const runs: OptimizeRun[] = [];
  let cancelled = false;
  for (const seed of seeds) {
    for (const strategy of strategies) {
      if (opts.shouldCancel?.()) {
        cancelled = true;
        break;
      }
      const started = Date.now();
      const result = nestPieces({ ...input, options: { ...(input.options ?? {}), seed, strategy } });
      const elapsedMs = Date.now() - started;
      runs.push({ seed, strategy, result, metrics: measureResult(result, area), elapsedMs });
    }
    if (cancelled) break;
  }
  if (runs.length === 0) throw new PatternCadError("invalid-transform", "optimization produced no runs");
  let best = runs[0];
  for (const run of runs.slice(1)) {
    if (compareResults({ result: run.result, metrics: run.metrics }, { result: best.result, metrics: best.metrics }, objective) < 0) {
      best = run;
    }
  }
  return { best, runs, cancelled, objective };
}

// ---------------------------------------------------------------------------
// Cost model (all prices explicit — nothing hardcoded)
// ---------------------------------------------------------------------------

export interface CostInputs {
  /** Fabric price per linear metre of full roll width. */
  fabricPricePerM: number;
  /** Fixed labor/handling estimate per marker. */
  laborPerMarker: number;
  /** Waste disposal/missed-value price per square metre of waste. */
  wastePricePerM2: number;
}

export interface CostEstimate {
  fabricCost: number;
  wasteCost: number;
  laborCost: number;
  totalCost: number;
  fabricConsumedM: number;
}

export function estimateCost(metrics: MarkerMetrics, prices: CostInputs): CostEstimate {
  for (const [key, value] of Object.entries(prices)) {
    if (!Number.isFinite(value) || value < 0) {
      throw new PatternCadError("invalid-transform", `cost input '${key}' must be finite and >= 0`);
    }
  }
  const fabricCost = metrics.markerLengthM * prices.fabricPricePerM;
  const wasteCost = metrics.wasteM2 * prices.wastePricePerM2;
  return {
    fabricCost,
    wasteCost,
    laborCost: prices.laborPerMarker,
    totalCost: fabricCost + wasteCost + prices.laborPerMarker,
    fabricConsumedM: metrics.markerLengthM,
  };
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

export interface BenchmarkRecord {
  name: string;
  elapsedMs: number;
  utilization: number;
  markerLengthM: number;
  placedCount: number;
  unplacedCount: number;
  iterations: number;
  seed: number;
  strategy: string;
}

/** Time one nesting run and record the manufacturing-relevant statistics. */
export function benchmarkNesting(name: string, input: NestInput, seed = 0): BenchmarkRecord {
  const started = Date.now();
  const result = nestPieces({ ...input, options: { ...(input.options ?? {}), seed } });
  const elapsedMs = Date.now() - started;
  return {
    name,
    elapsedMs,
    utilization: result.utilization,
    markerLengthM: result.markerLengthM,
    placedCount: result.placements.length,
    unplacedCount: result.unplaced.length,
    iterations: result.iterations,
    seed: result.seed,
    strategy: result.strategy,
  };
}
