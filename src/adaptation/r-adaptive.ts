/**
 * Fixed-connectivity r-adaptive relocation prototype.
 *
 * `referencePositions` are 2D material/reference coordinates in metres. The
 * optional `deformedPositions` are current 3D positions in metres and are used
 * only to estimate fold curvature. This module does not touch solver state,
 * constitutive laws, FEM data, or contact equations.
 */

export interface RAdaptiveInput {
  referencePositions: ArrayLike<number>; // n * 2, metres
  indices: Uint32Array; // triangle connectivity, preserved by reference
  deformedPositions?: ArrayLike<number>; // n * 3, metres
  wrinkle?: ArrayLike<number>; // dimensionless nonnegative indicator per vertex
  contact?: ArrayLike<number>; // dimensionless contact proximity per vertex
  pinned?: ArrayLike<number | boolean>; // optional fixed-vertex mask
}

export interface RAdaptiveOptions {
  iterations?: number; // fixed cap; default 4
  featureGain?: number; // dimensionless
  minimumResolutionScale?: number; // target spacing / local input spacing
  relaxation?: number; // 0..1
  maxMoveFraction?: number; // per-iteration move / local input spacing
  qualityWeight?: number;
  softMeanRatio?: number;
  maxMeanRatio?: number;
  minimumAreaFraction?: number;
  minimumEdgeFraction?: number;
  maxBacktracks?: number;
}

export interface RAdaptiveResult {
  accepted: boolean;
  reason: string | null;
  positions: Float64Array; // n * 2, metres
  indices: Uint32Array;
  targetResolution: Float64Array; // n, metres
  curvature: Float64Array; // n, 1/metre
  fixed: Uint8Array;
  iterations: number; // successfully accepted relocation iterations
  qualityEnergyBefore: number;
  qualityEnergyAfter: number;
  maxMeanRatioBefore: number;
  maxMeanRatioAfter: number;
}

interface Edge {
  a: number;
  b: number;
  faces: number[];
  initialLength: number;
  degreeA: number;
  degreeB: number;
}

interface MeshMeasure {
  valid: boolean;
  reason: string | null;
  qualityEnergy: number;
  maxMeanRatio: number;
}

const DEFAULTS: Required<RAdaptiveOptions> = {
  iterations: 4,
  featureGain: 2,
  minimumResolutionScale: 0.4,
  relaxation: 0.35,
  maxMoveFraction: 0.18,
  qualityWeight: 1,
  softMeanRatio: 1.8,
  maxMeanRatio: 4,
  minimumAreaFraction: 0.12,
  minimumEdgeFraction: 0.08,
  maxBacktracks: 10,
};

function mergedOptions(options: RAdaptiveOptions): Required<RAdaptiveOptions> {
  const out = { ...DEFAULTS, ...options };
  if (!Number.isInteger(out.iterations) || out.iterations < 0 || out.iterations > 32) {
    throw new RangeError("r-adaptive iterations must be an integer in [0, 32]");
  }
  if (!Object.values(out).every(Number.isFinite) ||
      !(out.featureGain >= 0) || !(out.minimumResolutionScale > 0 && out.minimumResolutionScale <= 1) ||
      !(out.relaxation > 0 && out.relaxation <= 1) || !(out.maxMoveFraction > 0) ||
      !(out.qualityWeight >= 0) || !(out.softMeanRatio >= 1) ||
      !(out.maxMeanRatio > out.softMeanRatio) || !(out.minimumAreaFraction > 0 && out.minimumAreaFraction < 1) ||
      !(out.minimumEdgeFraction > 0 && out.minimumEdgeFraction < 1) ||
      !Number.isInteger(out.maxBacktracks) || out.maxBacktracks < 0 || out.maxBacktracks > 20) {
    throw new RangeError("invalid r-adaptive option range");
  }
  return out;
}

function signedDoubleArea(p: ArrayLike<number>, a: number, b: number, c: number): number {
  const ax = p[a * 2], ay = p[a * 2 + 1];
  const bx = p[b * 2], by = p[b * 2 + 1];
  const cx = p[c * 2], cy = p[c * 2 + 1];
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

function triangleMeanRatio(p: ArrayLike<number>, a: number, b: number, c: number): number {
  const abx = p[a * 2] - p[b * 2], aby = p[a * 2 + 1] - p[b * 2 + 1];
  const bcx = p[b * 2] - p[c * 2], bcy = p[b * 2 + 1] - p[c * 2 + 1];
  const cax = p[c * 2] - p[a * 2], cay = p[c * 2 + 1] - p[a * 2 + 1];
  const sumEdgeSq = abx * abx + aby * aby + bcx * bcx + bcy * bcy + cax * cax + cay * cay;
  const area = Math.abs(signedDoubleArea(p, a, b, c)) * 0.5;
  if (!(area > 0)) return Infinity;
  // 1 for an equilateral triangle; grows as its aspect ratio degrades.
  return sumEdgeSq / (4 * Math.sqrt(3) * area);
}

function edgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function vector3(p: ArrayLike<number>, i: number): [number, number, number] {
  return [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];
}

function faceNormal(p: ArrayLike<number>, a: number, b: number, c: number): [number, number, number] {
  const A = vector3(p, a), B = vector3(p, b), C = vector3(p, c);
  const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2];
  const vx = C[0] - A[0], vy = C[1] - A[1], vz = C[2] - A[2];
  const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
  const len = Math.hypot(x, y, z);
  return len > 0 ? [x / len, y / len, z / len] : [0, 0, 0];
}

function estimateCurvature(
  positions: ArrayLike<number> | undefined,
  indices: Uint32Array,
  edges: Edge[],
  vertexCount: number,
): Float64Array {
  const curvature = new Float64Array(vertexCount);
  if (!positions) return curvature;
  const triCount = indices.length / 3;
  const normals = new Float64Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const n = faceNormal(positions, indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
    normals[t * 3] = n[0]; normals[t * 3 + 1] = n[1]; normals[t * 3 + 2] = n[2];
  }
  const counts = new Uint32Array(vertexCount);
  for (const edge of edges) {
    if (edge.faces.length !== 2) continue;
    const p0 = vector3(positions, edge.a), p1 = vector3(positions, edge.b);
    const length = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    if (!(length > 0)) continue;
    const t0 = edge.faces[0] * 3, t1 = edge.faces[1] * 3;
    const dot = Math.max(-1, Math.min(1,
      normals[t0] * normals[t1] + normals[t0 + 1] * normals[t1 + 1] + normals[t0 + 2] * normals[t1 + 2]));
    const curvatureContribution = Math.acos(dot) / length;
    curvature[edge.a] += curvatureContribution;
    curvature[edge.b] += curvatureContribution;
    counts[edge.a]++;
    counts[edge.b]++;
  }
  for (let i = 0; i < vertexCount; i++) {
    if (counts[i] > 0) curvature[i] /= counts[i];
  }
  return curvature;
}

function measureMesh(
  p: ArrayLike<number>,
  indices: Uint32Array,
  initialAreas: Float64Array,
  signs: Float64Array,
  initialLengths: Map<string, number>,
  scales: Float64Array,
  options: Required<RAdaptiveOptions>,
  enforceTargetBounds: boolean,
): MeshMeasure {
  let energy = 0;
  let maxMeanRatio = 0;
  for (let t = 0; t < indices.length / 3; t++) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const signedArea2 = signedDoubleArea(p, a, b, c);
    const area = Math.abs(signedArea2) * 0.5;
    const targetScale = (scales[a] + scales[b] + scales[c]) / 3;
    const targetArea = initialAreas[t] * targetScale * targetScale;
    if (!(area > 0) || signedArea2 * signs[t] <= 0) {
      return { valid: false, reason: "inverted-or-zero-area-triangle", qualityEnergy: Infinity, maxMeanRatio: Infinity };
    }
    if (enforceTargetBounds && area < options.minimumAreaFraction * targetArea) {
      return { valid: false, reason: "triangle-area-collapse", qualityEnergy: Infinity, maxMeanRatio: Infinity };
    }
    const q = triangleMeanRatio(p, a, b, c);
    maxMeanRatio = Math.max(maxMeanRatio, q);
    if (q > options.maxMeanRatio) {
      return { valid: false, reason: "triangle-aspect-ratio", qualityEnergy: Infinity, maxMeanRatio };
    }
    // Activate the quality penalty only once an element leaves the well-shaped
    // region; the hard cap below keeps this soft regularizer from masking failure.
    energy += Math.max(0, q - options.softMeanRatio) ** 2;
    if (enforceTargetBounds) {
      const areaRatio = area / targetArea;
      energy += Math.max(0, Math.log(1 / areaRatio)) ** 2;
    }
    for (const [i, j] of [[a, b], [b, c], [c, a]] as const) {
      const dx = p[i * 2] - p[j * 2], dy = p[i * 2 + 1] - p[j * 2 + 1];
      const len = Math.hypot(dx, dy);
      const initialLength = initialLengths.get(edgeKey(i, j)) ?? 0;
      const targetLength = initialLength * (scales[i] + scales[j]) * 0.5;
      if (!(len > 0)) {
        return { valid: false, reason: "collapsed-edge", qualityEnergy: Infinity, maxMeanRatio };
      }
      if (enforceTargetBounds && len < options.minimumEdgeFraction * targetLength) {
        return { valid: false, reason: "collapsed-edge", qualityEnergy: Infinity, maxMeanRatio };
      }
      const error = (len - targetLength) / Math.max(targetLength, 1e-30);
      energy += 0.5 * error * error;
    }
  }
  return { valid: true, reason: null, qualityEnergy: energy, maxMeanRatio };
}

/**
 * Relocate interior reference vertices toward a fixed-count target resolution
 * field. Boundary and explicitly pinned vertices are immutable. Each candidate
 * iteration uses a bounded edge-length descent proposal, a soft quality-energy
 * penalty, and backtracking against hard orientation/area/edge/aspect limits.
 */
export function adaptRMesh(input: RAdaptiveInput, userOptions: RAdaptiveOptions = {}): RAdaptiveResult {
  const options = mergedOptions(userOptions);
  const n = input.referencePositions.length / 2;
  if (!Number.isInteger(n) || n < 3 || input.indices.length < 3 || input.indices.length % 3 !== 0) {
    throw new RangeError("r-adaptive requires 2D vertices and triangle indices");
  }
  if (input.deformedPositions && input.deformedPositions.length !== n * 3) {
    throw new RangeError("deformedPositions must contain three SI coordinates per vertex");
  }
  for (const [name, values] of [["wrinkle", input.wrinkle], ["contact", input.contact], ["pinned", input.pinned]] as const) {
    if (values && values.length !== n) throw new RangeError(`${name} must contain one value per vertex`);
  }
  const positions = Float64Array.from(input.referencePositions);
  if (![...positions].every(Number.isFinite)) throw new RangeError("referencePositions must be finite");
  if (input.deformedPositions && !Array.from(input.deformedPositions).every(Number.isFinite)) {
    throw new RangeError("deformedPositions must be finite");
  }
  const triCount = input.indices.length / 3;
  const initialAreas = new Float64Array(triCount);
  const signs = new Float64Array(triCount);
  const edgesByKey = new Map<string, Edge>();
  const boundary = new Uint8Array(n);
  const vertexLengthSum = new Float64Array(n);
  const vertexDegree = new Uint32Array(n);
  const initialLengths = new Map<string, number>();

  for (let t = 0; t < triCount; t++) {
    const a = input.indices[t * 3], b = input.indices[t * 3 + 1], c = input.indices[t * 3 + 2];
    if (a >= n || b >= n || c >= n || a === b || b === c || c === a) {
      throw new RangeError(`invalid vertex index in triangle ${t}`);
    }
    const area2 = signedDoubleArea(positions, a, b, c);
    if (!Number.isFinite(area2) || Math.abs(area2) <= 1e-24) {
      return rejectedResult(input.indices, positions, new Float64Array(n), new Float64Array(n), boundary,
        "zero-area-triangle");
    }
    signs[t] = Math.sign(area2);
    initialAreas[t] = Math.abs(area2) * 0.5;
    for (const [i, j] of [[a, b], [b, c], [c, a]] as const) {
      const key = edgeKey(i, j);
      let edge = edgesByKey.get(key);
      if (!edge) {
        const dx = positions[i * 2] - positions[j * 2], dy = positions[i * 2 + 1] - positions[j * 2 + 1];
        const length = Math.hypot(dx, dy);
        if (!(length > 1e-12)) {
          return rejectedResult(input.indices, positions, new Float64Array(n), new Float64Array(n), boundary,
            "collapsed-edge");
        }
        edge = { a: Math.min(i, j), b: Math.max(i, j), faces: [], initialLength: length, degreeA: 0, degreeB: 0 };
        edgesByKey.set(key, edge);
        initialLengths.set(key, length);
        vertexLengthSum[i] += length; vertexLengthSum[j] += length;
        vertexDegree[i]++; vertexDegree[j]++;
      }
      edge.faces.push(t);
    }
  }

  const edges = [...edgesByKey.values()];
  for (const edge of edges) {
    if (edge.faces.length > 2) {
      return rejectedResult(input.indices, positions, new Float64Array(n), new Float64Array(n), boundary,
        "non-manifold-edge");
    }
    if (edge.faces.length === 1) { boundary[edge.a] = 1; boundary[edge.b] = 1; }
  }
  const fixed = Uint8Array.from(boundary);
  if (input.pinned) {
    for (let i = 0; i < n; i++) if (input.pinned[i]) fixed[i] = 1;
  }

  const initialScale = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    initialScale[i] = vertexDegree[i] ? vertexLengthSum[i] / vertexDegree[i] : 0;
    if (!(initialScale[i] > 0)) {
      return rejectedResult(input.indices, positions, new Float64Array(n), new Float64Array(n), fixed,
        "isolated-vertex");
    }
  }
  const curvature = estimateCurvature(input.deformedPositions, input.indices, edges, n);
  const scales = new Float64Array(n);
  const targetResolution = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const wrinkle = Math.max(0, Number(input.wrinkle?.[i] ?? 0));
    const contact = Math.max(0, Number(input.contact?.[i] ?? 0));
    if (!Number.isFinite(wrinkle) || !Number.isFinite(contact)) throw new RangeError("feature indicators must be finite");
    const feature = wrinkle + contact + curvature[i] * initialScale[i];
    scales[i] = Math.max(options.minimumResolutionScale, 1 / Math.sqrt(1 + options.featureGain * feature));
    targetResolution[i] = initialScale[i] * scales[i];
  }

  const initialMeasure = measureMesh(positions, input.indices, initialAreas, signs, initialLengths,
    new Float64Array(n).fill(1), options, false);
  if (!initialMeasure.valid) {
    return rejectedResult(input.indices, positions, targetResolution, curvature, fixed, initialMeasure.reason!);
  }
  let current = positions;
  let currentMeasure = measureMesh(current, input.indices, initialAreas, signs, initialLengths, scales, options, true);
  // Initial elements may not yet match the requested resolution, but they must
  // still be admissible. Bounds are enforced on every proposed relocation.
  if (!currentMeasure.valid) {
    currentMeasure = measureMesh(current, input.indices, initialAreas, signs, initialLengths, scales, options, false);
  }
  const qualityEnergyBefore = currentMeasure.qualityEnergy;
  const maxMeanRatioBefore = currentMeasure.maxMeanRatio;
  let acceptedIterations = 0;

  for (let iteration = 0; iteration < options.iterations; iteration++) {
    const force = new Float64Array(n * 2);
    const degree = new Uint32Array(n);
    for (const edge of edges) {
      const a = edge.a, b = edge.b;
      const dx = current[b * 2] - current[a * 2], dy = current[b * 2 + 1] - current[a * 2 + 1];
      const length = Math.hypot(dx, dy);
      const target = edge.initialLength * (scales[a] + scales[b]) * 0.5;
      if (!(length > 0)) continue;
      const correction = (length - target) / length;
      force[a * 2] += correction * dx; force[a * 2 + 1] += correction * dy;
      force[b * 2] -= correction * dx; force[b * 2 + 1] -= correction * dy;
      degree[a]++; degree[b]++;
    }
    const direction = new Float64Array(n * 2);
    let maxDisplacement = 0;
    for (let i = 0; i < n; i++) {
      if (fixed[i] || degree[i] === 0) continue;
      let dx = options.relaxation * force[i * 2] / degree[i];
      let dy = options.relaxation * force[i * 2 + 1] / degree[i];
      const cap = options.maxMoveFraction * initialScale[i];
      const length = Math.hypot(dx, dy);
      if (length > cap) { dx *= cap / length; dy *= cap / length; }
      direction[i * 2] = dx; direction[i * 2 + 1] = dy;
      maxDisplacement = Math.max(maxDisplacement, Math.hypot(dx, dy));
    }
    if (!(maxDisplacement > 0)) break;

    let accepted = false;
    for (let backtrack = 0; backtrack <= options.maxBacktracks; backtrack++) {
      const alpha = 2 ** -backtrack;
      const candidate = new Float64Array(current);
      for (let i = 0; i < n; i++) {
        if (fixed[i]) continue;
        candidate[i * 2] += alpha * direction[i * 2];
        candidate[i * 2 + 1] += alpha * direction[i * 2 + 1];
      }
      const measure = measureMesh(candidate, input.indices, initialAreas, signs, initialLengths, scales, options, true);
      if (measure.valid && measure.qualityEnergy < currentMeasure.qualityEnergy - 1e-14) {
        current = candidate;
        currentMeasure = measure;
        accepted = true;
        acceptedIterations++;
        break;
      }
    }
    if (!accepted) break;
  }

  return {
    accepted: true,
    reason: null,
    positions: current,
    indices: input.indices,
    targetResolution,
    curvature,
    fixed,
    iterations: acceptedIterations,
    qualityEnergyBefore,
    qualityEnergyAfter: currentMeasure.qualityEnergy,
    maxMeanRatioBefore,
    maxMeanRatioAfter: currentMeasure.maxMeanRatio,
  };
}

function rejectedResult(
  indices: Uint32Array,
  positions: Float64Array,
  targetResolution: Float64Array,
  curvature: Float64Array,
  fixed: Uint8Array,
  reason: string,
): RAdaptiveResult {
  return {
    accepted: false,
    reason,
    positions: Float64Array.from(positions),
    indices,
    targetResolution,
    curvature,
    fixed,
    iterations: 0,
    qualityEnergyBefore: Infinity,
    qualityEnergyAfter: Infinity,
    maxMeanRatioBefore: Infinity,
    maxMeanRatioAfter: Infinity,
  };
}
