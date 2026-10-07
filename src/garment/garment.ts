import type {
  BoundaryLoop,
  Garment,
  MaterialAssignment,
  PatternBoundaryCurve,
  PatternPanel,
  Seam,
  SeamCorrespondence,
  SeamSide,
} from "./types.js";

const PARAM_EPSILON = 1e-8;
const GEOMETRY_EPSILON_M = 1e-6;

export class GarmentValidationError extends Error {
  readonly issues: string[];

  constructor(issues: string[]) {
    super(`Invalid garment:\n${issues.map((issue) => `- ${issue}`).join("\n")}`);
    this.name = "GarmentValidationError";
    this.issues = issues;
  }
}

function failIf(issues: string[], condition: boolean, message: string): void {
  if (condition) issues.push(message);
}

function finiteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function pointDistance(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function endpoint(curve: PatternBoundaryCurve, end: "start" | "end"): readonly number[] {
  return curve.controlPoints[end === "start" ? 0 : curve.controlPoints.length - 1];
}

function evaluateBezier(controlPoints: readonly (readonly number[])[], t: number): readonly number[] {
  const work = controlPoints.map((point) => [point[0], point[1]]);
  for (let level = work.length - 1; level > 0; level--) {
    for (let i = 0; i < level; i++) {
      work[i][0] = work[i][0] * (1 - t) + work[i + 1][0] * t;
      work[i][1] = work[i][1] * (1 - t) + work[i + 1][1] * t;
    }
  }
  return work[0];
}

function curveLabel(panelId: string, loop: BoundaryLoop, curve: PatternBoundaryCurve): string {
  return `panel '${panelId}' loop '${loop.id}' curve '${curve.id}'`;
}

function validateLoop(
  panel: PatternPanel,
  loop: BoundaryLoop,
  ids: Set<string>,
  boundaryEdges: Set<string>,
  issues: string[],
): void {
  failIf(issues, !loop.id.trim(), `panel '${panel.id}' has an empty boundary-loop id`);
  failIf(issues, ids.has(loop.id), `duplicate entity id '${loop.id}'`);
  ids.add(loop.id);
  failIf(issues, loop.curves.length === 0, `panel '${panel.id}' loop '${loop.id}' has no curves`);

  for (let i = 0; i < loop.curves.length; i++) {
    const curve = loop.curves[i];
    const label = curveLabel(panel.id, loop, curve);
    failIf(issues, !curve.id.trim(), `${label} has an empty id`);
    failIf(issues, ids.has(curve.id), `duplicate entity id '${curve.id}'`);
    ids.add(curve.id);
    failIf(
      issues,
      curve.controlPoints.length < 2 || curve.controlPoints.length > 4,
      `${label} must have 2, 3, or 4 Bezier control points`,
    );
    for (const [pointIndex, point] of curve.controlPoints.entries()) {
      failIf(
        issues,
        point.length !== 2 || !finiteNumber(point[0]) || !finiteNumber(point[1]),
        `${label} control point ${pointIndex} must be finite 2D coordinates in meters`,
      );
    }
    failIf(issues, curve.meshSamples.length < 2, `${label} needs at least two mesh samples`);
    for (let j = 0; j < curve.meshSamples.length; j++) {
      const sample = curve.meshSamples[j];
      failIf(
        issues,
        !finiteNumber(sample.t) || sample.t < -PARAM_EPSILON || sample.t > 1 + PARAM_EPSILON,
        `${label} sample ${j} parameter must be in [0, 1]`,
      );
      failIf(
        issues,
        !Number.isInteger(sample.vertex) || sample.vertex < 0 || sample.vertex >= panel.mesh.positions.length / 3,
        `${label} sample ${j} references an invalid panel-mesh vertex`,
      );
      if (j > 0) {
        failIf(
          issues,
          sample.t <= curve.meshSamples[j - 1].t,
          `${label} mesh-sample parameters must be strictly increasing`,
        );
      }
      if (
        curve.controlPoints.length >= 2 &&
        curve.controlPoints.length <= 4 &&
        sample.vertex >= 0 &&
        sample.vertex < panel.mesh.patternCoordinates.length / 2 &&
        finiteNumber(sample.t)
      ) {
        const expected = evaluateBezier(curve.controlPoints, sample.t);
        const actual = [
          panel.mesh.patternCoordinates[sample.vertex * 2],
          panel.mesh.patternCoordinates[sample.vertex * 2 + 1],
        ];
        failIf(
          issues,
          pointDistance(expected, actual) > 1e-5,
          `${label} sample ${j} pattern coordinate does not lie on its Bezier curve`,
        );
      }
    }
    if (curve.meshSamples.length >= 2) {
      failIf(
        issues,
        Math.abs(curve.meshSamples[0].t) > PARAM_EPSILON ||
          Math.abs(curve.meshSamples[curve.meshSamples.length - 1].t - 1) > PARAM_EPSILON,
        `${label} mesh samples must include t=0 and t=1`,
      );
      for (let j = 1; j < curve.meshSamples.length; j++) {
        const a = curve.meshSamples[j - 1].vertex;
        const b = curve.meshSamples[j].vertex;
        const edge = a < b ? `${a}:${b}` : `${b}:${a}`;
        failIf(
          issues,
          a === b || !boundaryEdges.has(edge),
          `${label} mesh samples ${j - 1}-${j} are not a panel-boundary edge`,
        );
      }
    }

    const next = loop.curves[(i + 1) % loop.curves.length];
    if (curve.controlPoints.length >= 2 && next.controlPoints.length >= 2) {
      failIf(
        issues,
        pointDistance(endpoint(curve, "end"), endpoint(next, "start")) > GEOMETRY_EPSILON_M,
        `${label} is not geometrically closed/continuous with curve '${next.id}'`,
      );
    }
    if (curve.meshSamples.length >= 2 && next.meshSamples.length >= 2) {
      failIf(
        issues,
        curve.meshSamples[curve.meshSamples.length - 1].vertex !== next.meshSamples[0].vertex,
        `${label} endpoint vertex does not join curve '${next.id}'`,
      );
    }
  }
}

function resolveSide(
  side: SeamSide,
  panels: Map<string, PatternPanel>,
  issues: string[],
  seamId: string,
): { panel: PatternPanel; loop: BoundaryLoop; curve: PatternBoundaryCurve } | null {
  const panel = panels.get(side.panelId);
  if (!panel) {
    issues.push(`seam '${seamId}' references unknown panel '${side.panelId}'`);
    return null;
  }
  const loop = [panel.outerBoundary, ...panel.holes].find((item) => item.id === side.boundaryLoopId);
  if (!loop) {
    issues.push(`seam '${seamId}' references unknown boundary loop '${side.boundaryLoopId}'`);
    return null;
  }
  const curve = loop.curves.find((item) => item.id === side.curveId);
  if (!curve) {
    issues.push(`seam '${seamId}' references unknown curve '${side.curveId}'`);
    return null;
  }
  return { panel, loop, curve };
}

export function mapSeamParameter(correspondence: SeamCorrespondence, parameterA: number): number {
  const knots = correspondence.knots;
  if (parameterA <= knots[0].sideA) return knots[0].sideB;
  for (let i = 1; i < knots.length; i++) {
    const right = knots[i];
    if (parameterA <= right.sideA) {
      const left = knots[i - 1];
      const fraction = (parameterA - left.sideA) / (right.sideA - left.sideA);
      return left.sideB + fraction * (right.sideB - left.sideB);
    }
  }
  return knots[knots.length - 1].sideB;
}

function validateCorrespondence(seam: Seam, issues: string[]): void {
  const { correspondence: c } = seam;
  const knots = c.knots;
  failIf(issues, knots.length < 2, `seam '${seam.id}' correspondence needs at least two knots`);
  if (knots.length < 2) return;
  for (let i = 0; i < knots.length; i++) {
    const knot = knots[i];
    failIf(
      issues,
      !finiteNumber(knot.sideA) || !finiteNumber(knot.sideB) ||
        knot.sideA < -PARAM_EPSILON || knot.sideA > 1 + PARAM_EPSILON ||
        knot.sideB < -PARAM_EPSILON || knot.sideB > 1 + PARAM_EPSILON,
      `seam '${seam.id}' knot ${i} parameters must be in [0, 1]`,
    );
    if (i > 0) {
      failIf(
        issues,
        knot.sideA <= knots[i - 1].sideA,
        `seam '${seam.id}' side-A knot parameters must be strictly increasing`,
      );
    }
  }

  failIf(
    issues,
    Math.abs(knots[0].sideA) > PARAM_EPSILON ||
      Math.abs(knots[knots.length - 1].sideA - 1) > PARAM_EPSILON,
    `seam '${seam.id}' correspondence must span side A from 0 to 1`,
  );
  const firstB = knots[0].sideB;
  const lastB = knots[knots.length - 1].sideB;
  failIf(
    issues,
    Math.abs(firstB - (c.orientation === "same" ? 0 : 1)) > PARAM_EPSILON ||
      Math.abs(lastB - (c.orientation === "same" ? 1 : 0)) > PARAM_EPSILON,
    `seam '${seam.id}' orientation does not match its side-B endpoints`,
  );

  const bDeltas = knots.slice(1).map((knot, i) => knot.sideB - knots[i].sideB);
  const direction = c.orientation === "same" ? 1 : -1;
  const hasPlateau = bDeltas.some((delta) => Math.abs(delta) <= PARAM_EPSILON);
  failIf(
    issues,
    bDeltas.some((delta) => delta * direction < -PARAM_EPSILON),
    `seam '${seam.id}' side-B mapping must be monotonic for ${c.orientation} orientation`,
  );

  if (c.mode === "one-to-one") {
    failIf(issues, knots.length !== 2, `seam '${seam.id}' one-to-one mapping requires exactly two knots`);
    failIf(issues, hasPlateau, `seam '${seam.id}' one-to-one mapping cannot contain repeated parameters`);
  } else if (c.mode === "segmented") {
    failIf(issues, knots.length < 3, `seam '${seam.id}' segmented mapping requires at least three knots`);
    failIf(issues, hasPlateau, `seam '${seam.id}' segmented mapping cannot contain repeated parameters`);
  } else if (c.mode === "many-to-one") {
    failIf(issues, knots.length < 3, `seam '${seam.id}' many-to-one mapping requires at least three knots`);
    failIf(issues, !hasPlateau, `seam '${seam.id}' many-to-one mapping must contain a repeated side-B parameter`);
  } else {
    issues.push(`seam '${seam.id}' has an unknown correspondence mode`);
  }
}

function boundaryKey(side: SeamSide): string {
  return `${side.panelId}\u0000${side.boundaryLoopId}\u0000${side.curveId}`;
}

export function validateGarment(garment: Garment): void {
  const issues: string[] = [];
  const ids = new Set<string>();
  failIf(issues, !garment.id.trim(), "garment id must not be empty");
  ids.add(garment.id);

  const assignments = new Map<string, MaterialAssignment>();
  for (const assignment of garment.materialAssignments) {
    failIf(issues, !assignment.id.trim(), "material assignment id must not be empty");
    failIf(issues, ids.has(assignment.id), `duplicate entity id '${assignment.id}'`);
    ids.add(assignment.id);
    failIf(issues, !assignment.materialRef.trim(), `material assignment '${assignment.id}' has an empty material reference`);
    if (assignments.has(assignment.id)) issues.push(`duplicate material assignment '${assignment.id}'`);
    assignments.set(assignment.id, assignment);
  }

  const panels = new Map<string, PatternPanel>();
  for (const panel of garment.panels) {
    failIf(issues, !panel.id.trim(), "panel id must not be empty");
    failIf(issues, ids.has(panel.id), `duplicate entity id '${panel.id}'`);
    ids.add(panel.id);
    if (panels.has(panel.id)) issues.push(`duplicate panel '${panel.id}'`);
    panels.set(panel.id, panel);
    failIf(
      issues,
      !assignments.has(panel.materialAssignmentId),
      `panel '${panel.id}' references unknown material assignment '${panel.materialAssignmentId}'`,
    );
    const vertexCount = panel.mesh.positions.length / 3;
    failIf(
      issues,
      !Number.isInteger(vertexCount) || vertexCount < 3 ||
        panel.mesh.patternCoordinates.length !== vertexCount * 2,
      `panel '${panel.id}' mesh needs xyz positions and one pattern-coordinate pair per vertex`,
    );
    failIf(
      issues,
      panel.mesh.indices.length < 3 || panel.mesh.indices.length % 3 !== 0,
      `panel '${panel.id}' mesh indices must contain complete triangles`,
    );
    for (const [i, value] of panel.mesh.positions.entries()) {
      failIf(issues, !finiteNumber(value), `panel '${panel.id}' position component ${i} is not finite`);
    }
    for (const [i, value] of panel.mesh.patternCoordinates.entries()) {
      failIf(issues, !finiteNumber(value), `panel '${panel.id}' pattern coordinate ${i} is not finite`);
    }
    for (const [i, vertex] of panel.mesh.indices.entries()) {
      failIf(issues, vertex >= vertexCount, `panel '${panel.id}' index ${i} references invalid vertex ${vertex}`);
    }
    const edgeUse = new Map<string, number>();
    for (let i = 0; i + 2 < panel.mesh.indices.length; i += 3) {
      const triangle = [panel.mesh.indices[i], panel.mesh.indices[i + 1], panel.mesh.indices[i + 2]];
      for (let e = 0; e < 3; e++) {
        const a = triangle[e], b = triangle[(e + 1) % 3];
        const edge = a < b ? `${a}:${b}` : `${b}:${a}`;
        edgeUse.set(edge, (edgeUse.get(edge) ?? 0) + 1);
      }
    }
    const boundaryEdges = new Set([...edgeUse].filter(([, uses]) => uses === 1).map(([edge]) => edge));
    validateLoop(panel, panel.outerBoundary, ids, boundaryEdges, issues);
    for (const hole of panel.holes) validateLoop(panel, hole, ids, boundaryEdges, issues);
  }

  const seamPairs = new Set<string>();
  for (const seam of garment.seams) {
    failIf(issues, !seam.id.trim(), "seam id must not be empty");
    failIf(issues, ids.has(seam.id), `duplicate entity id '${seam.id}'`);
    ids.add(seam.id);
    const aKey = boundaryKey(seam.sideA);
    const bKey = boundaryKey(seam.sideB);
    failIf(issues, aKey === bKey, `seam '${seam.id}' cannot join a boundary to itself`);
    const pairKey = aKey < bKey ? `${aKey}\u0001${bKey}` : `${bKey}\u0001${aKey}`;
    failIf(issues, seamPairs.has(pairKey), `duplicate seam between '${aKey.replaceAll("\u0000", "/")}' and '${bKey.replaceAll("\u0000", "/")}'`);
    seamPairs.add(pairKey);

    const sideA = resolveSide(seam.sideA, panels, issues, seam.id);
    const sideB = resolveSide(seam.sideB, panels, issues, seam.id);
    validateCorrespondence(seam, issues);
    failIf(
      issues,
      !finiteNumber(seam.allowance.sideA_m) || seam.allowance.sideA_m < 0 ||
        !finiteNumber(seam.allowance.sideB_m) || seam.allowance.sideB_m < 0,
      `seam '${seam.id}' allowances must be finite nonnegative meters`,
    );

    failIf(
      issues,
      seam.stitchTopology.pairs.length === 0,
      `seam '${seam.id}' must define at least one explicit stitch pair`,
    );
    const stitchPairs = new Set<string>();
    for (let i = 0; i < seam.stitchTopology.pairs.length; i++) {
      const pair = seam.stitchTopology.pairs[i];
      const id = pair.id ?? deterministicStitchId(seam.id, i);
      failIf(issues, !id.trim(), `seam '${seam.id}' stitch ${i} has an empty id`);
      failIf(issues, ids.has(id), `duplicate entity id '${id}'`);
      ids.add(id);
      failIf(
        issues,
        !finiteNumber(pair.parameterA) || !finiteNumber(pair.parameterB) ||
          pair.parameterA < -PARAM_EPSILON || pair.parameterA > 1 + PARAM_EPSILON ||
          pair.parameterB < -PARAM_EPSILON || pair.parameterB > 1 + PARAM_EPSILON,
        `seam '${seam.id}' stitch '${id}' parameters must be in [0, 1]`,
      );
      const pairKey = `${pair.parameterA.toPrecision(12)}:${pair.parameterB.toPrecision(12)}`;
      failIf(issues, stitchPairs.has(pairKey), `seam '${seam.id}' has duplicate stitch pair '${pairKey}'`);
      stitchPairs.add(pairKey);
      if (sideA && sideB) {
        failIf(
          issues,
          !hasMeshSample(sideA.curve, pair.parameterA) || !hasMeshSample(sideB.curve, pair.parameterB),
          `seam '${seam.id}' stitch '${id}' must reference existing boundary mesh samples`,
        );
      }
      if (finiteNumber(pair.parameterA) && finiteNumber(pair.parameterB) && seam.correspondence.knots.length >= 2) {
        failIf(
          issues,
          Math.abs(mapSeamParameter(seam.correspondence, pair.parameterA) - pair.parameterB) > 1e-6,
          `seam '${seam.id}' stitch '${id}' disagrees with its parameter correspondence`,
        );
      }
    }
  }

  if (issues.length > 0) throw new GarmentValidationError(issues);
}

function hasMeshSample(curve: PatternBoundaryCurve, parameter: number): boolean {
  return curve.meshSamples.some((sample) => Math.abs(sample.t - parameter) <= PARAM_EPSILON);
}

export function deterministicStitchId(seamId: string, pairIndex: number): string {
  return `stitch:${encodeURIComponent(seamId)}:${pairIndex.toString().padStart(6, "0")}`;
}

function canonicalPanel(panel: PatternPanel): PatternPanel {
  return {
    ...panel,
    mesh: {
      positions: Float32Array.from(panel.mesh.positions),
      patternCoordinates: Float32Array.from(panel.mesh.patternCoordinates),
      indices: Uint32Array.from(panel.mesh.indices),
    },
    holes: [...panel.holes].sort((a, b) => a.id.localeCompare(b.id)),
  };
}

export function canonicalGarment(garment: Garment): Garment {
  validateGarment(garment);
  return {
    id: garment.id,
    materialAssignments: [...garment.materialAssignments]
      .sort((a, b) => a.id.localeCompare(b.id))
      .map((assignment) => ({ ...assignment })),
    panels: [...garment.panels].sort((a, b) => a.id.localeCompare(b.id)).map(canonicalPanel),
    seams: [...garment.seams]
      .sort((a, b) => a.id.localeCompare(b.id))
      .map((seam) => ({
        ...seam,
        sideA: { ...seam.sideA },
        sideB: { ...seam.sideB },
        correspondence: {
          ...seam.correspondence,
          knots: seam.correspondence.knots.map((knot) => ({ ...knot })),
        },
        allowance: { ...seam.allowance },
        stitchTopology: {
          pairs: seam.stitchTopology.pairs.map((pair, i) => ({
            id: pair.id ?? deterministicStitchId(seam.id, i),
            parameterA: pair.parameterA,
            parameterB: pair.parameterB,
          })),
        },
      })),
  };
}

export function serializeGarment(garment: Garment): string {
  const canonical = canonicalGarment(garment);
  return JSON.stringify({
    ...canonical,
    panels: canonical.panels.map((panel) => ({
      ...panel,
      mesh: {
        positions: Array.from(panel.mesh.positions),
        patternCoordinates: Array.from(panel.mesh.patternCoordinates),
        indices: Array.from(panel.mesh.indices),
      },
    })),
  });
}

export function deserializeGarment(serialized: string): Garment {
  const raw: unknown = JSON.parse(serialized);
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new GarmentValidationError(["serialized garment must be a JSON object"]);
  }
  const candidate = raw as Record<string, unknown>;
  if (
    typeof candidate.id !== "string" ||
    !Array.isArray(candidate.panels) ||
    !Array.isArray(candidate.seams) ||
    !Array.isArray(candidate.materialAssignments)
  ) {
    throw new GarmentValidationError(["serialized garment requires an id and panel, seam, and material-assignment arrays"]);
  }
  const source = raw as Garment;
  const panels = source.panels.map((panel, index) => {
    const mesh = panel?.mesh as unknown as Record<string, unknown> | undefined;
    if (
      !mesh ||
      !Array.isArray(mesh.positions) ||
      !Array.isArray(mesh.patternCoordinates) ||
      !Array.isArray(mesh.indices)
    ) {
      throw new GarmentValidationError([`serialized panel ${index} mesh buffers must be arrays`]);
    }
    return {
      ...panel,
      mesh: {
        positions: Float32Array.from(mesh.positions as number[]),
        patternCoordinates: Float32Array.from(mesh.patternCoordinates as number[]),
        indices: Uint32Array.from(mesh.indices as number[]),
      },
    };
  });
  const garment: Garment = {
    ...source,
    panels,
  };
  validateGarment(garment);
  return garment;
}
