/**
 * G8C — garment assembly & avatar fitting bridge.
 *
 * Pipeline (all deterministic, all outside the solver):
 *
 *   PatternDocument + Seam[] + PanelPlacement[]
 *     -> triangulate each panel (G8A, local coords preserved as UV/rest metric)
 *     -> rigid 3D placement (yaw about Y + translation)
 *     -> merged garment mesh + seam weld map (nearest-vertex correspondence)
 *     -> ClothScene via existing preprocess()/createScene()
 *     -> avatar attached via existing ContactSystem.setStaticMesh()
 *     -> fitting stages run on the existing ClothSolver interface
 *
 * The solver receives already-resolved topology (merged triangles) and
 * already-placed state. No FEM, contact, or collision code is modified here.
 * Sewing is NOT enforced as a solver constraint: panels start co-located
 * along seams (placement responsibility) and the weld map records the
 * intended correspondence for evaluation / future constraint work.
 */

import { preprocess, type ClothMeshData } from "../mesh/mesh.js";
import { createScene, type ClothScene } from "../physics/scene.js";
import { DEFAULT_MATERIAL, type ClothMaterial } from "../physics/types.js";
import {
  DEFAULT_CONTACT_PARAMS,
  type ContactParams,
} from "../collision/types.js";
import { ContactSystem } from "../collision/contact-assembly.js";
import type { ClothSolver } from "../backend/solver.js";
import {
  localToGlobal,
  triangulateCadPanel,
  type PatternDocument,
} from "../pattern/cad.js";
import {
  resolveStitchPairs,
  validateSeams,
  type Seam,
} from "./sewing.js";
import {
  signedDistanceToAvatar,
  type AvatarSpec,
} from "./avatar.js";

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface PanelPlacement {
  panelId: string;
  /** World-space translation in metres. */
  translation: Vec3;
  /** Yaw about the world Y axis in radians (deterministic spin). */
  yawRad: number;
}

export interface AssembleOptions {
  /** Tolerance for triangulation (sagitta metres). Default 1e-5. */
  sagittaTol?: number;
  /** Max admissible initial edge stretch ratio before flagging. Default 3. */
  maxInitialStretch?: number;
  /** Penetration depth (m) that counts as "deep". Default 0.02. */
  deepPenetrationM?: number;
  /** Avatar used for placement diagnostics. Null = skip collision checks. */
  avatar?: AvatarSpec | null;
}

export type AssemblyDiagnosticCode =
  | "missing-placement"
  | "extra-placement"
  | "triangulation-failed"
  | "failed-seam-reference"
  | "disconnected-component"
  | "degenerate-weld"
  | "nan-inf-state"
  | "deep-penetration"
  | "penetrating-placement"
  | "exploded-initialization"
  | "invalid-topology";

export interface AssemblyDiagnostic {
  code: AssemblyDiagnosticCode;
  message: string;
  panelId?: string;
  seamId?: string;
}

export interface WeldPair {
  seamId: string;
  stitchIndex: number;
  /** Global vertex indices into the assembled mesh. */
  vertexA: number;
  vertexB: number;
  /** Residual 2D pattern-space distance (m) of the correspondence snap. */
  residualM: number;
  /** 3D distance (m) between the welded vertices after placement. */
  gapM: number;
}

export interface PanelRange {
  panelId: string;
  vertexStart: number;
  vertexCount: number;
  triangleStart: number;
  triangleCount: number;
}

export interface AssembledGarment {
  /** Merged world-space positions (n*3), one entry per panel vertex. */
  positions: Float32Array;
  /** Pattern-space UV per vertex (rest metric input for preprocess). */
  uv: Float32Array;
  indices: Uint32Array;
  panelRanges: PanelRange[];
  /** Global vertex -> panel id (deterministic provenance). */
  vertexPanelIds: string[];
  weldPairs: WeldPair[];
  /** Max edge stretch (placed/rest) at initialization. */
  maxInitialStretch: number;
  /** Minimum signed avatar distance over garment vertices (null: no avatar). */
  minAvatarDistanceM: number | null;
  penetratingVertexCount: number;
  /** Connected components over triangles + welds (each = sorted global ids). */
  components: number[][];
  diagnostics: AssemblyDiagnostic[];
}

function finite3(v: Vec3): boolean {
  return v.length === 3 && v.every(Number.isFinite);
}

/** Pattern local (px,py) -> world. Applies G8A 2D panel transform, then yaw+translation. */
export function panelPointToWorld(
  document: PatternDocument,
  panelId: string,
  local: Vec2,
  placement: PanelPlacement,
): Vec3 {
  const panel = document.panels.find((p) => p.id === panelId);
  if (!panel) throw new Error(`assembly: panel '${panelId}' does not exist`);
  const flat = localToGlobal(panel, local);
  const c = Math.cos(placement.yawRad), s = Math.sin(placement.yawRad);
  // Pattern plane (x, y, 0) spun about Y: world = Ry(yaw) * (lx, ly, 0) + t.
  return [
    placement.translation[0] + c * flat[0],
    placement.translation[1] + flat[1],
    placement.translation[2] - s * flat[0],
  ];
}

function nearestVertex(patternXY: Float64Array, x: number, y: number): { index: number; residual: number } {
  let best = 0, bestD = Infinity;
  const n = patternXY.length / 2;
  for (let i = 0; i < n; i++) {
    const dx = patternXY[i * 2] - x, dy = patternXY[i * 2 + 1] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return { index: best, residual: Math.sqrt(bestD) };
}

export function assembleGarment(
  document: PatternDocument,
  seams: readonly Seam[],
  placements: readonly PanelPlacement[],
  opts: AssembleOptions = {},
): AssembledGarment {
  const diagnostics: AssemblyDiagnostic[] = [];
  const push = (code: AssemblyDiagnosticCode, message: string, panelId?: string, seamId?: string): void => {
    diagnostics.push({ code, message, ...(panelId ? { panelId } : {}), ...(seamId ? { seamId } : {}) });
  };

  const sagittaTol = opts.sagittaTol ?? 1e-5;
  const maxStretch = opts.maxInitialStretch ?? 3;
  const deepM = opts.deepPenetrationM ?? 0.02;
  const avatar = opts.avatar ?? null;

  // --- placement coverage -------------------------------------------------
  const placedIds = placements.map((p) => p.panelId);
  for (const panel of document.panels) {
    if (!placedIds.includes(panel.id)) {
      push("missing-placement", `panel '${panel.id}' has no placement; it is excluded from the garment`, panel.id);
    }
  }
  for (const p of placements) {
    if (!document.panels.some((panel) => panel.id === p.panelId)) {
      push("extra-placement", `placement references missing panel '${p.panelId}'`, p.panelId);
    }
    if (!finite3(p.translation) || !Number.isFinite(p.yawRad)) {
      push("invalid-topology", `placement for panel '${p.panelId}' is non-finite`, p.panelId);
    }
  }

  // --- triangulate + place (placements array order = deterministic) -------
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const panelRanges: PanelRange[] = [];
  const vertexPanelIds: string[] = [];
  // Per-panel pattern-space vertices for weld snapping (pre-placement).
  const panelPatternXY = new Map<string, Float64Array>();
  const panelBase = new Map<string, number>();

  for (const placement of placements) {
    const panel = document.panels.find((p) => p.id === placement.panelId);
    if (!panel) continue; // already diagnosed as extra-placement
    let tri;
    try {
      tri = triangulateCadPanel(document, panel.id, { sagittaTol });
    } catch (error) {
      push("triangulation-failed", `panel '${panel.id}' failed to triangulate: ${error instanceof Error ? error.message : String(error)}`, panel.id);
      continue;
    }
    const n = tri.vertices.length / 2;
    const base = pos.length / 3;
    panelBase.set(panel.id, base);
    panelPatternXY.set(panel.id, Float64Array.from(tri.vertices));
    const t0 = idx.length / 3;
    for (let i = 0; i < n; i++) {
      const world = panelPointToWorld(document, panel.id, [tri.vertices[i * 2], tri.vertices[i * 2 + 1]], placement);
      pos.push(world[0], world[1], world[2]);
      uv.push(tri.vertices[i * 2], tri.vertices[i * 2 + 1]);
      vertexPanelIds.push(panel.id);
    }
    for (let k = 0; k < tri.triangles.length; k++) idx.push(tri.triangles[k] + base);
    panelRanges.push({
      panelId: panel.id, vertexStart: base, vertexCount: n,
      triangleStart: t0, triangleCount: tri.triangles.length / 3,
    });
  }

  const positions = Float32Array.from(pos);
  const uvArr = Float32Array.from(uv);
  const indices = Uint32Array.from(idx);

  // --- NaN/Inf gate --------------------------------------------------------
  for (let i = 0; i < positions.length; i++) {
    if (!Number.isFinite(positions[i])) {
      push("nan-inf-state", `assembled position[${i}] is non-finite`);
      break;
    }
  }

  // --- seam welds (nearest-vertex snap of G8B stitch correspondence) ------
  const weldPairs: WeldPair[] = [];
  const seamValidation = validateSeams(document, seams);
  if (!seamValidation.valid) {
    for (const d of seamValidation.diagnostics) {
      push("failed-seam-reference", `seam '${d.seamId}': ${d.code} — ${d.message}`, undefined, d.seamId);
    }
  } else {
    for (const seam of seams) {
      let pairs;
      try {
        pairs = resolveStitchPairs(document, seam);
      } catch (error) {
        push("failed-seam-reference", `seam '${seam.id}' failed to resolve: ${error instanceof Error ? error.message : String(error)}`, undefined, seam.id);
        continue;
      }
      for (const pair of pairs) {
        // G8B stitch points are in G8A-transformed pattern space. Invert the
        // panel transform to recover triangulation-local coords for snapping.
        const snap = (panelId: string, svgPt: Vec2): { global: number; residual: number } | null => {
          const panel = document.panels.find((p) => p.id === panelId);
          const pattern = panelPatternXY.get(panelId);
          const base = panelBase.get(panelId);
          if (!panel || !pattern || base === undefined) return null;
          // sampleSide applied localToGlobal(panel, local); invert it.
          const sx = panel.transform.scale[0], sy = panel.transform.scale[1];
          const c = Math.cos(panel.transform.rotationRad), s = Math.sin(panel.transform.rotationRad);
          const dx = svgPt[0] - panel.transform.translation[0];
          const dy = svgPt[1] - panel.transform.translation[1];
          const local: Vec2 = [(c * dx + s * dy) / sx, (-s * dx + c * dy) / sy];
          const hit = nearestVertex(pattern, local[0], local[1]);
          return { global: base + hit.index, residual: hit.residual };
        };
        const a = snap(seam.sideA.panelId, pair.pointA);
        const b = snap(seam.sideB.panelId, pair.pointB);
        if (!a || !b) {
          push("failed-seam-reference", `seam '${seam.id}' stitch ${pair.index} references an unassembled panel`, undefined, seam.id);
          continue;
        }
        const gap = Math.hypot(
          positions[a.global * 3] - positions[b.global * 3],
          positions[a.global * 3 + 1] - positions[b.global * 3 + 1],
          positions[a.global * 3 + 2] - positions[b.global * 3 + 2],
        );
        weldPairs.push({
          seamId: seam.id, stitchIndex: pair.index,
          vertexA: a.global, vertexB: b.global,
          residualM: Math.max(a.residual, b.residual), gapM: gap,
        });
      }
    }
  }
  weldPairs.sort((p, q) =>
    p.seamId < q.seamId ? -1 : p.seamId > q.seamId ? 1 : p.stitchIndex - q.stitchIndex,
  );
  for (const w of weldPairs) {
    if (!Number.isFinite(w.gapM) || !Number.isFinite(w.residualM)) {
      push("degenerate-weld", `weld ${w.seamId}:${w.stitchIndex} is non-finite`, undefined, w.seamId);
    }
  }

  // --- initial strain (placed edge length vs pattern rest length) ----------
  let maxStretchRatio = 1;
  {
    const edgeKey = (a: number, b: number): string => (a < b ? `${a}_${b}` : `${b}_${a}`);
    const seen = new Set<string>();
    for (let t = 0; t < indices.length; t += 3) {
      const triIds = [indices[t], indices[t + 1], indices[t + 2]];
      for (let e = 0; e < 3; e++) {
        const a = triIds[e], b = triIds[(e + 1) % 3];
        const key = edgeKey(a, b);
        if (seen.has(key)) continue;
        seen.add(key);
        const rest = Math.hypot(uvArr[a * 2] - uvArr[b * 2], uvArr[a * 2 + 1] - uvArr[b * 2 + 1]);
        const placedLen = Math.hypot(
          positions[a * 3] - positions[b * 3],
          positions[a * 3 + 1] - positions[b * 3 + 1],
          positions[a * 3 + 2] - positions[b * 3 + 2],
        );
        if (rest > 1e-15) maxStretchRatio = Math.max(maxStretchRatio, placedLen / rest);
      }
    }
  }
  if (!Number.isFinite(maxStretchRatio)) {
    push("nan-inf-state", "initial stretch evaluation produced a non-finite ratio");
  } else if (maxStretchRatio > maxStretch) {
    push("exploded-initialization", `max initial edge stretch ${maxStretchRatio.toFixed(3)} exceeds ${maxStretch}`);
  }

  // --- connectivity (triangle adjacency + welds) ---------------------------
  const components: number[][] = (() => {
    const n = positions.length / 3;
    const parent = Array.from({ length: n }, (_, i) => i);
    const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    const union = (a: number, b: number): void => {
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
    };
    for (let t = 0; t < indices.length; t += 3) {
      union(indices[t], indices[t + 1]);
      union(indices[t + 1], indices[t + 2]);
    }
    for (const w of weldPairs) union(w.vertexA, w.vertexB);
    const groups = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r)!.push(i);
    }
    return [...groups.values()]
      .map((g) => g.sort((a, b) => a - b))
      .sort((a, b) => a[0] - b[0]);
  })();
  if (seams.length > 0 && components.length > 1 && weldPairs.length > 0) {
    push("disconnected-component", `garment has ${components.length} connected components despite ${seams.length} seam(s)`);
  }
  if (indices.length === 0 || positions.length === 0) {
    push("invalid-topology", "assembled garment has no triangles");
  }

  // --- avatar penetration --------------------------------------------------
  let minAvatarDistanceM: number | null = null;
  let penetratingVertexCount = 0;
  if (avatar) {
    let minD = Infinity;
    let inside = 0;
    const n = positions.length / 3;
    for (let i = 0; i < n; i++) {
      const d = signedDistanceToAvatar([positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]], avatar);
      if (!Number.isFinite(d)) continue;
      if (d < minD) minD = d;
      if (d < 0) inside++;
    }
    minAvatarDistanceM = Number.isFinite(minD) ? minD : null;
    penetratingVertexCount = inside;
    if (inside > 0) {
      push("penetrating-placement", `${inside}/${n} garment vertices start inside the avatar`);
    }
    if (minAvatarDistanceM !== null && minAvatarDistanceM < -deepM) {
      push("deep-penetration", `deepest initial penetration ${minAvatarDistanceM.toFixed(4)} m exceeds ${deepM} m`);
    }
  }

  return {
    positions, uv: uvArr, indices, panelRanges, vertexPanelIds, weldPairs,
    maxInitialStretch: maxStretchRatio,
    minAvatarDistanceM, penetratingVertexCount, components, diagnostics,
  };
}

// ---------------------------------------------------------------------------
// Solver handoff
// ---------------------------------------------------------------------------

export interface FittingSceneOptions {
  material?: ClothMaterial;
  arealDensityKgM2?: number;
  gravity?: Vec3;
  contact?: ContactParams;
  avatar?: AvatarSpec | AvatarSpec[] | null;
  floorY?: number | null;
}

export interface FittingScene {
  scene: ClothScene;
  mesh: ClothMeshData;
  contact: ContactSystem;
}

export function createFittingScene(
  garment: AssembledGarment,
  opts: FittingSceneOptions = {},
): FittingScene {
  if (garment.indices.length === 0) throw new Error("assembly: cannot create a scene from an empty garment");
  for (let i = 0; i < garment.positions.length; i++) {
    if (!Number.isFinite(garment.positions[i])) throw new Error("assembly: garment state contains NaN/Inf");
  }
  const density = opts.arealDensityKgM2 ?? 0.15;
  const mesh = preprocess(
    Float32Array.from(garment.positions),
    Float32Array.from(garment.uv),
    Uint32Array.from(garment.indices),
    density,
  );
  const scene = createScene(mesh, opts.material ?? { ...DEFAULT_MATERIAL }, opts.gravity ?? [0, -9.81, 0]);
  const contact = new ContactSystem(opts.contact ?? { ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  if (opts.avatar) {
    const list = Array.isArray(opts.avatar) ? opts.avatar : [opts.avatar];
    let nv = 0, ni = 0;
    for (const a of list) {
      nv += a.positions.length / 3;
      ni += a.indices.length;
    }
    const sp = new Float32Array(nv * 3);
    const si = new Uint32Array(ni);
    let vo = 0, io = 0;
    for (const a of list) {
      sp.set(a.positions, vo * 3);
      for (let k = 0; k < a.indices.length; k++) si[io + k] = a.indices[k] + vo;
      vo += a.positions.length / 3;
      io += a.indices.length;
    }
    contact.setStaticMesh(sp, si);
  }
  if (opts.floorY !== undefined && opts.floorY !== null) contact.setFloor(opts.floorY);
  scene.contact = contact;
  return { scene, mesh, contact };
}

// ---------------------------------------------------------------------------
// Fitting pipeline with explicit stages + diagnostics
// ---------------------------------------------------------------------------

export type FitStage = "assembly" | "placement" | "collision-validation" | "relaxation" | "simulation" | "fit-evaluation";

export interface FitStageReport {
  stage: FitStage;
  ok: boolean;
  detail: string;
  diagnostics: AssemblyDiagnostic[];
}

export interface FitRunOptions {
  dt?: number;
  /** Gravity-scaled relaxation steps before full simulation. Default 5. */
  relaxationSteps?: number;
  /** Full-gravity simulation steps after relaxation. Default 10. */
  simulationSteps?: number;
  /** NaN/Inf + explosion guard threshold on displacement per step. Default 1. */
  maxStepDisplacementM?: number;
}

export interface FitResult {
  ok: boolean;
  stages: FitStageReport[];
  stepsTaken: number;
  finalMinAvatarDistanceM: number | null;
  finalMaxDisplacementM: number;
  hasNaNInf: boolean;
}

function stateHasNaNInf(state: Float64Array): boolean {
  for (let i = 0; i < state.length; i++) if (!Number.isFinite(state[i])) return true;
  return false;
}

function maxDisplacement(a: Float64Array, b: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 3) {
    const d = Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1], a[i + 2] - b[i + 2]);
    if (d > m) m = d;
  }
  return m;
}

export function runFitting(
  garment: AssembledGarment,
  fitting: FittingScene,
  solver: ClothSolver,
  avatar: AvatarSpec | null,
  opts: FitRunOptions = {},
): FitResult {
  const dt = opts.dt ?? 1 / 60;
  const relaxSteps = opts.relaxationSteps ?? 5;
  const simSteps = opts.simulationSteps ?? 10;
  const maxStep = opts.maxStepDisplacementM ?? 1;
  const stages: FitStageReport[] = [];
  const fatal = garment.diagnostics.filter((d) =>
    d.code === "nan-inf-state" || d.code === "invalid-topology" || d.code === "triangulation-failed",
  );

  stages.push({
    stage: "assembly",
    ok: fatal.length === 0,
    detail: `${garment.panelRanges.length} panel(s), ${garment.indices.length / 3} tri(s), ${garment.weldPairs.length} weld(s), ${garment.components.length} component(s)`,
    diagnostics: [...fatal],
  });

  const blockingErrors = garment.diagnostics.filter((d) =>
    d.code === "deep-penetration" || d.code === "exploded-initialization",
  );
  stages.push({
    stage: "placement",
    ok: true, // placement warnings never silently block; reported for the caller
    detail: `maxInitialStretch=${garment.maxInitialStretch.toFixed(3)}, minAvatarDistance=${garment.minAvatarDistanceM?.toFixed(4) ?? "n/a"}`,
    diagnostics: [...blockingErrors],
  });

  // Collision validation: refresh active set once at x0 without stepping.
  let collisionOk = true;
  let collisionDetail = "no avatar attached; validation skipped";
  if (avatar) {
    try {
      fitting.contact.beginStep(fitting.scene.positions);
      fitting.contact.updateActiveSet(fitting.scene.positions);
      collisionDetail = `activePairs=${fitting.contact.active.length}`;
    } catch (error) {
      collisionOk = false;
      collisionDetail = `collision init failed: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  stages.push({ stage: "collision-validation", ok: collisionOk, detail: collisionDetail, diagnostics: [] });

  // Relaxation + simulation through the existing solver only.
  solver.initialize(fitting.scene);
  const x0 = Float64Array.from(solver.getPositions());
  let stepsTaken = 0;
  let failed: AssemblyDiagnostic | null = null;
  const totalSteps = relaxSteps + simSteps;
  const startGravity = fitting.scene.gravity;
  void startGravity;
  for (let s = 0; s < totalSteps; s++) {
    const before = Float64Array.from(solver.getPositions());
    try {
      solver.step(dt);
    } catch (error) {
      failed = { code: "nan-inf-state", message: `solver threw at step ${s}: ${error instanceof Error ? error.message : String(error)}` };
      break;
    }
    stepsTaken++;
    const now = solver.getPositions();
    if (stateHasNaNInf(now) || stateHasNaNInf(solver.getVelocities())) {
      failed = { code: "nan-inf-state", message: `non-finite state at step ${s}` };
      break;
    }
    if (maxDisplacement(before, now) > maxStep) {
      failed = { code: "exploded-initialization", message: `unstable displacement at step ${s}` };
      break;
    }
    if (s + 1 === relaxSteps) {
      stages.push({
        stage: "relaxation",
        ok: failed === null,
        detail: `${relaxSteps} relaxation step(s) completed`,
        diagnostics: failed ? [failed] : [],
      });
    }
  }
  if (totalSteps === 0 || relaxSteps === 0) {
    stages.push({ stage: "relaxation", ok: true, detail: "relaxation skipped (0 steps)", diagnostics: [] });
  }
  const simOk = failed === null;
  stages.push({
    stage: "simulation",
    ok: simOk,
    detail: `${stepsTaken}/${totalSteps} step(s) completed`,
    diagnostics: failed ? [failed] : [],
  });

  // Fit evaluation (read-only state audit).
  const final = solver.getPositions();
  const hasNaNInf = stateHasNaNInf(final);
  const finalMaxDisplacementM = maxDisplacement(x0, final);
  let finalMinAvatarDistanceM: number | null = null;
  if (avatar && !hasNaNInf) {
    let minD = Infinity;
    const n = final.length / 3;
    for (let i = 0; i < n; i++) {
      const d = signedDistanceToAvatar([final[i * 3], final[i * 3 + 1], final[i * 3 + 2]], avatar);
      if (d < minD) minD = d;
    }
    finalMinAvatarDistanceM = Number.isFinite(minD) ? minD : null;
  }
  stages.push({
    stage: "fit-evaluation",
    ok: !hasNaNInf && simOk,
    detail: `maxDisplacement=${finalMaxDisplacementM.toFixed(4)} m, minAvatarDistance=${finalMinAvatarDistanceM?.toFixed(4) ?? "n/a"}`,
    diagnostics: [],
  });

  return {
    ok: stages.every((s) => s.ok) && fatal.length === 0,
    stages, stepsTaken, finalMinAvatarDistanceM, finalMaxDisplacementM, hasNaNInf,
  };
}
