import { describe, expect, it } from "vitest";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { assembleGarment, createFittingScene, runFitting, type PanelPlacement } from "../../src/garment/assembly.js";
import { makeBoxAvatar, makeCapsuleAvatar, signedDistanceToAvatar } from "../../src/garment/avatar.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { buildAssemblyGraph, type Seam } from "../../src/garment/sewing.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";

function singlePanel() {
  let document = createPatternDocument("g8c", "single");
  const r = addRectPanel(document, "panel", [0, 0], 0.4, 0.3);
  document = r.document;
  return { document, refs: r.refs };
}

function twoPanels() {
  let document = createPatternDocument("g8c-two", "two");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3);
  document = b.document;
  const seam: Seam = {
    id: "seam/ab",
    sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
    sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: true },
    stitchCount: 5,
  };
  return { document, a: a.refs, b: b.refs, seam };
}

describe("G8C garment assembly", () => {
  it("assembles one panel with provenance, rest metric, and no diagnostics", () => {
    const { document, refs } = singlePanel();
    const placements: PanelPlacement[] = [{ panelId: refs.panelId, translation: [0, 1, 0.3], yawRad: 0 }];
    const g = assembleGarment(document, [], placements);
    expect(g.diagnostics).toEqual([]);
    expect(g.positions.length / 3).toBe(4);
    expect(g.indices.length / 3).toBe(2);
    expect(g.panelRanges).toHaveLength(1);
    expect(g.vertexPanelIds).toEqual([refs.panelId, refs.panelId, refs.panelId, refs.panelId]);
    // UV preserves pattern rest metric; world placement is translation only.
    // (Float32 rounding: compare with tolerance, not bitwise equality.)
    const uv = Array.from(g.uv);
    const want = [0, 0, 0.4, 0, 0.4, 0.3, 0, 0.3];
    expect(uv).toHaveLength(want.length);
    for (let i = 0; i < want.length; i++) expect(uv[i]).toBeCloseTo(want[i], 6);
    const first3 = Array.from(g.positions.slice(0, 3));
    expect(first3[0]).toBeCloseTo(0, 6);
    expect(first3[1]).toBeCloseTo(1, 6);
    expect(first3[2]).toBeCloseTo(0.3, 6);
    expect(g.components).toHaveLength(1);
    expect(g.maxInitialStretch).toBeCloseTo(1, 12);
  });

  it("joins two stitched panels with one connected component and stitch-count welds", () => {
    const { document, a, b, seam } = twoPanels();
    const placements: PanelPlacement[] = [
      { panelId: a.panelId, translation: [0, 1, 0.2], yawRad: 0 },
      { panelId: b.panelId, translation: [0.4, 1, -0.2], yawRad: Math.PI },
    ];
    const g = assembleGarment(document, [seam], placements);
    expect(g.diagnostics).toEqual([]);
    expect(g.weldPairs).toHaveLength(5);
    expect(g.components).toHaveLength(1);
    expect(g.weldPairs.map((w) => w.stitchIndex)).toEqual([0, 1, 2, 3, 4]);
    for (const w of g.weldPairs) {
      expect(Number.isFinite(w.residualM)).toBe(true);
      expect(Number.isFinite(w.gapM)).toBe(true);
    }
    // Cross-check against the G8B construction graph: same stitch count.
    const graph = buildAssemblyGraph(document, [seam]);
    expect(graph.seams[0].stitchPairs).toHaveLength(g.weldPairs.length);
  });

  it("reports disconnected components when a panel has no seam", () => {
    const { document, a, b } = twoPanels();
    // Seam exists but only references panel A twice? No — use no seams at all.
    const placements: PanelPlacement[] = [
      { panelId: a.panelId, translation: [0, 1, 0.2], yawRad: 0 },
      { panelId: b.panelId, translation: [5, 1, 5], yawRad: 0 },
    ];
    const g = assembleGarment(document, [], placements);
    expect(g.components).toHaveLength(2);
  });

  it("flags disconnected components when seams fail to connect all panels", () => {
    let document = createPatternDocument("g8c-disc", "disc");
    const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3); document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3); document = b.document;
    const c = addRectPanel(document, "C", [0, 0], 0.4, 0.3); document = c.document;
    const seam: Seam = {
      id: "seam/ab",
      sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
      sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: false },
      stitchCount: 3,
    };
    const placements: PanelPlacement[] = [a, b, c].map((p) => ({
      panelId: p.refs.panelId, translation: [0, 1, 0.3], yawRad: 0,
    }));
    const g = assembleGarment(document, [seam], placements);
    expect(g.components).toHaveLength(2);
    expect(g.diagnostics.map((d) => d.code)).toContain("disconnected-component");
  });

  it("detects initial penetration and deep penetration against the avatar", () => {
    const { document, refs } = singlePanel();
    // Radius 0.25: panel corners (radial 0.2 from the axis) are all inside.
    const avatar = makeCapsuleAvatar({ radiusM: 0.25, cylinderLengthM: 0.5, center: [0.2, 0.15, 0] });
    // Panel spans x in [0,0.4], y in [0,0.3] placed at the avatar core.
    const placements: PanelPlacement[] = [{ panelId: refs.panelId, translation: [0, 0, 0], yawRad: 0 }];
    const g = assembleGarment(document, [], placements, { avatar });
    expect(g.penetratingVertexCount).toBeGreaterThan(0);
    const codes = g.diagnostics.map((d) => d.code);
    expect(codes).toContain("penetrating-placement");
    expect(codes).toContain("deep-penetration");
    expect(g.minAvatarDistanceM).not.toBeNull();
    expect(g.minAvatarDistanceM!).toBeLessThan(0);
  });

  it("passes a clear-of-avatar placement with no penetration diagnostics", () => {
    const { document, refs } = singlePanel();
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 0.15, 0] });
    const placements: PanelPlacement[] = [{ panelId: refs.panelId, translation: [0, 1, 0.5], yawRad: 0 }];
    const g = assembleGarment(document, [], placements, { avatar });
    expect(g.penetratingVertexCount).toBe(0);
    expect(g.diagnostics).toEqual([]);
    expect(g.minAvatarDistanceM!).toBeGreaterThan(0);
  });

  it("reports failed seam references without dropping valid panels", () => {
    const { document, refs } = singlePanel();
    const bad: Seam = {
      id: "seam/bad",
      sideA: { panelId: refs.panelId, loopId: refs.loopId, segmentIds: [refs.segmentIds[0]], reversed: false },
      sideB: { panelId: "missing-panel", loopId: "missing-loop", segmentIds: ["missing-seg"], reversed: false },
      stitchCount: 3,
    };
    const g = assembleGarment(document, [bad], [
      { panelId: refs.panelId, translation: [0, 1, 0.3], yawRad: 0 },
    ]);
    expect(g.diagnostics.map((d) => d.code)).toContain("failed-seam-reference");
    expect(g.panelRanges).toHaveLength(1);
    expect(g.weldPairs).toEqual([]);
  });

  it("reports non-finite placements as invalid topology", () => {
    const { document, refs } = singlePanel();
    const g = assembleGarment(document, [], [
      { panelId: refs.panelId, translation: [NaN, 1, 0], yawRad: 0 },
    ]);
    expect(g.diagnostics.map((d) => d.code)).toContain("invalid-topology");
  });

  it("is deterministic across repeated assembly", () => {
    const { document, a, b, seam } = twoPanels();
    const placements: PanelPlacement[] = [
      { panelId: a.panelId, translation: [0, 1, 0.2], yawRad: 0 },
      { panelId: b.panelId, translation: [0.4, 1, -0.2], yawRad: Math.PI },
    ];
    const g1 = assembleGarment(document, [seam], placements);
    const g2 = assembleGarment(document, [seam], placements);
    expect(Array.from(g1.positions)).toEqual(Array.from(g2.positions));
    expect(Array.from(g1.indices)).toEqual(Array.from(g2.indices));
    expect(g1.weldPairs).toEqual(g2.weldPairs);
    expect(g1.diagnostics).toEqual(g2.diagnostics);
  });

  it("signed avatar distance is negative inside and positive outside", () => {
    const avatar = makeBoxAvatar({ halfExtentsM: [0.2, 0.2, 0.2], center: [0, 0, 0] });
    expect(signedDistanceToAvatar([0, 0, 0], avatar)).toBeLessThan(0);
    expect(signedDistanceToAvatar([5, 5, 5], avatar)).toBeGreaterThan(1);
  });
});

describe("G8C fitting pipeline", () => {
  it("runs assembly->placement->collision->relaxation->simulation->evaluation with no NaN", () => {
    const { document, refs } = singlePanel();
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 0.15, 0] });
    const g = assembleGarment(document, [], [
      { panelId: refs.panelId, translation: [0, 1, 0.5], yawRad: 0 },
    ], { avatar });
    const fitting = createFittingScene(g, { avatar });
    const solver = new CpuSolver();
    const result = runFitting(g, fitting, solver, avatar, {
      dt: 1 / 60, relaxationSteps: 2, simulationSteps: 3,
    });
    expect(result.stages.map((s) => s.stage)).toEqual([
      "assembly", "placement", "collision-validation", "relaxation", "simulation", "fit-evaluation",
    ]);
    expect(result.hasNaNInf).toBe(false);
    expect(result.stepsTaken).toBe(5);
    expect(result.ok).toBe(true);
    // Garment falls under gravity: final positions differ from initial.
    expect(result.finalMaxDisplacementM).toBeGreaterThan(0);
  });

  it("is deterministic across repeated simulation initialization", () => {
    const { document, refs } = singlePanel();
    const placements: PanelPlacement[] = [{ panelId: refs.panelId, translation: [0, 1, 0.5], yawRad: 0 }];
    const runOnce = (): number[] => {
      const g = assembleGarment(document, [], placements);
      const fitting = createFittingScene(g, {});
      const solver = new CpuSolver();
      runFitting(g, fitting, solver, null, { relaxationSteps: 2, simulationSteps: 2 });
      return Array.from(solver.getPositions());
    };
    expect(runOnce()).toEqual(runOnce());
  });

  it("refuses to build a scene from an empty garment", () => {
    const { document } = singlePanel();
    const g = assembleGarment(document, [], []);
    expect(g.diagnostics.map((d) => d.code)).toContain("missing-placement");
    expect(() => createFittingScene(g, {})).toThrow(/empty garment/);
  });
});
