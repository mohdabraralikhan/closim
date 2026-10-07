// G16 FINAL — collared-shirt commercial acceptance.
//
// frontL/frontR + back + sleeves shaped with darts/pleat/gather, collar +
// cuffs + pocket + buttons/buttonholes as registered components, sewn through
// the existing seam graph, graded by stable-id edits, simulated, production-
// engineered, rendered, and exported — then an upstream neckline edit
// propagates neckline -> collar -> seams -> production -> 3D -> simulation
// with no manual reconstruction.
import { describe, expect, it } from "vitest";
import {
  createPatternDocument,
  movePoint,
  serializePatternDocument,
  triangulateCadPanel,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import { buildPanelFromRing } from "../../src/cad/draft.js";
import { getPanel } from "../../src/cad/queries.js";
import {
  addDart,
  addGather,
  addPleat,
  createConstructionSet,
  deriveConstruction,
  deserializeConstructionSet,
  serializeConstructionSet,
} from "../../src/construction/features.js";
import {
  addComponent,
  createComponentSet,
  deriveComponents,
  deserializeComponentSet,
  serializeComponentSet,
  validateComponents,
} from "../../src/construction/components.js";
import { splitBoundarySegment } from "../../src/pattern/cad.js";
import { validateSeams, type Seam } from "../../src/garment/sewing.js";
import { assembleGarment, runFitting, createFittingScene } from "../../src/garment/assembly.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import {
  addAllowance,
  addCutLine,
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { pointInPanel } from "../../src/cad/queries.js";
import type { Vec2 } from "../../src/cad/geom.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import { buildRenderGarment, syncRenderPositions } from "../../src/render/representation.js";
import { createGarmentProject } from "../../src/garment/project.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import {
  addSize,
  createGradingDocument,
  createMasterPattern,
  createRuleTable,
  createSize,
  createSizeSet,
} from "../../src/grading/model.js";
import {
  addCutItem,
  createCutPlan,
  expandCutPlan,
  type Marker,
} from "../../src/marker/model.js";
import {
  createProductionSpecification,
  generateTechPack,
} from "../../src/production/specification.js";
import { createBOMFromSpecification, generateBOMRequirements, validateBOM } from "../../src/production/bom.js";
import { calculateCutPlan, createProductionRun, markerConsumptionForRun } from "../../src/production/run.js";
import { buildManufacturingReports } from "../../src/production/reporting.js";
import {
  createRevision,
  createRevisionLedger,
  recordRevisionArtifact,
  setRevisionStatus,
} from "../../src/production/revision.js";
import { generateProductionPackage } from "../../src/production/package.js";

interface ShirtRefs {
  frontL: string; frontR: string; back: string; sleeveL: string; sleeveR: string;
  loops: Record<string, string>;
  segs: Record<string, string[]>;
}

function ringDoc(): { document: PatternDocument; refs: ShirtRefs } {
  let document = createPatternDocument("g16-shirt", "Collared shirt");
  const mk = (name: string, x0: number, w: number): string => {
    const r = buildPanelFromRing(document, name, "cotton", 0, [
      [x0, 0], [x0 + w, 0], [x0 + w, 0.62], [x0, 0.62],
    ]);
    document = r.document;
    return r.panelId;
  };
  const frontL = mk("front-left", 0, 0.23);
  const frontR = mk("front-right", 0.23, 0.23);
  const back = mk("back", 0.6, 0.46);
  const sleeveL = mk("sleeve-left", 1.2, 0.3);
  const sleeveR = mk("sleeve-right", 1.6, 0.3);
  const loops: Record<string, string> = {};
  const segs: Record<string, string[]> = {};
  for (const pid of [frontL, frontR, back, sleeveL, sleeveR]) {
    const loop = getPanel(document, pid).boundaryLoops[0];
    loops[pid] = loop.id;
    segs[pid] = [...loop.segmentIds];
  }
  // Split the back top edge for two shoulder seams (segment order preserved).
  const split = splitBoundarySegment(document, back, loops[back], segs[back][2], 0.5);
  document = split.document;
  segs[back] = [...getPanel(document, back).boundaryLoops[0].segmentIds];
  return { document, refs: { frontL, frontR, back, sleeveL, sleeveR, loops, segs } };
}

function shirtSeams(refs: ShirtRefs, extra: Seam[] = []): Seam[] {
  const side = (panelId: string, k: number, reversed = false) => ({
    panelId, loopId: refs.loops[panelId], segmentIds: [refs.segs[panelId][k]], reversed,
  });
  return [
    { id: "seam/shoulder-l", sideA: side(refs.frontL, 2), sideB: side(refs.back, 2, true), stitchCount: 5 },
    { id: "seam/shoulder-r", sideA: side(refs.frontR, 2), sideB: side(refs.back, 3, true), stitchCount: 5 },
    { id: "seam/side-l", sideA: side(refs.frontL, 3), sideB: side(refs.back, 5 - 1, true), stitchCount: 7 },
    { id: "seam/side-r", sideA: side(refs.frontR, 1), sideB: side(refs.back, 1, true), stitchCount: 7 },
    { id: "seam/sleeve-l", sideA: side(refs.frontL, 3), sideB: side(refs.sleeveL, 1), stitchCount: 5 },
    { id: "seam/sleeve-r", sideA: side(refs.frontR, 1), sideB: side(refs.sleeveR, 3, true), stitchCount: 5 },
    ...extra,
  ];
}

describe("G16 final integration — collared shirt", () => {
  it("builds, sews, grades, simulates, engineers, renders, and exports", () => {
    // Create pattern.
    const { document: base, refs } = ringDoc();
    expect(validatePatternDocument(base).valid).toBe(true);

    // Shape garment: waist darts, back pleat, sleeve gather.
    let features = createConstructionSet();
    const d1 = addDart(features, refs.frontL, refs.loops[refs.frontL], refs.segs[refs.frontL][0], 0.3, 0.7, [0.115, 0.3], "open");
    features = d1.set;
    const d2 = addDart(features, refs.frontR, refs.loops[refs.frontR], refs.segs[refs.frontR][0], 0.3, 0.7, [0.345, 0.3], "open");
    features = d2.set;
    const d3 = addDart(features, refs.back, refs.loops[refs.back], refs.segs[refs.back][0], 0.35, 0.65, [0.83, 0.3], "closed");
    features = d3.set;
    // Box pleat on the unsewn sleeveR right edge (no feature, component,
    // or seam touches it; its extended end only slants the unsewn top edge).
    const pl = addPleat(features, refs.sleeveR, refs.loops[refs.sleeveR], refs.segs[refs.sleeveR][1], 0.5, "box", 0.015);
    features = pl.set;
    const ga = addGather(features, refs.sleeveL, refs.loops[refs.sleeveL], refs.segs[refs.sleeveL][2], { targetLengthM: 0.2, notchCount: 3 });
    features = ga.set;
    const shaped = deriveConstruction(base, features);
    expect(shaped.failed).toEqual([]);
    expect(validatePatternDocument(shaped.document).valid).toBe(true);

    // Components: collar (frontR neckline), cuffs, pocket, buttons, holes.
    let components = createComponentSet();
    const collar = addComponent(components, refs.frontR, "collar-band", {
      edgeLoopId: refs.loops[refs.frontR], edgeSegmentId: refs.segs[refs.frontR][2], heightM: 0.07, name: "collar",
    }, [refs.segs[refs.frontR][2], refs.loops[refs.frontR], refs.frontR]);
    components = collar.set;
    const cuffL = addComponent(components, refs.sleeveL, "cuff-band", {
      edgeLoopId: refs.loops[refs.sleeveL], edgeSegmentId: refs.segs[refs.sleeveL][0], heightM: 0.06, name: "cuff-left",
    }, [refs.segs[refs.sleeveL][0]]);
    components = cuffL.set;
    const cuffR = addComponent(components, refs.sleeveR, "cuff-band", {
      edgeLoopId: refs.loops[refs.sleeveR], edgeSegmentId: refs.segs[refs.sleeveR][0], heightM: 0.06, name: "cuff-right",
    }, [refs.segs[refs.sleeveR][0]]);
    components = cuffR.set;
    const pocket = addComponent(components, refs.frontL, "patch-pocket", {
      center: [0.115, 0.4], widthM: 0.1, heightM: 0.12, name: "pocket",
    }, [refs.frontL]);
    components = pocket.set;
    const buttons = addComponent(components, refs.frontR, "button", { pos: [0.25, 0.5], diameterM: 0.012 }, [refs.frontR]);
    components = buttons.set;
    const holes = addComponent(components, refs.frontL, "buttonhole", { pos: [0.21, 0.5], lengthM: 0.018, angleRad: 0 }, [refs.frontL]);
    components = holes.set;
    const built = deriveComponents(shaped.document, components);
    expect(built.failed).toEqual([]);
    expect(built.document.panels.length).toBeGreaterThanOrEqual(8); // 5 base + collar + 2 cuffs + pocket
    expect(validatePatternDocument(built.document).valid).toBe(true);

    // Sew through the existing seam graph (feature + component seams included).
    const collarSeam = built.seams.find((s) => s.id === `seam/${collar.id}`)!;
    const cuffSeams = built.seams.filter((s) => s.id === `seam/${cuffL.id}` || s.id === `seam/${cuffR.id}`);
    expect(cuffSeams).toHaveLength(2);
    const seams = shirtSeams(refs, [...shaped.seams, ...built.seams]);
    expect(validateSeams(built.document, seams).valid).toBe(true);

    // Grade by stable-id edits on the BASE, then re-derive (features follow the ids).
    let gradedBase = base;
    for (const pid of [refs.frontL, refs.frontR, refs.back, refs.sleeveL, refs.sleeveR]) {
      for (const pt of gradedBase.points.filter((p) => p.panelId === pid && p.y === 0)) {
        gradedBase = movePoint(gradedBase, pid, pt.id, [pt.x, pt.y - 0.02]);
      }
    }
    const reShaped = deriveConstruction(gradedBase, features);
    expect(reShaped.failed).toEqual([]);
    const reBuilt = deriveComponents(reShaped.document, built.set);
    expect(reBuilt.failed).toEqual([]);
    expect(validatePatternDocument(reBuilt.document).valid).toBe(true);
    const reSeams = shirtSeams(refs, [...reShaped.seams, ...reBuilt.seams]);
    // Collar/cuff bands regenerate deterministically (same ids, new geometry).
    expect(reBuilt.set.components.map((c) => c.id)).toEqual(built.set.components.map((c) => c.id));

    // Simulate the graded garment (derived bands placed explicitly).
    const bandPlacement = (type: string, translation: [number, number, number], yawRad = 0) => {
      const feature = reBuilt.set.components.find((c) => c.componentType === type)!;
      return { panelId: feature.derived.panels[0], translation, yawRad };
    };
    const placements = [
      { panelId: refs.frontL, translation: [0, 0.6, 0.2] as [number, number, number], yawRad: 0 },
      { panelId: refs.frontR, translation: [0.23, 0.6, 0.2] as [number, number, number], yawRad: 0 },
      { panelId: refs.back, translation: [1.06, 0.6, -0.2] as [number, number, number], yawRad: Math.PI },
      { panelId: refs.sleeveL, translation: [-0.06, 0.95, 0.65] as [number, number, number], yawRad: Math.PI / 2 },
      { panelId: refs.sleeveR, translation: [0.52, 0.95, -0.95] as [number, number, number], yawRad: -Math.PI / 2 },
      bandPlacement("collar-band", [0.1, 1.45, 0]),
      bandPlacement("cuff-band", [-0.2, 0.7, 0.2]),
    ];
    // Two cuffs share the type: place the second explicitly.
    const cuffs = reBuilt.set.components.filter((c) => c.componentType === "cuff-band");
    placements.push({ panelId: cuffs[1].derived.panels[0], translation: [0.66, 0.7, 0.2], yawRad: 0 });
    // Pocket rides as an unsewn swatch beside the body.
    const pocketFeature = reBuilt.set.components.find((c) => c.componentType === "patch-pocket")!;
    placements.push({ panelId: pocketFeature.derived.panels[0], translation: [-0.35, 1.0, 0.3], yawRad: 0 });
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.23, 0.95, 0] });
    const assembled = assembleGarment(reBuilt.document, reSeams, placements, { avatar });
    expect(assembled.diagnostics.filter((d) => d.code === "failed-seam-reference")).toEqual([]);
    const fitting = createFittingScene(assembled, { avatar });
    const solver = new CpuSolver();
    const fit = runFitting(assembled, fitting, solver, avatar, { relaxationSteps: 2, simulationSteps: 2 });
    expect(fit.hasNaNInf).toBe(false);

    // Production engineer: allowances, grainlines, markings, metadata.
    // Grainlines nudge sideways past dart cutouts (deterministic offsets).
    const grainlineFor = (doc: PatternDocument, panelId: string): { from: Vec2; to: Vec2 } => {
      const base = centeredGrainline(doc, panelId);
      for (const dx of [0, -0.02, 0.02, -0.04, 0.04, -0.06, 0.06, -0.08, 0.08, -0.1, 0.1]) {
        const from: Vec2 = [base.from[0] + dx, base.from[1]];
        const to: Vec2 = [base.to[0] + dx, base.to[1]];
        if (pointInPanel(doc, panelId, from) && pointInPanel(doc, panelId, to)) return { from, to };
      }
      throw new Error(`no valid grainline for panel ${panelId}`);
    };
    let prod = createProductionSet();
    for (const panel of reBuilt.document.panels) {
      const loop = panel.boundaryLoops[0];
      prod = addAllowance(prod, panel.id, loop.id, 0.01).set;
      const grain = grainlineFor(reBuilt.document, panel.id);
      prod = addGrainline(prod, panel.id, grain.from, grain.to).set;
      prod = setPanelMeta(prod, { panelId: panel.id, cutQuantity: 1 });
      prod = addCutLine(prod, panel.id, loop.id, "sewing").set;
    }
    for (const fold of [...reShaped.folds, ...reBuilt.folds]) {
      prod = addFoldLine(prod, fold.panelId, fold.a, fold.b, "valley", "press").set;
    }
    for (const notch of [...reShaped.notches, ...reBuilt.notches]) {
      prod = addNotch(prod, notch.panelId, notch.loopId, notch.segmentId, notch.t, notch.kind, notch.depthM).set;
    }
    for (const drill of reBuilt.drills) {
      prod = addDrillMark(prod, drill.panelId, drill.pos, drill.mark, drill.radiusM).set;
    }
    for (const internal of reBuilt.internals) {
      prod = addInternalLine(prod, internal.panelId, internal.points, internal.kind, internal.label).set;
    }
    const readiness = productionReadiness(reBuilt.document, reSeams, prod);
    expect(readiness.errorCount).toBe(0);

    // Render representation follows the simulated state.
    const render = buildRenderGarment(
      "garment/shirt", assembled, Float32Array.from(solver.getPositions()), {},
    );
    expect(render.panelUVs.length).toBeGreaterThanOrEqual(8);

    // Export: every panel triangulates; sets round-trip deterministically.
    for (const panel of reBuilt.document.panels) {
      expect(triangulateCadPanel(reBuilt.document, panel.id).triangles.length).toBeGreaterThan(0);
    }
    expect(serializeConstructionSet(deserializeConstructionSet(serializeConstructionSet(features))))
      .toBe(serializeConstructionSet(features));
    expect(serializeComponentSet(deserializeComponentSet(serializeComponentSet(built.set))))
      .toBe(serializeComponentSet(built.set));
    expect(serializePatternDocument(reBuilt.document).length).toBeGreaterThan(0);

    // Run the G16 collared shirt through the G18 production chain for XS–XXL.
    const sizeOrders = [
      ["xs", "XS", 20], ["s", "S", 50], ["m", "M", 100],
      ["l", "L", 100], ["xl", "XL", 60], ["xxl", "XXL", 20],
    ] as const;
    const gradingId = "grading/g16-shirt";
    let grading = createGradingDocument({
      id: gradingId,
      name: "Collared shirt XS–XXL",
      master: createMasterPattern("master/g16-shirt", "Collared shirt", reBuilt.document),
      sizeSet: createSizeSet("size-set/g16-shirt", "XS–XXL"),
      ruleTable: createRuleTable("rule-table/g16-shirt", "Collared shirt grade rules"),
      seams: reSeams,
    });
    for (const [id, label] of sizeOrders) {
      grading = addSize(grading, createSize({ id: `size/${id}`, label }));
    }
    const fabricIds = [...new Set(reBuilt.document.panels.map((panel) => panel.materialId).filter((id): id is string => Boolean(id)))];
    const garment = createGarmentProject("garment/g16-shirt", "Collared shirt", reBuilt.document, {
      seams: reSeams,
      materials: Object.fromEntries(fabricIds.map((id) => [id, { ...DEFAULT_MATERIAL }])),
    });
    const specification = createProductionSpecification({
      id: "spec/g16-shirt",
      garmentId: garment.id,
      garmentName: garment.metadata.name,
      styleNumber: "SHIRT-001",
      revision: "A",
      sizeRange: sizeOrders.map(([id]) => `size/${id}`),
      materials: fabricIds.map((id) => ({ id, name: "Cotton shell", category: "shell" })),
      colorways: [],
      panels: reBuilt.document.panels.map((panel) => ({
        panelId: panel.id,
        ...(panel.materialId ? { materialId: panel.materialId } : {}),
      })),
      construction: reSeams.map((seam) => ({
        id: seam.id,
        kind: "seam",
        title: seam.id,
        data: {
          panelA: seam.sideA.panelId,
          panelB: seam.sideB.panelId,
          segmentIdsA: seam.sideA.segmentIds.join(","),
          segmentIdsB: seam.sideB.segmentIds.join(","),
        },
      })),
      measurements: [],
      productionNotes: ["G16 collared-shirt production acceptance"],
      finishingNotes: [],
    });
    const techPack = generateTechPack(garment, specification, { grading });
    expect(techPack.validation.errors).toEqual([]);
    expect(techPack.sizes.map((size) => size.label)).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);

    const markers: Marker[] = [];
    const batchAssignments: Array<{ markerId: string; fabricId: string; repeats: number }> = [];
    for (const [shortId, , quantity] of sizeOrders) {
      const sizeId = `size/${shortId}`;
      let cutPlan = createCutPlan(`cut-plan/${sizeId}`, `${sizeId} shirt cut`, grading.id);
      for (const panel of reBuilt.document.panels) {
        cutPlan = addCutItem(cutPlan, {
          panelId: panel.id,
          sizeId,
          quantity: 1,
          mirror: "allowed",
          ...(panel.materialId ? { materialId: panel.materialId } : {}),
        }).plan;
      }
      const pieces = expandCutPlan(grading, cutPlan);
      for (const fabricId of fabricIds) {
        const fabricPieces = pieces.filter((piece) =>
          reBuilt.document.panels.find((panel) => panel.id === piece.panelId)?.materialId === fabricId);
        if (!fabricPieces.length) continue;
        let cursorY = 0.01;
        const placements = fabricPieces.map((piece) => {
          const minY = Math.min(...piece.polygon.map((point) => point[1]));
          const maxY = Math.max(...piece.polygon.map((point) => point[1]));
          const placement = {
            instanceId: piece.instanceId,
            x: 0.01,
            y: cursorY,
            rotationDeg: 0,
            mirrored: false,
            manual: false,
          };
          cursorY += maxY - minY + 0.02;
          return placement;
        });
        const marker: Marker = {
          schemaVersion: 1,
          id: `marker/${shortId}/${fabricId}`,
          name: `${sizeId} ${fabricId}`,
          fabric: {
            id: fabricId,
            name: "Cotton shell",
            widthM: 1.5,
            usableWidthM: 1.4,
            lengthM: 50,
            materialType: "shell",
          },
          cutPlanId: cutPlan.id,
          constraint: {
            spacingM: 0.01,
            marginM: 0.01,
            rotationsDeg: [0],
            mirrorDefault: "as-authored",
            grainToleranceDeg: 0,
            directionalFabric: false,
          },
          pieces: fabricPieces,
          placements,
          revision: 1,
        };
        markers.push(marker);
        batchAssignments.push({ markerId: marker.id, fabricId, repeats: quantity });
      }
    }
    const sizeQuantities = Object.fromEntries(sizeOrders.map(([id, , quantity]) => [`size/${id}`, quantity]));
    const run = createProductionRun({
      id: "run/g16-shirt/A",
      garmentId: garment.id,
      styleNumber: specification.styleNumber,
      revision: specification.revision,
      gradingId: grading.id,
      status: "planned",
      sizeQuantities,
      panelMaterialIds: Object.fromEntries(reBuilt.document.panels.flatMap((panel) =>
        panel.materialId ? [[panel.id, panel.materialId]] : [])),
      batches: [{ id: "batch/g16-shirt/1", markerAssignments: batchAssignments }],
    });
    const cutPlan = calculateCutPlan(run, grading, markers);
    expect(cutPlan.diagnostics).toEqual([]);
    expect(cutPlan.complete).toBe(true);
    expect(cutPlan.produced).toEqual(sizeQuantities);
    expect(Object.values(cutPlan.requiredPieces).flatMap(Object.values).reduce((sum, value) => sum + value, 0))
      .toBe(reBuilt.document.panels.length * 350);

    const bom = createBOMFromSpecification(specification, {
      id: "bom/g16-shirt/A",
      revision: specification.revision,
      materialUnits: Object.fromEntries(fabricIds.map((id) => [id, "m"])),
      markerReferences: markers.map((marker) => ({ markerId: marker.id, fabricId: marker.fabric.id })),
    });
    const markerMetrics = markerConsumptionForRun(run, markers);
    const panelQuantities = Object.values(cutPlan.cutPieces).reduce<Record<string, number>>((totals, panels) => {
      for (const [panelId, quantity] of Object.entries(panels)) totals[panelId] = (totals[panelId] ?? 0) + quantity;
      return totals;
    }, {});
    const bomRequirements = generateBOMRequirements(bom, {
      garmentQuantity: 350,
      sizeQuantities,
      panelQuantities,
      markerMetrics,
    });
    const requiredArtifactIds = [
      "pattern",
      `grading:${grading.id}`,
      ...markers.map((marker) => marker.id),
      bom.id,
      run.id,
      `tech-pack:${techPack.sourceFingerprint}`,
      `cut-plan:${run.id}`,
    ];
    const revisionId = "revision/g16-shirt/A";
    const traceArtifacts = requiredArtifactIds.map((id) => ({
      id,
      kind: id.startsWith("marker/") ? "marker" as const
        : id.startsWith("grading:") ? "grading" as const
          : id.startsWith("tech-pack:") ? "tech-pack" as const
            : id.startsWith("cut-plan:") ? "cut-plan" as const
              : id === "pattern" ? "pattern" as const : "bom" as const,
      dependencies: [],
      sourceFingerprint: `fingerprint/${id}`,
      currentSourceFingerprint: `fingerprint/${id}`,
      sourceRevisionId: revisionId,
      state: "current" as const,
    }));
    let ledger = createRevisionLedger(garment.id);
    ledger = createRevision({
      ledger,
      id: revisionId,
      author: "G18 acceptance",
      timestamp: "2026-01-01T00:00:00Z",
      changeSummary: "Release collared shirt for production",
      sourceProjectVersion: "g16/1",
      sourceFingerprint: techPack.sourceFingerprint,
    });
    for (const artifact of traceArtifacts) {
      ledger = recordRevisionArtifact(ledger, revisionId, {
        id: artifact.id,
        kind: artifact.kind,
        fingerprint: artifact.sourceFingerprint,
      }, { author: "G18 acceptance", timestamp: "2026-01-01T00:01:00Z" });
    }
    ledger = setRevisionStatus(ledger, revisionId, "Released", {
      author: "G18 acceptance",
      timestamp: "2026-01-01T00:02:00Z",
    });
    const revision = ledger.revisions[0];
    const report = buildManufacturingReports({
      garmentId: garment.id,
      garmentName: garment.metadata.name,
      garmentRevision: garment.metadata.revision,
      pattern: garment.pattern,
      grading,
      specification,
      techPack,
      bom,
      bomRequirements,
      bomIssues: validateBOM(bom, { markers, knownPanels: reBuilt.document.panels.map((panel) => panel.id) }),
      markers,
      run,
      cutPlan,
      artifacts: traceArtifacts,
      production: prod,
    });
    expect(report.dashboard.plannedQuantity).toBe(350);
    const pkg = generateProductionPackage({
      garment,
      specification,
      revision,
      grading,
      markers,
      bom,
      run,
      cutPlan,
      techPack,
      dashboard: report.dashboard,
      reports: report.reports,
      artifacts: traceArtifacts,
    });
    const manifest = JSON.parse(pkg.files.find((file) => file.path.endsWith("/metadata/manifest.json"))?.bytes as string) as {
      revisionId: string;
      productionRunId: string;
      sizes: string[];
    };
    expect(manifest).toMatchObject({
      revisionId,
      productionRunId: run.id,
      sizes: sizeOrders.map(([id]) => `size/${id}`),
    });
    expect(pkg.files.filter((file) => file.path.includes("/patterns/")).length).toBe(6);
  });

  it("propagates an upstream neckline edit with no manual reconstruction", () => {
    const { document: base, refs } = ringDoc();
    let features = createConstructionSet();
    const d = addDart(features, refs.frontL, refs.loops[refs.frontL], refs.segs[refs.frontL][0], 0.3, 0.7, [0.115, 0.3]);
    features = d.set;
    let components = createComponentSet();
    const collar = addComponent(components, refs.frontR, "collar-band", {
      edgeLoopId: refs.loops[refs.frontR], edgeSegmentId: refs.segs[refs.frontR][2], heightM: 0.07, name: "collar",
    }, [refs.segs[refs.frontR][2]]);
    components = collar.set;
    const shaped = deriveConstruction(base, features);
    const built = deriveComponents(shaped.document, components);
    const bandId = built.set.components[0].derived.panels[0];
    const widthBefore = Math.max(...built.document.points.filter((p) => p.panelId === bandId).map((p) => p.x));

    // Upstream edit: lengthen the neckline edge.
    const topSeg = base.segments.find((s) => s.id === refs.segs[refs.frontR][2])!;
    let edited = { ...base, points: base.points.map((p) => ({ ...p })) };
    edited.points.find((p) => p.id === topSeg.startPointId)!.x += 0.05;

    // Re-derive everything from the edited base: no manual reconstruction.
    const reShaped = deriveConstruction(edited, features);
    expect(reShaped.failed).toEqual([]);
    const reBuilt = deriveComponents(reShaped.document, built.set);
    expect(reBuilt.failed).toEqual([]);
    expect(reBuilt.set.components[0].derived.panels[0]).toBe(bandId);
    const widthAfter = Math.max(...reBuilt.document.points.filter((p) => p.panelId === bandId).map((p) => p.x));
    expect(widthAfter).toBeGreaterThan(widthBefore);
    // Seams still resolve; simulation stays valid.
    const seams = shirtSeams(refs, [...reShaped.seams, ...reBuilt.seams]);
    expect(validateSeams(reBuilt.document, seams).valid).toBe(true);
    const collarPanel = reBuilt.set.components[0].derived.panels[0];
    const assembled = assembleGarment(reBuilt.document, seams, [
      { panelId: refs.frontL, translation: [0, 0.6, 0.2] as [number, number, number], yawRad: 0 },
      { panelId: refs.frontR, translation: [0.23, 0.6, 0.2] as [number, number, number], yawRad: 0 },
      { panelId: refs.back, translation: [1.06, 0.6, -0.2] as [number, number, number], yawRad: Math.PI },
      { panelId: refs.sleeveL, translation: [-0.06, 0.95, 0.65] as [number, number, number], yawRad: Math.PI / 2 },
      { panelId: refs.sleeveR, translation: [0.52, 0.95, -0.95] as [number, number, number], yawRad: -Math.PI / 2 },
      { panelId: collarPanel, translation: [0.1, 1.45, 0] as [number, number, number], yawRad: 0 },
    ]);
    expect(assembled.diagnostics.filter((d) => d.code === "failed-seam-reference")).toEqual([]);
    const fitting = createFittingScene(assembled, {});
    const solver = new CpuSolver();
    const fit = runFitting(assembled, fitting, solver, null, { relaxationSteps: 1, simulationSteps: 1 });
    expect(fit.hasNaNInf).toBe(false);
  });
});
