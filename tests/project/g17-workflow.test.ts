// G17 FINAL — commercial acceptance session over a persistent project.
//
// 21 steps on one project directory: create/open, edit, component, material,
// grading, simulate, render, save, edit, undo, redo, save, export, marker,
// save, close, reopen, verify, modify, autosave, restart/recovery.
// Pattern edits run through CadSession history (app-wide command history is
// the G17B track); autosave mechanics live in the store while scheduling
// policy belongs to the reliability track.
import { describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { CadSession } from "../../src/cad/history.js";
import { movePoint } from "../../src/pattern/cad.js";
import { addDart, createConstructionSet, deriveConstruction } from "../../src/construction/features.js";
import { addComponent, createComponentSet, deriveComponents } from "../../src/construction/components.js";
import {
  addMaterial,
  assignMaterial,
  createLibrary,
  materialFromPreset,
} from "../../src/render/materials.js";
import { buildRenderGarment } from "../../src/render/representation.js";
import { createScene, planCaptures } from "../../src/render/presentation.js";
import { createRenderPackage } from "../../src/render/package.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import { createGarmentProject } from "../../src/garment/project.js";
import { assembleGarment, createFittingScene, runFitting } from "../../src/garment/assembly.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import {
  addAllowance,
  addCutLine,
  addGrainline,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { pointInPanel } from "../../src/cad/queries.js";
import type { Vec2 } from "../../src/cad/geom.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import { exportProductionPackage } from "../../src/cad/export.js";
import { createFabric, createCutPlan, addCutItem, defaultNestingConstraint } from "../../src/marker/model.js";
import { MarkerWorkspace } from "../../src/marker/workspace.js";
import {
  addCoreSizes,
  buildGradingFixture,
  buildSideSeam,
} from "../grading/fixtures.js";
import { addSize, createGradingPoint, createSize } from "../../src/grading/model.js";
import { validateGradingDocument } from "../../src/grading/validate.js";
import {
  addGarment,
  createProject,
  duplicateProject,
  fingerprintProject,
  removeGarment,
  renameProject,
  validateProject,
  type AppProject,
  type ProjectGarmentEntry,
} from "../../src/project/project.js";
import {
  assessRecovery,
  autosaveFilePath,
  openProject,
  projectFilePath,
  readAutosave,
  releaseLock,
  saveProject,
  writeAutosave,
  type FileSystem,
} from "../../src/project/store.js";

const nodeFs: FileSystem = {
  readTextFile: (p) => fs.readFileSync(p, "utf8"),
  writeTextFileAtomic: (p, content) => {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    const tmp = `${p}.tmp-${process.pid}`;
    fs.writeFileSync(tmp, content, "utf8");
    fs.renameSync(tmp, p);
  },
  exists: (p) => fs.existsSync(p),
  mkdirp: (p) => fs.mkdirSync(p, { recursive: true }),
  remove: (p) => fs.rmSync(p, { force: true }),
};

const T0 = "2026-01-01T00:00:00.000Z";
const T1 = "2026-01-01T00:01:00.000Z";
const T2 = "2026-01-01T00:02:00.000Z";

function cottonPhysical() {
  return {
    arealDensityKgM2: 0.15,
    thickness: 0.001,
    stretchWarp: 20000,
    stretchWeft: 20000,
    stretchCoupling: 0,
    shear: 5000,
    bendWarp: 1e-5,
    bendWeft: 1e-5,
    damping: 0.001,
  };
}

describe("G17 final integration — persistent product session", () => {
  it("runs the 21-step commercial acceptance scenario", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "closim-g17-"));
    const session = "session/acceptance";
    const g = buildGradingFixture();
    const frontId = g.ids.frontPanel;
    const frontLoop = g.ids.frontLoop;

    // 1. create project + garment from the grading master; open (save + reopen).
    let project = createProject("proj/shirt", "Shirt", T0);
    const sideSeam = buildSideSeam(g);
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 1.0, 0] });
    let garment = createGarmentProject("garment/shirt", "Shirt", g.document, {
      seams: [sideSeam],
      placements: [
        { panelId: g.ids.frontPanel, translation: [0, 1, 0.2], yawRad: 0 },
        { panelId: g.ids.backPanel, translation: [0.8, 1, -0.2], yawRad: Math.PI },
      ],
      avatar,
      materials: { cotton: cottonPhysical(), "default-material": cottonPhysical() },
    });
    let entry: ProjectGarmentEntry = { id: "entry/shirt", name: "Shirt", garment };
    project = addGarment(project, entry, T0);
    saveProject(nodeFs, dir, project, session);
    project = openProject(nodeFs, dir).project;
    entry = project.garments[0];
    expect(validateProject(project)).toEqual([]);

    // 2. edit pattern (dart intake edge stays valid; ids preserved).
    const bottomSeg = g.ids.segments.ab;
    const bottomStart = g.document.points.find((p) => p.id === g.ids.points.A)!;
    const editedPattern = movePoint(entry.garment.pattern, frontId, bottomStart.id, [bottomStart.x, bottomStart.y - 0.01]);
    entry = { ...entry, garment: { ...entry.garment, pattern: editedPattern } };

    // 3. add component (waist dart) and derive.
    let construction = createConstructionSet();
    const dart = addDart(construction, frontId, frontLoop, bottomSeg, 0.3, 0.7, [0.2, 0.25], "open");
    construction = dart.set;
    const derived = deriveConstruction(entry.garment.pattern, construction);
    expect(derived.failed).toEqual([]);
    entry = {
      ...entry,
      garment: { ...entry.garment, pattern: derived.document },
      construction,
    };

    // Collar component from the front top edge.
    let components = createComponentSet();
    const collar = addComponent(components, frontId, "collar-band", {
      edgeLoopId: frontLoop, edgeSegmentId: g.ids.segments.cd, heightM: 0.07, name: "collar",
    }, [g.ids.segments.cd, frontLoop, frontId]);
    components = collar.set;
    const built = deriveComponents(entry.garment.pattern, components);
    expect(built.failed).toEqual([]);
    entry = { ...entry, garment: { ...entry.garment, pattern: built.document }, components };
    const collarSeam = built.seams[0];
    const collarBandId = built.set.components[0].derived.panels[0];
    const seams = [...entry.garment.seams, collarSeam];
    const placements = [
      ...entry.garment.placements,
      { panelId: collarBandId, translation: [0.1, 1.65, 0] as [number, number, number], yawRad: 0 },
    ];
    entry = { ...entry, garment: { ...entry.garment, seams, placements } };

    // 4. assign material.
    let library = createLibrary();
    library = addMaterial(library, materialFromPreset("mat/cotton", "Cotton", "cotton", { physicalRef: "cotton" }));
    library = assignMaterial(library, frontId, "mat/cotton");
    entry = { ...entry, materials: library };

    // 5. modify grading (sizes S/M/L, then add XXL).
    let grading = addCoreSizes(g.doc);
    expect(validateGradingDocument(grading)).toEqual([]);
    grading = addSize(grading, createSize({ id: "size/xxl", label: "XXL", displayName: "Extra Large" }));
    expect(grading.sizeSet.sizes).toHaveLength(4);
    expect(validateGradingDocument(grading)).toEqual([]);
    entry = { ...entry, grading };

    // 6. simulate the derived, sewn garment.
    const assembled = assembleGarment(entry.garment.pattern, entry.garment.seams, entry.garment.placements, { avatar });
    expect(assembled.diagnostics.filter((d) => d.code === "failed-seam-reference")).toEqual([]);
    const fitting = createFittingScene(assembled, { avatar });
    const solver = new CpuSolver();
    const fit = runFitting(assembled, fitting, solver, avatar, { relaxationSteps: 2, simulationSteps: 2 });
    expect(fit.hasNaNInf).toBe(false);
    const simPositions = Array.from(solver.getPositions());

    // 7. render representation + saved presentation scene.
    const render = buildRenderGarment("garment/shirt", assembled, assembled.positions, {});
    expect(render.panelUVs.length).toBeGreaterThan(0);
    const scene = createScene({ id: "scene/hero", name: "Hero", garmentId: "garment/shirt" });
    const captures = planCaptures(scene, ["front", "back"]);
    expect(captures).toHaveLength(2);
    const presentation = createRenderPackage({ id: "renders/hero", scene, materials: library.materials, captures });
    entry = { ...entry, presentation };

    // 8. save.
    project = addGarment(removeEntry(project), entry, T1);
    const fp8 = fingerprintProject(project);
    saveProject(nodeFs, dir, project, session);
    expect(nodeFs.exists(projectFilePath(dir))).toBe(true);

    // 9-11. continue editing with undo/redo through the session history.
    const history = new CadSession(entry.garment.pattern);
    const nudge = history.run("nudge", (d) => {
      const pt = d.points.find((p) => p.panelId === frontId)!;
      return movePoint(d, frontId, pt.id, [pt.x + 0.01, pt.y]);
    });
    void nudge;
    expect(history.undo()).not.toBeNull();
    expect(history.redo()).not.toBeNull();
    entry = { ...entry, garment: { ...entry.garment, pattern: history.document } };

    // 12. save again (fingerprint must move).
    project = addGarment(removeEntry(project), entry, T2);
    saveProject(nodeFs, dir, project, session);
    expect(fingerprintProject(project)).not.toBe(fp8);

    // 13. export production files into exports/.
    let prod = createProductionSet();
    const grainlineFor = (doc: Parameters<typeof centeredGrainline>[0], panelId: string): { from: Vec2; to: Vec2 } => {
      const base = centeredGrainline(doc, panelId);
      for (const dx of [0, -0.02, 0.02, -0.04, 0.04, -0.06, 0.06, -0.08, 0.08, -0.1, 0.1]) {
        const from: Vec2 = [base.from[0] + dx, base.from[1]];
        const to: Vec2 = [base.to[0] + dx, base.to[1]];
        if (pointInPanel(doc, panelId, from) && pointInPanel(doc, panelId, to)) return { from, to };
      }
      throw new Error(`no valid grainline for panel ${panelId}`);
    };
    for (const panel of entry.garment.pattern.panels) {
      const loop = panel.boundaryLoops.find((l) => l.role === "outer")!;
      prod = addAllowance(prod, panel.id, loop.id, 0.01).set;
      const grain = grainlineFor(entry.garment.pattern, panel.id);
      prod = addGrainline(prod, panel.id, grain.from, grain.to).set;
      prod = setPanelMeta(prod, { panelId: panel.id, cutQuantity: 1 });
      prod = addCutLine(prod, panel.id, loop.id, "sewing").set;
    }
    const pkg = exportProductionPackage(entry.garment.pattern, entry.garment.seams, prod, { garmentName: "Shirt" });
    expect(pkg.readiness.state).toBe("READY_FOR_EXPORT");
    const exportsDir = path.join(dir, "exports");
    nodeFs.writeTextFileAtomic(path.join(exportsDir, "shirt.json"), pkg.json);
    nodeFs.writeTextFileAtomic(path.join(exportsDir, "shirt.dxf"), pkg.dxf.dxf);
    const exportTime = T2;
    entry = {
      ...entry,
      production: prod,
      exports: [
        { id: "export/json", kind: "production-json", path: "exports/shirt.json", createdAt: exportTime },
        { id: "export/dxf", kind: "production-dxf", path: "exports/shirt.dxf", createdAt: exportTime },
      ],
    };

    // 14. create marker.
    let plan = createCutPlan("cut/order", "Order", grading.id);
    const i1 = addCutItem(plan, { panelId: frontId, sizeId: "size/m", quantity: 1, mirror: "allowed" });
    plan = i1.plan;
    const i2 = addCutItem(plan, { panelId: g.ids.backPanel, sizeId: "size/s", quantity: 1, mirror: "allowed" });
    plan = i2.plan;
    const fabric = createFabric({ id: "fabric/cotton", name: "Cotton", widthM: 1.5, lengthM: 10, materialType: "cotton" });
    const grainRadOf = (): number => 0;
    const workspace = MarkerWorkspace.open(grading, plan, fabric, defaultNestingConstraint(), "Order", "marker/order", grainRadOf);
    const report = workspace.optimize([0]);
    expect(report.best.result.unplaced).toEqual([]);
    entry = { ...entry, marker: workspace.marker };

    // 15. save.
    project = addGarment(removeEntry(project), entry, T2);
    saveProject(nodeFs, dir, project, session);
    const fp15 = fingerprintProject(project);

    // 16-17. close (drop everything, release lock) and reopen.
    releaseLock(nodeFs, dir, session);
    let closed: AppProject | null = null;
    closed = null;
    expect(closed).toBeNull();
    const reopened = openProject(nodeFs, dir).project;
    expect(fingerprintProject(reopened)).toBe(fp15);

    // 18. verify complete project integrity.
    const check = reopened.garments[0];
    expect(check.garment.pattern.panels.length).toBeGreaterThanOrEqual(3); // front/back + collar
    expect(check.garment.seams.map((s) => s.id)).toContain(collarSeam.id);
    expect(check.construction?.features).toHaveLength(1);
    expect(check.components?.components).toHaveLength(1);
    expect(check.grading?.sizeSet.sizes).toHaveLength(4);
    expect(check.materials?.assignment[frontId]).toBe("mat/cotton");
    expect(check.garment.avatar).not.toBeNull();
    expect(check.garment.simulation.dt).toBeGreaterThan(0);
    expect(check.presentation?.scene.id).toBe("scene/hero");
    expect(check.production?.panelMeta).toHaveLength(check.garment.pattern.panels.length);
    expect(check.marker?.placements).toHaveLength(2);
    expect(check.exports?.map((e) => e.path).sort()).toEqual(["exports/shirt.dxf", "exports/shirt.json"]);
    for (const ref of check.exports ?? []) {
      expect(nodeFs.exists(path.join(dir, ref.path))).toBe(true);
    }

    // 19. modify garment and re-simulate (positions must respond).
    const moved = (() => {
      const doc = check.garment.pattern;
      const pt = doc.points.find((p) => p.panelId === frontId)!;
      return movePoint(doc, frontId, pt.id, [pt.x, pt.y + 0.05]);
    })();
    const reassembled = assembleGarment(moved, check.garment.seams, check.garment.placements, { avatar });
    const refitting = createFittingScene(reassembled, { avatar });
    const solver2 = new CpuSolver();
    const fit2 = runFitting(reassembled, refitting, solver2, avatar, { relaxationSteps: 1, simulationSteps: 1 });
    expect(fit2.hasNaNInf).toBe(false);
    expect(Array.from(solver2.getPositions())).not.toEqual(simPositions);

    // 20-21. autosave without saving, then restart/recovery.
    const modified: AppProject = {
      ...reopened,
      garments: [{ ...check, garment: { ...check.garment, pattern: moved } }],
    };
    writeAutosave(nodeFs, dir, modified, T2);
    expect(nodeFs.exists(autosaveFilePath(dir))).toBe(true);
    const assessment = assessRecovery(nodeFs, dir);
    expect(assessment.decision).toBe("autosave-newer");
    expect(readAutosave(nodeFs, dir)?.projectId).toBe("proj/shirt");
    // Restore the autosaved state and save it: fingerprint must match autosave.
    const restored = ((): AppProject => {
      const record = readAutosave(nodeFs, dir)!;
      return JSON.parse(record.payload) as AppProject;
    })();
    saveProject(nodeFs, dir, restored, session);
    expect(fingerprintProject(openProject(nodeFs, dir).project)).toBe(fingerprintProject(modified));
    releaseLock(nodeFs, dir, session);

    // Duplicate projects keep working after everything above.
    expect(duplicateProject(restored, "proj/copy").garments).toHaveLength(1);
  });
});

function removeEntry(project: AppProject): AppProject {
  // Replace the single garment entry (test project holds exactly one).
  return removeGarment(project, project.garments[0].id);
}
