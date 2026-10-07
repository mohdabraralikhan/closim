// G15 FINAL INTEGRATION — commercial visualization acceptance.
//
// Graded/production T-shirt -> visual materials -> 3D garment -> avatar fit
// -> simulate -> camera/lighting choice -> preview -> front/back/
// three-quarter/garment-only captures -> saved presentation setup.
// Rendering stays downstream: visual edits never touch physics or pattern.
import { describe, expect, it } from "vitest";
import {
  assignMaterial,
  addMaterial,
  createLibrary,
  materialFromPreset,
  replaceMaterial,
  serializeLibrary,
} from "../../src/render/materials.js";
import { generatePanelUVs } from "../../src/render/uvgen.js";
import {
  buildRenderAvatar,
  buildRenderGarment,
  defaultVisibility,
  qualitySettings,
  syncRenderPositions,
} from "../../src/render/representation.js";
import {
  buildRenderObjects,
  disposeRenderObjects,
} from "../../src/render/three-adapter.js";
import {
  createScene,
  frameCamera,
  lightingPreset,
  planCaptures,
} from "../../src/render/presentation.js";
import {
  createRenderPackage,
  deserializeRenderPackage,
  serializeRenderPackage,
} from "../../src/render/package.js";
import { computeBounds } from "../../src/view/types.js";
import { buildTshirtProject } from "../../src/garment/tshirt.js";
import { assembleGarment, createFittingScene, runFitting } from "../../src/garment/assembly.js";
import { rebuildGarment } from "../../src/garment/project.js";
import { serializePatternDocument } from "../../src/pattern/cad.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";

describe("G15 final integration — T-shirt visualization", () => {
  it("renders a simulated garment into saved presentation captures", () => {
    const { project, refs } = buildTshirtProject();
    const patternBefore = serializePatternDocument(project.pattern);

    // 1-2. open 3D garment + select presentation materials.
    const { assembled, fitting } = rebuildGarment(project);
    let library = createLibrary();
    const cotton = materialFromPreset("mat/cotton", "Cotton", "cotton", { physicalRef: "cotton" });
    library = addMaterial(library, cotton);
    library = assignMaterial(library, refs.front.panelId, "mat/cotton");
    library = assignMaterial(library, refs.back.panelId, "mat/cotton");
    const silk = materialFromPreset("mat/silk", "Silk", "silk");
    library = addMaterial(library, silk);
    library = assignMaterial(library, refs.sleeveL.panelId, "mat/silk");
    library = assignMaterial(library, refs.sleeveR.panelId, "mat/silk");

    // 3-4. fit on avatar + simulate, then sync render state explicitly.
    const solver = new CpuSolver();
    const fit = runFitting(assembled, fitting, solver, project.avatar, { relaxationSteps: 2, simulationSteps: 3 });
    expect(fit.ok).toBe(true);
    const materialOf: Record<string, string> = {};
    for (const panelId of [refs.front.panelId, refs.back.panelId, refs.sleeveL.panelId, refs.sleeveR.panelId]) {
      materialOf[panelId] = library.assignment[panelId];
    }
    const render = buildRenderGarment("garment/tshirt", assembled, assembled.positions, materialOf);
    expect(syncRenderPositions(render, assembled, Float32Array.from(solver.getPositions()), 5)).toBe("updated");
    // UVs are pattern-derived and stable across the simulated deformation.
    for (const panelId of Object.keys(materialOf)) {
      expect(generatePanelUVs(assembled, panelId).uv.length).toBe(
        render.panelUVs.find((p) => p.panelId === panelId)!.uv.length,
      );
    }

    // 5-7. camera + lighting + preview objects (headless scene graph).
    const bounds = computeBounds(render.positions);
    const scene = createScene({
      id: "scene/hero", name: "Hero", garmentId: "garment/tshirt",
      camera: frameCamera(bounds, { name: "Three-quarter", view: "three-quarter", yawRad: Math.PI / 4, pitchRad: 0.18, frameMargin: 1.35 }),
      cameraView: "three-quarter",
      lights: lightingPreset("studio"),
      quality: "preview",
    });
    const resolved = Object.fromEntries(
      Object.entries(materialOf).map(([panel, id]) => [panel, library.materials.find((m) => m.id === id)!]),
    );
    const avatar = project.avatar ? buildRenderAvatar(project.avatar) : null;
    const objects = buildRenderObjects(render, avatar, assembled, {
      visibility: defaultVisibility(), quality: qualitySettings(scene.quality), materials: resolved,
    });
    expect(objects.scene.children.length).toBeGreaterThanOrEqual(4); // garment + avatar + seams + boundary + lights
    expect(objects.textureQueue).toEqual([]);

    // 8-11. capture plans for the commercial set (pixels execute in-browser).
    const plans = planCaptures(scene, ["front", "back", "three-quarter"]);
    expect(plans).toHaveLength(3);
    const ghostScene = createScene({ id: "scene/ghost", name: "Ghost", garmentId: "garment/tshirt", showAvatar: false });
    const solo = planCaptures(ghostScene, ["front"]);
    expect(solo[0].garmentOnly).toBe(true);

    // Visual edits stay visual: new roughness, same physics + pattern.
    const physicsBefore = Array.from(solver.getPositions());
    library = replaceMaterial(library, { ...cotton, roughness: 0.6 });
    expect(Array.from(solver.getPositions())).toEqual(physicsBefore);
    expect(serializePatternDocument(project.pattern)).toBe(patternBefore);
    expect(serializeLibrary(library)).toContain('"version":2');

    // 12. save presentation setup (renders/ manifest, separate from patterns/).
    // One package per scene: mixed-scene captures are refused by construction.
    const pkg = createRenderPackage({
      id: "renders/tshirt-hero",
      scene,
      materials: library.materials,
      captures: plans,
      simEpoch: render.epoch,
    });
    expect(pkg.source).toEqual({ garmentId: "garment/tshirt", simEpoch: 5 });
    expect(deserializeRenderPackage(serializeRenderPackage(pkg))).toEqual(pkg);
    expect(() => createRenderPackage({
      id: "renders/bad", scene, materials: library.materials, captures: [...plans, ...solo],
    })).toThrowError(/another scene/);
    const ghostPkg = createRenderPackage({
      id: "renders/tshirt-ghost", scene: ghostScene, materials: library.materials, captures: solo,
    });
    expect(ghostPkg.captures[0].garmentOnly).toBe(true);
    disposeRenderObjects(objects);
  });
});
