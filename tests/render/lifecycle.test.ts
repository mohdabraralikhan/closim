// G15E — rendering adversarial QA (data-level, no pixels needed).
// Attacks lifecycle, materials, cameras, rebuilds, and restarts; every case
// records expected behavior inline. Severity of a data-corruption failure
// here is critical; pixel assertions belong to the browser path.
import { describe, expect, it } from "vitest";
import {
  buildRenderObjects,
  disposeRenderObjects,
  updateRenderObjects,
} from "../../src/render/three-adapter.js";
import {
  buildRenderGarment,
  defaultVisibility,
  qualitySettings,
  syncRenderPositions,
} from "../../src/render/representation.js";
import {
  assignMaterial,
  addMaterial,
  createLibrary,
  materialFromPreset,
  replaceMaterial,
  resolveMaterial,
  serializeLibrary,
} from "../../src/render/materials.js";
import {
  createScene,
  frameCamera,
  planCaptures,
} from "../../src/render/presentation.js";
import { computeBounds } from "../../src/view/types.js";
import { createPatternDocument, serializePatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { assembleGarment, createFittingScene, runFitting } from "../../src/garment/assembly.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";

function garment() {
  let document = createPatternDocument("qa", "QA");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const g = assembleGarment(document, [], [{ panelId: a.refs.panelId, translation: [0, 0, 0], yawRad: 0 }]);
  return { document, garment: g, panelId: a.refs.panelId };
}

describe("G15E lifecycle", () => {
  it("load → render → unload → reload reproduces state without leaks", () => {
    const { garment: g, panelId } = garment();
    const mats = { [panelId]: materialFromPreset("mat/c", "C", "cotton") };
    const first = buildRenderGarment("g", g, g.positions, { [panelId]: "mat/c" });
    const o1 = buildRenderObjects(first, null, g, {
      visibility: defaultVisibility(), quality: qualitySettings("preview"), materials: mats,
    });
    const before = Array.from((o1.garmentGeometry.getAttribute("position") as { array: Float32Array }).array);
    const counts = disposeRenderObjects(o1);
    expect(counts.geometries).toBeGreaterThan(0);
    const second = buildRenderGarment("g", g, g.positions, { [panelId]: "mat/c" });
    const o2 = buildRenderObjects(second, null, g, {
      visibility: defaultVisibility(), quality: qualitySettings("preview"), materials: mats,
    });
    const after = Array.from((o2.garmentGeometry.getAttribute("position") as { array: Float32Array }).array);
    expect(after).toEqual(before);
    disposeRenderObjects(o2);
  });

  it("load → simulate → render advances epochs without corrupting sources", () => {
    const { document, garment: g, panelId } = garment();
    const patternBefore = serializePatternDocument(document);
    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 0.4, 0.5] });
    const fitting = createFittingScene(g, { avatar });
    const solver = new CpuSolver();
    const render = buildRenderGarment("g", g, g.positions, {});
    for (let epoch = 1; epoch <= 3; epoch++) {
      solver.initialize(fitting.scene);
      solver.step(1 / 60);
      const positions = Float32Array.from(solver.getPositions());
      expect(syncRenderPositions(render, g, positions, epoch)).toBe("updated");
      expect(render.epoch).toBe(epoch);
    }
    expect(serializePatternDocument(document)).toBe(patternBefore);
    expect(g.positions.length).toBe(render.positions.length);
  });

  it("render → rebuild → render resynchronizes across topology change", () => {
    const { garment: g } = garment();
    const render = buildRenderGarment("g", g, g.positions, {});
    let document = createPatternDocument("qa2", "QA2");
    const r = addRectPanel(document, "B", [0, 0], 1, 1);
    document = r.document;
    const g2 = assembleGarment(document, [], [{ panelId: r.refs.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    expect(syncRenderPositions(render, g2, g2.positions, 1)).toBe("topology-changed");
    const rebuilt = buildRenderGarment("g", g2, g2.positions, {});
    expect(syncRenderPositions(rebuilt, g2, g2.positions, 1)).toBe("updated");
  });

  it("render → change material → render picks up the new version", () => {
    const { garment: g, panelId } = garment();
    let lib = createLibrary();
    const cotton = materialFromPreset("mat/c", "C", "cotton");
    lib = addMaterial(lib, cotton);
    const render = buildRenderGarment("g", g, g.positions, { [panelId]: "mat/c" });
    const o1 = buildRenderObjects(render, null, g, {
      visibility: defaultVisibility(), quality: qualitySettings("draft"),
      materials: { [panelId]: cotton },
    });
    expect(o1.materialByPanel[panelId].userData.renderMaterialVersion).toBe(1);
    lib = replaceMaterial(lib, { ...cotton, roughness: 0.5 });
    lib = assignMaterial(lib, panelId, "mat/c");
    const updated = lib.materials.find((m) => m.id === "mat/c")!;
    const o2 = buildRenderObjects(render, null, g, {
      visibility: defaultVisibility(), quality: qualitySettings("draft"),
      materials: { [panelId]: updated },
    });
    expect(o2.materialByPanel[panelId].userData.renderMaterialVersion).toBe(2);
    expect(o2.materialByPanel[panelId].roughness).toBeCloseTo(0.5, 9);
    // Saved library round-trips with the new version.
    expect(serializeLibrary(lib)).toContain('"version":2');
    disposeRenderObjects(o1);
    disposeRenderObjects(o2);
  });
});

describe("G15E adversarial rendering", () => {
  it("extreme cameras, hidden subjects, and transparency plan deterministically", () => {
    const { garment: g } = garment();
    const bounds = computeBounds(g.positions);
    const far = frameCamera(bounds, { name: "X", view: "front", yawRad: 0, pitchRad: 0, frameMargin: 100 });
    expect(far.distanceM).toBeLessThanOrEqual(200);
    const hidden = createScene({ id: "s", name: "S", garmentId: "g", showAvatar: false, showGarment: false, background: null });
    const plans = planCaptures(hidden, ["front", "three-quarter"]);
    expect(plans.every((p) => p.transparent && p.garmentOnly)).toBe(true);
    const render = buildRenderGarment("g", g, g.positions, {});
    const mats = { [render.panelRanges[0].panelId]: materialFromPreset("m", "M", "wool") };
    const objects = buildRenderObjects(render, null, g, {
      visibility: { ...defaultVisibility(), garment: false, avatar: false },
      quality: qualitySettings("draft"),
      materials: mats,
    });
    expect(objects.garment.visible).toBe(false);
    disposeRenderObjects(objects);
  });

  it("missing textures and materials fall back explicitly", () => {
    const lib = createLibrary();
    const miss = resolveMaterial(lib, "ghost");
    expect(miss.missed).toBe(true);
    const textured = materialFromPreset("mat/t", "T", "denim", { colorMap: { kind: "file", ref: "tex/missing.png" } });
    const { garment: g, panelId } = garment();
    const render = buildRenderGarment("g", g, g.positions, {});
    const objects = buildRenderObjects(render, null, g, {
      visibility: defaultVisibility(), quality: qualitySettings("draft"),
      materials: { [panelId]: textured },
    });
    // Queued for browser load — never fetched headlessly, never blocking.
    expect(objects.textureQueue).toEqual(["tex/missing.png"]);
    disposeRenderObjects(objects);
  });

  it("repeated simulation restarts stay in sync", () => {
    const { garment: g } = garment();
    const render = buildRenderGarment("g", g, g.positions, {});
    for (let epoch = 1; epoch <= 5; epoch++) {
      const drifted = Float32Array.from(g.positions, (v, i) => (i % 3 === 1 ? v + epoch * 0.01 : v));
      expect(syncRenderPositions(render, g, drifted, epoch)).toBe("updated");
    }
    expect(render.epoch).toBe(5);
    expect(render.positions[1]).toBeCloseTo(g.positions[1] + 0.05, 6);
  });

  it("visual edits never reach physics or pattern state", () => {
    const { document, garment: g, panelId } = garment();
    const patternBefore = serializePatternDocument(document);
    const positionsBefore = Array.from(g.positions);
    const render = buildRenderGarment("g", g, g.positions, { [panelId]: "mat/c" });
    render.positions.fill(12345);
    render.normals.fill(0);
    expect(Array.from(g.positions)).toEqual(positionsBefore);
    expect(serializePatternDocument(document)).toBe(patternBefore);
  });
});
