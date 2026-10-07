// three.js adapter tests (scene graph only — no GL context required).
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  applyLightRig,
  applyOrbitCamera,
  buildRenderObjects,
  disposeRenderObjects,
  updateRenderObjects,
  visualToStandard,
} from "../../src/render/three-adapter.js";
import {
  buildRenderGarment,
  defaultVisibility,
  qualitySettings,
} from "../../src/render/representation.js";
import { createMaterial, materialFromPreset } from "../../src/render/materials.js";
import { lightingPreset } from "../../src/render/presentation.js";
import { DEFAULT_CAMERA_STATE } from "../../src/view/camera.js";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { assembleGarment } from "../../src/garment/assembly.js";
import { makeBoxAvatar } from "../../src/garment/avatar.js";
import { buildRenderAvatar } from "../../src/render/representation.js";

function garment() {
  let document = createPatternDocument("ad", "Adapter");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const b = addRectPanel(document, "B", [0.5, 0], 0.2, 0.2);
  document = b.document;
  const seam = {
    id: "seam/ab",
    sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
    sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: false },
    stitchCount: 3,
  };
  const g = assembleGarment(document, [seam], [
    { panelId: a.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
    { panelId: b.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
  ]);
  const render = buildRenderGarment("g", g, g.positions, { [a.refs.panelId]: "mat/a", [b.refs.panelId]: "mat/b" });
  const mats = {
    [a.refs.panelId]: materialFromPreset("mat/a", "A", "cotton"),
    [b.refs.panelId]: materialFromPreset("mat/b", "B", "denim", { colorMap: { kind: "file", ref: "tex/denim.png" } }),
  };
  return { assembled: g, render, mats };
}

describe("three.js adapter", () => {
  it("converts visual materials without physics leakage", () => {
    const visual = createMaterial({
      id: "m", name: "M", baseColor: [0.2, 0.4, 0.6],
      roughness: 0.5, metallic: 0.1, opacity: 0.9, physicalRef: "cotton",
    });
    const three = visualToStandard(visual);
    expect(three.roughness).toBe(0.5);
    expect((three.color as THREE.Color).b).toBeCloseTo(0.6, 6);
    expect(three.userData.renderMaterialId).toBe("m");
    // No physics fields cross the boundary.
    expect("physicalRef" in three.userData).toBe(false);
    expect((three as unknown as Record<string, unknown>).friction).toBeUndefined();
    three.dispose();
  });

  it("builds grouped meshes, overlays, and lights", () => {
    const { assembled, render, mats } = garment();
    const avatar = buildRenderAvatar(makeBoxAvatar({ halfExtentsM: [0.2, 0.3, 0.15], center: [0.2, 0.5, 0] }));
    const objects = buildRenderObjects(render, avatar, assembled, {
      visibility: { ...defaultVisibility(), wireframe: true },
      quality: qualitySettings("preview"),
      materials: mats,
    });
    // One material group per panel.
    expect(objects.garmentGeometry.groups).toHaveLength(2);
    expect(Object.keys(objects.materialByPanel)).toHaveLength(2);
    expect(objects.garment.material).toHaveLength(2);
    // UV attribute covers every vertex exactly once.
    expect((objects.garmentGeometry.getAttribute("uv") as THREE.BufferAttribute).count)
      .toBe(render.positions.length / 3);
    expect(objects.avatar).not.toBeNull();
    expect(objects.wireframe).not.toBeNull();
    expect(objects.boundary).not.toBeNull(); // preview enables boundary overlay
    expect(objects.seamGroup.children.length).toBeGreaterThan(0);
    // Texture needing browser load is queued, not fetched.
    expect(objects.textureQueue).toEqual(["tex/denim.png"]);
    applyLightRig(objects, lightingPreset("studio"));
    expect(objects.lightGroup.children.length).toBe(4); // key + fill + rim + ambient
    const counts = disposeRenderObjects(objects);
    expect(objects.disposed).toBe(true);
    expect(counts.geometries).toBeGreaterThanOrEqual(4);
    expect(counts.materials).toBeGreaterThanOrEqual(4);
    expect(objects.scene.children).toHaveLength(0);
  });

  it("updates positions in place and refuses topology changes", () => {
    const { assembled, render, mats } = garment();
    const objects = buildRenderObjects(render, null, assembled, {
      visibility: defaultVisibility(),
      quality: qualitySettings("draft"),
      materials: mats,
    });
    expect(objects.wireframe).toBeNull(); // draft disables overlay
    expect(objects.boundary).toBeNull();
    const moved = Float32Array.from(render.positions);
    moved[1] += 0.1;
    const next = buildRenderGarment("g", assembled, moved, {});
    updateRenderObjects(objects, next);
    const attr = objects.garmentGeometry.getAttribute("position") as THREE.BufferAttribute;
    expect(attr.array[1]).toBeCloseTo(0.1, 6);
    expect(() => updateRenderObjects(objects, { ...next, positions: new Float32Array(3) })).toThrowError(/vertex count/);
    disposeRenderObjects(objects);
    expect(() => updateRenderObjects(objects, next)).toThrowError(/disposed/);
  });

  it("positions cameras from orbit state", () => {
    const perspective = new THREE.PerspectiveCamera();
    applyOrbitCamera(perspective, { ...DEFAULT_CAMERA_STATE, target: [0, 0.9, 0], yawRad: 0, pitchRad: 0, distanceM: 2 }, 1);
    expect(perspective.position.toArray()).toEqual([0, 0.9, 2]);
    const ortho = new THREE.OrthographicCamera();
    applyOrbitCamera(ortho, { ...DEFAULT_CAMERA_STATE, orthoHeightM: 2 }, 2);
    expect(ortho.right).toBeCloseTo(2, 12);
    expect(ortho.top).toBeCloseTo(1, 12);
  });

  it("requires resolved materials per panel", () => {
    const { assembled, render, mats } = garment();
    const missing = { ...mats };
    delete (missing as Record<string, unknown>)[Object.keys(missing)[0]];
    expect(() => buildRenderObjects(render, null, assembled, {
      visibility: defaultVisibility(), quality: qualitySettings("draft"), materials: missing,
    })).toThrowError(/no material resolved/);
  });
});
