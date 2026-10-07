// G15D tests: presets, framing, rigs, scenes, capture plans, serialization.
import { describe, expect, it } from "vitest";
import {
  CAMERA_PRESETS,
  createScene,
  deserializeScene,
  frameCamera,
  LIGHTING_PRESETS,
  lightingPreset,
  planCaptures,
  serializeScene,
} from "../../src/render/presentation.js";
import { computeBounds } from "../../src/view/types.js";
import { DEFAULT_CAMERA_STATE } from "../../src/view/camera.js";

describe("G15D presentation", () => {
  it("frames bounds at every preset", () => {
    const bounds = computeBounds(new Float32Array([0, 0.6, -0.2, 0.46, 1.22, 0.2]));
    for (const key of ["front", "back", "left", "right", "three-quarter", "top"] as const) {
      const state = frameCamera(bounds, CAMERA_PRESETS[key]);
      expect(state.target[0]).toBeCloseTo(0.23, 6);
      expect(state.distanceM).toBeGreaterThan(0.3);
      expect(state.distanceM).toBeLessThan(200);
    }
    const threeQuarter = frameCamera(bounds, CAMERA_PRESETS["three-quarter"]);
    expect(threeQuarter.yawRad).toBeCloseTo(Math.PI / 4, 12);
    expect(() => frameCamera(bounds, CAMERA_PRESETS.front, 0)).toThrowError(/field of view/);
  });

  it("ships data-driven lighting rigs", () => {
    for (const name of ["studio", "soft", "product", "technical", "neutral"]) {
      expect(LIGHTING_PRESETS[name].name.length).toBeGreaterThan(0);
      const rig = lightingPreset(name);
      expect(rig).toEqual(LIGHTING_PRESETS[name]);
      expect(rig).not.toBe(LIGHTING_PRESETS[name]); // cloned, not aliased
      for (const light of [rig.key, rig.fill, rig.rim]) {
        const len = Math.hypot(light.direction[0], light.direction[1], light.direction[2]);
        expect(len).toBeCloseTo(1, 12);
      }
    }
    expect(() => lightingPreset("noir")).toThrowError(/lighting preset/);
  });

  it("creates scenes referencing (never duplicating) garments", () => {
    const scene = createScene({ id: "scene/1", name: "Hero", garmentId: "garment/tshirt" });
    expect(scene.garmentId).toBe("garment/tshirt");
    expect(scene.camera).toEqual(DEFAULT_CAMERA_STATE);
    expect(scene.lights.name).toBe("Studio");
    expect(scene.background).toEqual([0.96, 0.96, 0.97]);
    expect(scene.materialOverrides).toEqual({});
    const custom = createScene({
      id: "s", name: "N", garmentId: "g", background: null,
      showAvatar: false, renderMode: "wireframe", quality: "high",
      materialOverrides: { p: "mat/silk" },
    });
    expect(custom.background).toBeNull();
    expect(() => createScene({ id: "", name: "N", garmentId: "g" })).toThrowError(/id, name/);
  });

  it("plans deterministic commercial captures", () => {
    const scene = createScene({ id: "scene/1", name: "Hero", garmentId: "g" });
    const plans = planCaptures(scene);
    expect(plans.map((p) => p.view)).toEqual(["front", "back", "three-quarter"]);
    expect(plans.map((p) => p.id)).toEqual([
      "scene/1/capture/front", "scene/1/capture/back", "scene/1/capture/three-quarter",
    ]);
    expect(plans[0].camera.yawRad).toBe(0);
    expect(plans[1].camera.yawRad).toBeCloseTo(Math.PI, 12);
    expect(plans.every((p) => p.widthPx === 1024 && !p.transparent && !p.garmentOnly)).toBe(true);
    const ghost = createScene({ id: "s", name: "N", garmentId: "g", background: null, showAvatar: false });
    const solo = planCaptures(ghost, ["front"], { widthPx: 512, heightPx: 256 });
    expect(solo[0].transparent).toBe(true);
    expect(solo[0].garmentOnly).toBe(true);
    expect(solo[0].widthPx).toBe(512);
    expect(planCaptures(scene)).toEqual(plans); // deterministic
    expect(() => planCaptures(scene, [])).toThrowError(/views/);
  });

  it("round-trips scenes", () => {
    const scene = createScene({ id: "scene/1", name: "Hero", garmentId: "g" });
    expect(deserializeScene(serializeScene(scene))).toEqual(scene);
    expect(() => deserializeScene("nope")).toThrowError(/JSON/);
    expect(() => deserializeScene('{"id":"x"}')).toThrowError(/shape/);
  });
});
