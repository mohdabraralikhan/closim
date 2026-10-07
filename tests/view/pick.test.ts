import { describe, expect, it } from "vitest";
import { panelPointToWorld } from "../../src/garment/assembly.js";
import { OrbitCamera } from "../../src/view/camera.js";
import {
  pickAvatar,
  pickPanel,
  pickSeam,
  pickVertex,
  rayIntersectsBox,
  rayTriangleDistance,
  screenRay,
} from "../../src/view/pick.js";
import { buildTshirtFixture } from "./fixtures.js";

describe("ray primitives", () => {
  it("hits a triangle head-on and misses behind", () => {
    const t = rayTriangleDistance([0.2, 0.2, 1], [0, 0, -1], 0, 0, 0, 1, 0, 0, 0, 1, 0);
    expect(t).toBeCloseTo(1, 12);
    expect(rayTriangleDistance([0.2, 0.2, -1], [0, 0, -1], 0, 0, 0, 1, 0, 0, 0, 1, 0)).toBeNull();
    expect(rayTriangleDistance([0.2, 0.2, 1], [1, 0, 0], 0, 0, 0, 1, 0, 0, 0, 1, 0)).toBeNull();
  });

  it("box test accepts and rejects", () => {
    const ray = { origin: [0, 0, 5] as [number, number, number], dir: [0, 0, -1] as [number, number, number] };
    expect(rayIntersectsBox(ray, [-1, -1, -1], [1, 1, 1])).toBe(true);
    expect(rayIntersectsBox(ray, [2, 2, -1], [3, 3, 1])).toBe(false);
    const insideRay = { origin: [0, 0, 0] as [number, number, number], dir: [1, 0, 0] as [number, number, number] };
    expect(rayIntersectsBox(insideRay, [-1, -1, -1], [1, 1, 1])).toBe(true);
  });
});

describe("picking against the t-shirt assembly", () => {
  const fixture = buildTshirtFixture();
  const frontPanel = fixture.project.pattern.panels[0];
  const frontPlacement = fixture.project.placements.find((p) => p.panelId === frontPanel.id)!;
  const centerWorld = panelPointToWorld(fixture.project.pattern, frontPanel.id, [0.23, 0.31], frontPlacement);

  it("screenRay through the viewport center hits the front panel", () => {
    const cam = new OrbitCamera({ target: centerWorld, yawRad: 0, pitchRad: 0, distanceM: 2 });
    const vp = cam.viewProjectionMatrix(16 / 9);
    const ray = screenRay(vp, 0, 0);
    expect(ray.dir[2]).toBeLessThan(0);
    const hit = pickPanel(fixture.assembled, fixture.assembled.positions, ray);
    expect(hit).not.toBeNull();
    expect(hit!.panelId).toBe(frontPanel.id);
    expect(hit!.distance).toBeGreaterThan(0);
  });

  it("a ray aimed at empty space misses", () => {
    const ray = { origin: [5, 5, 5] as [number, number, number], dir: [0, 0, -1] as [number, number, number] };
    expect(pickPanel(fixture.assembled, fixture.assembled.positions, ray)).toBeNull();
    expect(pickVertex(fixture.assembled, fixture.assembled.positions, ray, 0.01)).toBeNull();
  });

  it("pickVertex returns the nearest vertex within radius", () => {
    const ray = {
      origin: [centerWorld[0], centerWorld[1], centerWorld[2] + 1] as [number, number, number],
      dir: [0, 0, -1] as [number, number, number],
    };
    const hit = pickVertex(fixture.assembled, fixture.assembled.positions, ray, 0.3);
    expect(hit).not.toBeNull();
    const v = hit!.vertex;
    const px = fixture.assembled.positions[v * 3];
    const py = fixture.assembled.positions[v * 3 + 1];
    expect(Math.hypot(px - centerWorld[0], py - centerWorld[1])).toBeLessThanOrEqual(0.3);
  });

  it("pickSeam finds a weld pair crossed by the ray", () => {
    const weld = fixture.assembled.weldPairs[0];
    const a = weld.vertexA * 3;
    const b = weld.vertexB * 3;
    const mid: [number, number, number] = [
      (fixture.assembled.positions[a] + fixture.assembled.positions[b]) / 2,
      (fixture.assembled.positions[a + 1] + fixture.assembled.positions[b + 1]) / 2,
      (fixture.assembled.positions[a + 2] + fixture.assembled.positions[b + 2]) / 2,
    ];
    const ray = { origin: [mid[0], mid[1], mid[2] + 2] as [number, number, number], dir: [0, 0, -1] as [number, number, number] };
    const hit = pickSeam(fixture.assembled, fixture.assembled.positions, ray, 0.05);
    expect(hit).not.toBeNull();
    expect(hit!.seamId).toBe(weld.seamId);
    expect(hit!.weldIndex).toBe(0);
  });

  it("pickAvatar hits the capsule torso", () => {
    const ray = { origin: [0.23, 0.95, 5] as [number, number, number], dir: [0, 0, -1] as [number, number, number] };
    const hit = pickAvatar(fixture.avatar, ray);
    expect(hit).not.toBeNull();
    expect(hit!.point[2]).toBeLessThan(5);
    expect(hit!.point[2]).toBeGreaterThan(0);
  });

  it("repeated picks are deterministic", () => {
    const cam = new OrbitCamera({ target: centerWorld, yawRad: 0.3, pitchRad: 0.2, distanceM: 1.5 });
    const vp = cam.viewProjectionMatrix(1.6);
    const run = () => {
      const ray = screenRay(vp, 0.1, -0.2);
      const hit = pickPanel(fixture.assembled, fixture.assembled.positions, ray);
      return hit ? `${hit.panelId}:${hit.triangle}:${hit.vertex}` : "null";
    };
    expect(run()).toBe(run());
  });
});
