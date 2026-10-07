import { describe, expect, it } from "vitest";
import {
  DEFAULT_CAMERA_STATE,
  MAX_DISTANCE_M,
  MIN_DISTANCE_M,
  OrbitCamera,
  PITCH_LIMIT_RAD,
  multiply4,
  normalizeAngle,
  perspective,
  viewDirection,
} from "../../src/view/camera.js";

function applyMatrix(m: Float64Array, p: [number, number, number]): [number, number, number, number] {
  const x = p[0], y = p[1], z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

describe("orbit camera state", () => {
  it("starts from documented defaults", () => {
    const cam = new OrbitCamera();
    expect(cam.state.mode).toBe("perspective");
    expect(cam.state.target).toEqual(DEFAULT_CAMERA_STATE.target);
    expect(cam.state.distanceM).toBeGreaterThan(0);
  });

  it("orbit clamps pitch and normalizes yaw", () => {
    const cam = new OrbitCamera();
    cam.orbit(0, 10);
    expect(cam.state.pitchRad).toBeCloseTo(PITCH_LIMIT_RAD, 12);
    cam.orbit(0, -20);
    expect(cam.state.pitchRad).toBeCloseTo(-PITCH_LIMIT_RAD, 12);
    cam.orbit(-3 * Math.PI, 0);
    expect(cam.state.yawRad).toBeGreaterThanOrEqual(0);
    expect(cam.state.yawRad).toBeLessThan(2 * Math.PI);
  });

  it("pan moves the target in the camera plane", () => {
    const cam = new OrbitCamera({ yawRad: 0, pitchRad: 0, target: [0, 0, 0] });
    cam.pan(0.5, 0.25);
    expect(cam.state.target[0]).toBeCloseTo(0.5, 12);
    expect(cam.state.target[1]).toBeCloseTo(0.25, 12);
    expect(cam.state.target[2]).toBeCloseTo(0, 12);
  });

  it("zoom divides distance and clamps", () => {
    const cam = new OrbitCamera({ distanceM: 2 });
    cam.zoom(2);
    expect(cam.state.distanceM).toBeCloseTo(1, 12);
    cam.zoom(1e12);
    expect(cam.state.distanceM).toBe(MIN_DISTANCE_M);
    cam.zoom(0);
    expect(cam.state.distanceM).toBe(MIN_DISTANCE_M);
    cam.zoom(1e-12);
    expect(cam.state.distanceM).toBe(MAX_DISTANCE_M);
  });

  it("frame centers on the bounds and guarantees coverage", () => {
    const cam = new OrbitCamera({ mode: "perspective", fovYRad: Math.PI / 3 });
    cam.frame([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 1.5, 1.0);
    expect(cam.state.target).toEqual([0, 0, 0]);
    const radius = 0.5 * Math.hypot(1, 1, 1);
    expect(cam.state.distanceM).toBeGreaterThanOrEqual(radius / Math.sin(Math.PI / 6) - 1e-9);
    expect(cam.state.orthoHeightM).toBeGreaterThan(0);
  });

  it("frame in orthographic mode covers the bounding sphere at small aspect", () => {
    const cam = new OrbitCamera({ mode: "orthographic" });
    cam.frame([-1, -1, -1], [1, 1, 1], 0.5, 1.0);
    const radius = Math.sqrt(3);
    expect(cam.state.orthoHeightM).toBeCloseTo((2 * radius) / 0.5, 9);
  });

  it("mode switch preserves approximate screen coverage", () => {
    const cam = new OrbitCamera({ distanceM: 2, fovYRad: Math.PI / 3 });
    const distBefore = cam.state.distanceM;
    cam.setMode("orthographic");
    expect(cam.state.mode).toBe("orthographic");
    expect(cam.state.orthoHeightM).toBeCloseTo(2 * distBefore * Math.tan(Math.PI / 6), 12);
    cam.setMode("perspective");
    expect(cam.state.distanceM).toBeCloseTo(distBefore, 9);
  });

  it("view presets set yaw/pitch without touching target or distance", () => {
    const cam = new OrbitCamera({ target: [1, 2, 3], distanceM: 5 });
    cam.setView("front");
    expect(cam.state.yawRad).toBe(0);
    expect(cam.state.pitchRad).toBe(0);
    cam.setView("top");
    expect(cam.state.pitchRad).toBeCloseTo(PITCH_LIMIT_RAD, 12);
    expect(cam.state.target).toEqual([1, 2, 3]);
    expect(cam.state.distanceM).toBe(5);
  });

  it("reset restores defaults", () => {
    const cam = new OrbitCamera();
    cam.orbit(2, 1);
    cam.zoom(10);
    cam.reset();
    expect(cam.snapshot()).toEqual(DEFAULT_CAMERA_STATE);
  });

  it("basis is orthonormal and eye sits behind the target", () => {
    const cam = new OrbitCamera({ yawRad: 0.7, pitchRad: 0.3, target: [1, 1, 1], distanceM: 3 });
    const b = cam.basis();
    const d = viewDirection(cam.state.yawRad, cam.state.pitchRad);
    expect(b.eye[0]).toBeCloseTo(1 + 3 * d[0], 12);
    expect(b.eye[1]).toBeCloseTo(1 + 3 * d[1], 12);
    expect(b.eye[2]).toBeCloseTo(1 + 3 * d[2], 12);
    const dot = (a: number[], c: number[]) => a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
    expect(dot(b.right, b.up)).toBeCloseTo(0, 12);
    expect(dot(b.right, b.forward)).toBeCloseTo(0, 12);
    expect(dot(b.up, b.forward)).toBeCloseTo(0, 12);
  });

  it("worldPerPixel is positive and shrinks when zooming in", () => {
    const cam = new OrbitCamera();
    const a = cam.worldPerPixel(800);
    cam.zoom(2);
    const b = cam.worldPerPixel(800);
    expect(a).toBeGreaterThan(0);
    expect(b).toBeCloseTo(a / 2, 12);
  });
});

describe("camera matrices", () => {
  it("view matrix maps the target to -distance on Z and the eye to the origin", () => {
    const cam = new OrbitCamera({ yawRad: 0, pitchRad: 0, target: [1, 2, 3], distanceM: 2 });
    const view = cam.viewMatrix();
    const t = applyMatrix(view, [1, 2, 3]);
    expect(t[0]).toBeCloseTo(0, 12);
    expect(t[1]).toBeCloseTo(0, 12);
    expect(t[2]).toBeCloseTo(-2, 12);
    expect(t[3]).toBeCloseTo(1, 12);
    const b = cam.basis();
    const e = applyMatrix(view, b.eye);
    expect(e[0]).toBeCloseTo(0, 12);
    expect(e[1]).toBeCloseTo(0, 12);
    expect(e[2]).toBeCloseTo(0, 12);
  });

  it("view-projection maps the frame target inside clip space", () => {
    const cam = new OrbitCamera({ yawRad: 0.4, pitchRad: 0.2, distanceM: 3 });
    const vp = cam.viewProjectionMatrix(1.5);
    const c = applyMatrix(vp, cam.state.target);
    expect(c[3]).toBeGreaterThan(0);
    const ndcX = c[0] / c[3];
    const ndcY = c[1] / c[3];
    const ndcZ = c[2] / c[3];
    expect(Math.abs(ndcX)).toBeLessThan(1e-12);
    expect(Math.abs(ndcY)).toBeLessThan(1e-12);
    expect(ndcZ).toBeGreaterThan(-1);
    expect(ndcZ).toBeLessThan(1);
  });

  it("orthographic projection keeps vertical extent = orthoHeight at target depth", () => {
    const cam = new OrbitCamera({ mode: "orthographic", orthoHeightM: 2, target: [0, 0, 0], yawRad: 0, pitchRad: 0 });
    const vp = cam.viewProjectionMatrix(1);
    const top = applyMatrix(vp, [0, 1, -cam.state.distanceM]);
    const bottom = applyMatrix(vp, [0, -1, -cam.state.distanceM]);
    expect(top[1] / top[3]).toBeCloseTo(1, 9);
    expect(bottom[1] / bottom[3]).toBeCloseTo(-1, 9);
  });

  it("perspective near plane maps to ndc -1", () => {
    const m = perspective(Math.PI / 3, 1, 0.1, 100);
    const nearPoint = applyMatrix(m, [0, 0, -0.1]);
    expect(nearPoint[2] / nearPoint[3]).toBeCloseTo(-1, 12);
  });

  it("multiply4 composes translation matrices", () => {
    const t1 = new Float64Array(16);
    t1[0] = t1[5] = t1[10] = t1[15] = 1;
    t1[12] = 1;
    const t2 = new Float64Array(16);
    t2[0] = t2[5] = t2[10] = t2[15] = 1;
    t2[13] = 2;
    const m = multiply4(t1, t2);
    const p = applyMatrix(m, [0, 0, 0]);
    expect(p[0]).toBeCloseTo(1, 12);
    expect(p[1]).toBeCloseTo(2, 12);
  });

  it("normalizeAngle wraps into [0, 2pi)", () => {
    expect(normalizeAngle(-Math.PI / 2)).toBeCloseTo(3 * Math.PI / 2, 12);
    expect(normalizeAngle(2 * Math.PI + 0.5)).toBeCloseTo(0.5, 12);
    expect(normalizeAngle(0)).toBe(0);
  });
});
