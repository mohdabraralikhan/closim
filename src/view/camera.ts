// G10A orbit camera: pure state + matrix math, no DOM and no rendering deps.
// Convention: right-handed, Y-up, column-major matrices (WebGL layout).
// eye = target + distance * [cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw)]

import { type Vec3 } from "./types.js";

export type CameraMode = "perspective" | "orthographic";

export type ViewPreset = "front" | "back" | "left" | "right" | "top" | "bottom";

export interface OrbitCameraState {
  target: Vec3;
  yawRad: number;
  pitchRad: number;
  distanceM: number;
  mode: CameraMode;
  fovYRad: number;
  orthoHeightM: number;
  nearM: number;
  farM: number;
}

export const PITCH_LIMIT_RAD = Math.PI / 2 - 1e-3;
export const MIN_DISTANCE_M = 0.02;
export const MAX_DISTANCE_M = 200;
export const MIN_ORTHO_HEIGHT_M = 0.02;
export const MAX_ORTHO_HEIGHT_M = 200;

export const DEFAULT_CAMERA_STATE: OrbitCameraState = {
  target: [0, 0.9, 0],
  yawRad: 0.6,
  pitchRad: 0.35,
  distanceM: 2.5,
  mode: "perspective",
  fovYRad: (45 * Math.PI) / 180,
  orthoHeightM: 2,
  nearM: 0.01,
  farM: 100,
};

export function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

export function viewDirection(yawRad: number, pitchRad: number): Vec3 {
  const cp = Math.cos(pitchRad);
  return [cp * Math.sin(yawRad), Math.sin(pitchRad), cp * Math.cos(yawRad)];
}

export type Mat4 = Float64Array;

export function identity4(): Mat4 {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz);
  if (len < 1e-12) { zx = 0; zy = 0; zz = 1; len = 1; }
  zx /= len; zy /= len; zz /= len;
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz);
  if (len < 1e-12) {
    // view direction parallel to up: pick a perpendicular fallback
    xx = zz; xy = 0; xz = -zx;
    len = Math.hypot(xx, xy, xz) || 1;
  }
  xx /= len; xy /= len; xz /= len;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  const m = new Float64Array(16);
  m[0] = xx; m[1] = yx; m[2] = zx; m[3] = 0;
  m[4] = xy; m[5] = yy; m[6] = zy; m[7] = 0;
  m[8] = xz; m[9] = yz; m[10] = zz; m[11] = 0;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}

export function perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const m = new Float64Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

export function orthographic(heightM: number, aspect: number, near: number, far: number): Mat4 {
  const halfH = heightM / 2;
  const halfW = halfH * aspect;
  const m = new Float64Array(16);
  m[0] = 1 / halfW;
  m[5] = 1 / halfH;
  m[10] = -2 / (far - near);
  m[14] = -((far + near) / (far - near));
  m[15] = 1;
  return m;
}

export function multiply4(a: Mat4, b: Mat4): Mat4 {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export interface CameraBasis {
  eye: Vec3;
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}

export class OrbitCamera {
  readonly state: OrbitCameraState;
  private readonly onDirty?: () => void;

  constructor(state: Partial<OrbitCameraState> = {}, onDirty?: () => void) {
    this.onDirty = onDirty;
    this.state = { ...DEFAULT_CAMERA_STATE, ...state, target: [...(state.target ?? DEFAULT_CAMERA_STATE.target)] as Vec3 };
  }

  private changed(): void {
    this.onDirty?.();
  }

  snapshot(): OrbitCameraState {
    return { ...this.state, target: [...this.state.target] as Vec3 };
  }

  reset(): void {
    const t = [...DEFAULT_CAMERA_STATE.target] as Vec3;
    Object.assign(this.state, { ...DEFAULT_CAMERA_STATE, target: t });
    this.changed();
  }

  orbit(deltaYawRad: number, deltaPitchRad: number): void {
    this.state.yawRad += deltaYawRad;
    this.state.pitchRad = clamp(this.state.pitchRad + deltaPitchRad, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD);
    this.state.yawRad = normalizeAngle(this.state.yawRad);
    this.changed();
  }

  /** Move the orbit target in the camera plane (world meters). */
  pan(rightM: number, upM: number): void {
    const b = this.basis();
    const t = this.state.target;
    t[0] += b.right[0] * rightM + b.up[0] * upM;
    t[1] += b.right[1] * rightM + b.up[1] * upM;
    t[2] += b.right[2] * rightM + b.up[2] * upM;
    this.changed();
  }

  /** factor > 1 zooms in. */
  zoom(factor: number): void {
    if (!(Number.isFinite(factor) && factor > 0)) return;
    this.state.distanceM = clamp(this.state.distanceM / factor, MIN_DISTANCE_M, MAX_DISTANCE_M);
    this.state.orthoHeightM = clamp(this.state.orthoHeightM / factor, MIN_ORTHO_HEIGHT_M, MAX_ORTHO_HEIGHT_M);
    this.changed();
  }

  setMode(mode: CameraMode): void {
    if (mode === this.state.mode) return;
    if (mode === "orthographic") {
      this.state.orthoHeightM = clamp(2 * this.state.distanceM * Math.tan(this.state.fovYRad / 2), MIN_ORTHO_HEIGHT_M, MAX_ORTHO_HEIGHT_M);
    } else {
      this.state.distanceM = clamp(this.state.orthoHeightM / (2 * Math.tan(this.state.fovYRad / 2)), MIN_DISTANCE_M, MAX_DISTANCE_M);
    }
    this.state.mode = mode;
    this.changed();
  }

  setView(preset: ViewPreset): void {
    switch (preset) {
      case "front": this.state.yawRad = 0; this.state.pitchRad = 0; break;
      case "back": this.state.yawRad = Math.PI; this.state.pitchRad = 0; break;
      case "right": this.state.yawRad = Math.PI / 2; this.state.pitchRad = 0; break;
      case "left": this.state.yawRad = -Math.PI / 2; this.state.pitchRad = 0; break;
      case "top": this.state.yawRad = normalizeAngle(this.state.yawRad); this.state.pitchRad = PITCH_LIMIT_RAD; break;
      case "bottom": this.state.yawRad = normalizeAngle(this.state.yawRad); this.state.pitchRad = -PITCH_LIMIT_RAD; break;
    }
    this.changed();
  }

  /** Frame an AABB so it fits the viewport with a margin. */
  frame(min: Vec3, max: Vec3, aspect: number, margin = 1.2): void {
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const radius = Math.max(
      0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]),
      1e-3,
    );
    const safeAspect = Math.max(aspect, 1e-6);
    this.state.target = [cx, cy, cz];
    if (this.state.mode === "perspective") {
      const hFov = 2 * Math.atan(Math.tan(this.state.fovYRad / 2) * safeAspect);
      const fov = Math.min(this.state.fovYRad, hFov);
      this.state.distanceM = clamp((radius * margin) / Math.sin(fov / 2), MIN_DISTANCE_M, MAX_DISTANCE_M);
      this.state.orthoHeightM = clamp(2 * this.state.distanceM * Math.tan(this.state.fovYRad / 2), MIN_ORTHO_HEIGHT_M, MAX_ORTHO_HEIGHT_M);
    } else {
      const height = (2 * radius * margin) / Math.min(1, safeAspect);
      this.state.orthoHeightM = clamp(height, MIN_ORTHO_HEIGHT_M, MAX_ORTHO_HEIGHT_M);
      this.state.distanceM = clamp(this.state.orthoHeightM / (2 * Math.tan(this.state.fovYRad / 2)), MIN_DISTANCE_M, MAX_DISTANCE_M);
    }
    this.state.nearM = Math.max(1e-3, Math.min(this.state.distanceM - 4 * radius, this.state.distanceM * 0.5));
    this.state.farM = this.state.distanceM + 8 * radius + 1;
    this.changed();
  }

  basis(): CameraBasis {
    const { target, yawRad, pitchRad, distanceM } = this.state;
    const d = viewDirection(yawRad, pitchRad);
    const eye: Vec3 = [
      target[0] + distanceM * d[0],
      target[1] + distanceM * d[1],
      target[2] + distanceM * d[2],
    ];
    const worldUp: Vec3 = [0, 1, 0];
    let fx = -d[0], fy = -d[1], fz = -d[2];
    let rx = fy * worldUp[2] - fz * worldUp[1];
    let ry = fz * worldUp[0] - fx * worldUp[2];
    let rz = fx * worldUp[1] - fy * worldUp[0];
    let len = Math.hypot(rx, ry, rz);
    if (len < 1e-9) { rx = 1; ry = 0; rz = 0; len = 1; }
    rx /= len; ry /= len; rz /= len;
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;
    return { eye, forward: [fx, fy, fz], right: [rx, ry, rz], up: [ux, uy, uz] };
  }

  /** World meters covered by one vertical viewport unit (use with pixel height). */
  worldPerPixel(viewportHeightPx: number): number {
    if (this.state.mode === "orthographic") return this.state.orthoHeightM / Math.max(viewportHeightPx, 1);
    return (2 * this.state.distanceM * Math.tan(this.state.fovYRad / 2)) / Math.max(viewportHeightPx, 1);
  }

  viewMatrix(): Mat4 {
    const b = this.basis();
    return lookAt(b.eye, this.state.target, [0, 1, 0]);
  }

  projectionMatrix(aspect: number): Mat4 {
    const s = this.state;
    if (s.mode === "orthographic") return orthographic(s.orthoHeightM, aspect, s.nearM, s.farM);
    return perspective(s.fovYRad, aspect, s.nearM, s.farM);
  }

  viewProjectionMatrix(aspect: number): Mat4 {
    return multiply4(this.projectionMatrix(aspect), this.viewMatrix());
  }
}

export function normalizeAngle(a: number): number {
  const twoPi = 2 * Math.PI;
  let x = a % twoPi;
  if (x < 0) x += twoPi;
  return x;
}
