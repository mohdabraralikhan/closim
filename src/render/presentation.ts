// G15D — garment presentation system (data layer).
//
// Cameras, lighting rigs, backgrounds, saved scenes, and capture plans are
// pure data referencing garments — they never duplicate garment geometry.
// Camera presets map onto the view-core OrbitCameraState, so framing math
// stays in one tested place. Pixel production happens in the browser
// adapter; here capture plans are deterministic and serializable.

import { PatternCadError } from "../pattern/cad.js";
import {
  DEFAULT_CAMERA_STATE,
  MAX_DISTANCE_M,
  MIN_DISTANCE_M,
  type OrbitCameraState,
} from "../view/camera.js";
import type { Bounds } from "../view/types.js";
import { boundsCenter } from "../view/types.js";
import type { RGB } from "./materials.js";
import type { QualityLevel, RenderMode } from "./representation.js";

export type PresentationView =
  | "front" | "back" | "left" | "right" | "three-quarter" | "top" | "custom";

export interface CameraPreset {
  name: string;
  view: PresentationView;
  yawRad: number;
  pitchRad: number;
  /** Fraction of the bounding sphere kept in frame (1 = exact fit). */
  frameMargin: number;
}

export const CAMERA_PRESETS: Record<Exclude<PresentationView, "custom">, CameraPreset> = {
  front: { name: "Front", view: "front", yawRad: 0, pitchRad: 0.08, frameMargin: 1.25 },
  back: { name: "Back", view: "back", yawRad: Math.PI, pitchRad: 0.08, frameMargin: 1.25 },
  left: { name: "Left", view: "left", yawRad: -Math.PI / 2, pitchRad: 0.08, frameMargin: 1.25 },
  right: { name: "Right", view: "right", yawRad: Math.PI / 2, pitchRad: 0.08, frameMargin: 1.25 },
  "three-quarter": { name: "Three-quarter", view: "three-quarter", yawRad: Math.PI / 4, pitchRad: 0.18, frameMargin: 1.35 },
  top: { name: "Top", view: "top", yawRad: 0, pitchRad: Math.PI / 2 - 0.02, frameMargin: 1.25 },
};

function boundingRadius(bounds: Bounds): number {
  const dx = bounds.max[0] - bounds.min[0];
  const dy = bounds.max[1] - bounds.min[1];
  const dz = bounds.max[2] - bounds.min[2];
  return Math.hypot(dx, dy, dz) / 2;
}

/**
 * Frame bounds with a preset: target = bounds center, distance fits the
 * bounding sphere at the default 40° field of view times the margin.
 */
export function frameCamera(
  bounds: Bounds,
  preset: CameraPreset,
  fovYRad = (40 * Math.PI) / 180,
): OrbitCameraState {
  if (!(fovYRad > 0) || !(fovYRad < Math.PI)) {
    throw new PatternCadError("invalid-transform", "field of view must be in (0, pi)");
  }
  const radius = Math.max(boundingRadius(bounds), 1e-6);
  const distanceM = Math.min(
    MAX_DISTANCE_M,
    Math.max(MIN_DISTANCE_M, (radius * preset.frameMargin) / Math.tan(fovYRad / 2)),
  );
  return {
    ...DEFAULT_CAMERA_STATE,
    target: boundsCenter(bounds),
    yawRad: preset.yawRad,
    pitchRad: preset.pitchRad,
    distanceM,
  };
}

// ---------------------------------------------------------------------------
// Lighting rigs (data-driven; nothing hardcoded into garments)
// ---------------------------------------------------------------------------

export interface DirectionalLight {
  direction: [number, number, number];
  intensity: number;
  color: RGB;
}

export interface LightRig {
  name: string;
  key: DirectionalLight;
  fill: DirectionalLight;
  rim: DirectionalLight;
  ambientIntensity: number;
  ambientColor: RGB;
}

function dirLight(direction: [number, number, number], intensity: number, color: RGB): DirectionalLight {
  const len = Math.hypot(direction[0], direction[1], direction[2]);
  if (!(len > 0) || !Number.isFinite(intensity) || intensity < 0) {
    throw new PatternCadError("invalid-transform", "light direction/intensity invalid");
  }
  return { direction: [direction[0] / len, direction[1] / len, direction[2] / len], intensity, color };
}

export const LIGHTING_PRESETS: Record<string, LightRig> = {
  studio: {
    name: "Studio",
    key: dirLight([0.5, 1, 0.75], 2.2, [1, 1, 1]),
    fill: dirLight([-0.6, 0.3, 0.5], 0.7, [0.9, 0.95, 1]),
    rim: dirLight([0, 0.4, -1], 1.1, [1, 1, 1]),
    ambientIntensity: 0.5,
    ambientColor: [1, 1, 1],
  },
  soft: {
    name: "Soft",
    key: dirLight([0.3, 1, 0.4], 1.2, [1, 0.98, 0.95]),
    fill: dirLight([-0.5, 0.4, 0.6], 0.9, [0.95, 0.97, 1]),
    rim: dirLight([0, 0.3, -1], 0.4, [1, 1, 1]),
    ambientIntensity: 0.8,
    ambientColor: [1, 1, 1],
  },
  product: {
    name: "Product",
    key: dirLight([0, 1, 0.35], 2.6, [1, 1, 1]),
    fill: dirLight([-0.8, 0.2, 0.3], 0.5, [1, 1, 1]),
    rim: dirLight([0.6, 0.5, -0.8], 1.4, [1, 1, 1]),
    ambientIntensity: 0.35,
    ambientColor: [1, 1, 1],
  },
  technical: {
    name: "Technical",
    key: dirLight([0.2, 1, 0.2], 1.6, [1, 1, 1]),
    fill: dirLight([-0.2, 0.5, 0.8], 1.6, [1, 1, 1]),
    rim: dirLight([0, 0, -1], 0.2, [1, 1, 1]),
    ambientIntensity: 0.9,
    ambientColor: [1, 1, 1],
  },
  neutral: {
    name: "Neutral",
    key: dirLight([0.4, 0.8, 0.6], 1.8, [1, 1, 1]),
    fill: dirLight([-0.4, 0.4, 0.6], 0.8, [1, 1, 1]),
    rim: dirLight([0, 0.5, -0.9], 0.8, [1, 1, 1]),
    ambientIntensity: 0.6,
    ambientColor: [1, 1, 1],
  },
};

export function lightingPreset(name: string): LightRig {
  const rig = LIGHTING_PRESETS[name];
  if (!rig) throw new PatternCadError("invalid-document", `unknown lighting preset '${name}'`);
  return JSON.parse(JSON.stringify(rig)) as LightRig;
}

// ---------------------------------------------------------------------------
// Presentation scenes + capture plans
// ---------------------------------------------------------------------------

export interface PresentationScene {
  id: string;
  name: string;
  /** Garment reference (id only — scenes never duplicate geometry). */
  garmentId: string;
  camera: OrbitCameraState;
  cameraView: PresentationView;
  lights: LightRig;
  /** Solid background, or null for transparent output. */
  background: RGB | null;
  showAvatar: boolean;
  showGarment: boolean;
  renderMode: RenderMode;
  quality: QualityLevel;
  materialOverrides: Record<string, string>;
}

export function createScene(partial: {
  id: string;
  name: string;
  garmentId: string;
  camera?: OrbitCameraState;
  cameraView?: PresentationView;
  lights?: LightRig;
  background?: RGB | null;
  showAvatar?: boolean;
  showGarment?: boolean;
  renderMode?: RenderMode;
  quality?: QualityLevel;
  materialOverrides?: Record<string, string>;
}): PresentationScene {
  if (!partial.id || !partial.name || !partial.garmentId) {
    throw new PatternCadError("invalid-document", "scene needs id, name, and garment reference");
  }
  return {
    id: partial.id,
    name: partial.name,
    garmentId: partial.garmentId,
    camera: partial.camera ?? { ...DEFAULT_CAMERA_STATE },
    cameraView: partial.cameraView ?? "three-quarter",
    lights: partial.lights ?? lightingPreset("studio"),
    background: partial.background === undefined ? [0.96, 0.96, 0.97] : partial.background,
    showAvatar: partial.showAvatar ?? true,
    showGarment: partial.showGarment ?? true,
    renderMode: partial.renderMode ?? "shaded",
    quality: partial.quality ?? "preview",
    materialOverrides: { ...(partial.materialOverrides ?? {}) },
  };
}

export interface CapturePlan {
  id: string;
  sceneId: string;
  view: PresentationView;
  camera: OrbitCameraState;
  widthPx: number;
  heightPx: number;
  transparent: boolean;
  garmentOnly: boolean;
}

/**
 * Deterministic capture plans for a scene: named views at fixed
 * resolutions. Front/back/three-quarter/garment-only are the commercial set.
 */
export function planCaptures(
  scene: PresentationScene,
  views: PresentationView[] = ["front", "back", "three-quarter"],
  size: { widthPx?: number; heightPx?: number } = {},
): CapturePlan[] {
  const widthPx = size.widthPx ?? 1024;
  const heightPx = size.heightPx ?? 1024;
  if (!views.length || !(widthPx > 0) || !(heightPx > 0)) {
    throw new PatternCadError("invalid-transform", "capture plan needs views and positive size");
  }
  return views.map((view) => {
    const preset = view === "custom" ? null : CAMERA_PRESETS[view];
    const camera = preset
      ? { ...scene.camera, yawRad: preset.yawRad, pitchRad: preset.pitchRad }
      : { ...scene.camera };
    return {
      id: `${scene.id}/capture/${view}`,
      sceneId: scene.id,
      view,
      camera,
      widthPx,
      heightPx,
      transparent: scene.background === null,
      garmentOnly: scene.showAvatar === false,
    };
  });
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export function serializeScene(scene: PresentationScene): string {
  return canonicalJson(scene);
}

export function deserializeScene(serialized: string): PresentationScene {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized scene is not valid JSON");
  }
  const scene = parsed as PresentationScene;
  if (!scene || !scene.id || !scene.garmentId || !scene.camera || !scene.lights) {
    throw new PatternCadError("invalid-document", "presentation scene shape is invalid");
  }
  return JSON.parse(JSON.stringify(scene)) as PresentationScene;
}
