// G15 browser capture path: executes CapturePlans against a live WebGL
// renderer and resolves queued textures. Browser-only (DOM + GL); the data
// layer (src/render/*) stays headless-testable and owns all decisions.

import * as THREE from "three";
import type { SceneObjects } from "../src/render/three-adapter.js";
import { applyOrbitCamera } from "../src/render/three-adapter.js";
import type { CapturePlan, PresentationScene } from "../src/render/presentation.js";

export interface CaptureResult {
  captureId: string;
  widthPx: number;
  heightPx: number;
  /** PNG data URL (preserveDrawingBuffer or synchronous readback required). */
  dataUrl: string;
  elapsedMs: number;
}

/**
 * Execute one capture plan: camera, background/alpha, size, render, readback.
 * The renderer must have been created with preserveDrawingBuffer: true (or
 * this must run synchronously inside the RAF that rendered).
 */
export function executeCapture(
  renderer: THREE.WebGLRenderer,
  objects: SceneObjects,
  scene: PresentationScene,
  plan: CapturePlan,
): CaptureResult {
  const started = performance.now();
  const camera = new THREE.PerspectiveCamera();
  applyOrbitCamera(camera, plan.camera, plan.widthPx / plan.heightPx);
  renderer.setSize(plan.widthPx, plan.heightPx, false);
  if (plan.transparent) {
    renderer.setClearAlpha(0);
  } else if (scene.background) {
    renderer.setClearColor(new THREE.Color(scene.background[0], scene.background[1], scene.background[2]), 1);
    renderer.setClearAlpha(1);
  }
  const avatarWasVisible = objects.avatar?.visible ?? true;
  if (objects.avatar) objects.avatar.visible = !plan.garmentOnly;
  renderer.render(objects.scene, camera);
  if (objects.avatar) objects.avatar.visible = avatarWasVisible;
  const dataUrl = (renderer.domElement as HTMLCanvasElement).toDataURL("image/png");
  return { captureId: plan.id, widthPx: plan.widthPx, heightPx: plan.heightPx, dataUrl, elapsedMs: performance.now() - started };
}

/** Resolve queued texture refs onto converted materials (browser-side fetch). */
export async function resolveTextureQueue(
  objects: SceneObjects,
  refs: string[],
  loader: THREE.TextureLoader = new THREE.TextureLoader(),
): Promise<{ loaded: string[]; failed: string[] }> {
  const loaded: string[] = [];
  const failed: string[] = [];
  for (const ref of refs) {
    try {
      const texture = await loader.loadAsync(ref);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 4;
      for (const material of Object.values(objects.materialByPanel)) {
        if (material.userData.pendingColorMap === ref) {
          material.map = texture;
          material.needsUpdate = true;
        }
      }
      loaded.push(ref);
    } catch {
      failed.push(ref);
    }
  }
  return { loaded, failed };
}
