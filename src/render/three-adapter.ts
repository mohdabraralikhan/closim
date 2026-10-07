// three.js scene adapter (G15A/D browser path, headless-testable).
//
// Builds and updates a THREE.Scene graph from render representations.
// NEVER instantiates a renderer or loads a texture here: materials map
// color/roughness/metalness/opacity/double-sided only, and texture refs that
// need loading are reported in `textureQueue` for the browser capture path
// (app/render-g15.ts) to resolve. No physics enters this module.

import * as THREE from "three";
import type { OrbitCameraState } from "../view/camera.js";
import type { VisualMaterial } from "./materials.js";
import {
  boundarySegments,
  seamPolylines,
  type QualitySettings,
  type RenderAvatar,
  type RenderGarment,
  type RenderVisibility,
} from "./representation.js";
import type { AssembledGarment } from "../garment/assembly.js";
import type { LightRig } from "./presentation.js";

export interface AdapterOptions {
  visibility: RenderVisibility;
  quality: QualitySettings;
  /** Panel id -> resolved visual material (fallback pre-resolved by caller). */
  materials: Record<string, VisualMaterial>;
  /** World-space seam polylines override (default: derived from welds). */
  seamOverride?: Map<string, Array<[number, number, number]>>;
}

export interface SceneObjects {
  scene: THREE.Scene;
  garment: THREE.Mesh;
  garmentGeometry: THREE.BufferGeometry;
  wireframe: THREE.LineSegments | null;
  avatar: THREE.Mesh | null;
  seamGroup: THREE.Group;
  boundary: THREE.LineSegments | null;
  lightGroup: THREE.Group;
  materialByPanel: Record<string, THREE.MeshStandardMaterial>;
  /** Texture refs needing browser-side loading (kind !== none). */
  textureQueue: string[];
  disposed: boolean;
}

export function visualToStandard(material: VisualMaterial): THREE.MeshStandardMaterial {
  const [r, g, b] = material.baseColor;
  const three = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, g, b),
    roughness: material.roughness,
    metalness: material.metallic,
    transparent: material.opacity < 1,
    opacity: material.opacity,
    side: THREE.DoubleSide,
  });
  three.userData.renderMaterialId = material.id;
  three.userData.renderMaterialVersion = material.version;
  return three;
}

function concatUVs(garment: RenderGarment): Float32Array {
  const total = garment.panelUVs.reduce((s, p) => s + p.uv.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const panel of garment.panelUVs) {
    out.set(panel.uv, offset);
    offset += panel.uv.length;
  }
  return out;
}

export function buildRenderObjects(
  garment: RenderGarment,
  avatar: RenderAvatar | null,
  assembled: AssembledGarment,
  opts: AdapterOptions,
): SceneObjects {
  const scene = new THREE.Scene();
  const textureQueue = new Set<string>();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(garment.positions), 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(Float32Array.from(garment.normals), 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(concatUVs(garment), 2));
  geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(garment.indices), 1));

  const materialByPanel: Record<string, THREE.MeshStandardMaterial> = {};
  const materials: THREE.MeshStandardMaterial[] = [];
  garment.panelRanges.forEach((range, groupIndex) => {
    const visual = opts.materials[range.panelId];
    if (!visual) throw new Error(`adapter: no material resolved for panel '${range.panelId}'`);
    const three = visualToStandard(visual);
    if (visual.colorMap.kind !== "none") {
      three.userData.pendingColorMap = visual.colorMap.ref;
    }
    materialByPanel[range.panelId] = three;
    materials.push(three);
    geometry.addGroup(range.triangleStart * 3, range.triangleCount * 3, groupIndex);
    for (const tex of [visual.colorMap, visual.normalMap, visual.roughnessMap]) {
      if (tex.kind !== "none") textureQueue.add(tex.ref);
    }
  });
  const mesh = new THREE.Mesh(geometry, materials);
  mesh.visible = opts.visibility.garment;
  mesh.frustumCulled = false;
  scene.add(mesh);

  let wireframe: THREE.LineSegments | null = null;
  if (opts.visibility.wireframe) {
    wireframe = new THREE.LineSegments(
      new THREE.WireframeGeometry(geometry),
      new THREE.LineBasicMaterial({ color: 0x222222 }),
    );
    wireframe.frustumCulled = false;
    scene.add(wireframe);
  }

  let avatarMesh: THREE.Mesh | null = null;
  if (avatar && opts.visibility.avatar) {
    const ageo = new THREE.BufferGeometry();
    ageo.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(avatar.positions), 3));
    ageo.setAttribute("normal", new THREE.BufferAttribute(Float32Array.from(avatar.normals), 3));
    ageo.setIndex(new THREE.BufferAttribute(Uint32Array.from(avatar.indices), 1));
    avatarMesh = new THREE.Mesh(
      ageo,
      new THREE.MeshStandardMaterial({ color: 0x495057, roughness: 0.9, metalness: 0 }),
    );
    avatarMesh.frustumCulled = false;
    scene.add(avatarMesh);
  }

  const seamGroup = new THREE.Group();
  if (opts.visibility.seams && opts.quality.seamOverlay) {
    const polylines = opts.seamOverride ?? seamPolylines(garment, assembled);
    const mat = new THREE.LineBasicMaterial({ color: 0xffd27a });
    for (const [, points] of polylines) {
      if (points.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
      seamGroup.add(new THREE.Line(geo, mat));
    }
  }
  seamGroup.visible = opts.visibility.seams;
  scene.add(seamGroup);

  let boundary: THREE.LineSegments | null = null;
  if (opts.visibility.boundaries && opts.quality.boundaryOverlay) {
    const segments = boundarySegments(garment);
    const flat = new Float32Array(segments.length * 6);
    segments.forEach((s, i) => flat.set(s, i * 6));
    const bgeo = new THREE.BufferGeometry();
    bgeo.setAttribute("position", new THREE.BufferAttribute(flat, 3));
    boundary = new THREE.LineSegments(bgeo, new THREE.LineBasicMaterial({ color: 0xd8dde4 }));
    boundary.frustumCulled = false;
    scene.add(boundary);
  }

  const lightGroup = new THREE.Group();
  scene.add(lightGroup);

  return {
    scene, garment: mesh, garmentGeometry: geometry, wireframe,
    avatar: avatarMesh, seamGroup, boundary, lightGroup,
    materialByPanel, textureQueue: [...textureQueue].sort(), disposed: false,
  };
}

/**
 * Sync fresh simulation state into live GPU attributes. Same vertex count
 * required — topology changes must rebuild (never silently remap).
 */
export function updateRenderObjects(objects: SceneObjects, render: RenderGarment): void {
  if (objects.disposed) throw new Error("adapter: scene objects are disposed");
  const position = objects.garmentGeometry.getAttribute("position") as THREE.BufferAttribute;
  const normal = objects.garmentGeometry.getAttribute("normal") as THREE.BufferAttribute;
  if (position.count * 3 !== render.positions.length) {
    throw new Error("adapter: vertex count changed — rebuild the scene objects");
  }
  (position.array as Float32Array).set(render.positions);
  (normal.array as Float32Array).set(render.normals);
  position.needsUpdate = true;
  normal.needsUpdate = true;
  if (objects.wireframe) {
    const rebuilt = new THREE.WireframeGeometry(objects.garmentGeometry);
    const old = objects.wireframe.geometry;
    objects.wireframe.geometry = rebuilt;
    old.dispose();
  }
}

export function applyLightRig(objects: SceneObjects, rig: LightRig): void {
  while (objects.lightGroup.children.length > 0) {
    objects.lightGroup.remove(objects.lightGroup.children[0]);
  }
  const make = (light: LightRig["key"]): THREE.DirectionalLight => {
    const l = new THREE.DirectionalLight(
      new THREE.Color(light.color[0], light.color[1], light.color[2]),
      light.intensity,
    );
    l.position.set(light.direction[0], light.direction[1], light.direction[2]);
    return l;
  };
  objects.lightGroup.add(make(rig.key), make(rig.fill), make(rig.rim));
  objects.lightGroup.add(new THREE.AmbientLight(
    new THREE.Color(rig.ambientColor[0], rig.ambientColor[1], rig.ambientColor[2]),
    rig.ambientIntensity,
  ));
}

/** Position a THREE camera from view-core orbit state (no control takeover). */
export function applyOrbitCamera(
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
  state: OrbitCameraState,
  aspect: number,
): void {
  const cp = Math.cos(state.pitchRad);
  const dir: [number, number, number] = [cp * Math.sin(state.yawRad), Math.sin(state.pitchRad), cp * Math.cos(state.yawRad)];
  camera.position.set(
    state.target[0] + dir[0] * state.distanceM,
    state.target[1] + dir[1] * state.distanceM,
    state.target[2] + dir[2] * state.distanceM,
  );
  camera.up.set(0, 1, 0);
  camera.lookAt(state.target[0], state.target[1], state.target[2]);
  if (camera instanceof THREE.OrthographicCamera) {
    const halfH = state.orthoHeightM / 2;
    camera.left = -halfH * aspect;
    camera.right = halfH * aspect;
    camera.top = halfH;
    camera.bottom = -halfH;
    camera.near = state.nearM;
    camera.far = state.farM;
    camera.updateProjectionMatrix();
  } else {
    camera.aspect = aspect;
    camera.near = state.nearM;
    camera.far = state.farM;
    camera.fov = (state.fovYRad * 180) / Math.PI;
    camera.updateProjectionMatrix();
  }
}

/** Dispose every GPU-side resource owned by the scene objects. */
export function disposeRenderObjects(objects: SceneObjects): { geometries: number; materials: number } {
  let geometries = 0, materials = 0;
  objects.garmentGeometry.dispose();
  geometries++;
  for (const key of Object.keys(objects.materialByPanel)) {
    objects.materialByPanel[key].dispose();
    materials++;
  }
  if (objects.wireframe) {
    objects.wireframe.geometry.dispose();
    geometries++;
    (objects.wireframe.material as THREE.Material).dispose();
    materials++;
  }
  if (objects.avatar) {
    objects.avatar.geometry.dispose();
    geometries++;
    ((objects.avatar as THREE.Mesh).material as THREE.Material).dispose();
    materials++;
  }
  const disposeGroup = (group: THREE.Group): void => {
    for (const child of [...group.children]) {
      const mesh = child as THREE.Mesh | THREE.Line;
      (mesh.geometry as THREE.BufferGeometry)?.dispose?.();
      geometries++;
    }
  };
  disposeGroup(objects.seamGroup);
  if (objects.boundary) {
    objects.boundary.geometry.dispose();
    geometries++;
    (objects.boundary.material as THREE.Material).dispose();
    materials++;
  }
  objects.scene.clear();
  objects.disposed = true;
  return { geometries, materials };
}
