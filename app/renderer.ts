// G10A three.js adapter. Thin by design: all state lives in the pure view-core
// (workspace + camera + selection); this class only mirrors that state into
// GPU objects and never feeds anything back into it.

import * as THREE from "three";
import { panelPointToWorld, type AssembledGarment, type PanelPlacement } from "../src/garment/assembly.js";
import { resolveStitchPairs, type Seam } from "../src/garment/sewing.js";
import { signedDistanceToAvatar, type AvatarSpec } from "../src/garment/avatar.js";
import type { PatternDocument } from "../src/pattern/cad.js";
import type { PinManager } from "../src/view/manipulation.js";
import type { GarmentWorkspace } from "../src/view/viewport.js";
import type { OrbitCamera } from "../src/view/camera.js";
import { WorkspaceError } from "../src/view/types.js";

const COL_FABRIC = new THREE.Color(0xe8e0d0);
const COL_HIGHLIGHT = new THREE.Color(0xffc04d);
const COL_SEAM = 0xffd27a;
const COL_SEAM_SELECTED = 0xff8c42;
const COL_BOUNDARY = 0xd8dde4;
const COL_BOUNDARY_SELECTED = 0xffc04d;
const COL_AVATAR = 0x9aa0a8;
const COL_HEAT_OK = new THREE.Color(0x2f9e44);
const COL_HEAT_NEAR = new THREE.Color(0xf59f00);
const COL_HEAT_INSIDE = new THREE.Color(0xe03131);

function boundaryEdgePositions(assembled: AssembledGarment, panelId: string | null): Float32Array {
  const segments: number[] = [];
  for (const range of assembled.panelRanges) {
    if (panelId !== null && range.panelId !== panelId) continue;
    const counts = new Map<string, number>();
    const edge = new Map<string, [number, number]>();
    for (let t = range.triangleStart; t < range.triangleStart + range.triangleCount; t++) {
      for (let e = 0; e < 3; e++) {
        const a = assembled.indices[t * 3 + e];
        const b = assembled.indices[t * 3 + (e + 1) % 3];
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        counts.set(key, (counts.get(key) ?? 0) + 1);
        edge.set(key, [a, b]);
      }
    }
    for (const [key, count] of counts) {
      if (count !== 1) continue;
      const [a, b] = edge.get(key)!;
      segments.push(
        assembled.positions[a * 3], assembled.positions[a * 3 + 1], assembled.positions[a * 3 + 2],
        assembled.positions[b * 3], assembled.positions[b * 3 + 1], assembled.positions[b * 3 + 2],
      );
    }
  }
  return new Float32Array(segments);
}

export class ViewportRenderer {
  readonly three: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();

  private perspective = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  private orthographic = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);

  private garmentGeometry: THREE.BufferGeometry | null = null;
  private garmentMesh: THREE.Mesh | null = null;
  private garmentMaterial: THREE.MeshStandardMaterial;
  private positionAttribute: THREE.BufferAttribute | null = null;
  private colorAttribute: THREE.BufferAttribute | null = null;

  private avatarMesh: THREE.Mesh | null = null;
  private grid: THREE.GridHelper;
  private seamLines = new Map<string, THREE.LineSegments>();
  private seamGroup = new THREE.Group();
  private boundaryLines: THREE.LineSegments | null = null;
  private boundarySelectedLines: THREE.LineSegments | null = null;
  private normalLines: THREE.LineSegments | null = null;
  private pinGroup = new THREE.Group();
  private pinMeshes: THREE.Mesh[] = [];
  private pinGeometry = new THREE.SphereGeometry(0.008, 10, 8);
  private pinMaterial = new THREE.MeshBasicMaterial({ color: 0xff5c5c });

  private avatar: AvatarSpec | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.three = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.three.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.three.shadowMap.enabled = true;
    this.three.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = new THREE.Color(0x20242a);

    const hemi = new THREE.HemisphereLight(0xcfd8e3, 0x30343b, 0.9);
    hemi.position.set(0.5, 1, 0.2);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.1);
    dir.position.set(1.5, 3, 2);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.left = -1.5;
    dir.shadow.camera.right = 1.5;
    dir.shadow.camera.top = 2;
    dir.shadow.camera.bottom = -0.5;
    dir.shadow.camera.near = 0.5;
    dir.shadow.camera.far = 8;
    dir.shadow.bias = -0.0002;
    this.scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0x8899aa, 0.4);
    dir2.position.set(-2, 1, -1.5);
    this.scene.add(dir2);

    this.grid = new THREE.GridHelper(4, 20, 0x3c4450, 0x2b313a);
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.6;
    this.scene.add(this.grid);
    // Shadow-catcher ground just under the grid (avoids z-fighting).
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(3, 48),
      new THREE.ShadowMaterial({ opacity: 0.32 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.001;
    ground.receiveShadow = true;
    this.scene.add(ground);
    this.scene.add(this.seamGroup);
    this.scene.add(this.pinGroup);

    this.garmentMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      color: COL_FABRIC,
      roughness: 0.85,
      metalness: 0,
      side: THREE.DoubleSide,
    });
  }

  resize(width: number, height: number): void {
    this.three.setSize(width, height, false);
  }

  /** Recreate all garment-derived scene objects. Call after setGarment. */
  rebuild(ws: GarmentWorkspace, seams: readonly Seam[], document: PatternDocument, placements: PanelPlacement[]): void {
    const assembled = ws.requireAssembled();
    this.disposeGarmentObjects();

    this.avatar = ws.project?.avatar ?? null;
    const avatar = this.avatar;
    const vertexCount = assembled.positions.length / 3;

    const geometry = new THREE.BufferGeometry();
    const positionAttr = new THREE.BufferAttribute(ws.renderPositions, 3);
    positionAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", positionAttr);
    const colorArray = new Float32Array(vertexCount * 3);
    const colorAttr = new THREE.BufferAttribute(colorArray, 3);
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("color", colorAttr);
    geometry.setIndex(new THREE.BufferAttribute(assembled.indices, 1));
    geometry.computeVertexNormals();
    this.garmentGeometry = geometry;
    this.positionAttribute = positionAttr;
    this.colorAttribute = colorAttr;
    this.garmentMesh = new THREE.Mesh(geometry, this.garmentMaterial);
    this.garmentMesh.frustumCulled = false;
    this.garmentMesh.castShadow = true;
    this.garmentMesh.receiveShadow = true;
    this.scene.add(this.garmentMesh);
    this.updateColors(ws);

    // avatar
    if (avatar) {
      const avatarGeometry = new THREE.BufferGeometry();
      avatarGeometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(avatar.positions), 3));
      avatarGeometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(avatar.indices), 1));
      avatarGeometry.computeVertexNormals();
      this.avatarMesh = new THREE.Mesh(avatarGeometry, new THREE.MeshStandardMaterial({
        color: COL_AVATAR,
        roughness: 0.55,
        metalness: 0,
        side: THREE.DoubleSide,
      }));
      this.avatarMesh.castShadow = true;
      this.avatarMesh.receiveShadow = true;
      this.scene.add(this.avatarMesh);
    }

    // seams: stitch polylines per side, placed into world space
    for (const seam of seams) {
      const points: number[] = [];
      try {
        const stitches = resolveStitchPairs(document, seam);
        const placementA = placementFor(placements, seam.sideA.panelId);
        const placementB = placementFor(placements, seam.sideB.panelId);
        for (let i = 0; i < stitches.length; i++) {
          const wa = panelPointToWorld(document, seam.sideA.panelId, stitches[i].pointA, placementA);
          const wb = panelPointToWorld(document, seam.sideB.panelId, stitches[i].pointB, placementB);
          if (i > 0) {
            const prevA = panelPointToWorld(document, seam.sideA.panelId, stitches[i - 1].pointA, placementA);
            const prevB = panelPointToWorld(document, seam.sideB.panelId, stitches[i - 1].pointB, placementB);
            points.push(prevA[0], prevA[1], prevA[2], wa[0], wa[1], wa[2]);
            points.push(prevB[0], prevB[1], prevB[2], wb[0], wb[1], wb[2]);
          }
          points.push(wa[0], wa[1], wa[2], wb[0], wb[1], wb[2]);
        }
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        // A stale seam must never block the viewport; skip its overlay only.
        continue;
      }
      const seamGeometry = new THREE.BufferGeometry();
      seamGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points), 3));
      const lines = new THREE.LineSegments(seamGeometry, new THREE.LineBasicMaterial({ color: COL_SEAM }));
      lines.userData.seamId = seam.id;
      lines.frustumCulled = false;
      this.seamLines.set(seam.id, lines);
      this.seamGroup.add(lines);
    }

    // panel boundaries
    this.boundaryLines = makeLineSegments(boundaryEdgePositions(assembled, null), COL_BOUNDARY);
    this.boundarySelectedLines = makeLineSegments(new Float32Array(0), COL_BOUNDARY_SELECTED);
    this.scene.add(this.boundaryLines, this.boundarySelectedLines);
    this.updateSelectedBoundaries(ws);
  }

  /** Re-upload the position buffer when the workspace published new state. */
  sync(ws: GarmentWorkspace): void {
    if (!ws.viewDirty || !this.garmentGeometry || !this.positionAttribute) return;
    this.positionAttribute.needsUpdate = true;
    this.garmentGeometry.computeVertexNormals();
    this.garmentGeometry.computeBoundingSphere();
    if (ws.visibility.normals) this.rebuildNormalLines(ws);
    if (ws.visibility.penetrationHeat) this.updateColors(ws);
    ws.clearViewDirty();
  }

  render(ws: GarmentWorkspace): void {
    const aspect = this.three.domElement.width / Math.max(this.three.domElement.height, 1);
    this.syncCamera(ws.camera, aspect);
    const visibility = ws.visibility;
    if (this.garmentMesh) this.garmentMesh.visible = visibility.garment;
    this.garmentMaterial.wireframe = visibility.wireframe;
    if (this.avatarMesh) this.avatarMesh.visible = visibility.avatar;
    this.grid.visible = visibility.grid;
    this.seamGroup.visible = visibility.seams;
    if (this.boundaryLines) this.boundaryLines.visible = visibility.panelBoundaries;
    if (this.boundarySelectedLines) this.boundarySelectedLines.visible = visibility.panelBoundaries;
    if (this.normalLines) this.normalLines.visible = visibility.normals;
    this.pinGroup.visible = visibility.pins;
    const active = ws.camera.state.mode === "orthographic" ? this.orthographic : this.perspective;
    this.three.render(this.scene, active);
  }

  refreshSelection(ws: GarmentWorkspace): void {
    this.updateColors(ws);
    this.updateSelectedBoundaries(ws);
    if (ws.visibility.normals) this.rebuildNormalLines(ws);
    for (const [seamId, lines] of this.seamLines) {
      const material = lines.material as THREE.LineBasicMaterial;
      material.color.set(ws.selection.seamIds.includes(seamId) ? COL_SEAM_SELECTED : COL_SEAM);
    }
  }

  updatePins(pins: PinManager): void {
    const list = pins.list();
    while (this.pinMeshes.length < list.length) {
      const mesh = new THREE.Mesh(this.pinGeometry, this.pinMaterial);
      this.pinMeshes.push(mesh);
      this.pinGroup.add(mesh);
    }
    for (let i = 0; i < this.pinMeshes.length; i++) {
      const mesh = this.pinMeshes[i];
      if (i < list.length) {
        mesh.visible = true;
        mesh.position.set(list[i].target[0], list[i].target[1], list[i].target[2]);
      } else {
        mesh.visible = false;
      }
    }
  }

  private syncCamera(camera: OrbitCamera, aspect: number): void {
    const b = camera.basis();
    const state = camera.state;
    const active = state.mode === "orthographic" ? this.orthographic : this.perspective;
    active.matrixAutoUpdate = false;
    active.matrixWorldAutoUpdate = false;
    const back = new THREE.Vector3(-b.forward[0], -b.forward[1], -b.forward[2]);
    active.matrixWorld.makeBasis(
      new THREE.Vector3(b.right[0], b.right[1], b.right[2]),
      new THREE.Vector3(b.up[0], b.up[1], b.up[2]),
      back,
    );
    active.matrixWorld.setPosition(b.eye[0], b.eye[1], b.eye[2]);
    active.matrixWorldInverse.copy(active.matrixWorld).invert();
    active.projectionMatrix.fromArray(camera.projectionMatrix(aspect));
    active.projectionMatrixInverse.copy(active.projectionMatrix).invert();
  }

  private updateColors(ws: GarmentWorkspace): void {
    const assembled = ws.assembled;
    const geometry = this.garmentGeometry;
    const colorAttr = this.colorAttribute;
    if (!assembled || !geometry || !colorAttr) return;
    const colors = colorAttr.array as Float32Array;
    const n = assembled.positions.length / 3;
    if (ws.visibility.penetrationHeat && this.avatar) {
      const positions = ws.renderPositions;
      for (let v = 0; v < n; v++) {
        const d = signedDistanceToAvatar([positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]], this.avatar);
        const c = d < 0 ? COL_HEAT_INSIDE : d < 0.008 ? COL_HEAT_NEAR : COL_HEAT_OK;
        colors[v * 3] = c.r; colors[v * 3 + 1] = c.g; colors[v * 3 + 2] = c.b;
      }
    } else {
      // Single neutral fabric tone; selection is the only tint. The panel
      // index zebra striping is gone on purpose: it read as toy plastic.
      const white = new THREE.Color(0xffffff);
      for (let v = 0; v < n; v++) {
        const c = ws.selection.panelIds.includes(assembled.vertexPanelIds[v]) ? COL_HIGHLIGHT : white;
        colors[v * 3] = c.r; colors[v * 3 + 1] = c.g; colors[v * 3 + 2] = c.b;
      }
    }
    colorAttr.needsUpdate = true;
  }

  private updateSelectedBoundaries(ws: GarmentWorkspace): void {
    const assembled = ws.assembled;
    if (!assembled || !this.boundarySelectedLines) return;
    if (ws.selection.panelIds.length === 0) {
      this.boundarySelectedLines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
      return;
    }
    const parts: Float32Array[] = [];
    for (const panelId of ws.selection.panelIds) {
      parts.push(boundaryEdgePositions(assembled, panelId));
    }
    let total = 0;
    for (const p of parts) total += p.length;
    const merged = new Float32Array(total);
    let offset = 0;
    for (const p of parts) {
      merged.set(p, offset);
      offset += p.length;
    }
    this.boundarySelectedLines.geometry.setAttribute("position", new THREE.BufferAttribute(merged, 3));
  }

  private rebuildNormalLines(ws: GarmentWorkspace): void {
    const geometry = this.garmentGeometry;
    if (!geometry || !this.garmentMesh) return;
    const normalAttr = geometry.getAttribute("normal") as THREE.BufferAttribute;
    const positionAttr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const n = positionAttr.count;
    const linePositions = new Float32Array(n * 6);
    const scale = 0.02;
    for (let v = 0; v < n; v++) {
      const x = positionAttr.array[v * 3];
      const y = positionAttr.array[v * 3 + 1];
      const z = positionAttr.array[v * 3 + 2];
      const nx = normalAttr.array[v * 3];
      const ny = normalAttr.array[v * 3 + 1];
      const nz = normalAttr.array[v * 3 + 2];
      linePositions[v * 6] = x;
      linePositions[v * 6 + 1] = y;
      linePositions[v * 6 + 2] = z;
      linePositions[v * 6 + 3] = x + nx * scale;
      linePositions[v * 6 + 4] = y + ny * scale;
      linePositions[v * 6 + 5] = z + nz * scale;
    }
    if (this.normalLines) {
      this.normalLines.geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    } else {
      this.normalLines = makeLineSegments(linePositions, 0x9fd35f);
      this.scene.add(this.normalLines);
    }
    this.normalLines.visible = ws.visibility.normals;
  }

  private disposeGarmentObjects(): void {
    const disposeGeometry = (object: THREE.Object3D | null) => {
      if (!object) return;
      this.scene.remove(object);
      const mesh = object as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    };
    disposeGeometry(this.garmentMesh);
    disposeGeometry(this.avatarMesh);
    disposeGeometry(this.boundaryLines);
    disposeGeometry(this.boundarySelectedLines);
    disposeGeometry(this.normalLines);
    if (this.avatarMesh) (this.avatarMesh.material as THREE.Material).dispose();
    for (const lines of [this.boundaryLines, this.boundarySelectedLines, this.normalLines]) {
      if (lines) (lines.material as THREE.Material).dispose();
    }
    for (const lines of this.seamLines.values()) {
      this.seamGroup.remove(lines);
      lines.geometry.dispose();
      (lines.material as THREE.Material).dispose();
    }
    this.seamLines.clear();
    this.garmentGeometry = null;
    this.garmentMesh = null;
    this.positionAttribute = null;
    this.colorAttribute = null;
    this.avatarMesh = null;
    this.boundaryLines = null;
    this.boundarySelectedLines = null;
    this.normalLines = null;
  }
}

function makeLineSegments(positions: Float32Array, color: number): THREE.LineSegments {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
  lines.frustumCulled = false;
  return lines;
}

function placementFor(placements: PanelPlacement[], panelId: string): PanelPlacement {
  const placement = placements.find((p) => p.panelId === panelId);
  if (!placement) throw new WorkspaceError("invalid-entity", `no placement for panel ${panelId}`);
  return placement;
}
