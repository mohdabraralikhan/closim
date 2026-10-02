import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "./types.js";
import type { ContactSystem } from "../collision/contact-assembly.js";

export interface ClothScene {
  mesh: ClothMeshData;
  material: ClothMaterial;
  gravity: [number, number, number];
  pinned: Map<number, [number, number, number]>; // vertex -> fixed position
  positions: Float64Array; // current x (3n)
  velocities: Float64Array; // current v (3n)
  /** Optional variational contact (Phase 1). Null = free flight (V0 behavior). */
  contact: ContactSystem | null;
}

export function createScene(mesh: ClothMeshData, material: ClothMaterial, gravity: [number, number, number] = [0, -9.81, 0]): ClothScene {
  return {
    mesh,
    material,
    gravity,
    pinned: new Map(),
    positions: Float64Array.from(mesh.positions),
    velocities: new Float64Array(mesh.count * 3),
    contact: null,
  };
}

export function pinColumn(scene: ClothScene, predicate: (x: number, y: number, z: number, id: number) => boolean): void {
  const n = scene.mesh.count;
  for (let i = 0; i < n; i++) {
    const x = scene.positions[i * 3], y = scene.positions[i * 3 + 1], z = scene.positions[i * 3 + 2];
    if (predicate(x, y, z, i)) {
      scene.pinned.set(i, [x, y, z]);
    }
  }
}

export function enforcePins(scene: ClothScene, x: Float64Array): void {
  for (const [id, p] of scene.pinned) {
    x[id * 3] = p[0];
    x[id * 3 + 1] = p[1];
    x[id * 3 + 2] = p[2];
  }
}

export function makePinFilter(pinned: Set<number> | Map<number, unknown>): (v: Float64Array) => void {
  const set = pinned instanceof Map ? new Set(pinned.keys()) : pinned;
  return (v: Float64Array) => {
    for (const id of set) {
      v[id * 3] = 0;
      v[id * 3 + 1] = 0;
      v[id * 3 + 2] = 0;
    }
  };
}
