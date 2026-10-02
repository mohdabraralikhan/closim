import type { ClothMaterial } from "../physics/types.js";
import type { ClothScene } from "../physics/scene.js";

/** Stable backend contract — CPU/WASM/GPU must all satisfy this. */
export interface ClothSolver {
  initialize(scene: ClothScene): void;
  step(dt: number): void;
  setMaterial(material: ClothMaterial): void;
  pinVertex(vertexId: number, position?: [number, number, number]): void;
  unpinVertex(vertexId: number): void;
  getPositions(): Float64Array;
  getVelocities(): Float64Array;
}
