import type { ClothSolver } from "./solver.js";
import type { ClothMaterial } from "../physics/types.js";
import type { ClothScene } from "../physics/scene.js";
import { enforcePins } from "../physics/scene.js";
import { implicitStep, type StepStats } from "../solver/newton.js";

export class CpuSolver implements ClothSolver {
  private scene: ClothScene | null = null;
  lastStats: StepStats | null = null;

  initialize(scene: ClothScene): void {
    this.scene = scene;
  }

  step(dt: number): void {
    if (!this.scene) throw new Error("CpuSolver not initialized");
    this.lastStats = implicitStep(this.scene, dt);
  }

  setMaterial(material: ClothMaterial): void {
    if (!this.scene) throw new Error("CpuSolver not initialized");
    this.scene.material = { ...material };
  }

  pinVertex(vertexId: number, position?: [number, number, number]): void {
    if (!this.scene) throw new Error("CpuSolver not initialized");
    const p: [number, number, number] = position ?? [
      this.scene.positions[vertexId * 3],
      this.scene.positions[vertexId * 3 + 1],
      this.scene.positions[vertexId * 3 + 2],
    ];
    this.scene.pinned.set(vertexId, p);
    enforcePins(this.scene, this.scene.positions);
  }

  unpinVertex(vertexId: number): void {
    this.scene?.pinned.delete(vertexId);
  }

  getPositions(): Float64Array {
    if (!this.scene) throw new Error("CpuSolver not initialized");
    return this.scene.positions;
  }

  getVelocities(): Float64Array {
    if (!this.scene) throw new Error("CpuSolver not initialized");
    return this.scene.velocities;
  }
}
