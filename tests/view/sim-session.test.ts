import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import type { ClothSolver } from "../../src/backend/solver.js";
import type { ClothScene } from "../../src/physics/scene.js";
import { SimulationSession } from "../../src/view/sim-session.js";
import { buildTshirtFixture } from "./fixtures.js";

class ScriptedSolver implements ClothSolver {
  scene: ClothScene | null = null;
  steps = 0;
  /** Writes NaN into a vertex once `failAfter` steps have run. */
  failAfter = Infinity;

  initialize(scene: ClothScene): void {
    this.scene = scene;
    this.steps = 0;
  }

  step(dt: number): void {
    if (!this.scene) throw new Error("not initialized");
    this.steps++;
    if (this.steps > this.failAfter) {
      this.scene.positions[0] = Number.NaN;
      return;
    }
    for (let i = 0; i < this.scene.positions.length; i++) {
      this.scene.positions[i] += 0.001 * dt * 60;
    }
  }

  setMaterial(): void {}
  pinVertex(): void {}
  unpinVertex(): void {}
  getPositions(): Float64Array {
    if (!this.scene) throw new Error("not initialized");
    return this.scene.positions;
  }
  getVelocities(): Float64Array {
    if (!this.scene) throw new Error("not initialized");
    return this.scene.velocities;
  }
}

function makeSession(solver: ClothSolver) {
  const fixture = buildTshirtFixture();
  const published: Float64Array[] = [];
  const session = new SimulationSession(solver, (pos) => published.push(pos.slice()));
  session.attach(fixture.fitting, fixture.project.simulation);
  return { fixture, session, published };
}

describe("simulation session", () => {
  it("attaches paused and publishes the initial state", () => {
    const { session, published } = makeSession(new CpuSolver());
    expect(session.status).toBe("ready");
    expect(session.stepCount).toBe(0);
    expect(published.length).toBe(1);
    expect(session.stableStep).toBe(0);
    expect(session.hasStableState()).toBe(true);
  });

  it("stepOnce advances exactly N steps and publishes each", () => {
    const { session, published } = makeSession(new CpuSolver());
    session.stepOnce(5);
    expect(session.stepCount).toBe(5);
    expect(published.length).toBe(6);
    expect(session.isPlaying).toBe(false);
  });

  it("tick only advances while playing and caps the backlog", () => {
    const { session } = makeSession(new CpuSolver());
    session.tick(0.5);
    expect(session.stepCount).toBe(0);
    session.play();
    expect(session.status).toBe("playing");
    session.tick(0.2);
    expect(session.stepCount).toBeLessThanOrEqual(8);
    expect(session.stepCount).toBeGreaterThan(0);
    const afterBurst = session.stepCount;
    session.tick(0);
    expect(session.stepCount).toBe(afterBurst);
    session.pause();
    session.tick(0.5);
    expect(session.stepCount).toBe(afterBurst);
  });

  it("stops stepping and keeps the last stable snapshot when state goes NaN", () => {
    const solver = new ScriptedSolver();
    solver.failAfter = 2;
    const { session } = makeSession(solver);
    session.stepOnce(10);
    expect(session.lastStepNaNFree).toBe(false);
    expect(session.stepCount).toBe(3);
    expect(session.stableStep).toBe(2);
  });

  it("restoreStable restarts from the last NaN-free state through initialize", () => {
    const solver = new ScriptedSolver();
    solver.failAfter = 2;
    const { session, fixture, published } = makeSession(solver);
    session.stepOnce(10);
    expect(published.length).toBe(4);
    const stableSnapshot = published[2];
    for (let i = 0; i < fixture.fitting.scene.positions.length; i++) {
      fixture.fitting.scene.velocities[i] = 7;
    }
    expect(session.restoreStable()).toBe(true);
    expect(session.isPlaying).toBe(false);
    expect(session.lastStepNaNFree).toBe(true);
    const restored = fixture.fitting.scene.positions;
    for (let i = 0; i < restored.length; i++) {
      expect(restored[i]).toBeCloseTo(stableSnapshot[i], 12);
      expect(Number.isFinite(restored[i])).toBe(true);
      expect(fixture.fitting.scene.velocities[i]).toBe(0);
    }
    expect(solver.scene).toBe(fixture.fitting.scene);
  });

  it("restoreStable returns false with no stable state", () => {
    const fixture = buildTshirtFixture();
    const session = new SimulationSession(new CpuSolver(), () => {});
    expect(session.status).toBe("detached");
    expect(session.restoreStable()).toBe(false);
    session.attach(fixture.fitting, fixture.project.simulation);
    session.detach();
    expect(session.restoreStable()).toBe(false);
    expect(session.status).toBe("detached");
  });

  it("real solver steps keep state finite on the t-shirt", () => {
    const { session, fixture } = makeSession(new CpuSolver());
    session.stepOnce(3);
    expect(session.lastStepNaNFree).toBe(true);
    const positions = fixture.fitting.scene.positions;
    let maxDisp = 0;
    for (let i = 0; i < positions.length; i++) {
      expect(Number.isFinite(positions[i])).toBe(true);
      maxDisp = Math.max(maxDisp, Math.abs(positions[i] - fixture.assembled.positions[i]));
    }
    expect(maxDisp).toBeLessThan(1);
    expect(session.lastStats).not.toBeNull();
  });
});
