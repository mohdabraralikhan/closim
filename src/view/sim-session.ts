// G10A/B explicit simulation scheduling. The render loop never starts a solve;
// this session is the only place solver.step() is called, and every completed
// step is published to the workspace through the sanctioned transfer path.

import type { ClothSolver } from "../backend/solver.js";
import type { FittingScene } from "../garment/assembly.js";
import type { SimulationConfig } from "../garment/project.js";
import type { StepStats } from "../solver/newton.js";

export type SimSessionStatus = "detached" | "ready" | "playing" | "paused";

export const MAX_STEPS_PER_TICK = 8;

/** Concrete solvers expose per-step diagnostics as a public field (CpuSolver.lastStats). */
type SolverStatsSource = { lastStats?: StepStats | null };

export class SimulationSession {
  readonly solver: ClothSolver;
  private publish: (positions: Float64Array) => void;

  private fitting_: FittingScene | null = null;
  private config_: SimulationConfig | null = null;
  private playing_ = false;
  private stepCount_ = 0;
  private accumulator = 0;
  private lastStats_: StepStats | null = null;
  private lastStepNaNFree_ = true;
  private stablePositions_: Float64Array | null = null;
  private stableStep_ = -1;

  constructor(solver: ClothSolver, publish: (positions: Float64Array) => void) {
    this.solver = solver;
    this.publish = publish;
  }

  get status(): SimSessionStatus {
    if (!this.fitting_) return "detached";
    return this.playing_ ? "playing" : "ready";
  }

  get isPlaying(): boolean {
    return this.playing_;
  }

  get stepCount(): number {
    return this.stepCount_;
  }

  get lastStats(): StepStats | null {
    return this.lastStats_;
  }

  get lastStepNaNFree(): boolean {
    return this.lastStepNaNFree_;
  }

  get config(): SimulationConfig | null {
    return this.config_;
  }

  get stableStep(): number {
    return this.stableStep_;
  }

  /** Attach to a freshly rebuilt garment. Always starts paused. */
  attach(fitting: FittingScene, config: SimulationConfig): void {
    this.fitting_ = fitting;
    this.config_ = config;
    this.solver.initialize(fitting.scene);
    this.playing_ = false;
    this.stepCount_ = 0;
    this.accumulator = 0;
    this.lastStats_ = null;
    this.lastStepNaNFree_ = true;
    this.stablePositions_ = fitting.scene.positions.slice();
    this.stableStep_ = 0;
    this.publish(fitting.scene.positions);
  }

  detach(): void {
    this.fitting_ = null;
    this.config_ = null;
    this.playing_ = false;
    this.stepCount_ = 0;
    this.lastStats_ = null;
    this.stablePositions_ = null;
    this.stableStep_ = -1;
  }

  play(): void {
    if (!this.fitting_) return;
    this.playing_ = true;
  }

  pause(): void {
    this.playing_ = false;
  }

  toggle(): boolean {
    if (this.playing_) this.pause();
    else this.play();
    return this.playing_;
  }

  /** Advance exactly `count` fixed steps regardless of play state. */
  stepOnce(count = 1): void {
    for (let i = 0; i < count; i++) {
      if (!this.doStep()) return;
    }
  }

  /**
   * Drive from the app frame loop with real elapsed time. Fixed-dt accumulator
   * with a hard cap; a stall drops the backlog instead of spiraling.
   */
  tick(realDtSeconds: number): void {
    if (!this.playing_ || !this.fitting_ || !this.config_) return;
    this.accumulator += Math.max(0, Math.min(realDtSeconds, 0.25));
    const dt = this.config_.dt;
    let steps = 0;
    while (this.accumulator >= dt && steps < MAX_STEPS_PER_TICK) {
      if (!this.doStep()) {
        this.accumulator = 0;
        return;
      }
      this.accumulator -= dt;
      steps++;
    }
    if (steps >= MAX_STEPS_PER_TICK) this.accumulator = 0;
  }

  private doStep(): boolean {
    const fitting = this.fitting_;
    const config = this.config_;
    if (!fitting || !config) return false;
    this.solver.step(config.dt);
    this.stepCount_++;
    this.lastStats_ = (this.solver as SolverStatsSource).lastStats ?? null;
    const positions = this.solver.getPositions();
    let nanFree = true;
    for (let i = 0; i < positions.length; i++) {
      if (!Number.isFinite(positions[i])) {
        nanFree = false;
        break;
      }
    }
    this.lastStepNaNFree_ = nanFree;
    if (nanFree) {
      this.stablePositions_ = positions.slice();
      this.stableStep_ = this.stepCount_;
    }
    this.publish(positions);
    return nanFree;
  }

  hasStableState(): boolean {
    return this.stablePositions_ !== null;
  }

  /**
   * Restart from the last NaN-free state: copy the stable snapshot back into
   * the scene (positions from the solver boundary, velocities zeroed), then
   * re-initialize through the solver interface. Pins are not re-applied here;
   * the caller owns pin re-attachment.
   */
  restoreStable(): boolean {
    const fitting = this.fitting_;
    if (!fitting || !this.stablePositions_) return false;
    fitting.scene.positions.set(this.stablePositions_);
    fitting.scene.velocities.fill(0);
    this.accumulator = 0;
    this.playing_ = false;
    this.lastStats_ = null;
    this.lastStepNaNFree_ = true;
    this.solver.initialize(fitting.scene);
    this.publish(fitting.scene.positions);
    return true;
  }
}
