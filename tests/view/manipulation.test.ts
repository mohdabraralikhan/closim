import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import {
  DragController,
  PinManager,
  defaultPlacements,
  movePanelPlacement,
  repositionAroundAvatar,
  rotatePanelPlacement,
  transformGarmentPlacements,
  withPlacements,
} from "../../src/view/manipulation.js";
import { GarmentWorkspace } from "../../src/view/viewport.js";
import { SimulationSession } from "../../src/view/sim-session.js";
import { WorkspaceError } from "../../src/view/types.js";
import { buildTshirtFixture } from "./fixtures.js";

function makeStack() {
  const fixture = buildTshirtFixture();
  const ws = new GarmentWorkspace();
  const solver = new CpuSolver();
  const session = new SimulationSession(solver, (pos) => ws.publishSimPositions(pos));
  ws.setGarment(fixture.project, fixture.assembled, "initial-load");
  session.attach(fixture.fitting, fixture.project.simulation);
  const pins = new PinManager();
  pins.attach(ws, solver);
  const drag = new DragController(pins, ws);
  return { fixture, ws, solver, session, pins, drag };
}

describe("placement manipulation", () => {
  it("moves one panel without touching the 2D pattern", () => {
    const { fixture } = makeStack();
    const before = JSON.stringify(fixture.project.pattern);
    const next = movePanelPlacement(fixture.project, fixture.project.pattern.panels[0].id, [0.1, 0, -0.05]);
    expect(JSON.stringify(next.pattern)).toBe(before);
    expect(next.metadata.revision).toBe(fixture.project.metadata.revision + 1);
    const placement = next.placements.find((p) => p.panelId === fixture.project.pattern.panels[0].id)!;
    const original = fixture.project.placements.find((p) => p.panelId === fixture.project.pattern.panels[0].id)!;
    expect(placement.translation[0]).toBeCloseTo(original.translation[0] + 0.1, 12);
    expect(placement.translation[2]).toBeCloseTo(original.translation[2] - 0.05, 12);
    expect(fixture.project.placements.find((p) => p.panelId === fixture.project.pattern.panels[0].id)!.translation[0])
      .toBeCloseTo(original.translation[0], 12);
  });

  it("rotates a single panel yaw", () => {
    const { fixture } = makeStack();
    const panelId = fixture.project.pattern.panels[0].id;
    const next = rotatePanelPlacement(fixture.project, panelId, Math.PI / 6);
    const placement = next.placements.find((p) => p.panelId === panelId)!;
    const original = fixture.project.placements.find((p) => p.panelId === panelId)!;
    expect(placement.yawRad).toBeCloseTo(original.yawRad + Math.PI / 6, 12);
    expect(placement.translation).toEqual(original.translation);
  });

  it("translates and yaws the whole garment around a pivot", () => {
    const { fixture } = makeStack();
    const next = transformGarmentPlacements(fixture.project, [0, 0.2, 0], Math.PI / 2, [0, 0, 0]);
    expect(next.placements.length).toBe(fixture.project.placements.length);
    const original = fixture.project.placements[0];
    const rotated = next.placements.find((p) => p.panelId === original.panelId)!;
    expect(rotated.translation[0]).toBeCloseTo(original.translation[2], 12);
    expect(rotated.translation[2]).toBeCloseTo(-original.translation[0], 12);
    expect(rotated.translation[1]).toBeCloseTo(original.translation[1] + 0.2, 12);
    expect(rotated.yawRad).toBeCloseTo(original.yawRad + Math.PI / 2, 12);
  });

  it("recenters the garment on the avatar XZ", () => {
    const { fixture } = makeStack();
    const shifted = transformGarmentPlacements(fixture.project, [0.8, 0, 0.5], 0, [0, 0, 0]);
    const ap = Float32Array.from(fixture.avatar.positions);
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < ap.length; i += 3) {
      minX = Math.min(minX, ap[i]); maxX = Math.max(maxX, ap[i]);
      minZ = Math.min(minZ, ap[i + 2]); maxZ = Math.max(maxZ, ap[i + 2]);
    }
    const bounds = { min: [0.8, 0, 0.5] as [number, number, number], max: [1.26, 0.62, 1.12] as [number, number, number] };
    const next = repositionAroundAvatar(shifted, bounds, fixture.avatar);
    const dx = (next.placements[0].translation[0] - shifted.placements[0].translation[0]);
    const dz = (next.placements[0].translation[2] - shifted.placements[0].translation[2]);
    expect(dx).toBeCloseTo((minX + maxX) / 2 - (bounds.min[0] + bounds.max[0]) / 2, 9);
    expect(dz).toBeCloseTo((minZ + maxZ) / 2 - (bounds.min[2] + bounds.max[2]) / 2, 9);
  });

  it("rejects unknown panels and keeps defaults restorable", () => {
    const { fixture } = makeStack();
    expect(() => movePanelPlacement(fixture.project, "no-such-panel", [0, 0, 0])).toThrowError(WorkspaceError);
    const defaults = defaultPlacements(fixture.project);
    const moved = transformGarmentPlacements(fixture.project, [1, 1, 1], 0, [0, 0, 0]);
    const restored = withPlacements(moved, defaults);
    expect(restored.placements).toEqual(fixture.project.placements);
  });
});

describe("pin manager", () => {
  it("pins through the solver constraint interface", () => {
    const { fixture, solver, session, pins } = makeStack();
    void fixture;
    const record = pins.create(3);
    expect(record.id).toBe("pin/1");
    expect(record.garmentEpoch).toBe(1);
    // Hard yank (half a metre on a ~2.5 cm mesh) — the pin must still hold
    // the vertex exactly through a solver step without blowing up.
    const yanked: [number, number, number] = [0.6, 1.1, 0.5];
    pins.updateTarget("pin/1", yanked);
    session.stepOnce(1);
    const positions = solver.getPositions();
    expect(positions[9]).toBe(yanked[0]);
    expect(positions[10]).toBe(yanked[1]);
    expect(positions[11]).toBe(yanked[2]);
    expect(pins.remove("pin/1")).toBe(true);
    expect(pins.remove("pin/1")).toBe(false);
  });

  it("rejects invalid pins", () => {
    const { pins } = makeStack();
    expect(() => pins.create(-1)).toThrowError(WorkspaceError);
    expect(() => pins.create(99999)).toThrowError(WorkspaceError);
    expect(() => pins.create(0, [Number.NaN, 0, 0])).toThrowError(WorkspaceError);
    expect(() => pins.updateTarget("missing", [0, 0, 0])).toThrowError(WorkspaceError);
  });

  it("drops all pins when the garment is replaced", () => {
    const { fixture, ws, pins } = makeStack();
    pins.create(0);
    pins.create(1);
    expect(pins.list().length).toBe(2);
    ws.setGarment(fixture.project, fixture.assembled, "rebuild");
    expect(pins.list().length).toBe(0);
    expect(pins.invalidation).not.toBeNull();
    expect(pins.invalidation!.removed.length).toBe(2);
    expect(pins.invalidation!.reason).toBe("garment-replaced");
  });

  it("serializes and restores pins for the current epoch only", () => {
    const { fixture, ws, pins } = makeStack();
    pins.create(2);
    pins.create(5);
    const serialized = pins.serialize();
    pins.clear();
    expect(pins.list().length).toBe(0);
    const restored = pins.restore(serialized);
    expect(restored.length).toBe(2);
    expect(pins.list().length).toBe(2);
    ws.setGarment(fixture.project, fixture.assembled, "rebuild");
    expect(pins.restore(serialized).length).toBe(0);
  });
});

describe("drag controller", () => {
  it("drags a vertex through a temporary pin and releases cleanly", () => {
    const { fixture, solver, ws, pins, drag } = makeStack();
    const vertex = 0;
    const grabPoint: [number, number, number] = [
      ws.renderPositions[0], ws.renderPositions[1], ws.renderPositions[2],
    ];
    const state = drag.begin(vertex, grabPoint);
    expect(drag.isActive).toBe(true);
    expect(pins.list().length).toBe(1);
    drag.moveTo([0.5, 0.5, 0.5]);
    expect(state.pin.target).toEqual([0.5, 0.5, 0.5]);
    expect(solver.getPositions().length).toBeGreaterThan(0);
    void fixture;
    const released = drag.end(false);
    expect(released).toBeNull();
    expect(pins.list().length).toBe(0);
    expect(drag.isActive).toBe(false);
  });

  it("keeps the pin when dropping with keepPin", () => {
    const { ws, pins, drag } = makeStack();
    drag.begin(0, [ws.renderPositions[0], ws.renderPositions[1], ws.renderPositions[2]]);
    const kept = drag.end(true);
    expect(kept).not.toBeNull();
    expect(pins.list().length).toBe(1);
  });

  it("preserves the grab offset while dragging", () => {
    const { ws, drag } = makeStack();
    const px = ws.renderPositions[0];
    const py = ws.renderPositions[1];
    const pz = ws.renderPositions[2];
    drag.begin(0, [px + 0.05, py, pz]);
    drag.moveTo([0, 0, 0]);
    expect(drag.state!.pin.target[0]).toBeCloseTo(-0.05, 9);
    expect(drag.state!.pin.target[1]).toBeCloseTo(0, 9);
    expect(drag.state!.pin.target[2]).toBeCloseTo(0, 9);
    drag.cancel();
    expect(drag.isActive).toBe(false);
  });

  it("kills the drag when the garment is rebuilt underneath it", () => {
    const { fixture, ws, pins, drag } = makeStack();
    drag.begin(0, [ws.renderPositions[0], ws.renderPositions[1], ws.renderPositions[2]]);
    ws.setGarment(fixture.project, fixture.assembled, "rebuild");
    expect(drag.isActive).toBe(false);
    expect(pins.list().length).toBe(0);
    expect(() => drag.moveTo([0, 0, 0])).toThrowError(WorkspaceError);
  });

  it("refuses a second concurrent drag", () => {
    const { ws, drag } = makeStack();
    drag.begin(0, [ws.renderPositions[0], ws.renderPositions[1], ws.renderPositions[2]]);
    expect(() => drag.begin(1, [0, 0, 0])).toThrowError(WorkspaceError);
    drag.cancel();
  });
});
