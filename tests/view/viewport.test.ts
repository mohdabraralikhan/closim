import { describe, expect, it } from "vitest";
import { GarmentWorkspace } from "../../src/view/viewport.js";
import { WorkspaceError } from "../../src/view/types.js";
import { buildTshirtFixture } from "./fixtures.js";

describe("garment workspace", () => {
  const fixture = buildTshirtFixture();
  const frontId = fixture.project.pattern.panels[0].id;
  const vertexCount = fixture.assembled.positions.length / 3;

  it("loads a garment and exposes render state", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    expect(ws.garmentEpoch).toBe(1);
    expect(ws.vertexCount).toBe(vertexCount);
    expect(ws.renderPositions).not.toBe(fixture.assembled.positions);
    expect(Array.from(ws.renderPositions)).toEqual(Array.from(fixture.assembled.positions));
    expect(ws.viewDirty).toBe(true);
    ws.clearViewDirty();
    expect(ws.viewDirty).toBe(false);
  });

  it("rejects non-finite assembly data", () => {
    const ws = new GarmentWorkspace();
    const corrupted = { ...fixture.assembled, positions: fixture.assembled.positions.slice() };
    corrupted.positions[0] = Number.NaN;
    expect(() => ws.setGarment(fixture.project, corrupted, "initial-load")).toThrowError(WorkspaceError);
  });

  it("publishes simulation positions as the only solver->viewport path", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    ws.clearViewDirty();
    const moved = new Float64Array(vertexCount * 3);
    for (let i = 0; i < moved.length; i++) moved[i] = fixture.assembled.positions[i] + 0.01;
    ws.publishSimPositions(moved);
    expect(ws.simEpoch).toBe(1);
    expect(ws.viewDirty).toBe(true);
    expect(ws.renderPositions[0]).toBeCloseTo(fixture.assembled.positions[0] + 0.01, 8);
    expect(() => ws.publishSimPositions(new Float64Array(3))).toThrowError(WorkspaceError);
  });

  it("fires garment-replaced listeners and prunes stale selection", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    const events: string[] = [];
    ws.onGarmentReplaced((e) => events.push(`${e.reason}:${e.garmentEpoch}`));
    ws.selection.garment = true;
    setStaleSelection(ws);
    ws.setGarment(fixture.project, fixture.assembled, "rebuild");
    expect(events).toEqual(["rebuild:2"]);
    expect(ws.selection.panelIds).toEqual([]);
    expect(ws.garmentEpoch).toBe(2);
    function setStaleSelection(workspace: GarmentWorkspace) {
      workspace.selection.panelIds = ["ghost-panel"];
    }
  });

  it("resetToRestPositions discards simulation output", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    const moved = new Float64Array(vertexCount * 3).fill(0.5);
    ws.publishSimPositions(moved);
    ws.resetToRestPositions();
    expect(ws.simEpoch).toBe(0);
    expect(Array.from(ws.renderPositions)).toEqual(Array.from(fixture.assembled.positions));
  });

  it("frames the garment and flags the camera dirty", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    ws.frameGarment(1.5);
    expect(ws.cameraDirty).toBe(true);
    const b = ws.currentBounds();
    expect(ws.camera.state.target[0]).toBeCloseTo((b.min[0] + b.max[0]) / 2, 9);
    expect(ws.camera.state.target[1]).toBeCloseTo((b.min[1] + b.max[1]) / 2, 9);
    ws.clearCameraDirty();
    expect(ws.cameraDirty).toBe(false);
    ws.camera.orbit(0.1, 0);
    expect(ws.cameraDirty).toBe(true);
  });

  it("frames a panel selection around that panel only", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    const range = fixture.assembled.panelRanges.find((r) => r.panelId === frontId)!;
    ws.selection.panelIds = [frontId];
    expect(ws.frameSelection(1.0)).toBe(true);
    const target = ws.camera.state.target;
    expect(target[0]).toBeGreaterThanOrEqual(fixture.assembled.positions[range.vertexStart * 3] - 1e-9);
    const garmentBounds = ws.currentBounds();
    const fits = (v: number, axis: number) => v >= garmentBounds.min[axis] - 1e-9 && v <= garmentBounds.max[axis] + 1e-9;
    expect(fits(target[0], 0)).toBe(true);
    expect(fits(target[1], 1)).toBe(true);
  });

  it("tracks vertex->panel provenance", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    const range = fixture.assembled.panelRanges.find((r) => r.panelId === frontId)!;
    expect(ws.vertexPanelId(range.vertexStart)).toBe(frontId);
    expect(ws.vertexPanelId(vertexCount + 10)).toBeNull();
  });

  it("reset clears the garment and notifies listeners", () => {
    const ws = new GarmentWorkspace();
    ws.setGarment(fixture.project, fixture.assembled, "initial-load");
    const events: number[] = [];
    ws.onGarmentReplaced((e) => events.push(e.garmentEpoch));
    ws.selection.garment = true;
    ws.reset();
    expect(ws.assembled).toBeNull();
    expect(ws.selection.garment).toBe(false);
    expect(ws.vertexCount).toBe(0);
    expect(events.length).toBe(1);
    expect(() => ws.currentBounds()).toThrowError(WorkspaceError);
  });
});
