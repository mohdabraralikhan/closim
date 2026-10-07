// G15A tests: representation sync, normals, visibility, quality, overlays.
import { describe, expect, it } from "vitest";
import {
  boundarySegments,
  buildRenderAvatar,
  buildRenderGarment,
  computeNormals,
  contactHeat,
  defaultVisibility,
  qualitySettings,
  seamPolylines,
  setRenderVisibility,
  syncRenderPositions,
  topologyKeyOf,
} from "../../src/render/representation.js";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { assembleGarment } from "../../src/garment/assembly.js";
import { makeBoxAvatar } from "../../src/garment/avatar.js";

function garment() {
  let document = createPatternDocument("rep", "Rep");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const b = addRectPanel(document, "B", [0.5, 0], 0.2, 0.2);
  document = b.document;
  const g = assembleGarment(document, [], [
    { panelId: a.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
    { panelId: b.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
  ]);
  return { garment: g, panels: [a.refs.panelId, b.refs.panelId] };
}

describe("G15A representation", () => {
  it("computes area-weighted unit normals", () => {
    const normals = computeNormals(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
      new Uint32Array([0, 1, 2]),
    );
    expect(normals.length).toBe(9);
    for (let i = 0; i < 3; i++) {
      const len = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
      expect(len).toBeCloseTo(1, 12);
    }
    // Degenerate triangle falls back to +Y without NaN.
    const flat = computeNormals(new Float32Array(9), new Uint32Array([0, 1, 2]));
    expect(Array.from(flat.slice(0, 3))).toEqual([0, 1, 0]);
  });

  it("builds render state as copies and syncs explicitly", () => {
    const { garment: g, panels } = garment();
    const render = buildRenderGarment("g1", g, g.positions, { [panels[0]]: "m1", [panels[1]]: "m2" });
    expect(render.epoch).toBe(0);
    expect(render.topologyKey).toBe(topologyKeyOf(g));
    expect(render.positions).not.toBe(g.positions);
    expect(render.panelUVs).toHaveLength(2);
    // Mutating the source does not move the render state.
    g.positions[0] = 999;
    expect(render.positions[0]).not.toBe(999);
    g.positions[0] = 0;
    // Explicit sync refreshes in place.
    const moved = Float32Array.from(g.positions);
    moved[1] += 0.25;
    expect(syncRenderPositions(render, g, moved, 7)).toBe("updated");
    expect(render.positions[1]).toBeCloseTo(0.25, 6);
    expect(render.epoch).toBe(7);
    // Count mismatch is a topology change, never a silent remap.
    expect(syncRenderPositions(render, g, new Float32Array(3), 8)).toBe("topology-changed");
    expect(render.epoch).toBe(7);
    expect(() => buildRenderGarment("g", g, new Float32Array(3), {})).toThrowError(/vertex count/);
  });

  it("detects topology changes across rebuilds", () => {
    const { garment: g } = garment();
    let document = createPatternDocument("rep2", "Rep2");
    const r = addRectPanel(document, "A", [0, 0], 1, 1);
    document = r.document;
    const g2 = assembleGarment(document, [], [{ panelId: r.refs.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    expect(topologyKeyOf(g2)).not.toBe(topologyKeyOf(g));
    const render = buildRenderGarment("g1", g, g.positions, {});
    expect(syncRenderPositions(render, g2, g2.positions, 1)).toBe("topology-changed");
  });

  it("toggles visibility and validates quality levels", () => {
    const v = defaultVisibility();
    expect(v.garment).toBe(true);
    expect(setRenderVisibility(v, "wireframe", true).wireframe).toBe(true);
    expect(() => setRenderVisibility(v, "nope" as "garment", true)).toThrowError(/visibility key/);
    expect(qualitySettings("draft").shadows).toBe(false);
    expect(qualitySettings("high").textureAnisotropy).toBe(8);
    expect(qualitySettings("preview").boundaryOverlay).toBe(true);
    expect(() => qualitySettings("ultra" as "draft")).toThrowError(/quality level/);
  });

  it("derives seam and boundary overlays from construction data", () => {
    let document = createPatternDocument("seam", "Seam");
    const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
    document = a.document;
    const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3);
    document = b.document;
    const g = assembleGarment(document, [{
      id: "seam/ab",
      sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
      sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: false },
      stitchCount: 4,
    }], [
      { panelId: a.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
      { panelId: b.refs.panelId, translation: [0, 0, 0], yawRad: 0 },
    ]);
    const render = buildRenderGarment("g", g, g.positions, {});
    const seams = seamPolylines(render, g);
    expect(seams.get("seam/ab")).toHaveLength(4);
    const bounds = boundarySegments(render);
    expect(bounds.length).toBeGreaterThan(0);
    // Boundary segments lie on the garment surface (z=0 plane here).
    for (const s of bounds) expect(s[2]).toBeCloseTo(0, 9);
  });

  it("computes collision heat without touching the solver", () => {
    const { garment: g } = garment();
    const avatar = makeBoxAvatar({ halfExtentsM: [5, 5, 5], center: [0, 0, 0] });
    const heat = contactHeat(g.positions, avatar, 0.002);
    expect(heat.length).toBe(g.positions.length / 3);
    for (const h of heat) expect(h).toBe(1); // deep inside
    const far = makeBoxAvatar({ halfExtentsM: [0.1, 0.1, 0.1], center: [50, 50, 50] });
    const cold = contactHeat(g.positions, far, 0.002);
    for (const h of cold) expect(h).toBe(0);
    expect(() => contactHeat(g.positions, far, 0)).toThrowError(/heat band/);
  });

  it("builds avatar representations", () => {
    const avatar = makeBoxAvatar({ halfExtentsM: [0.2, 0.3, 0.15], center: [0, 0.9, 0] });
    const render = buildRenderAvatar(avatar);
    expect(render.indices.length).toBe(36);
    expect(render.normals.length).toBe(render.positions.length);
  });
});
