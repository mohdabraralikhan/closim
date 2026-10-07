// G9A snapping: fixed priority (vertex > intersection > midpoint >
// on-segment > grid > none), tolerance gating, panel scoping, and
// byte-stable determinism across repeated calls.
import { describe, it, expect } from "vitest";
import { createConstructionLine, createPoint, movePanel } from "../../src/pattern/cad.js";
import { snapPosition, type SnapKind } from "../../src/cad/snap.js";
import { rectFixture } from "./fixtures.js";

const W = 0.4;
const H = 0.3;
const TOL = 0.02;

/** Panel with two crossing construction diagonals (for intersection snaps). */
function crossedDiagonals() {
  const f = rectFixture();
  const p1 = createPoint(f.document, f.panelId, [0, 0], "construction");
  const p2 = createPoint(p1.document, f.panelId, [W, H], "construction");
  const d1 = createConstructionLine(p2.document, f.panelId, p1.pointId, p2.pointId);
  const p3 = createPoint(d1.document, f.panelId, [0, H], "construction");
  const p4 = createPoint(p3.document, f.panelId, [W, 0], "construction");
  const d2 = createConstructionLine(p4.document, f.panelId, p3.pointId, p4.pointId);
  return { ...f, document: d2.document, diag1: d1.segmentId, diag2: d2.segmentId };
}

function kindOf(result: { kind: SnapKind }): SnapKind {
  return result.kind;
}

describe("snap priority", () => {
  it("vertex wins when the cursor sits on a point", () => {
    const f = rectFixture();
    const r = snapPosition(f.document, [0.003, -0.002], { toleranceM: TOL });
    expect(kindOf(r)).toBe("vertex");
    expect(r.pos).toEqual([0, 0]);
    expect(r.sourceIds).toEqual([f.points.bl]);
  });

  it("intersection wins over on-segment near a crossing", () => {
    const f = crossedDiagonals();
    // Near the X center (0.2, 0.15) but off both diagonals slightly.
    const r = snapPosition(f.document, [0.205, 0.155], { toleranceM: TOL });
    expect(kindOf(r)).toBe("intersection");
    expect(r.pos[0]).toBeCloseTo(W / 2, 9);
    expect(r.pos[1]).toBeCloseTo(H / 2, 9);
    expect(r.sourceIds.sort()).toEqual([f.diag1, f.diag2].sort());
  });

  it("midpoint wins over plain on-segment", () => {
    const f = rectFixture();
    // Slightly off the middle of the bottom edge.
    const r = snapPosition(f.document, [W / 2 + 0.004, 0.006], { toleranceM: TOL });
    expect(kindOf(r)).toBe("midpoint");
    expect(r.pos).toEqual([W / 2, 0]);
    expect(r.sourceIds).toEqual([f.segments.bottom]);
  });

  it("on-segment projects when nothing better is nearby", () => {
    const f = rectFixture();
    const r = snapPosition(f.document, [0.33, 0.008], { toleranceM: TOL });
    expect(kindOf(r)).toBe("on-segment");
    expect(r.pos[0]).toBeCloseTo(0.33, 12);
    expect(r.pos[1]).toBeCloseTo(0, 12);
  });

  it("grid is the lowest-priority fallback", () => {
    const f = rectFixture();
    // 0.012 is closer to the 0.01 grid point than to any panel geometry
    // far away from the rectangle (move query away from the panel).
    const r = snapPosition(f.document, [1.004, 2.003], { toleranceM: TOL, gridM: 0.01 });
    expect(kindOf(r)).toBe("grid");
    expect(r.pos[0]).toBeCloseTo(1, 12);
    expect(r.pos[1]).toBeCloseTo(2, 12);
    expect(r.sourceIds).toEqual([]);

    // Near a vertex the vertex still beats the grid.
    const v = snapPosition(f.document, [0.004, 0.004], { toleranceM: TOL, gridM: 0.01 });
    expect(kindOf(v)).toBe("vertex");
  });

  it("returns kind none with the raw position beyond tolerance", () => {
    const f = rectFixture();
    const raw: [number, number] = [10, 10];
    const r = snapPosition(f.document, raw, { toleranceM: TOL });
    expect(kindOf(r)).toBe("none");
    expect(r.pos).toBe(raw); // raw passthrough
    expect(r.sourceIds).toEqual([]);
  });

  it("rejects non-positive tolerance", () => {
    const f = rectFixture();
    expect(() => snapPosition(f.document, [0, 0], { toleranceM: 0 })).toThrow();
  });
});

describe("determinism + scoping", () => {
  it("repeated calls produce byte-identical results", () => {
    const f = crossedDiagonals();
    for (const q of [[0.2, 0.15], [0.001, 0.001], [W / 2, 0.001], [5, 5]] as Array<[number, number]>) {
      const a = snapPosition(f.document, q, { toleranceM: TOL, gridM: 0.05 });
      const b = snapPosition(f.document, q, { toleranceM: TOL, gridM: 0.05 });
      expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    }
  });

  it("panelIds restricts candidates", () => {
    const f = rectFixture();
    const other = rectFixture(0.4, 0.3, [5, 5], "other-doc");
    const merged = {
      ...f.document,
      panels: [...f.document.panels, ...other.document.panels],
      points: [...f.document.points, ...other.document.points],
      segments: [...f.document.segments, ...other.document.segments],
    };
    // Cursor at the other panel's corner: without scoping it snaps there.
    const unscoped = snapPosition(merged, [5.002, 5.002], { toleranceM: TOL });
    expect(unscoped.kind).toBe("vertex");

    const scoped = snapPosition(merged, [5.002, 5.002], { toleranceM: TOL, panelIds: [f.panelId] });
    expect(scoped.kind).toBe("none"); // first panel is far away
  });

  it("snaps to a vertex of a transformed panel in workspace space", () => {
    const f = rectFixture();
    const movedDoc = movePanel(f.document, f.panelId, [2, 0]);
    const r = snapPosition(movedDoc, [2.002, -0.002], { toleranceM: TOL });
    expect(r.kind).toBe("vertex");
    expect(r.pos[0]).toBeCloseTo(2, 12);
    expect(r.pos[1]).toBeCloseTo(0, 12);
  });
});
