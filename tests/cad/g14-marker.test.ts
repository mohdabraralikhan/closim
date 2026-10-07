// G14 tests — marker building, nesting invariants, grain constraints,
// efficiency math, SVG rendering, and gate refusal.
import { describe, expect, it } from "vitest";
import { buildMarker } from "../../src/cad/marker.js";
import { exportMarkerSVG } from "../../src/cad/marker-svg.js";
import { engineeredGarment, skewedGrainFixture } from "./g14-fixtures.js";

const W = 1.4; // marker width (m) — fits both rotated 0.64-wide pieces side by side

describe("G14 marker: construction and invariants", () => {
  it("places all requested pieces inside the frame without collisions", () => {
    const f = engineeredGarment();
    const marker = buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: W });
    expect(marker.placements).toHaveLength(2);
    expect(marker.declined).toHaveLength(0);
    expect(marker.footprints).toHaveLength(2);
    for (const p of marker.placements) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeGreaterThanOrEqual(0);
      // Cut-boundary footprint is the sewing ring + 10 mm allowance: 0.48 x 0.64.
      const w = p.rotation % 180 === 0 ? 0.48 : 0.64;
      expect(p.x + w).toBeLessThanOrEqual(W + 1e-9);
      const fp = marker.footprints.find((q) => q.pieceId === p.pieceId)!;
      for (const pt of fp.ring) {
        expect(pt[0]).toBeGreaterThanOrEqual(-1e-9);
        expect(pt[0]).toBeLessThanOrEqual(W + 1e-9);
        expect(pt[1]).toBeGreaterThanOrEqual(-1e-9);
      }
    }
  });

  it("is deterministic: identical inputs give identical markers", () => {
    const a = engineeredGarment();
    const b = engineeredGarment();
    const m1 = buildMarker(a.document, a.seams, a.set, markerPiecesFor(a), { widthM: W });
    const m2 = buildMarker(b.document, b.seams, b.set, markerPiecesFor(b), { widthM: W });
    expect(JSON.stringify({ p: m1.placements, e: m1.efficiency, l: m1.lengthM }))
      .toBe(JSON.stringify({ p: m2.placements, e: m2.efficiency, l: m2.lengthM }));
  });

  it("honours quantities and mirror flags with unique piece ids", () => {
    const f = engineeredGarment();
    const marker = buildMarker(
      f.document,
      f.seams,
      f.set,
      [
        { panelId: f.refs.front.panelId, quantity: 2 },
        { panelId: f.refs.back.panelId, mirrored: true },
      ],
      { widthM: W },
    );
    expect(marker.placements).toHaveLength(3);
    const ids = marker.placements.map((p) => p.pieceId);
    expect(new Set(ids).size).toBe(3);
    expect(ids).toContain(`${f.refs.back.panelId}-m`);
    expect(ids.filter((id) => id.startsWith(`${f.refs.front.panelId}#`))).toHaveLength(2);
    expect(marker.placements.find((p) => p.mirrored)).toBeDefined();
  });

  it("keeps grain-constrained pieces warp-parallel (fixture grain is vertical, so rotation is required)", () => {
    const f = engineeredGarment();
    const marker = buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: W });
    for (const p of marker.placements) {
      expect([90, 270]).toContain(p.rotation);
    }
  });

  it("declines a piece too wide for the frame with a reason (relaxed mode)", () => {
    const f = engineeredGarment();
    const marker = buildMarker(
      f.document,
      f.seams,
      f.set,
      markerPiecesFor(f),
      { widthM: 0.5, relaxed: true },
    );
    expect(marker.declined).toHaveLength(2);
    expect(marker.declined[0].reason).toMatch(/orientation fits the marker width/);
    expect(marker.placements).toHaveLength(0);
  });

  it("throws in strict mode when a piece cannot be placed", () => {
    const f = engineeredGarment();
    expect(() =>
      buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: 0.4 }),
    ).toThrow(/could not be placed/);
  });

  it("respects a fixed frame length and reports efficiency on that frame", () => {
    const f = engineeredGarment();
    const marker = buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), {
      widthM: W,
      lengthM: 1.6,
    });
    expect(marker.lengthM).toBe(1.6);
    expect(marker.efficiency.frameAreaM2).toBeCloseTo(1.6 * W, 9);
    expect(marker.efficiency.pieceAreaM2).toBeCloseTo(2 * 0.48 * 0.64, 9); // cut-boundary area
    expect(marker.efficiency.efficiency).toBeCloseTo((2 * 0.48 * 0.64) / (1.6 * W), 9);
    expect(marker.efficiency.efficiency).toBeGreaterThan(0.2);
    expect(marker.efficiency.efficiency).toBeLessThan(1);
  });

  it("rejects invalid frames and empty piece lists", () => {
    const f = engineeredGarment();
    expect(() => buildMarker(f.document, f.seams, f.set, [], { widthM: W })).toThrow(/at least one piece/);
    expect(() => buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: -1 })).toThrow(/width/);
    expect(() => buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: Number.NaN })).toThrow(/width/);
    expect(() =>
      buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: W, clearanceM: -0.1 }),
    ).toThrow(/clearance/);
  });

  it("refuses to build when the gate blocks a production-invalid document", () => {
    const f = engineeredGarment();
    // Dropping all panel meta makes every panel "missing cut quantity" — a
    // readiness error, so the gate state is INVALID and the marker refuses.
    const broken = { ...f.set, panelMeta: [] };
    expect(() =>
      buildMarker(f.document, f.seams, broken, markerPiecesFor(f), { widthM: W }),
    ).toThrow(/marker refused/);
  });

  it("warns instead of rotating a piece whose grain cannot reach the warp", () => {
    const skew = skewedGrainFixture();
    const marker = buildMarker(
      skew.document,
      skew.seams,
      skew.set,
      [{ panelId: skew.panelId }],
      { widthM: 1.0, skipGate: true },
    );
    expect(marker.warnings.some((w) => w.includes("cannot be aligned"))).toBe(true);
    expect(marker.placements[0].rotation).toBe(0); // never silently off-grain
  });

  it("allows free rotation for grainless pieces when configured", () => {
    const skew = skewedGrainFixture();
    const noGrain = { ...skew.set, grainlines: [] };
    const marker = buildMarker(
      skew.document,
      skew.seams,
      noGrain,
      [{ panelId: skew.panelId }],
      { widthM: 1.0, skipGate: true },
    );
    expect(marker.placements).toHaveLength(1);
    expect(marker.warnings.some((w) => w.includes("cannot be aligned"))).toBe(false);
  });

  it("nests a six-piece mixed set with clearance and deterministic ordering", () => {
    const f = engineeredGarment();
    const pieces = [
      { panelId: f.refs.front.panelId, quantity: 2 },
      { panelId: f.refs.back.panelId, quantity: 2 },
      { panelId: f.refs.front.panelId, mirrored: true },
      { panelId: f.refs.back.panelId, mirrored: true },
    ];
    const marker = buildMarker(f.document, f.seams, f.set, pieces, { widthM: 1.8 });
    expect(marker.placements).toHaveLength(6);
    // Independent pairwise gap check on the emitted footprints (rectangular
    // pieces: the bbox gap IS the polygon gap). Every pair must keep the
    // configured clearance in at least one axis.
    const boxes = marker.footprints.map((fp) => {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const q of fp.ring) {
        if (q[0] < minX) minX = q[0];
        if (q[1] < minY) minY = q[1];
        if (q[0] > maxX) maxX = q[0];
        if (q[1] > maxY) maxY = q[1];
      }
      return { minX, minY, maxX, maxY };
    });
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const gapX = Math.max(a.minX - b.maxX, b.minX - a.maxX);
        const gapY = Math.max(a.minY - b.maxY, b.minY - a.maxY);
        expect(Math.max(gapX, gapY)).toBeGreaterThanOrEqual(0.005 - 1e-9);
      }
    }
    expect(marker.efficiency.efficiency).toBeGreaterThan(0.4);
    expect(marker.efficiency.efficiency).toBeLessThan(1);
  });
});

describe("G14 marker: SVG", () => {
  it("renders a true-scale deterministic SVG with efficiency metadata", () => {
    const f = engineeredGarment();
    const marker = buildMarker(f.document, f.seams, f.set, markerPiecesFor(f), { widthM: W });
    const a = exportMarkerSVG(marker, { title: "G14 marker" });
    const b = exportMarkerSVG(marker, { title: "G14 marker" });
    expect(a.svg).toBe(b.svg);
    expect(a.svg).toContain('data-true-scale="1"');
    expect(a.svg).toContain("width=\"1410mm\""); // 1.4 m frame + 2x5 mm margins
    expect(a.svg).toMatch(/efficiency [0-9.]+%/);
    expect(a.svg).toContain(`data-piece="${f.refs.front.panelId}"`);
    expect(a.svg).toContain(`data-piece="${f.refs.back.panelId}"`);
  });
});

/** Piece inputs bound to the fixture's kernel-allocated panel IDs. */
function markerPiecesFor(f: { refs: { front: { panelId: string }; back: { panelId: string } } }) {
  return [
    { panelId: f.refs.front.panelId },
    { panelId: f.refs.back.panelId },
  ];
}
