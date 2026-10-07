// G11C tests: readiness states, seam comparison, measurements, tolerances.
import { describe, expect, it } from "vitest";
import {
  addAllowance,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { addGrainline } from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import { createPatternDocument, createPanel } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import type { Seam } from "../../src/garment/sewing.js";
import { polygonFixture } from "./fixtures.js";

function twoPanelGarment() {
  let document = createPatternDocument("g11c", "two panels");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3);
  document = b.document;
  const seam: Seam = {
    id: "seam/ab",
    sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
    sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: false },
    stitchCount: 5,
  };
  return { document, a: a.refs, b: b.refs, seam };
}

function readySet(doc: ReturnType<typeof twoPanelGarment>["document"], refs: { panelId: string; loopId: string }[]) {
  let set = createProductionSet();
  for (const r of refs) {
    set = addAllowance(set, r.panelId, r.loopId, 0.01).set;
    const g = centeredGrainline(doc, r.panelId);
    set = addGrainline(set, r.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: r.panelId, cutQuantity: 2 });
  }
  return set;
}

describe("G11C readiness", () => {
  it("reports READY_FOR_EXPORT for a complete garment", () => {
    const { document, a, b, seam } = twoPanelGarment();
    const set = readySet(document, [a, b]);
    const report = productionReadiness(document, [seam], set);
    expect(report.state).toBe("READY_FOR_EXPORT");
    expect(report.errorCount).toBe(0);
    expect(report.panels).toHaveLength(2);
    expect(report.totalCutAreaM2).toBeCloseTo(2 * 0.4 * 0.3 * 2, 9);
    expect(report.seams).toHaveLength(1);
    expect(report.seams[0].withinTolerance).toBe(true);
    expect(report.seams[0].diffM).toBeCloseTo(0, 12);
  });

  it("flags seam mismatch as INVALID without touching geometry", () => {
    const { document, a, b, seam } = twoPanelGarment();
    // B side uses the 0.3 m edge vs A's 0.3 m edge... craft real mismatch:
    // A's right edge is 0.3; use A's bottom (0.4) vs B's left (0.3).
    const bad: Seam = {
      ...seam,
      sideA: { ...seam.sideA, segmentIds: [a.segmentIds[0]] },
    };
    const set = readySet(document, [a, b]);
    const before = JSON.stringify(document);
    const report = productionReadiness(document, [bad], set);
    expect(report.state).toBe("INVALID");
    expect(report.diagnostics.map((d) => d.code)).toContain("seam:mismatch");
    expect(JSON.stringify(document)).toBe(before); // untouched
  });

  it("warns on missing grainlines and unsewn panels", () => {
    const { document, a, b, seam } = twoPanelGarment();
    let set = createProductionSet();
    set = setPanelMeta(set, { panelId: a.panelId, cutQuantity: 1 });
    set = setPanelMeta(set, { panelId: b.panelId, cutQuantity: 1 });
    const report = productionReadiness(document, [seam], set);
    expect(report.state).toBe("WARNINGS");
    expect(report.diagnostics.map((d) => d.code)).toContain("panel:missing-grainline");
  });

  it("errors on missing cut quantity and broken kernel geometry", () => {
    const { document, a, b, seam } = twoPanelGarment();
    const set = readySet(document, [a, b]);
    // Drop B's metadata.
    const stripped = { ...set, panelMeta: set.panelMeta.filter((m) => m.panelId === a.panelId) };
    const report = productionReadiness(document, [seam], stripped);
    expect(report.state).toBe("INVALID");
    expect(report.diagnostics.map((d) => d.code)).toContain("panel:missing-meta");
    // Broken kernel: empty panel.
    let bad = createPatternDocument("bad");
    const p = createPanel(bad, "ghost");
    bad = p.document;
    const report2 = productionReadiness(bad, [], createProductionSet());
    expect(report2.state).toBe("INVALID");
    expect(report2.diagnostics[0].severity).toBe("error");
    expect(report2.diagnostics[0].suggestedAction.length).toBeGreaterThan(0);
  });

  it("maps allowance spikes to warnings and self-intersections to errors", () => {
    // Needle triangle: acute corner forces a miter spike (warning, not error).
    const needle = polygonFixture([[0, 0], [1, 0], [0, 0.01]]);
    let set = createProductionSet();
    set = addAllowance(set, needle.panelId, needle.loopId, 0.05).set;
    const g = centeredGrainline(needle.document, needle.panelId);
    set = addGrainline(set, needle.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: needle.panelId, cutQuantity: 1 });
    const spiky = productionReadiness(needle.document, [], set);
    expect(spiky.diagnostics.map((d) => d.code)).toContain("allowance:spike");
    expect(spiky.state).toBe("WARNINGS");
    // V bite with a crossing offset (error).
    const v = polygonFixture([[0, 0], [2, 0], [2, 1], [1.2, 1], [1, 0.2], [0.8, 1], [0, 1]]);
    let set2 = createProductionSet();
    set2 = addAllowance(set2, v.panelId, v.loopId, 0.3).set;
    const g2 = centeredGrainline(v.document, v.panelId);
    set2 = addGrainline(set2, v.panelId, g2.from, g2.to).set;
    set2 = setPanelMeta(set2, { panelId: v.panelId, cutQuantity: 1 });
    const crossed = productionReadiness(v.document, [], set2);
    expect(crossed.diagnostics.map((d) => d.code)).toContain("allowance:self-intersection");
    expect(crossed.state).toBe("INVALID");
  });

  it("honours custom tolerances", () => {
    const { document, a, b, seam } = twoPanelGarment();
    const bad: Seam = { ...seam, sideA: { ...seam.sideA, segmentIds: [a.segmentIds[0]] } };
    const set = readySet(document, [a, b]);
    const strict = productionReadiness(document, [bad], set, { seamMismatchM: 0.001 });
    expect(strict.state).toBe("INVALID");
    const lax = productionReadiness(document, [bad], set, { seamMismatchM: 10 });
    expect(lax.diagnostics.map((d) => d.code)).not.toContain("seam:mismatch");
  });

  it("sorts diagnostics errors-first and deterministically", () => {
    const { document, a, b, seam } = twoPanelGarment();
    const set = createProductionSet(); // nothing: meta+grainline missing everywhere
    const r1 = productionReadiness(document, [seam], set);
    const r2 = productionReadiness(document, [seam], set);
    expect(JSON.stringify(r1.diagnostics)).toBe(JSON.stringify(r2.diagnostics));
    expect(r1.diagnostics[0].severity).toBe("error");
  });
});
