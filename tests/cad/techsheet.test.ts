// G11D tests: tech-sheet rendering, toggles, selection, determinism.
import { describe, expect, it } from "vitest";
import {
  addAllowance,
  addAnnotation,
  addCutLine,
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  createProductionSet,
} from "../../src/cad/production.js";
import { renderTechSheet } from "../../src/cad/techsheet.js";
import { rectFixture } from "./fixtures.js";

function fullSheet() {
  const f = rectFixture(0.4, 0.3);
  let set = createProductionSet();
  set = addAllowance(set, f.panelId, f.loopId, 0.01).set;
  set = addNotch(set, f.panelId, f.loopId, f.segments.bottom, 0.5, "double", 0.005).set;
  set = addGrainline(set, f.panelId, [0.2, 0.05], [0.2, 0.25]).set;
  set = addFoldLine(set, f.panelId, [0.1, 0.05], [0.1, 0.25], "valley", "pleat").set;
  set = addDrillMark(set, f.panelId, [0.2, 0.15], "circle").set;
  set = addInternalLine(set, f.panelId, [[0.05, 0.05], [0.35, 0.25]], "dart", "dart").set;
  set = addAnnotation(set, f.panelId, [0.2, 0.15], "note").set;
  set = addLabelRegion(set, f.panelId, [0.05, 0.05], [0.35, 0.12], { panel: "front" }).set;
  set = addCutLine(set, f.panelId, f.loopId, "allowance").set;
  return { f, set };
}

describe("G11D tech sheet", () => {
  it("renders every production layer as SVG", () => {
    const { f, set } = fullSheet();
    const sheet = renderTechSheet(f.document, set, { title: "Top" });
    expect(sheet.svg.startsWith("<svg")).toBe(true);
    expect(sheet.panelIds).toEqual([f.panelId]);
    expect(sheet.svg).toContain("Top");
    expect(sheet.svg).toContain("<polygon"); // boundaries
    // Cut source is the allowance: sewing dashed + allowance as the cut edge.
    expect(sheet.svg).toContain('class="sew"');
    expect(sheet.svg).toContain('class="cut"');
    // Dashed allowance overlay appears when no cut line overrides it.
    const noCut = renderTechSheet(f.document, { ...set, cutLines: [] });
    expect(noCut.svg).toContain('class="allow"');
    expect(sheet.svg).toContain('class="notch"');
    expect(sheet.svg).toContain('class="grain"');
    expect(sheet.svg).toContain('class="fold"');
    expect(sheet.svg).toContain('class="drill"');
    expect(sheet.svg).toContain('class="internal"');
    expect(sheet.svg).toContain("400.0 mm"); // bottom edge dimension
    expect(sheet.svg).toContain("panel: front");
    expect(sheet.widthPx).toBeGreaterThan(0);
    expect(sheet.heightPx).toBeGreaterThan(0);
  });

  it("is deterministic and honours visibility toggles", () => {
    const { f, set } = fullSheet();
    const a = renderTechSheet(f.document, set);
    const b = renderTechSheet(f.document, set);
    expect(a.svg).toBe(b.svg);
    const bare = renderTechSheet(f.document, set, {
      showAllowance: false, showNotches: false, showGrainlines: false,
      showFolds: false, showDrills: false, showInternals: false,
      showAnnotations: false, showLabels: false, showDimensions: false,
    });
    for (const cls of ["allow", "notch", "grain", "fold", "drill", "internal", "dim", "lbl", "note"]) {
      expect(bare.svg).not.toContain(`class="${cls}"`);
    }
    expect(bare.svg).toContain("<polygon"); // cut boundary always present
  });

  it("highlights selection and reports hits", () => {
    const { f, set } = fullSheet();
    const notchId = set.notches[0].id;
    const sheet = renderTechSheet(f.document, set, { selectedIds: [notchId, "ghost", f.panelId] });
    expect(sheet.selectionHits).toContain(notchId);
    expect(sheet.selectionHits).toContain(f.panelId);
    expect(sheet.selectionHits).not.toContain("ghost");
    expect(sheet.svg).toContain('class="notch selected"');
  });

  it("refuses invalid patterns", () => {
    const { f, set } = fullSheet();
    const broken = { ...f.document, panels: [] };
    expect(() => renderTechSheet(broken, set)).toThrowError(/invalid/);
  });

  it("escapes label text", () => {
    const { f, set } = fullSheet();
    let next = set;
    next = addLabelRegion(next, f.panelId, [0.05, 0.2], [0.35, 0.28], { note: "<b>&\"quoted\"</b>" }).set;
    const sheet = renderTechSheet(f.document, next);
    expect(sheet.svg).toContain("&lt;b&gt;&amp;&quot;quoted&quot;&lt;/b&gt;");
    expect(sheet.svg).not.toContain("<b>");
  });
});
