import { describe, expect, it } from "vitest";
import {
  anySelected,
  clearSelection,
  createSelection,
  highlightVertexMask,
  pruneSelection,
  selectionStats,
  setPanelSelection,
  setRegionSelection,
  setSeamSelection,
  togglePanel,
  toggleSeam,
} from "../../src/view/selection.js";
import { buildTshirtFixture } from "./fixtures.js";

describe("selection state", () => {
  it("starts empty and reports stats", () => {
    const sel = createSelection();
    expect(anySelected(sel)).toBe(false);
    expect(selectionStats(sel)).toEqual({ garment: false, avatar: false, panels: 0, seams: 0, regionVertices: 0 });
  });

  it("toggles panels and seams without duplicates", () => {
    const sel = createSelection();
    togglePanel(sel, "p1");
    togglePanel(sel, "p1");
    togglePanel(sel, "p2");
    expect(sel.panelIds).toEqual(["p2"]);
    toggleSeam(sel, "s1");
    toggleSeam(sel, "s1");
    expect(sel.seamIds).toEqual([]);
    expect(anySelected(sel)).toBe(true);
    sel.panelIds = [];
    expect(anySelected(sel)).toBe(false);
  });

  it("set operations dedupe while preserving order", () => {
    const sel = createSelection();
    setPanelSelection(sel, ["b", "a", "b", "c"]);
    expect(sel.panelIds).toEqual(["b", "a", "c"]);
    setSeamSelection(sel, ["y", "x", "y"]);
    expect(sel.seamIds).toEqual(["y", "x"]);
  });

  it("clear resets everything", () => {
    const sel = createSelection();
    sel.garment = true;
    sel.avatar = true;
    togglePanel(sel, "p");
    toggleSeam(sel, "s");
    setRegionSelection(sel, { vertexStart: 0, vertexCount: 2, label: "r" });
    clearSelection(sel);
    expect(anySelected(sel)).toBe(false);
    expect(sel.region).toBeNull();
  });
});

describe("selection pruning against an assembly", () => {
  const fixture = buildTshirtFixture();
  const frontId = fixture.project.pattern.panels[0].id;

  it("keeps valid ids and reports removed ones", () => {
    const sel = createSelection();
    setPanelSelection(sel, [frontId, "ghost-panel"]);
    setSeamSelection(sel, [fixture.project.seams[0].id, "ghost-seam"]);
    const removed = pruneSelection(sel, fixture.assembled);
    expect(removed.panels).toEqual(["ghost-panel"]);
    expect(removed.seams).toEqual(["ghost-seam"]);
    expect(sel.panelIds).toEqual([frontId]);
    expect(sel.seamIds).toEqual([fixture.project.seams[0].id]);
  });

  it("drops a region outside the vertex range", () => {
    const sel = createSelection();
    const n = fixture.assembled.positions.length / 3;
    setRegionSelection(sel, { vertexStart: 0, vertexCount: n, label: "ok" });
    pruneSelection(sel, fixture.assembled);
    expect(sel.region).not.toBeNull();
    setRegionSelection(sel, { vertexStart: n - 1, vertexCount: 5, label: "bad" });
    const removed = pruneSelection(sel, fixture.assembled);
    expect(removed.region).toBe(true);
    expect(sel.region).toBeNull();
  });

  it("clears everything when the assembly is gone", () => {
    const sel = createSelection();
    sel.garment = true;
    setPanelSelection(sel, [frontId]);
    const removed = pruneSelection(sel, null);
    expect(removed.panels.length).toBe(1);
    expect(anySelected(sel)).toBe(false);
  });

  it("highlight mask covers exactly the selected panel vertices", () => {
    const sel = createSelection();
    setPanelSelection(sel, [frontId]);
    const mask = highlightVertexMask(fixture.assembled, sel);
    expect(mask.length).toBe(fixture.assembled.positions.length / 3);
    const range = fixture.assembled.panelRanges.find((r) => r.panelId === frontId)!;
    let hits = 0;
    for (let i = 0; i < mask.length; i++) hits += mask[i];
    expect(hits).toBe(range.vertexCount);
    expect(mask[range.vertexStart]).toBe(1);
    expect(mask[range.vertexStart + range.vertexCount]).toBe(0);
  });
});
