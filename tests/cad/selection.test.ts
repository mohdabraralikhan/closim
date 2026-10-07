// G9A selection state: dedupe/order, add/remove/toggle, pruning after
// entity deletion, and per-kind statistics.
import { describe, it, expect } from "vitest";
import { deletePanel, movePanel } from "../../src/pattern/cad.js";
import {
  addToSelection,
  clearSelection,
  isSelected,
  pruneSelection,
  removeFromSelection,
  selectionOf,
  selectionStats,
  setSelection,
  toggleSelection,
} from "../../src/cad/selection.js";
import { rectFixture } from "./fixtures.js";

describe("selection basics", () => {
  it("dedupes while preserving first order", () => {
    const sel = selectionOf(["b", "a", "b", "c", "a"]);
    expect(sel).toEqual(["b", "a", "c"]);
  });

  it("set/add/remove/toggle/clear behave", () => {
    const sel = setSelection(["p1", "p2"]);
    expect(isSelected(sel, "p1")).toBe(true);
    expect(isSelected(sel, "p3")).toBe(false);

    const added = addToSelection(sel, ["p3", "p1"]); // p1 dup ignored
    expect(added).toEqual(["p1", "p2", "p3"]);

    const removed = removeFromSelection(added, ["p2", "nope"]);
    expect(removed).toEqual(["p1", "p3"]);

    // p1 was IN the selection, so toggling removes it; p9 is new.
    const toggled = toggleSelection(removed, ["p1", "p9"]);
    expect(toggled).toEqual(["p3", "p9"]);

    // p1 is now absent, so toggling it again appends it.
    const toggledAgain = toggleSelection(toggled, ["p1"]);
    expect(toggledAgain).toEqual(["p3", "p9", "p1"]);

    expect(clearSelection()).toEqual([]);
  });
});

describe("prune + stats", () => {
  it("pruneSelection drops ids whose entities no longer exist", () => {
    const f = rectFixture();
    const sel = [
      f.panelId,
      f.points.bl,
      f.segments.bottom,
      f.loopId,
      "ghost/point",
    ];
    // Everything real survives; the never-valid id is already pruned.
    expect(pruneSelection(f.document, sel)).toEqual(
      sel.filter((id) => id !== "ghost/point"),
    );

    // Deleting the panel removes its point/segment/loop ids too; the
    // never-valid id was already gone. Everything is pruned.
    const withoutPanel = deletePanel(f.document, f.panelId);
    expect(pruneSelection(withoutPanel, sel)).toEqual([]);
  });

  it("selectionStats counts by kind and ignores unknown ids", () => {
    const f = rectFixture();
    const stats = selectionStats(f.document, [
      f.points.bl,
      f.points.br,
      f.segments.bottom,
      f.panelId,
      f.loopId,
      "unknown-id",
    ]);
    expect(stats).toEqual({
      points: 2,
      segments: 1,
      panels: 1,
      loops: 1,
      dimensions: 0,
      constraints: 0,
      total: 5, // unknown id ignored
    });
  });

  it("selection survives document transforms (ids are stable)", () => {
    const f = rectFixture();
    const sel = selectionOf([f.points.tr, f.segments.right]);
    const moved = movePanel(f.document, f.panelId, [3, 3]);
    expect(pruneSelection(moved, sel)).toEqual(sel);
    expect(isSelected(sel, f.points.tr)).toBe(true);
  });
});
