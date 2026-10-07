// G9A — selection state helpers.
//
// Selection is an ordered, duplicate-free list of entity ids held by the
// caller (G9D's interaction layer owns it; CadSession deliberately does not,
// so headless/tests can drive editing without selection bookkeeping).
// Every helper is pure and deterministic: order-preserving, store-order for
// anything that must enumerate entities.

import type { EntityId, PatternDocument } from "../pattern/cad.js";

export type Selection = EntityId[];
/** Read-only view of a selection (accepts Selection or readonly arrays). */
export type SelectionView = readonly EntityId[];

/** Build a selection from ids, deduplicating while preserving first order. */
export function selectionOf(ids: SelectionView): Selection {
  const seen = new Set<EntityId>();
  const out: Selection = [];
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

export function isSelected(selection: SelectionView, id: EntityId): boolean {
  return selection.includes(id);
}

/** Replace the selection (single-select when one id is given). */
export function setSelection(ids: readonly EntityId[]): Selection {
  return selectionOf(ids);
}

export function addToSelection(selection: SelectionView, ids: SelectionView): Selection {
  return selectionOf([...selection, ...ids]);
}

export function removeFromSelection(selection: SelectionView, ids: SelectionView): Selection {
  const drop = new Set(ids);
  return selection.filter((id) => !drop.has(id));
}

/**
 * Toggle membership. Toggle order matters for the result's ordering: ids
 * removed keep their original positions' absence; new ids append in the
 * order given.
 */
export function toggleSelection(selection: SelectionView, ids: SelectionView): Selection {
  let out: Selection = [...selection];
  for (const id of ids) {
    out = out.includes(id) ? out.filter((x) => x !== id) : [...out, id];
  }
  return out;
}

export function clearSelection(): Selection {
  return [];
}

/** Drop ids whose entities no longer exist (after undo/delete). */
export function pruneSelection(doc: PatternDocument, selection: SelectionView): Selection {
  const points = new Set(doc.points.map((p) => p.id));
  const segments = new Set(doc.segments.map((s) => s.id));
  const panels = new Set(doc.panels.map((p) => p.id));
  const loops = new Set<string>();
  const dimensions = new Set<string>();
  const constraints = new Set<string>();
  for (const panel of doc.panels) {
    for (const l of panel.boundaryLoops) loops.add(l.id);
    for (const d of panel.dimensions) dimensions.add(d.id);
    for (const c of panel.constraints) constraints.add(c.id);
  }
  const exists = (id: EntityId): boolean =>
    points.has(id) || segments.has(id) || panels.has(id) ||
    loops.has(id) || dimensions.has(id) || constraints.has(id);
  return selection.filter(exists);
}

export interface SelectionStats {
  points: number;
  segments: number;
  panels: number;
  loops: number;
  dimensions: number;
  constraints: number;
  total: number;
}

/** Count a selection by entity kind (store-order scan; unknown ids ignored). */
export function selectionStats(doc: PatternDocument, selection: SelectionView): SelectionStats {
  const stats: SelectionStats = { points: 0, segments: 0, panels: 0, loops: 0, dimensions: 0, constraints: 0, total: 0 };
  const wanted = new Set(selection);
  const bump = (bucket: keyof SelectionStats, id: EntityId): void => {
    if (wanted.has(id)) {
      stats[bucket]++;
      stats.total++;
    }
  };
  for (const p of doc.points) bump("points", p.id);
  for (const s of doc.segments) bump("segments", s.id);
  for (const panel of doc.panels) {
    bump("panels", panel.id);
    for (const l of panel.boundaryLoops) bump("loops", l.id);
    for (const d of panel.dimensions) bump("dimensions", d.id);
    for (const c of panel.constraints) bump("constraints", c.id);
  }
  return stats;
}
