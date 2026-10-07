// G10A selection state: stable entity IDs only, never array indices.
// Panel/seam IDs come from the pattern document and survive rebuilds;
// vertex-index regions are explicit and treated as unstable across rebuilds.

import type { AssembledGarment } from "../garment/assembly.js";

export interface RegionSelection {
  vertexStart: number;
  vertexCount: number;
  label: string;
}

export interface SelectionState {
  garment: boolean;
  avatar: boolean;
  panelIds: string[];
  seamIds: string[];
  region: RegionSelection | null;
}

export function createSelection(): SelectionState {
  return { garment: false, avatar: false, panelIds: [], seamIds: [], region: null };
}

export function clearSelection(sel: SelectionState): void {
  sel.garment = false;
  sel.avatar = false;
  sel.panelIds = [];
  sel.seamIds = [];
  sel.region = null;
}

export function anySelected(sel: SelectionState): boolean {
  return sel.garment || sel.avatar || sel.panelIds.length > 0 || sel.seamIds.length > 0 || sel.region !== null;
}

function pushUnique(list: string[], id: string): void {
  if (!list.includes(id)) list.push(id);
}

export function togglePanel(sel: SelectionState, panelId: string): void {
  if (sel.panelIds.includes(panelId)) {
    sel.panelIds = sel.panelIds.filter((id) => id !== panelId);
  } else {
    pushUnique(sel.panelIds, panelId);
  }
}

export function toggleSeam(sel: SelectionState, seamId: string): void {
  if (sel.seamIds.includes(seamId)) {
    sel.seamIds = sel.seamIds.filter((id) => id !== seamId);
  } else {
    pushUnique(sel.seamIds, seamId);
  }
}

export function setPanelSelection(sel: SelectionState, panelIds: string[]): void {
  sel.panelIds = [];
  for (const id of panelIds) pushUnique(sel.panelIds, id);
}

export function setSeamSelection(sel: SelectionState, seamIds: string[]): void {
  sel.seamIds = [];
  for (const id of seamIds) pushUnique(sel.seamIds, id);
}

export function setRegionSelection(sel: SelectionState, region: RegionSelection | null): void {
  sel.region = region;
}

export interface PrunedSelection {
  panels: string[];
  seams: string[];
  region: boolean;
}

/** Drop selection entries that no longer resolve in the given assembly. */
export function pruneSelection(sel: SelectionState, assembled: AssembledGarment | null): PrunedSelection {
  const removed: PrunedSelection = { panels: [], seams: [], region: false };
  if (!assembled) {
    removed.panels = [...sel.panelIds];
    removed.seams = [...sel.seamIds];
    removed.region = sel.region !== null;
    clearSelection(sel);
    return removed;
  }
  const panelIds = new Set(assembled.panelRanges.map((r) => r.panelId));
  const seamIds = new Set(assembled.weldPairs.map((w) => w.seamId));
  const vertexCount = assembled.positions.length / 3;
  removed.panels = sel.panelIds.filter((id) => !panelIds.has(id));
  removed.seams = sel.seamIds.filter((id) => !seamIds.has(id));
  sel.panelIds = sel.panelIds.filter((id) => panelIds.has(id));
  sel.seamIds = sel.seamIds.filter((id) => seamIds.has(id));
  if (sel.region) {
    if (sel.region.vertexStart < 0 || sel.region.vertexStart + sel.region.vertexCount > vertexCount) {
      sel.region = null;
      removed.region = true;
    }
  }
  return removed;
}

/** Per-vertex highlight mask (1 = highlighted) from panels + region. */
export function highlightVertexMask(assembled: AssembledGarment, sel: SelectionState): Uint8Array {
  const n = assembled.positions.length / 3;
  const mask = new Uint8Array(n);
  for (const panelId of sel.panelIds) {
    const range = assembled.panelRanges.find((r) => r.panelId === panelId);
    if (!range) continue;
    mask.fill(1, range.vertexStart, range.vertexStart + range.vertexCount);
  }
  if (sel.region) {
    mask.fill(1, sel.region.vertexStart, sel.region.vertexStart + sel.region.vertexCount);
  }
  return mask;
}

export interface SelectionStats {
  garment: boolean;
  avatar: boolean;
  panels: number;
  seams: number;
  regionVertices: number;
}

export function selectionStats(sel: SelectionState): SelectionStats {
  return {
    garment: sel.garment,
    avatar: sel.avatar,
    panels: sel.panelIds.length,
    seams: sel.seamIds.length,
    regionVertices: sel.region ? sel.region.vertexCount : 0,
  };
}
