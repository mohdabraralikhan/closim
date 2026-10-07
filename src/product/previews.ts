// G19D (integration surface) — product preview assets.
//
// Previews reference (never duplicate) the product revision, garment
// revision, and render configuration that produced them. A garment or config
// change marks affected previews stale explicitly — outdated previews are
// never presented as current. Pixel production stays in the browser render
// path; here generation is a deterministic batch over an injected renderer.

import { PatternCadError } from "../pattern/cad.js";
import type { Product } from "./product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type PreviewView =
  | "primary" | "front" | "back" | "side" | "three-quarter"
  | "technical" | "pattern" | "detail" | "thumbnail";

export type PreviewStatus = "current" | "stale" | "failed" | "missing";

export interface RenderConfigRef {
  camera?: string;
  lighting?: string;
  quality?: string;
  background?: string;
  showAvatar?: boolean;
}

export interface PreviewAsset {
  id: string;
  productId: string;
  productRevision: number;
  view: PreviewView;
  garmentRevision: number;
  garmentFingerprint: string;
  renderConfig: RenderConfigRef;
  filename: string;
  checksum: string;
  status: PreviewStatus;
  widthPx?: number;
  heightPx?: number;
}

/** Deterministic filename: collision-free across products/revisions/views. */
export function previewFilename(slug: string, productRevision: number, view: PreviewView, sizeSegment = "", ext = "png"): string {
  if (!slug) throw new PatternCadError("invalid-document", "preview filename needs a slug");
  const size = sizeSegment ? `-${sizeSegment}` : "";
  return `${slug}-r${productRevision}-${view}${size}.${ext}`;
}

export function createPreview(partial: {
  id: string;
  productId: string;
  productRevision: number;
  view: PreviewView;
  garmentRevision: number;
  garmentFingerprint: string;
  renderConfig?: RenderConfigRef;
  filename: string;
  checksum: string;
  widthPx?: number;
  heightPx?: number;
}): PreviewAsset {
  if (!partial.id || !partial.productId || !partial.filename || !partial.checksum) {
    throw new PatternCadError("invalid-document", "preview needs id, product, filename, and checksum");
  }
  if (!Number.isInteger(partial.productRevision) || partial.productRevision < 1 ||
    !Number.isInteger(partial.garmentRevision) || partial.garmentRevision < 1) {
    throw new PatternCadError("invalid-document", "preview revisions must be positive integers", partial.id);
  }
  return {
    ...clone(partial),
    renderConfig: clone(partial.renderConfig ?? {}),
    status: "current",
  };
}

/** Mark a preview on a preview list (products carry preview *ids*; assets live beside them). */
export function markPreviewAsset(previews: PreviewAsset[], previewId: string, status: PreviewStatus): PreviewAsset[] {
  const next = clone(previews);
  const found = next.find((p) => p.id === previewId);
  if (!found) throw new PatternCadError("missing-reference", `preview '${previewId}' does not exist`, previewId);
  found.status = status;
  return next;
}

/**
 * Staleness after a garment change: every preview pinned to an older
 * garment revision/fingerprint goes stale (returned list; caller persists).
 */
export function staleAfterGarmentChange(
  previews: PreviewAsset[],
  garmentRevision: number,
  garmentFingerprint: string,
): PreviewAsset[] {
  return clone(previews).map((p) =>
    p.garmentRevision !== garmentRevision || p.garmentFingerprint !== garmentFingerprint
      ? { ...p, status: "stale" as PreviewStatus }
      : p,
  );
}

export interface BatchReport {
  completed: string[];
  failed: Array<{ view: PreviewView; reason: string }>;
  stale: string[];
  skipped: string[];
  generated: PreviewAsset[];
}

export interface GeneratedView {
  filename: string;
  checksum: string;
  widthPx: number;
  heightPx: number;
}

/**
 * Batch-generate configured views. The render callback performs (or stubs)
 * actual rendering; failures and skips are reported, never thrown.
 */
export function generatePreviews(
  product: Product,
  views: PreviewView[],
  render: (view: PreviewView) => GeneratedView,
  renderConfig: RenderConfigRef = {},
  now?: string,
): BatchReport {
  void now;
  const completed: string[] = [];
  const failed: Array<{ view: PreviewView; reason: string }> = [];
  const generated: PreviewAsset[] = [];
  for (const view of views) {
    try {
      const out = render(view);
      if (!out || !out.filename || !out.checksum) throw new Error("renderer returned no file");
      generated.push({
        id: `${product.id}/preview/${view}`,
        productId: product.id,
        productRevision: product.revision,
        view,
        garmentRevision: product.garment.garmentRevision,
        garmentFingerprint: product.garment.garmentFingerprint,
        renderConfig: clone(renderConfig),
        filename: out.filename,
        checksum: out.checksum,
        status: "current",
        widthPx: out.widthPx,
        heightPx: out.heightPx,
      });
      completed.push(view);
    } catch (error) {
      failed.push({ view, reason: error instanceof Error ? error.message : String(error) });
    }
  }
  return { completed, failed, stale: [], skipped: [], generated };
}

/** Primary (catalog-display) preview asset, if any is current. */
export function primaryPreview(previews: PreviewAsset[]): PreviewAsset | null {
  const primary = previews.find((p) => p.view === "primary" && p.status === "current")
    ?? previews.find((p) => p.view === "thumbnail" && p.status === "current");
  return primary ?? null;
}
