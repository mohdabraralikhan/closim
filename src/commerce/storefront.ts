// G20 storefront data (integration surface): product pages, variant
// resolution, price lists, and purchase state. Presentation data only —
// checkout executes through the order/payment boundary, never here.

import { PatternCadError } from "../pattern/cad.js";
import type { Entitlement } from "./entitlements.js";
import { isEntitlementUsable } from "./entitlements.js";
import type { Product, ProductArtifact, ProductVariant } from "../product/product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export interface StorefrontPage {
  productId: string;
  name: string;
  description?: string;
  sizes: string[];
  formats: string[];
  files: Array<{ filename: string; type: string; format: string }>;
  requirements?: string;
  license: { type: string; commercialUse: boolean; attribution: boolean };
  version: number;
  previews: string[];
  purchasable: boolean;
  unavailableReason?: string;
}

export function productPage(product: Product, artifacts: ProductArtifact[]): StorefrontPage {
  const current = artifacts.filter((a) => a.status === "current");
  const purchasable = product.status === "published";
  return {
    productId: product.id,
    name: product.name,
    ...(product.description ? { description: product.description } : {}),
    sizes: [...product.sizes],
    formats: [...product.formats],
    files: current.map((a) => ({ filename: a.filename, type: a.type, format: a.format })),
    ...(product.requirements ? { requirements: product.requirements } : {}),
    license: {
      type: product.license.type,
      commercialUse: product.license.allowsCommercialUse,
      attribution: product.license.attributionRequired,
    },
    version: product.revision,
    previews: [...product.previewIds],
    purchasable,
    ...(purchasable ? {} : { unavailableReason: `product is ${product.status}` }),
  };
}

export interface VariantSelection {
  sizeId?: string;
  formatBundle?: string[];
}

/** Resolve a customer selection to an explicit product variant. */
export function resolveVariant(product: Product, selection: VariantSelection): ProductVariant {
  const candidates = product.variants.filter((v) => {
    if (selection.sizeId && !v.sizeIds.includes(selection.sizeId)) return false;
    if (selection.formatBundle && !selection.formatBundle.every((f) => v.formats.includes(f))) return false;
    return true;
  });
  if (candidates.length === 0) {
    throw new PatternCadError("invalid-document", "no variant matches the selection", product.id);
  }
  // Deterministic: prefer exact size+format match, then fewest extras, then id.
  const exact = candidates.filter((v) =>
    (selection.sizeId ? v.sizeIds.length === 1 : true) &&
    (selection.formatBundle ? v.formats.length === selection.formatBundle.length : true),
  );
  const pool = exact.length > 0 ? exact : candidates;
  const sorted = [...pool].sort((a, b) =>
    (a.sizeIds.length + a.formats.length) - (b.sizeIds.length + b.formats.length) ||
    (a.id < b.id ? -1 : 1),
  );
  return clone(sorted[0]);
}

export type PriceList = Record<string, { amountMinor: number; currency: string }>;

export function quoteVariant(
  variant: ProductVariant,
  prices: PriceList,
): { amountMinor: number; currency: string } {
  const quote = prices[variant.id];
  if (!quote) throw new PatternCadError("missing-reference", `no price for variant '${variant.id}'`, variant.id);
  if (!Number.isInteger(quote.amountMinor) || quote.amountMinor < 0 || !quote.currency) {
    throw new PatternCadError("invalid-transform", `invalid price for variant '${variant.id}'`, variant.id);
  }
  return { ...quote };
}

export type PurchaseState = "available" | "owned" | "unavailable" | "discontinued";

/** What the storefront offers this customer for this product. */
export function purchaseState(
  product: Product,
  entitlement: Entitlement | null,
  nowIso: string,
): { state: PurchaseState; entitlementId?: string } {
  if (product.status === "archived") return { state: "discontinued" };
  if (product.status !== "published") return { state: "unavailable" };
  if (entitlement && entitlement.productId === product.id && isEntitlementUsable(entitlement, nowIso)) {
    return { state: "owned", entitlementId: entitlement.id };
  }
  return { state: "available" };
}

export function customerLibrary(
  entitlements: Entitlement[],
  products: Product[],
  nowIso: string,
): Array<{ productId: string; name: string; revision: number; usable: boolean; entitlementId: string }> {
  const byId = new Map(products.map((p) => [p.id, p]));
  return entitlements
    .filter((e) => byId.has(e.productId))
    .map((e) => ({
      productId: e.productId,
      name: byId.get(e.productId)!.name,
      revision: e.productRevision,
      usable: isEntitlementUsable(e, nowIso),
      entitlementId: e.id,
    }))
    .sort((a, b) => (a.productId < b.productId ? -1 : 1));
}
