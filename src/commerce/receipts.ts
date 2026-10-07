// G21B — customer-facing order history, receipts, and license documents.
//
// All three are read faces over authoritative records: orders (G20) and
// product license metadata (G19). Nothing here mutates a purchase, and no
// document ever invents terms — license content comes strictly from product
// metadata (booleans restated as restrictions, creator terms verbatim),
// receipts strictly from the order snapshot. Issued documents are frozen:
// a later product change cannot rewrite what was purchased.

import { PatternCadError } from "../pattern/cad.js";
import type { Order, OrderStatus } from "./orders.js";
import type { Entitlement, ProductRelease } from "./entitlements.js";
import type { Product } from "../product/product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

function requireId(id: string | undefined, label: string): void {
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", `${label} must be non-empty`);
}

function requireTimestamp(value: string | undefined, label: string): void {
  if (!value || Number.isNaN(Date.parse(value))) {
    throw new PatternCadError("invalid-transform", `${label} must be a valid timestamp`);
  }
}

// ---------------------------------------------------------------------------
// Order history
// ---------------------------------------------------------------------------

export interface OrderLineView {
  productId: string;
  variantId: string;
  productRevision: number;
  quantity: number;
  unitPriceMinor: number;
  amountMinor: number;
  /** Enrichment from the product record; frozen into receipts at issue time. */
  name?: string;
  author?: string;
}

export interface OrderHistoryEntry {
  orderId: string;
  date: string;
  status: OrderStatus;
  currency: string;
  totalMinor: number;
  lines: OrderLineView[];
}

function lineView(item: Order["items"][number], product?: Product): OrderLineView {
  return {
    productId: item.productId,
    variantId: item.variantId,
    productRevision: item.productRevision,
    quantity: item.quantity,
    unitPriceMinor: item.unitPriceMinor,
    amountMinor: item.quantity * item.unitPriceMinor,
    ...(product?.name ? { name: product.name } : {}),
    ...(product?.author ? { author: product.author } : {}),
  };
}

/** The customer's own orders, newest first. Foreign orders are unreachable. */
export function orderHistory(
  customerId: string,
  orders: Order[],
  products?: Product[],
): OrderHistoryEntry[] {
  requireId(customerId, "customer id");
  if (!Array.isArray(orders)) throw new PatternCadError("invalid-document", "order history needs an orders array");
  const byId = new Map((products ?? []).map((p) => [p.id, p]));
  return orders
    .filter((order) => order.customerId === customerId)
    .map((order) => ({
      orderId: order.id,
      date: order.createdAt,
      status: order.status,
      currency: order.currency,
      totalMinor: order.totalMinor,
      lines: order.items.map((item) => lineView(item, byId.get(item.productId))),
    }))
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.orderId < b.orderId ? -1 : a.orderId > b.orderId ? 1 : 0));
}

/**
 * Fetch one order for one customer. Miss and foreign access raise the SAME
 * error so order ids cannot be enumerated across accounts.
 */
export function findOrder(orders: Order[], customerId: string, orderId: string): Order {
  requireId(customerId, "customer id");
  requireId(orderId, "order id");
  if (!Array.isArray(orders)) throw new PatternCadError("invalid-document", "order lookup needs an orders array");
  const order = orders.find((o) => o.id === orderId && o.customerId === customerId);
  if (!order) {
    throw new PatternCadError("missing-reference", "order was not found for this customer", orderId);
  }
  return clone(order);
}

// ---------------------------------------------------------------------------
// Receipt (frozen snapshot of the authoritative order)
// ---------------------------------------------------------------------------

export interface Receipt {
  orderId: string;
  issuedAt: string;
  orderDate: string;
  status: OrderStatus;
  customerEmail: string;
  lines: OrderLineView[];
  currency: string;
  totalMinor: number;
  /** Provider-side reference (e.g. payment id) — never card or credential data. */
  transactionReference?: string;
}

export function createReceipt(input: { order: Order; products?: Product[]; now: string }): Receipt {
  const { order, now } = input;
  if (!order || !order.id || !Array.isArray(order.items) || order.items.length === 0) {
    throw new PatternCadError("invalid-document", "receipt needs an order with items", order?.id);
  }
  requireTimestamp(now, "receipt issue time");
  const byId = new Map((input.products ?? []).map((p) => [p.id, p]));
  return {
    orderId: order.id,
    issuedAt: now,
    orderDate: order.createdAt,
    status: order.status,
    customerEmail: order.customerEmail,
    lines: order.items.map((item) => lineView(item, byId.get(item.productId))),
    currency: order.currency,
    totalMinor: order.totalMinor,
    ...(order.paymentId ? { transactionReference: order.paymentId } : {}),
  };
}

// ---------------------------------------------------------------------------
// License document (strictly product metadata — nothing invented)
// ---------------------------------------------------------------------------

export interface LicenseDocument {
  productId: string;
  productName: string;
  sku: string;
  orderId?: string;
  entitlementId: string;
  releaseId?: string;
  /** Product release this license covers (pinned at purchase). */
  releaseRevision: number;
  issuedAt: string;
  license: {
    type: string;
    version: string;
    permittedUse: {
      commercial: boolean;
      modification: boolean;
      redistribution: boolean;
      attribution: boolean;
    };
    restrictions: string[];
    /** Verbatim creator-entered terms, only when the product carries them. */
    terms?: string;
  };
}

/** Restrictions are restatements of recorded metadata — never invented clauses. */
function restrictionsOf(product: Product): { restrictions: string[]; terms?: string } {
  const license = product.license;
  const restrictions: string[] = [];
  if (!license.allowsCommercialUse) restrictions.push("commercial use is not permitted");
  if (!license.allowsModification) restrictions.push("modification is not permitted");
  if (!license.allowsRedistribution) restrictions.push("redistribution is not permitted");
  if (license.attributionRequired) restrictions.push("attribution is required");
  return {
    restrictions,
    ...(license.terms ? { terms: license.terms } : {}),
  };
}

export function createLicenseDocument(input: {
  entitlement: Entitlement;
  product: Product;
  release?: ProductRelease;
  order?: Order;
  now: string;
}): LicenseDocument {
  const { entitlement, product, release, order, now } = input;
  requireTimestamp(now, "license issue time");
  if (!product?.license || !product.license.type || !product.license.version) {
    throw new PatternCadError("invalid-document", "license missing: product carries no usable license metadata", product?.id);
  }
  if (product.license.type === "custom" && !product.license.terms) {
    throw new PatternCadError("invalid-document", "license missing: custom license has no creator-entered terms", product.id);
  }
  if (product.id !== entitlement.productId) {
    throw new PatternCadError("invalid-document", "product does not match the entitlement", product.id);
  }
  if (order && order.id !== entitlement.orderId) {
    throw new PatternCadError("invalid-document", "order does not match the entitlement", order.id);
  }
  if (release) {
    if (release.productId !== entitlement.productId) {
      throw new PatternCadError("invalid-document", `release '${release.id}' belongs to another product`, release.id);
    }
    if (release.revision !== entitlement.productRevision) {
      throw new PatternCadError(
        "invalid-document",
        `license covers purchased release ${entitlement.productRevision}, not ${release.revision}`,
        release.id,
      );
    }
  }
  const { restrictions, terms } = restrictionsOf(product);
  return {
    productId: product.id,
    productName: product.name,
    sku: product.sku,
    ...(order ? { orderId: order.id } : {}),
    entitlementId: entitlement.id,
    ...(release ? { releaseId: release.id } : {}),
    releaseRevision: entitlement.productRevision,
    issuedAt: now,
    license: {
      type: product.license.type,
      version: product.license.version,
      permittedUse: {
        commercial: product.license.allowsCommercialUse,
        modification: product.license.allowsModification,
        redistribution: product.license.allowsRedistribution,
        attribution: product.license.attributionRequired,
      },
      restrictions,
      ...(terms ? { terms } : {}),
    },
  };
}

// ---------------------------------------------------------------------------
// Persistence (documents are frozen value objects)
// ---------------------------------------------------------------------------

export function serializeReceipt(receipt: Receipt): string {
  return canonicalJson(receipt);
}

export function serializeLicenseDocument(document: LicenseDocument): string {
  return canonicalJson(document);
}
