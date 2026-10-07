// G20A orders: priced purchase records with an enforced state machine.
//
// Money is integer minor units + explicit currency code (no floats, no
// hard-coded currency). Historical items pin product/variant/revision/price
// at purchase time. Entitlement is granted only on trusted payment
// confirmation (see payments), never on browser-reported success.

import { PatternCadError } from "../pattern/cad.js";
import type { Product } from "../product/product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type OrderStatus = "pending" | "paid" | "failed" | "canceled" | "refunded";

export interface OrderItem {
  productId: string;
  variantId: string;
  productRevision: number;
  quantity: number;
  /** Unit price in minor units at purchase time. */
  unitPriceMinor: number;
  currency: string;
}

export interface Order {
  id: string;
  customerId: string;
  /** Email snapshot: financial records survive account deletion. */
  customerEmail: string;
  status: OrderStatus;
  currency: string;
  items: OrderItem[];
  totalMinor: number;
  paymentId?: string;
  createdAt: string;
  updatedAt: string;
}

function requireId(id: string, label: string): void {
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", `${label} must be non-empty`);
}

export function orderTotal(items: OrderItem[]): number {
  return items.reduce((sum, item) => sum + item.quantity * item.unitPriceMinor, 0);
}

export function createOrder(partial: {
  id: string;
  customerId: string;
  customerEmail: string;
  currency: string;
  items: Array<{ product: Product; variantId: string; quantity: number; unitPriceMinor: number }>;
  now?: string;
}): Order {
  requireId(partial.id, "order id");
  requireId(partial.customerId, "customer id");
  if (!partial.currency || !partial.currency.trim()) {
    throw new PatternCadError("invalid-document", "order currency must be non-empty");
  }
  if (partial.items.length === 0) throw new PatternCadError("invalid-document", "order needs at least one item");
  const items: OrderItem[] = partial.items.map((entry, index) => {
    const { product, variantId, quantity, unitPriceMinor } = entry;
    if (product.status !== "published") {
      throw new PatternCadError("invalid-document", `product '${product.id}' is ${product.status}, not published`, product.id);
    }
    const variant = product.variants.find((v) => v.id === variantId);
    if (!variant) throw new PatternCadError("missing-reference", `variant '${variantId}' does not exist`, variantId);
    if (!Number.isInteger(quantity) || quantity < 1) {
      throw new PatternCadError("invalid-transform", `item ${index} quantity must be a positive integer`);
    }
    if (!Number.isInteger(unitPriceMinor) || unitPriceMinor < 0) {
      throw new PatternCadError("invalid-transform", `item ${index} price must be integer minor units >= 0`);
    }
    return {
      productId: product.id, variantId, productRevision: product.revision,
      quantity, unitPriceMinor, currency: partial.currency,
    };
  });
  for (const item of items) {
    if (item.currency !== partial.currency) {
      throw new PatternCadError("invalid-document", "order items must share the order currency");
    }
  }
  const timestamp = partial.now ?? new Date().toISOString();
  return {
    id: partial.id, customerId: partial.customerId, customerEmail: partial.customerEmail,
    status: "pending", currency: partial.currency, items,
    totalMinor: orderTotal(items), createdAt: timestamp, updatedAt: timestamp,
  };
}

function transition(order: Order, status: OrderStatus, now?: string): Order {
  const allowed: Record<OrderStatus, OrderStatus[]> = {
    pending: ["paid", "failed", "canceled"],
    paid: ["refunded"],
    failed: [],
    canceled: [],
    refunded: [],
  };
  if (!allowed[order.status].includes(status)) {
    throw new PatternCadError("invalid-document", `order cannot move from '${order.status}' to '${status}'`, order.id);
  }
  return { ...clone(order), status, updatedAt: now ?? new Date().toISOString() };
}

/** Trusted payment confirmation only (provider webhook path, never the browser). */
export function markOrderPaid(order: Order, paymentId: string, now?: string): Order {
  requireId(paymentId, "payment id");
  const next = transition(order, "paid", now);
  next.paymentId = paymentId;
  return next;
}

export function markOrderFailed(order: Order, now?: string): Order {
  return transition(order, "failed", now);
}

export function cancelOrder(order: Order, now?: string): Order {
  return transition(order, "canceled", now);
}

export function refundOrder(order: Order, now?: string): Order {
  return transition(order, "refunded", now);
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

export function serializeOrder(order: Order): string {
  return canonicalJson(order);
}

export function deserializeOrder(serialized: string): Order {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized order is not valid JSON");
  }
  const order = parsed as Order;
  if (!order || !order.id || !order.customerId || !Array.isArray(order.items)) {
    throw new PatternCadError("invalid-document", "order shape is invalid");
  }
  return clone(order);
}
