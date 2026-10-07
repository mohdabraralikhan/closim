// G20A payments: provider-independent boundary.
//
// No card processing here. The provider is an injected interface; webhooks
// are untrusted input requiring signature verification, idempotency (event
// dedupe), and explicit state transitions. Entitlement follows only trusted
// confirmation, never browser-reported success.

import { PatternCadError } from "../pattern/cad.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type PaymentStatus = "pending" | "confirmed" | "failed" | "refunded";

export interface Payment {
  id: string;
  orderId: string;
  amountMinor: number;
  currency: string;
  status: PaymentStatus;
  provider: string;
  transactionRef?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CheckoutSession {
  checkoutId: string;
  orderId: string;
  amountMinor: number;
  currency: string;
}

export type WebhookType = "payment.succeeded" | "payment.failed" | "refund.issued";

export interface WebhookEvent {
  eventId: string;
  type: WebhookType;
  checkoutId: string;
  amountMinor: number;
  currency: string;
  receivedAt: string;
}

export interface PaymentProvider {
  name: string;
  createCheckout(session: { checkoutId: string; orderId: string; amountMinor: number; currency: string }): { checkoutId: string };
  /** Verify raw webhook body against signature; null = untrusted, reject. */
  verifyWebhook(rawBody: string, signature: string): WebhookEvent | null;
}

export function createPayment(partial: {
  id: string;
  orderId: string;
  amountMinor: number;
  currency: string;
  provider: string;
  now?: string;
}): Payment {
  if (!partial.id || !partial.orderId || !partial.provider) {
    throw new PatternCadError("invalid-document", "payment needs id, order, and provider");
  }
  if (!Number.isInteger(partial.amountMinor) || partial.amountMinor < 0 || !partial.currency) {
    throw new PatternCadError("invalid-transform", "payment amount/currency invalid", partial.id);
  }
  const timestamp = partial.now ?? new Date().toISOString();
  return {
    id: partial.id, orderId: partial.orderId, amountMinor: partial.amountMinor,
    currency: partial.currency, status: "pending", provider: partial.provider,
    createdAt: timestamp, updatedAt: timestamp,
  };
}

export interface WebhookOutcome {
  payment: Payment;
  /** True when this event changed state (false = duplicate/out-of-order no-op). */
  applied: boolean;
  ignoredReason?: string;
}

/**
 * Apply one verified webhook to a payment. Idempotent: replays and
 * out-of-order events are recorded as ignored, never double-applied.
 */
export function applyWebhook(
  payment: Payment,
  event: WebhookEvent,
  seenEventIds: Set<string>,
  now?: string,
): WebhookOutcome {
  const next = clone(payment);
  if (seenEventIds.has(event.eventId)) {
    return { payment: next, applied: false, ignoredReason: "duplicate-event" };
  }
  seenEventIds.add(event.eventId);
  if (event.amountMinor !== payment.amountMinor || event.currency !== payment.currency) {
    return { payment: next, applied: false, ignoredReason: "amount-mismatch" };
  }
  const timestamp = now ?? new Date().toISOString();
  if (event.type === "payment.succeeded") {
    if (next.status !== "pending") {
      return { payment: next, applied: false, ignoredReason: `already-${next.status}` };
    }
    next.status = "confirmed";
    next.transactionRef = event.eventId;
    next.updatedAt = timestamp;
    return { payment: next, applied: true };
  }
  if (event.type === "payment.failed") {
    if (next.status !== "pending") {
      return { payment: next, applied: false, ignoredReason: `already-${next.status}` };
    }
    next.status = "failed";
    next.updatedAt = timestamp;
    return { payment: next, applied: true };
  }
  // refund.issued: valid only after confirmation.
  if (next.status !== "confirmed") {
    return { payment: next, applied: false, ignoredReason: "refund-before-confirmation" };
  }
  next.status = "refunded";
  next.updatedAt = timestamp;
  return { payment: next, applied: true };
}

/** In-memory test double: HMAC-signed webhooks, scripted outcomes. */
export class StubPaymentProvider implements PaymentProvider {
  readonly name = "stub";
  private readonly checkouts = new Map<string, CheckoutSession>();
  constructor(
    private readonly sign: (body: string) => string,
    private readonly parse: (body: string) => WebhookEvent | null,
  ) {}

  createCheckout(session: { checkoutId: string; orderId: string; amountMinor: number; currency: string }): { checkoutId: string } {
    this.checkouts.set(session.checkoutId, { ...session });
    return { checkoutId: session.checkoutId };
  }

  verifyWebhook(rawBody: string, signature: string): WebhookEvent | null {
    if (this.sign(rawBody) !== signature) return null;
    return this.parse(rawBody);
  }

  /** Build a signed webhook body for tests (uses the injected signer). */
  signedEvent(event: WebhookEvent): { body: string; signature: string } {
    const body = JSON.stringify(event);
    return { body, signature: this.sign(body) };
  }
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

export function serializePayment(payment: Payment): string {
  return canonicalJson(payment);
}

export function deserializePayment(serialized: string): Payment {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized payment is not valid JSON");
  }
  const payment = parsed as Payment;
  if (!payment || !payment.id || !payment.orderId) {
    throw new PatternCadError("invalid-document", "payment shape is invalid");
  }
  return clone(payment);
}
