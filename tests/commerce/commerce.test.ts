// G20A tests: customers, orders, payments, entitlements, releases, delivery.
import { describe, expect, it } from "vitest";
import { createHmac, createHash } from "node:crypto";
import {
  authenticate,
  createCustomer,
  deserializeCustomer,
  serializeCustomer,
  setAccountStatus,
} from "../../src/commerce/customers.js";
import {
  cancelOrder,
  createOrder,
  deserializeOrder,
  markOrderFailed,
  markOrderPaid,
  refundOrder,
  serializeOrder,
} from "../../src/commerce/orders.js";
import {
  applyWebhook,
  createPayment,
  deserializePayment,
  serializePayment,
  StubPaymentProvider,
  type WebhookEvent,
} from "../../src/commerce/payments.js";
import {
  deserializeEntitlement,
  deserializeRelease,
  isEntitlementUsable,
  issueEntitlement,
  createRelease,
  revokeEntitlement,
  revokeRelease,
  revokeReleaseArtifact,
  resolveReleaseArtifact,
  serializeEntitlement,
  serializeRelease,
  updateEligible,
} from "../../src/commerce/entitlements.js";
import {
  addArtifact,
  addVariant,
  attachPreview,
  createProduct,
  setProductStatus,
} from "../../src/product/product.js";
import {
  authorizeDownload,
  fulfillDownload,
  memoryLedger,
  tokenBucketLimiter,
} from "../../src/commerce/delivery.js";
import {
  customerLibrary,
  productPage,
  purchaseState,
  quoteVariant,
  resolveVariant,
} from "../../src/commerce/storefront.js";

const SECRET = "test-secret";
const hmac = (body: string): string => createHmac("sha256", SECRET).update(body, "utf8").digest("hex");
const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const parse = (body: string): WebhookEvent | null => {
  try {
    return JSON.parse(body) as WebhookEvent;
  } catch {
    return null;
  }
};

const LICENSE = {
  type: "commercial-single", allowsCommercialUse: true, allowsModification: true,
  allowsRedistribution: false, attributionRequired: false, version: "1.0",
};

function publishedProduct() {
  let product = createProduct({
    id: "product/shirt", name: "Shirt", sku: "SHIRT-001", category: "shirts",
    author: "Atelier",
    garment: { projectId: "proj/1", garmentId: "garment/s", garmentRevision: 2, garmentFingerprint: "fp/g2" },
    sizes: ["size/s", "size/m"], formats: ["pdf", "dxf"],
    license: { ...LICENSE }, description: "A shirt.",
  });
  const a = addArtifact(product, { filename: "shirt.pdf", type: "pdf", sizeIds: ["size/s", "size/m"], format: "pdf", revision: 1, checksum: sha256("pdf-bytes"), generator: "g13" });
  product = a.product;
  const v = addVariant(product, { name: "Full", kind: "multi-size", sizeIds: ["size/s", "size/m"], formats: ["pdf"], artifactIds: [a.id] });
  product = v.product;
  const preview = addArtifact(product, { filename: "hero.png", type: "preview", sizeIds: [], format: "png", revision: 1, checksum: sha256("png"), generator: "g15" });
  product = preview.product;
  product = attachPreview(product, preview.id, true);
  product = setProductStatus(product, "ready");
  product = setProductStatus(product, "published");
  return { product, variantId: v.id, artifactId: a.id };
}

describe("G20A customers", () => {
  it("creates accounts and authenticates through providers", () => {
    const customer = createCustomer({ id: "cust/ana", email: "ana@example.com", displayName: "Ana" });
    expect(customer.status).toBe("active");
    expect(() => createCustomer({ id: "x", email: "not-an-email", displayName: "X" })).toThrowError(/address/);
    const provider = { name: "stub", verify: (token: string) => (token === "tok-ana" ? { customerId: "cust/ana" } : null) };
    expect(authenticate([customer], provider, "tok-ana").id).toBe("cust/ana");
    expect(() => authenticate([customer], provider, "tok-eve")).toThrowError(/unknown token/);
    expect(() => authenticate([customer], provider, "")).toThrowError(/missing/);
    const suspended = setAccountStatus(customer, "suspended");
    expect(() => authenticate([suspended], provider, "tok-ana")).toThrowError(/suspended/);
    expect(serializeCustomer(deserializeCustomer(serializeCustomer(customer)))).toBe(serializeCustomer(customer));
  });
});

describe("G20A orders", () => {
  it("prices, pins revisions, and enforces the state machine", () => {
    const { product, variantId } = publishedProduct();
    const customer = createCustomer({ id: "cust/ana", email: "ana@example.com", displayName: "Ana" });
    let order = createOrder({
      id: "order/1", customerId: customer.id, customerEmail: customer.email,
      currency: "USD",
      items: [{ product, variantId, quantity: 2, unitPriceMinor: 1500 }],
    });
    expect(order.totalMinor).toBe(3000);
    expect(order.status).toBe("pending");
    expect(order.items[0].productRevision).toBe(1);
    order = markOrderPaid(order, "pay/1");
    expect(order.status).toBe("paid");
    expect(order.paymentId).toBe("pay/1");
    order = refundOrder(order);
    expect(order.status).toBe("refunded");
    expect(() => cancelOrder(order)).toThrowError(/cannot move/);
    expect(serializeOrder(deserializeOrder(serializeOrder(order)))).toBe(serializeOrder(order));
  });

  it("refuses unpublished products, bad money, and bad transitions", () => {
    const { product } = publishedProduct();
    const draft = { ...product, status: "draft" as const };
    expect(() => createOrder({
      id: "o", customerId: "c", customerEmail: "a@b.c", currency: "USD",
      items: [{ product: draft, variantId: "v", quantity: 1, unitPriceMinor: 1 }],
    })).toThrowError(/not published/);
    expect(() => cancelOrder(markOrderFailed(createOrder({
      id: "o", customerId: "c", customerEmail: "a@b.c", currency: "USD",
      items: [{ product: publishedProduct().product, variantId: publishedProduct().variantId, quantity: 1, unitPriceMinor: 1 }],
    })))).toThrowError(/cannot move/);
  });
});

describe("G20A payments and webhooks", () => {
  function payment() {
    return createPayment({ id: "pay/1", orderId: "order/1", amountMinor: 3000, currency: "USD", provider: "stub" });
  }

  function event(overrides: Partial<WebhookEvent> = {}): WebhookEvent {
    return {
      eventId: "evt/1", type: "payment.succeeded", checkoutId: "co/1",
      amountMinor: 3000, currency: "USD", receivedAt: "2026-01-01T00:00:00.000Z", ...overrides,
    };
  }

  it("confirms only on trusted, matching, first-seen success", () => {
    const provider = new StubPaymentProvider(hmac, parse);
    provider.createCheckout({ checkoutId: "co/1", orderId: "order/1", amountMinor: 3000, currency: "USD" });
    const { body, signature } = provider.signedEvent(event());
    expect(provider.verifyWebhook(body, signature)?.eventId).toBe("evt/1");
    expect(provider.verifyWebhook(body, "tampered")).toBeNull();
    const seen = new Set<string>();
    const first = applyWebhook(payment(), event(), seen);
    expect(first.applied).toBe(true);
    expect(first.payment.status).toBe("confirmed");
    const replay = applyWebhook(first.payment, event(), seen);
    expect(replay.applied).toBe(false);
    expect(replay.ignoredReason).toBe("duplicate-event");
  });

  it("rejects forgery, mismatch, out-of-order, and post-refund events", () => {
    const seen = new Set<string>();
    // Amount mismatch.
    expect(applyWebhook(payment(), event({ eventId: "e2", amountMinor: 1 }), seen).ignoredReason).toBe("amount-mismatch");
    // Refund before confirmation is held, not applied.
    const held = applyWebhook(payment(), event({ eventId: "e3", type: "refund.issued" }), seen);
    expect(held.applied).toBe(false);
    expect(held.ignoredReason).toBe("refund-before-confirmation");
    // Failure then success: success after terminal failure is a no-op.
    const failed = applyWebhook(payment(), event({ eventId: "e4", type: "payment.failed" }), seen);
    expect(failed.payment.status).toBe("failed");
    const late = applyWebhook(failed.payment, event({ eventId: "e5" }), seen);
    expect(late.applied).toBe(false);
    // Refund after confirmation applies once.
    const confirmed = applyWebhook(payment(), event({ eventId: "e6" }), seen);
    const refunded = applyWebhook(confirmed.payment, event({ eventId: "e7", type: "refund.issued" }), seen);
    expect(refunded.payment.status).toBe("refunded");
    expect(serializePayment(deserializePayment(serializePayment(refunded.payment)))).toBe(serializePayment(refunded.payment));
  });

  it("validates payment amounts", () => {
    expect(() => createPayment({ id: "p", orderId: "o", amountMinor: -1, currency: "USD", provider: "s" })).toThrowError(/amount/);
  });
});

describe("G20A entitlements and releases", () => {
  function paidOrder() {
    const { product, variantId } = publishedProduct();
    const order = createOrder({
      id: "order/1", customerId: "cust/ana", customerEmail: "ana@example.com",
      currency: "USD", items: [{ product, variantId, quantity: 1, unitPriceMinor: 1500 }],
    });
    return { product, variantId, order: markOrderPaid(order, "pay/1") };
  }

  it("issues only against paid orders and tracks usability", () => {
    const { order } = paidOrder();
    const entitlement = issueEntitlement({ id: "ent/1", order, productId: "product/shirt", variantId: order.items[0].variantId });
    expect(entitlement.productRevision).toBe(1);
    expect(isEntitlementUsable(entitlement, "2026-06-01T00:00:00.000Z")).toBe(true);
    const expiring = issueEntitlement({
      id: "ent/2", order, productId: "product/shirt", variantId: order.items[0].variantId,
      expiresAt: "2026-01-02T00:00:00.000Z",
    });
    expect(isEntitlementUsable(expiring, "2026-06-01T00:00:00.000Z")).toBe(false);
    expect(isEntitlementUsable(revokeEntitlement(entitlement), "2026-06-01T00:00:00.000Z")).toBe(false);
    const unpaid = { ...order, status: "pending" as const };
    expect(() => issueEntitlement({ id: "ent/3", order: unpaid, productId: "x", variantId: "y" })).toThrowError(/paid order/);
    expect(serializeEntitlement(deserializeEntitlement(serializeEntitlement(entitlement)))).toBe(serializeEntitlement(entitlement));
  });

  it("gates updates by explicit policy", () => {
    const { order } = paidOrder();
    const base = { id: "e", order, productId: "product/shirt", variantId: order.items[0].variantId } as const;
    expect(updateEligible(issueEntitlement({ ...base, id: "e1" }), 1)).toBe(true);
    expect(updateEligible(issueEntitlement({ ...base, id: "e2" }), 2)).toBe(false);
    expect(updateEligible(issueEntitlement({ ...base, id: "e3", updatePolicy: "free-updates" }), 2)).toBe(true);
    expect(updateEligible(issueEntitlement({ ...base, id: "e4", updatePolicy: "paid-updates" }), 2)).toBe(false);
  });

  it("freezes immutable releases and enforces revocation", () => {
    const { product } = publishedProduct();
    const release = createRelease({
      id: "rel/1", product,
      artifacts: product.artifacts.filter((a) => a.type !== "preview"),
      updatePolicy: "original-only",
    });
    expect(release.revision).toBe(1);
    expect(resolveReleaseArtifact(release, product.artifacts[0].id).filename).toBe("shirt.pdf");
    expect(() => resolveReleaseArtifact(release, "ghost")).toThrowError(/not in release/);
    const revoked = revokeReleaseArtifact(release, product.artifacts[0].id);
    expect(() => resolveReleaseArtifact(revoked, product.artifacts[0].id)).toThrowError(/revoked/);
    // Original release object untouched (immutability).
    expect(resolveReleaseArtifact(release, product.artifacts[0].id).filename).toBe("shirt.pdf");
    expect(() => resolveReleaseArtifact(revokeRelease(release), product.artifacts[0].id)).toThrowError(/is revoked/);
    expect(() => createRelease({ id: "r", product: { ...product, status: "draft" }, artifacts: [] })).toThrowError(/ready\/published/);
    expect(serializeRelease(deserializeRelease(serializeRelease(release)))).toBe(serializeRelease(release));
  });
});

describe("G20A secure delivery", () => {
  function setup() {
    const { product, variantId } = publishedProduct();
    const order = markOrderPaid(createOrder({
      id: "order/1", customerId: "cust/ana", customerEmail: "ana@example.com",
      currency: "USD", items: [{ product, variantId, quantity: 1, unitPriceMinor: 1500 }],
    }), "pay/1");
    const entitlement = issueEntitlement({ id: "ent/1", order, productId: product.id, variantId });
    const release = createRelease({
      id: "rel/1", product, artifacts: product.artifacts.filter((a) => a.type !== "preview"),
    });
    return { product, entitlement, release };
  }

  const ctx = (overrides: Partial<Parameters<typeof authorizeDownload>[0]> = {}) => ({
    token: "tok/nonce-1",
    customerId: "cust/ana",
    ledger: memoryLedger(),
    limiter: tokenBucketLimiter(10, 60_000),
    nowIso: "2026-01-01T00:00:00.000Z",
    nowMs: Date.parse("2026-01-01T00:00:00.000Z"),
    ...overrides,
  });

  it("authorizes the exact entitled artifact and fulfills with checksum", () => {
    const { entitlement, release } = setup();
    const artifactId = release.artifacts[0].artifactId;
    const auth = authorizeDownload({ ...ctx(), entitlement, release, artifactId });
    expect(auth.filename).toBe("shirt.pdf");
    expect(auth.entitlementId).toBe("ent/1");
    // Replay of the same token fails.
    const ledger = memoryLedger();
    ledger.consume("tok/nonce-1");
    expect(() => authorizeDownload({ ...ctx({ ledger }), entitlement, release, artifactId }))
      .toThrowError(/already used/);
    // Fulfillment verifies checksum and records.
    const done = fulfillDownload({
      id: "dl/1", authorization: auth, payload: "pdf-bytes",
      expectedChecksum: sha256("pdf-bytes"), hashHex: sha256,
      nowIso: "2026-01-01T00:00:00.000Z", nowMs: Date.parse("2026-01-01T00:00:00.000Z"),
    });
    expect(done.record.status).toBe("completed");
    expect(done.bytes).toBe("pdf-bytes".length);
    const tampered = fulfillDownload({
      id: "dl/2", authorization: auth, payload: "evil-bytes",
      expectedChecksum: sha256("pdf-bytes"), hashHex: sha256,
      nowIso: "2026-01-01T00:00:00.000Z", nowMs: Date.parse("2026-01-01T00:00:00.000Z"),
    });
    expect(tampered.record.status).toBe("failed");
    expect(tampered.bytes).toBe(0);
  });

  it("refuses wrong customers, revoked/expired rights, and rate abuse", () => {
    const { entitlement, release } = setup();
    const artifactId = release.artifacts[0].artifactId;
    expect(() => authorizeDownload({ ...ctx(), customerId: "cust/eve", entitlement, release, artifactId }))
      .toThrowError(/another customer/);
    expect(() => authorizeDownload({ ...ctx(), entitlement: revokeEntitlement(entitlement), release, artifactId }))
      .toThrowError(/not usable/);
    const revoked = revokeReleaseArtifact(release, artifactId);
    expect(() => authorizeDownload({ ...ctx(), entitlement, release: revoked, artifactId }))
      .toThrowError(/revoked/);
    expect(() => authorizeDownload({ ...ctx(), entitlement, release: revokeRelease(release), artifactId }))
      .toThrowError(/is revoked/);
    expect(() => authorizeDownload({ ...ctx(), entitlement, release, artifactId: "ghost" }))
      .toThrowError(/not in release/);
    const strict = tokenBucketLimiter(1, 60_000);
    const first = authorizeDownload({ ...ctx({ limiter: strict, token: "t1" }), entitlement, release, artifactId });
    expect(first.token).toBe("t1");
    expect(() => authorizeDownload({ ...ctx({ limiter: strict, token: "t2" }), entitlement, release, artifactId }))
      .toThrowError(/rate limit/);
    expect(() => tokenBucketLimiter(0, 1)).toThrowError(/capacity/);
  });
});

describe("G20A storefront data", () => {
  it("pages, resolves variants, quotes, and states ownership", () => {
    const { product, variantId } = publishedProduct();
    const page = productPage(product, product.artifacts);
    expect(page.purchasable).toBe(true);
    expect(page.files.map((f) => f.filename)).toContain("shirt.pdf");
    expect(page.previews).toEqual(product.previewIds);
    const draftPage = productPage({ ...product, status: "draft" }, product.artifacts);
    expect(draftPage.purchasable).toBe(false);
    expect(draftPage.unavailableReason).toContain("draft");
    expect(resolveVariant(product, { sizeId: "size/s", formatBundle: ["pdf"] }).id).toBe(variantId);
    expect(() => resolveVariant(product, { sizeId: "size/xxxl" })).toThrowError(/no variant/);
    expect(quoteVariant(product.variants[0], { [variantId]: { amountMinor: 1500, currency: "USD" } }))
      .toEqual({ amountMinor: 1500, currency: "USD" });
    expect(() => quoteVariant(product.variants[0], {})).toThrowError(/no price/);
    const entitlement = issueEntitlement({
      id: "ent/1",
      order: markOrderPaid(createOrder({
        id: "order/1", customerId: "cust/ana", customerEmail: "a@b.c", currency: "USD",
        items: [{ product, variantId, quantity: 1, unitPriceMinor: 1500 }],
      }), "pay/1"),
      productId: product.id, variantId,
    });
    expect(purchaseState(product, entitlement, "2026-01-01T00:00:00.000Z").state).toBe("owned");
    expect(purchaseState(product, null, "2026-01-01T00:00:00.000Z").state).toBe("available");
    expect(purchaseState({ ...product, status: "archived" }, null, "2026-01-01T00:00:00.000Z").state).toBe("discontinued");
    const library = customerLibrary([entitlement], [product], "2026-01-01T00:00:00.000Z");
    expect(library).toEqual([{
      productId: product.id, name: product.name, revision: 1,
      usable: true, entitlementId: "ent/1",
    }]);
  });
});
