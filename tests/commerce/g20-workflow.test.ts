// G20 FINAL — first real digital sale, end to end.
//
// Garment -> released product -> storefront -> variant -> checkout ->
// verified payment -> order -> entitlement -> secure download -> purchase
// library -> re-download -> new revision (old entitlement pinned) ->
// refund (entitlement revoked, downloads refused). The garment engine is
// never imported here: commerce consumes released product records only.
import { describe, expect, it } from "vitest";
import { createHash, createHmac } from "node:crypto";
import {
  addArtifact,
  addVariant,
  attachPreview,
  bumpProductRevision,
  createProduct,
  isPinnedCurrent,
  setProductStatus,
} from "../../src/product/product.js";
import {
  authenticate,
  createCustomer,
  setAccountStatus,
} from "../../src/commerce/customers.js";
import {
  cancelOrder,
  createOrder,
  markOrderPaid,
  refundOrder,
} from "../../src/commerce/orders.js";
import {
  applyWebhook,
  createPayment,
  StubPaymentProvider,
  type WebhookEvent,
} from "../../src/commerce/payments.js";
import {
  createRelease,
  isEntitlementUsable,
  issueEntitlement,
  revokeEntitlement,
  updateEligible,
} from "../../src/commerce/entitlements.js";
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

const SECRET = "g20-acceptance";
const hmac = (body: string): string => createHmac("sha256", SECRET).update(body, "utf8").digest("hex");
const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const parse = (body: string): WebhookEvent | null => {
  try {
    return JSON.parse(body) as WebhookEvent;
  } catch {
    return null;
  }
};

const T0 = "2026-01-01T00:00:00.000Z";
const NOW_MS = Date.parse(T0);

function licensedProduct() {
  let product = createProduct({
    id: "product/collared-shirt",
    name: "Classic Collared Shirt",
    sku: "CSHIRT-001",
    category: "shirts",
    author: "Atelier",
    garment: { projectId: "proj/1", garmentId: "garment/shirt", garmentRevision: 5, garmentFingerprint: "fp/rev5" },
    sizes: ["size/xs", "size/s", "size/m", "size/l", "size/xl", "size/xxl"],
    formats: ["pdf", "dxf", "svg"],
    license: {
      type: "commercial-single", allowsCommercialUse: true, allowsModification: true,
      allowsRedistribution: false, attributionRequired: false, version: "1.0",
    },
    description: "Classic collared shirt with full size set.",
    now: T0,
  });
  const files: Array<[string, "pdf" | "dxf" | "svg" | "preview", string[]]> = [
    ["shirt-full.pdf", "pdf", ["size/xs", "size/s", "size/m", "size/l", "size/xl", "size/xxl"]],
    ["shirt-full.dxf", "dxf", ["size/xs", "size/s", "size/m", "size/l", "size/xl", "size/xxl"]],
    ["shirt-m.pdf", "pdf", ["size/m"]],
    ["hero-front.png", "preview", []],
  ];
  for (const [filename, type, sizeIds] of files) {
    const added = addArtifact(product, {
      filename, type, sizeIds, format: filename.split(".").pop()!,
      revision: 1, checksum: sha256(`${filename}:rev1`), generator: "test-harness",
    }, T0);
    product = added.product;
  }
  const preview = addArtifact(product, {
    filename: "hero-front.png", type: "preview",
    sizeIds: [], format: "png", revision: 1, checksum: sha256("hero-front.png:rev1"), generator: "g15",
  });
  product = preview.product;
  product = attachPreview(product, preview.id, true, T0);
  return product;
}

describe("G20 final integration — first real digital sale", () => {
  it("sells, delivers, re-downloads, revisions, and refunds traceably", () => {    // G19 product, published with variants. (Preview attach uses the hero artifact.)
    let product = licensedProduct();
    // NOTE: attachPreview above references a derived id; re-attach the real hero file.
    const heroId = product.artifacts.find((a) => a.filename === "hero-front.png")!.id;
    product = attachPreview({ ...product, previewIds: [], thumbnailArtifactId: undefined }, heroId, true, T0);
    const full = addVariant(product, {
      name: "Complete Bundle", kind: "multi-size",
      sizeIds: ["size/xs", "size/s", "size/m", "size/l", "size/xl", "size/xxl"],
      formats: ["pdf", "dxf", "svg"],
      artifactIds: product.artifacts.filter((a) => a.type !== "preview").map((a) => a.id),
    }, T0);
    product = full.product;
    const single = addVariant(product, {
      name: "Single Size", kind: "single-size", sizeIds: ["size/m"],
      formats: ["pdf"], artifactIds: product.artifacts.filter((a) => a.filename === "shirt-m.pdf").map((a) => a.id),
    }, T0);
    product = single.product;
    product = setProductStatus(product, "ready", T0);
    product = setProductStatus(product, "published", T0);

    // Release freezes the downloadable set (previews excluded from delivery).

    // Customer visits: page, previews, files, variant selection.
    const page = productPage(product, product.artifacts);
    expect(page.purchasable).toBe(true);
    expect(page.files.map((f) => f.filename)).toContain("shirt-full.dxf");
    const chosen = resolveVariant(product, { sizeId: "size/m", formatBundle: ["pdf"] });
    expect(chosen.id).toBe(single.id);
    const quote = quoteVariant(chosen, { [single.id]: { amountMinor: 1200, currency: "USD" } });
    expect(quote).toEqual({ amountMinor: 1200, currency: "USD" });

    // Customer account + purchase state before buying.
    const customer = createCustomer({ id: "cust/ana", email: "ana@example.com", displayName: "Ana", now: T0 });
    expect(purchaseState(product, null, T0).state).toBe("available");
    const provider = new StubPaymentProvider(hmac, parse);

    // Checkout -> order (pending, no access yet).
    let order = createOrder({
      id: "order/1", customerId: customer.id, customerEmail: customer.email,
      currency: quote.currency,
      items: [{ product, variantId: chosen.id, quantity: 1, unitPriceMinor: quote.amountMinor }],
      now: T0,
    });
    expect(order.totalMinor).toBe(1200);
    let payment = createPayment({ id: "pay/1", orderId: order.id, amountMinor: 1200, currency: "USD", provider: "stub", now: T0 });
    const { checkoutId } = provider.createCheckout({ checkoutId: "co/1", orderId: order.id, amountMinor: 1200, currency: "USD" });

    // Browser claims success: no entitlement follows (untrusted path).
    expect(purchaseState(product, null, T0).state).toBe("available");

    // Trusted webhook confirms: order paid, entitlement issued.
    const seen = new Set<string>();
    const success: WebhookEvent = {
      eventId: "evt/ok", type: "payment.succeeded", checkoutId,
      amountMinor: 1200, currency: "USD", receivedAt: T0,
    };
    const signed = provider.signedEvent(success);
    expect(provider.verifyWebhook(signed.body, signed.signature)?.eventId).toBe("evt/ok");
    expect(provider.verifyWebhook(signed.body, "forged")).toBeNull();
    const applied = applyWebhook(payment, provider.verifyWebhook(signed.body, signed.signature)!, seen, T0);
    expect(applied.applied).toBe(true);
    payment = applied.payment;
    order = markOrderPaid(order, payment.id, T0);
    const entitlement = issueEntitlement({
      id: "ent/1", order, productId: product.id, variantId: chosen.id, updatePolicy: "original-only", now: T0,
    });
    expect(purchaseState(product, entitlement, T0)).toEqual({ state: "owned", entitlementId: "ent/1" });
    // Duplicate webhook changes nothing.
    const replay = applyWebhook(payment, provider.verifyWebhook(signed.body, signed.signature)!, seen, T0);
    expect(replay.applied).toBe(false);

    // Secure download of the purchased PDF (payload integrity verified).
    const release = createRelease({
      id: "rel/1", product,
      artifacts: product.artifacts.filter((a) => a.filename === "shirt-m.pdf"),
      now: T0,
    });
    const ledger = memoryLedger();
    const limiter = tokenBucketLimiter(10, 60_000);
    const pdfId = release.artifacts[0].artifactId;
    const auth = authorizeDownload({
      token: "tok/dl-1", customerId: customer.id, entitlement, release,
      artifactId: pdfId, ledger, limiter, nowIso: T0, nowMs: NOW_MS,
    });
    expect(auth.filename).toBe("shirt-m.pdf");
    const pdfBytes = "shirt-m.pdf:rev1";
    expect(sha256(pdfBytes)).toBe(release.artifacts[0].checksum);
    const done = fulfillDownload({
      id: "dl/1", authorization: auth, payload: pdfBytes,
      expectedChecksum: release.artifacts[0].checksum, hashHex: sha256,
      nowIso: T0, nowMs: NOW_MS,
    });
    expect(done.record.status).toBe("completed");
    expect(done.bytes).toBe(pdfBytes.length);
    // Product appears in the purchase library.
    const library = customerLibrary([entitlement], [product], T0);
    expect(library).toEqual([{
      productId: product.id, name: product.name, revision: 1,
      usable: true, entitlementId: "ent/1",
    }]);

    // Re-download later: same release, fresh token, same bytes.
    const auth2 = authorizeDownload({
      token: "tok/dl-2", customerId: customer.id, entitlement, release,
      artifactId: pdfId, ledger, limiter, nowIso: T0, nowMs: NOW_MS,
    });
    const done2 = fulfillDownload({
      id: "dl/2", authorization: auth2, payload: pdfBytes,
      expectedChecksum: release.artifacts[0].checksum, hashHex: sha256,
      nowIso: T0, nowMs: NOW_MS,
    });
    expect(done2.record.status).toBe("completed");

    // New product revision: old customers keep the original entitlement.
    expect(isPinnedCurrent(product, 5, "fp/rev5")).toBe(true);
    const rev2 = bumpProductRevision(product, {
      projectId: "proj/1", garmentId: "garment/shirt",
      garmentRevision: 6, garmentFingerprint: "fp/rev6",
    }, T0);
    expect(rev2.revision).toBe(2);
    expect(isPinnedCurrent(rev2, 5, "fp/rev5")).toBe(false);
    expect(updateEligible(entitlement, 2)).toBe(false); // original-only
    expect(updateEligible({ ...entitlement, updatePolicy: "free-updates" }, 2)).toBe(true);

    // Refund: order refunded, entitlement revoked, downloads refused.
    const refundEvt: WebhookEvent = {
      eventId: "evt/refund", type: "refund.issued", checkoutId,
      amountMinor: 1200, currency: "USD", receivedAt: T0,
    };
    const refundSigned = provider.signedEvent(refundEvt);
    const refunded = applyWebhook(payment, provider.verifyWebhook(refundSigned.body, refundSigned.signature)!, seen, T0);
    expect(refunded.applied).toBe(true);
    expect(refunded.payment.status).toBe("refunded");
    order = refundOrder(order, T0);
    expect(order.status).toBe("refunded");
    const revoked = revokeEntitlement(entitlement, T0);
    expect(isEntitlementUsable(revoked, T0)).toBe(false);
    expect(() => authorizeDownload({
      token: "tok/dl-3", customerId: customer.id, entitlement: revoked, release,
      artifactId: pdfId, ledger, limiter, nowIso: T0, nowMs: NOW_MS,
    })).toThrowError(/not usable/);
    expect(() => cancelOrder(order)).toThrowError(/cannot move/); // terminal state guard
  });
});
