// G21B — orders, receipts, and license documents.
import { describe, expect, it } from "vitest";
import { createHmac } from "node:crypto";
import { bumpProductRevision, updateProductDetails } from "../../src/product/product.js";
import { cancelOrder, createOrder, markOrderFailed, markOrderPaid, refundOrder, type Order } from "../../src/commerce/orders.js";
import { applyWebhook, createPayment, StubPaymentProvider, type WebhookEvent } from "../../src/commerce/payments.js";
import {
  createLicenseDocument,
  createReceipt,
  findOrder,
  orderHistory,
  serializeLicenseDocument,
  serializeReceipt,
} from "../../src/commerce/receipts.js";
import { purchase, T } from "./fixtures.js";

const hmac = (body: string): string =>
  createHmac("sha256", "g21b-secret").update(body, "utf8").digest("hex");
const parse = (body: string): WebhookEvent | null => {
  try {
    return JSON.parse(body) as WebhookEvent;
  } catch {
    return null;
  }
};

describe("G21B order history and receipts", () => {
  it("completed order: history and receipt from the authoritative record", () => {
    const f = purchase({ productId: "product/shirt", name: "Classic Shirt", priceMinor: 1500, now: T(2) });
    const history = orderHistory("cust/ana", [f.order], [f.product]);
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      orderId: f.order.id,
      date: T(2),
      status: "paid",
      currency: "USD",
      totalMinor: 1500,
    });
    expect(history[0].lines[0]).toMatchObject({
      productId: "product/shirt",
      variantId: f.variantId,
      productRevision: 1,
      quantity: 1,
      unitPriceMinor: 1500,
      amountMinor: 1500,
      name: "Classic Shirt",
      author: "Atelier",
    });

    const receipt = createReceipt({ order: f.order, products: [f.product], now: T(5) });
    expect(receipt).toMatchObject({
      orderId: f.order.id,
      issuedAt: T(5),
      orderDate: T(2),
      status: "paid",
      customerEmail: "ana@example.com",
      currency: "USD",
      totalMinor: 1500,
      transactionReference: f.order.paymentId,
    });
    // Never any credential/card-shaped data — reference id only.
    expect(JSON.stringify(receipt)).not.toMatch(/card|cvv|pan"/i);
    expect(serializeReceipt(receipt)).toContain('"transactionReference"');
  });

  it("failed, refunded, and canceled orders keep their status in records", () => {
    const failed = purchase({ productId: "product/fail" });
    // A failed order never reaches payment: build it as pending first.
    const pending: Order = {
      ...failed.order,
      id: "order/pending-fail",
      status: "pending",
      paymentId: undefined,
      createdAt: T(2),
      updatedAt: T(2),
    };
    const failedReceipt = createReceipt({ order: markOrderFailed(pending, T(3)), now: T(4) });
    expect(failedReceipt.status).toBe("failed");
    expect(failedReceipt.transactionReference).toBeUndefined();

    const refunded = purchase({ productId: "product/refund" });
    const refundedOrder = refundOrder(refunded.order, T(3));
    expect(orderHistory("cust/ana", [refundedOrder])[0].status).toBe("refunded");
    expect(createReceipt({ order: refundedOrder, now: T(4) }).status).toBe("refunded");

    const canceled = purchase({ productId: "product/cancel" });
    const cancelPending: Order = { ...canceled.order, id: "order/pending-cancel", status: "pending", paymentId: undefined };
    expect(createReceipt({ order: cancelOrder(cancelPending, T(3)), now: T(4) }).status).toBe("canceled");

    // Terminal states refuse transitions (records stay honest).
    expect(() => markOrderPaid(refundedOrder, "pay/other", T(4))).toThrowError(/cannot move/);
    expect(() => markOrderFailed(refundedOrder, T(4))).toThrowError(/cannot move/);
  });

  it("multiple products: one order, several lines, summed total", () => {
    const a = purchase({ productId: "product/multi-a", name: "Alpha", priceMinor: 1000, now: T(2) });
    const b = purchase({ productId: "product/multi-b", name: "Beta", priceMinor: 2500, now: T(2) });
    let order = createOrder({
      id: "order/multi",
      customerId: "cust/ana",
      customerEmail: "ana@example.com",
      currency: "USD",
      items: [
        { product: a.product, variantId: a.variantId, quantity: 2, unitPriceMinor: 1000 },
        { product: b.product, variantId: b.variantId, quantity: 1, unitPriceMinor: 2500 },
      ],
      now: T(2),
    });
    order = markOrderPaid(order, "pay/multi", T(2));

    const receipt = createReceipt({ order, products: [a.product, b.product], now: T(3) });
    expect(receipt.lines).toHaveLength(2);
    expect(receipt.lines.map((l) => [l.name, l.amountMinor])).toEqual([
      ["Alpha", 2000],
      ["Beta", 2500],
    ]);
    expect(receipt.totalMinor).toBe(4500);

    const history = orderHistory("cust/ana", [order], [a.product, b.product]);
    expect(history[0].totalMinor).toBe(4500);
    expect(history[0].lines).toHaveLength(2);
  });

  it("old product release and updated product never rewrite issued documents", () => {
    const f = purchase({ productId: "product/history", name: "Original Name", now: T(2) });
    const receipt = createReceipt({ order: f.order, products: [f.product], now: T(3) });
    const license = createLicenseDocument({
      entitlement: f.entitlement,
      product: f.product,
      release: f.release,
      order: f.order,
      now: T(3),
    });
    const receiptBefore = serializeReceipt(receipt);
    const licenseBefore = serializeLicenseDocument(license);

    // Creator renames the product and re-releases at revision 2.
    const renamed = updateProductDetails(f.product, { name: "Renamed Later" }, T(4));
    const rebumped = bumpProductRevision(renamed, {
      projectId: "proj/1",
      garmentId: f.product.garment.garmentId,
      garmentRevision: 2,
      garmentFingerprint: "fp/rev2",
    }, T(4));
    expect(rebumped.revision).toBe(2);

    // Frozen documents are byte-identical after the change.
    expect(serializeReceipt(receipt)).toBe(receiptBefore);
    expect(serializeLicenseDocument(license)).toBe(licenseBefore);
    expect(license.releaseRevision).toBe(1);
    expect(receipt.lines[0].name).toBe("Original Name");
    expect(receipt.lines[0].productRevision).toBe(1);

    // A license may only cover the purchased release — never a newer one.
    expect(() => createLicenseDocument({
      entitlement: f.entitlement,
      product: rebumped,
      release: { ...f.release, revision: 2 },
      now: T(5),
    })).toThrowError(/covers purchased release 1, not 2/);
    expect(() => createLicenseDocument({
      entitlement: f.entitlement,
      product: { ...rebumped, id: "product/other" },
      now: T(5),
    })).toThrowError(/does not match the entitlement/);
    expect(() => createLicenseDocument({
      entitlement: f.entitlement,
      product: f.product,
      order: { ...f.order, id: "order/other" },
      now: T(5),
    })).toThrowError(/order does not match/);
  });

  it("license content comes only from product metadata", () => {
    const f = purchase({
      productId: "product/licensed",
      license: { type: "personal", allowsCommercialUse: false, allowsModification: false, attributionRequired: true, terms: "Personal use only. Contact the studio for redistribution rights." },
      now: T(2),
    });
    const doc = createLicenseDocument({
      entitlement: f.entitlement,
      product: f.product,
      release: f.release,
      order: f.order,
      now: T(3),
    });
    expect(doc.license.type).toBe("personal");
    expect(doc.license.version).toBe("1.0");
    expect(doc.license.permittedUse).toEqual({
      commercial: false,
      modification: false,
      redistribution: false,
      attribution: true,
    });
    expect(doc.license.restrictions).toEqual([
      "commercial use is not permitted",
      "modification is not permitted",
      "redistribution is not permitted",
      "attribution is required",
    ]);
    expect(doc.license.terms).toBe("Personal use only. Contact the studio for redistribution rights.");

    // Missing license metadata → refused, never invented.
    expect(() => createLicenseDocument({
      entitlement: f.entitlement,
      product: { ...f.product, license: { ...f.product.license, type: "", version: "" } },
      now: T(3),
    })).toThrowError(/license missing/);
    expect(() => createLicenseDocument({
      entitlement: f.entitlement,
      product: { ...f.product, license: (() => { const { terms: _omitted, ...rest } = f.product.license; return { ...rest, type: "custom" }; })() },
      now: T(3),
    })).toThrowError(/custom license has no creator-entered terms/);
  });

  it("duplicate order events change nothing", () => {
    const f = purchase({ productId: "product/dup", now: T(2) });
    const provider = new StubPaymentProvider(hmac, parse);
    provider.createCheckout({ checkoutId: "co/1", orderId: f.order.id, amountMinor: f.order.totalMinor, currency: "USD" });
    let payment = createPayment({ id: "pay/dup", orderId: f.order.id, amountMinor: f.order.totalMinor, currency: "USD", provider: "stub", now: T(2) });

    const event: WebhookEvent = {
      eventId: "evt/dup", type: "payment.succeeded", checkoutId: "co/1",
      amountMinor: f.order.totalMinor, currency: "USD", receivedAt: T(2),
    };
    const signed = provider.signedEvent(event);
    const verified = provider.verifyWebhook(signed.body, signed.signature)!;
    const seen = new Set<string>();
    const first = applyWebhook(payment, verified, seen, T(2));
    const second = applyWebhook(payment, verified, seen, T(2)); // replay against original
    expect(first.applied).toBe(true);
    expect(second.applied).toBe(false);
    payment = first.payment;
    expect(payment.status).toBe("confirmed");
    // Order untouched by duplicate events; receipt identical.
    const receipt = createReceipt({ order: f.order, products: [f.product], now: T(3) });
    expect(serializeReceipt(receipt)).toBe(serializeReceipt(createReceipt({ order: f.order, products: [f.product], now: T(3) })));
    expect(f.order.status).toBe("paid");
  });

  it("privacy: foreign orders unreachable, same error for miss and access", () => {
    const ana = purchase({ customerId: "cust/ana", productId: "product/ana-b" });
    const eve = purchase({ customerId: "cust/eve", email: "eve@example.com", productId: "product/eve-b" });
    expect(orderHistory("cust/ana", [ana.order, eve.order], [ana.product, eve.product]).map((h) => h.orderId))
      .toEqual([ana.order.id]);
    expect(() => findOrder([ana.order, eve.order], "cust/ana", eve.order.id)).toThrowError(/not found for this customer/);
    expect(() => findOrder([ana.order], "cust/ana", "order/ghost")).toThrowError(/not found for this customer/);
    expect(findOrder([ana.order, eve.order], "cust/ana", ana.order.id).id).toBe(ana.order.id);
    expect(() => orderHistory("", [ana.order])).toThrowError(/non-empty/);
    expect(() => orderHistory("cust/ana", undefined as never)).toThrowError(/orders array/);
    expect(() => findOrder(undefined as never, "cust/ana", "o")).toThrowError(/orders array/);
  });

  it("history sorts newest first with deterministic ties", () => {
    const older: Order = { ...purchase({ productId: "product/old-b", now: T(2) }).order, id: "order/z-old" };
    const newerA: Order = { ...purchase({ productId: "product/new-a", now: T(4) }).order, id: "order/a-new" };
    const newerB: Order = { ...purchase({ productId: "product/new-b", now: T(4) }).order, id: "order/b-new" };
    const history = orderHistory("cust/ana", [older, newerB, newerA]);
    expect(history.map((h) => h.orderId)).toEqual(["order/a-new", "order/b-new", "order/z-old"]);
  });
});
