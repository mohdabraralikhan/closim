// G21A — customer digital library.
//
// Covers the definition of done: a persistent, searchable library containing
// every legitimately purchased product — plus the required scenarios (empty,
// large, revoked, updated, deleted, repeated downloads, missing preview,
// account changes) and the privacy/entitlement invariants the library must
// never break.

import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  addArtifact,
  addVariant,
  attachPreview,
  createProduct,
  deserializeProduct,
  serializeProduct,
  setProductStatus,
  type LicenseInfo,
  type Product,
} from "../../src/product/product.js";
import { createCustomer, setAccountStatus } from "../../src/commerce/customers.js";
import { createOrder, markOrderPaid, type Order } from "../../src/commerce/orders.js";
import {
  createRelease,
  issueEntitlement,
  revokeEntitlement,
  revokeRelease,
  type Entitlement,
  type ProductRelease,
  type UpdatePolicy,
} from "../../src/commerce/entitlements.js";
import {
  fulfillDownload,
  memoryLedger,
  tokenBucketLimiter,
  type DownloadRecord,
} from "../../src/commerce/delivery.js";
import {
  addFavorite,
  buildLibrary,
  createFavorites,
  deserializeFavorites,
  deserializeLibrary,
  filterLibrary,
  isFavorite,
  libraryProductDetail,
  organizeLibrary,
  recentlyDownloaded,
  recentlyPurchased,
  removeFavorite,
  requestLibraryDownload,
  searchLibrary,
  serializeFavorites,
  serializeLibrary,
  sortLibrary,
  toggleFavorite,
  type Library,
  type LibraryEntry,
  type LibrarySort,
  type LibraryUpdateState,
} from "../../src/commerce/library.js";

const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const T = (day: number): string => `2026-01-${String(day).padStart(2, "0")}T00:00:00.000Z`;

interface Fixture {
  product: Product;
  variantId: string;
  pdfArtifactId: string;
  previewArtifactId: string;
  order: Order;
  entitlement: Entitlement;
  release: ProductRelease;
  payload: string;
}

interface PurchaseOptions {
  customerId?: string;
  email?: string;
  productId?: string;
  name?: string;
  sku?: string;
  category?: string;
  sizes?: string[];
  formats?: string[];
  priceMinor?: number;
  now?: string;
  updatePolicy?: UpdatePolicy;
  expiresAt?: string;
  supportContact?: string;
  requirements?: string;
}

let seq = 0;

/** Full G19→G20 pipeline: published product → paid order → entitlement → release. */
function purchase(options: PurchaseOptions = {}): Fixture {
  const n = ++seq;
  const customerId = options.customerId ?? "cust/ana";
  const email = options.email ?? "ana@example.com";
  const productId = options.productId ?? `product/item-${n}`;
  const name = options.name ?? `Garment ${n}`;
  const sku = options.sku ?? `SKU-${n}`;
  const category = options.category ?? "shirts";
  const sizes = options.sizes ?? ["size/s", "size/m"];
  const formats = options.formats ?? ["pdf"];
  const now = options.now ?? T(2);
  const priceMinor = options.priceMinor ?? 1500;
  const license: LicenseInfo = {
    type: "commercial-single",
    allowsCommercialUse: true,
    allowsModification: true,
    allowsRedistribution: false,
    attributionRequired: false,
    ...(options.supportContact ? { supportContact: options.supportContact } : {}),
    version: "1.0",
  };

  let product = createProduct({
    id: productId,
    name,
    sku,
    category,
    author: "Atelier",
    garment: { projectId: "proj/1", garmentId: `garment/${n}`, garmentRevision: 1, garmentFingerprint: `fp/${n}` },
    sizes,
    formats,
    license,
    description: `${name} digital sewing pattern.`,
    ...(options.requirements ? { requirements: options.requirements } : {}),
    now,
  });
  const payload = `payload:${productId}:rev1`;
  const pdf = addArtifact(product, {
    filename: `${sku}.pdf`, type: "pdf", sizeIds: sizes, format: "pdf",
    revision: 1, checksum: sha256(payload), generator: "test-harness",
  }, now);
  product = pdf.product;
  const preview = addArtifact(product, {
    filename: `${sku}-preview.png`, type: "preview", sizeIds: [], format: "png",
    revision: 1, checksum: sha256(`preview:${productId}`), generator: "test-harness",
  }, now);
  product = preview.product;
  product = attachPreview(product, preview.id, true, now);
  const variant = addVariant(product, {
    name: "Bundle", kind: "multi-size", sizeIds: sizes, formats, artifactIds: [pdf.id],
  }, now);
  product = variant.product;
  product = setProductStatus(product, "ready", now);
  product = setProductStatus(product, "published", now);

  let order = createOrder({
    id: `order/${n}`,
    customerId,
    customerEmail: email,
    currency: "USD",
    items: [{ product, variantId: variant.id, quantity: 1, unitPriceMinor: priceMinor }],
    now,
  });
  order = markOrderPaid(order, `pay/${n}`, now);
  const entitlement = issueEntitlement({
    id: `ent/${n}`,
    order,
    productId,
    variantId: variant.id,
    updatePolicy: options.updatePolicy ?? "original-only",
    ...(options.expiresAt ? { expiresAt: options.expiresAt } : {}),
    now,
  });
  const release = createRelease({
    id: `rel/${n}/1`,
    product,
    artifacts: product.artifacts.filter((a) => a.type !== "preview"),
    updatePolicy: entitlement.updatePolicy,
    now,
  });
  return {
    product,
    variantId: variant.id,
    pdfArtifactId: pdf.id,
    previewArtifactId: preview.id,
    order,
    entitlement,
    release,
    payload,
  };
}

/**
 * Build the completed re-release state (product revision 2 + frozen release).
 * G19's public API has no republish path after bumpProductRevision (stale
 * rev-1 artifacts block Ready) — G21C owns that transition. This fixture goes
 * through the validating deserializer, then freezes with the real
 * createRelease, so both records are shape-validated public types.
 */
function republishAtV2(fixture: Fixture): { product: Product; release: ProductRelease } {
  const raw = JSON.parse(serializeProduct(fixture.product)) as unknown as {
    revision: number;
    status: string;
    garment: { garmentRevision: number; garmentFingerprint: string };
    artifacts: Array<{ id: string; filename: string; revision: number; status: string; checksum: string }>;
    previewIds: string[];
    thumbnailArtifactId?: string;
    variants: Array<{ artifactIds: string[] }>;
  };
  raw.revision = 2;
  raw.status = "published";
  raw.garment.garmentRevision = 2;
  raw.garment.garmentFingerprint = "fp/rev2";
  const idMap = new Map<string, string>();
  for (const artifact of raw.artifacts) {
    const nextId = `${artifact.id}@v2`;
    idMap.set(artifact.id, nextId);
    artifact.id = nextId;
    artifact.revision = 2;
    artifact.status = "current";
    artifact.checksum = sha256(`${artifact.filename}:rev2`);
  }
  raw.previewIds = raw.previewIds.map((id) => idMap.get(id) ?? id);
  if (raw.thumbnailArtifactId) {
    raw.thumbnailArtifactId = idMap.get(raw.thumbnailArtifactId) ?? raw.thumbnailArtifactId;
  }
  for (const variant of raw.variants) {
    variant.artifactIds = variant.artifactIds.map((id) => idMap.get(id) ?? id);
  }
  const product = deserializeProduct(JSON.stringify(raw));
  const release = createRelease({
    id: `rel/${product.id}/2`,
    product,
    artifacts: product.artifacts.filter((a) => a.type !== "preview"),
    updatePolicy: fixture.entitlement.updatePolicy,
    now: T(6),
  });
  return { product, release };
}

function entryOf(library: Library, productId: string): LibraryEntry {
  const entry = library.entries.find((e) => e.productId === productId);
  if (!entry) throw new Error(`no library entry for '${productId}'`);
  return entry;
}

describe("G21A customer digital library", () => {
  it("empty library: nothing to show, organization stays safe, persists", () => {
    const library = buildLibrary({ customerId: "cust/ana", entitlements: [], nowIso: T(1) });
    expect(library.entries).toEqual([]);
    expect(library.customerId).toBe("cust/ana");
    expect(searchLibrary(library.entries, "anything")).toEqual([]);
    expect(filterLibrary(library.entries, { updateStates: ["current"] })).toEqual([]);
    expect(sortLibrary(library.entries, "purchased-desc")).toEqual([]);
    expect(organizeLibrary(library.entries, { limit: 10 })).toEqual([]);
    expect(recentlyPurchased(library.entries)).toEqual([]);
    expect(recentlyDownloaded(library.entries)).toEqual([]);
    expect(serializeLibrary(deserializeLibrary(serializeLibrary(library)))).toBe(serializeLibrary(library));
    expect(() => deserializeLibrary("not json")).toThrowError(/valid JSON/);
    expect(() => deserializeLibrary(JSON.stringify({ schemaVersion: 99, customerId: "c", entries: [] })))
      .toThrowError(/shape or schema/);
    expect(() => buildLibrary({ customerId: "", entitlements: [], nowIso: T(1) })).toThrowError(/non-empty/);
    expect(() => buildLibrary({ customerId: "cust/ana", entitlements: [], nowIso: "not-a-date" }))
      .toThrowError(/valid timestamp/);
    expect(() => buildLibrary({ customerId: "cust/ana", entitlements: undefined as never, nowIso: T(1) }))
      .toThrowError(/entitlements array/);
  });

  it("lists owned products with customer-facing display fields only", () => {
    const shirt = purchase({ productId: "product/shirt", name: "Classic Shirt", sku: "SHIRT-1", category: "shirts", priceMinor: 1500, now: T(2) });
    const dress = purchase({ productId: "product/dress", name: "Summer Dress", sku: "DRESS-1", category: "dresses", priceMinor: 2500, now: T(3) });
    const input = {
      customerId: "cust/ana",
      entitlements: [shirt.entitlement, dress.entitlement],
      orders: [shirt.order, dress.order],
      products: [shirt.product, dress.product],
      releases: [shirt.release, dress.release],
      nowIso: T(5),
    };
    const library = buildLibrary(input);
    expect(library.entries.map((e) => e.productId)).toEqual(["product/dress", "product/shirt"]);

    const entry = entryOf(library, "product/shirt");
    expect(entry).toMatchObject({
      productId: "product/shirt",
      name: "Classic Shirt",
      sku: "SHIRT-1",
      category: "shirts",
      purchasedAt: shirt.order.createdAt,
      orderId: shirt.order.id,
      entitlementId: shirt.entitlement.id,
      variantId: shirt.variantId,
      ownedRelease: 1,
      updateState: "current",
      updateAvailable: false,
      eligibleForUpdate: false,
      usable: true,
    });
    expect(entry.thumbnailArtifactId).toBe(shirt.previewArtifactId);
    expect(entry.license).toEqual({
      type: "commercial-single",
      commercialUse: true,
      modification: true,
      redistribution: false,
      attribution: false,
      version: "1.0",
    });
    expect(entry.downloads).toHaveLength(1);
    expect(entry.downloads[0]).toMatchObject({
      artifactId: shirt.pdfArtifactId,
      filename: "SHIRT-1.pdf",
      format: "pdf",
      available: true,
      sizeIds: ["size/s", "size/m"],
    });

    // Deterministic rebuild: same input, same output.
    expect(buildLibrary(input)).toEqual(library);

    // Customer-facing records never expose garment/project internals.
    const flat = JSON.stringify(library);
    expect(flat).not.toContain("projectId");
    expect(flat).not.toContain("garmentId");
    expect(flat).not.toContain("proj/");
  });

  it("revoked and expired entitlements: blocked downloads, accurate state", () => {
    const f = purchase({ productId: "product/revoke-me" });
    const revoked = revokeEntitlement(f.entitlement, T(4));
    const input = {
      customerId: "cust/ana",
      entitlements: [revoked],
      orders: [f.order],
      products: [f.product],
      releases: [f.release],
      nowIso: T(5),
    };
    const entry = entryOf(buildLibrary(input), f.product.id);
    expect(entry.updateState).toBe("revoked");
    expect(entry.usable).toBe(false);
    expect(entry.downloads[0].available).toBe(false);
    expect(entry.downloads[0].unavailableReason).toBe("entitlement-revoked");

    const detail = libraryProductDetail({ ...input, entitlementId: revoked.id });
    expect(detail.update.state).toBe("revoked");
    expect(detail.downloads[0].available).toBe(false);
    expect(detail.orderId).toBe(f.order.id); // history retained

    expect(() => requestLibraryDownload({
      customerId: "cust/ana",
      entitlement: revoked,
      release: f.release,
      artifactId: f.pdfArtifactId,
      token: "tok/revoked",
      ledger: memoryLedger(),
      limiter: tokenBucketLimiter(5, 60_000),
      nowIso: T(5),
      nowMs: Date.parse(T(5)),
    })).toThrowError(/not usable/);

    // Expired entitlement → download-unavailable (distinct from revoked).
    const expiring = purchase({ productId: "product/expiring", expiresAt: T(3) });
    const expiredEntry = entryOf(buildLibrary({
      customerId: "cust/ana",
      entitlements: [expiring.entitlement],
      products: [expiring.product],
      releases: [expiring.release],
      nowIso: T(5),
    }), expiring.product.id);
    expect(expiredEntry.updateState).toBe("download-unavailable");
    expect(expiredEntry.usable).toBe(false);
    expect(expiredEntry.downloads[0].unavailableReason).toBe("entitlement-expired");
  });

  it("updated product: newer release shows state, policy drives eligibility", () => {
    const original = purchase({ productId: "product/a", name: "Alpha", updatePolicy: "original-only" });
    const freebie = purchase({ productId: "product/b", name: "Beta", updatePolicy: "free-updates" });
    const a2 = republishAtV2(original);
    const b2 = republishAtV2(freebie);
    const base = {
      customerId: "cust/ana",
      entitlements: [original.entitlement, freebie.entitlement],
      orders: [original.order, freebie.order],
      products: [a2.product, b2.product],
      releases: [original.release, a2.release, freebie.release, b2.release],
      nowIso: T(5),
    };
    const library = buildLibrary(base);

    const a = entryOf(library, "product/a");
    expect(a.updateState).toBe("update-available");
    expect(a.updateAvailable).toBe(true);
    expect(a.eligibleForUpdate).toBe(false); // original-only: newer exists, not included
    expect(a.ownedRelease).toBe(1);
    expect(a.latestRelease).toBe(2);
    expect(a.downloads).toHaveLength(1); // owned release still listed

    const b = entryOf(library, "product/b");
    expect(b.updateState).toBe("update-available");
    expect(b.updateAvailable).toBe(true);
    expect(b.eligibleForUpdate).toBe(true); // free-updates

    // Explicit registry override wins over the release array.
    const overridden = entryOf(buildLibrary({ ...base, latestReleases: { "product/a": 3 } }), "product/a");
    expect(overridden.latestRelease).toBe(3);
    expect(overridden.eligibleForUpdate).toBe(false);

    // Revoked newest release: latest falls back to the newest ACTIVE release.
    const fallback = entryOf(buildLibrary({
      ...base,
      releases: [original.release, revokeRelease(a2.release), freebie.release, b2.release],
    }), "product/a");
    expect(fallback.latestRelease).toBe(1);
    expect(fallback.updateState).toBe("current");
    expect(fallback.updateAvailable).toBe(false);

    // Explicit registry values are validated.
    expect(() => buildLibrary({ ...base, latestReleases: { "product/a": 0 } }))
      .toThrowError(/positive integer/);
  });

  it("deleted product: record retained, owned release still delivers, no leak", () => {
    const f = purchase({ productId: "product/gone", name: "Gone Shirt" });
    const customerId = f.entitlement.customerId;

    // Product record vanished; owned release keeps the purchase usable.
    const library = buildLibrary({
      customerId,
      entitlements: [f.entitlement],
      orders: [f.order],
      releases: [f.release],
      nowIso: T(5),
    });
    const entry = entryOf(library, "product/gone");
    expect(entry.productMissing).toBe(true);
    expect(entry.name).toBe("product/gone");
    expect(entry.license).toBeUndefined();
    expect(entry.thumbnailArtifactId).toBeUndefined();
    expect(entry.updateState).toBe("deprecated");
    expect(entry.downloads).toHaveLength(1);
    expect(entry.downloads[0].available).toBe(true);

    const detail = libraryProductDetail({
      customerId,
      entitlements: [f.entitlement],
      orders: [f.order],
      releases: [f.release],
      entitlementId: f.entitlement.id,
      nowIso: T(5),
    });
    expect(detail.productMissing).toBe(true);
    expect(detail.sizes).toEqual([]);
    expect(detail.formats).toEqual([]);
    expect(detail.support).toMatchObject({
      productId: "product/gone",
      orderId: f.order.id,
      entitlementId: f.entitlement.id,
      variantId: f.variantId,
      releaseRevision: 1,
    });

    // Deleted product AND no release → nothing to deliver.
    const orphan = entryOf(buildLibrary({ customerId, entitlements: [f.entitlement], nowIso: T(5) }), "product/gone");
    expect(orphan.updateState).toBe("download-unavailable");
    expect(orphan.downloads).toEqual([]);

    // Archived (still in catalog, no longer sold) → deprecated, downloads intact.
    const archived = setProductStatus(f.product, "archived", T(4));
    const archivedEntry = entryOf(buildLibrary({
      customerId,
      entitlements: [f.entitlement],
      products: [archived],
      releases: [f.release],
      nowIso: T(5),
    }), f.product.id);
    expect(archivedEntry.updateState).toBe("deprecated");
    expect(archivedEntry.downloads[0].available).toBe(true);
  });

  it("missing preview: library renders without a thumbnail", () => {
    const f = purchase({ productId: "product/no-preview" });
    const raw = JSON.parse(serializeProduct(f.product)) as Product;
    raw.previewIds = [];
    delete raw.thumbnailArtifactId;
    const bare = deserializeProduct(JSON.stringify(raw));
    const entry = entryOf(buildLibrary({
      customerId: "cust/ana",
      entitlements: [f.entitlement],
      products: [bare],
      releases: [f.release],
      nowIso: T(5),
    }), f.product.id);
    expect(entry.thumbnailArtifactId).toBeUndefined();
    expect(entry.updateState).toBe("current");
    expect(entry.downloads[0].available).toBe(true);
    expect(entry.name).toBe(f.product.name);
  });

  it("repeated downloads: latest own success tracked, history never alters the entry", () => {
    const first = purchase({ productId: "product/first", now: T(2) });
    const second = purchase({ productId: "product/second", now: T(1) });
    const customerId = first.entitlement.customerId;
    const records: DownloadRecord[] = [
      { id: "dl/1", customerId, entitlementId: first.entitlement.id, artifactId: first.pdfArtifactId, releaseId: first.release.id, timestamp: T(3), status: "completed", bytesDelivered: 100 },
      { id: "dl/2", customerId, entitlementId: first.entitlement.id, artifactId: first.pdfArtifactId, releaseId: first.release.id, timestamp: T(6), status: "completed", bytesDelivered: 100 },
      { id: "dl/3", customerId, entitlementId: first.entitlement.id, artifactId: first.pdfArtifactId, releaseId: first.release.id, timestamp: T(4), status: "failed", bytesDelivered: 0 },
      { id: "dl/4", customerId: "cust/eve", entitlementId: first.entitlement.id, artifactId: first.pdfArtifactId, releaseId: first.release.id, timestamp: T(9), status: "completed", bytesDelivered: 100 },
      { id: "dl/5", customerId, entitlementId: second.entitlement.id, artifactId: second.pdfArtifactId, releaseId: second.release.id, timestamp: T(5), status: "completed", bytesDelivered: 100 },
    ];
    const library = buildLibrary({
      customerId,
      entitlements: [first.entitlement, second.entitlement],
      orders: [first.order, second.order],
      products: [first.product, second.product],
      releases: [first.release, second.release],
      downloads: records,
      nowIso: T(7),
    });
    const entry = entryOf(library, "product/first");
    expect(entry.lastDownloadedAt).toBe(T(6)); // failed + foreign records ignored
    expect(entry.downloads).toHaveLength(1);
    expect(entry.downloads[0].available).toBe(true);

    const recent = recentlyDownloaded(library.entries);
    expect(recent.map((e) => e.productId)).toEqual(["product/first", "product/second"]);
    expect(recentlyDownloaded(library.entries, 1).map((e) => e.productId)).toEqual(["product/first"]);
    expect(recentlyPurchased(library.entries).map((e) => e.productId)).toEqual(["product/first", "product/second"]);
  });

  it("large library: search, filter, and sort stay correct at 120 entries", () => {
    const entitlements: Entitlement[] = [];
    const orders: Order[] = [];
    const products: Product[] = [];
    const releases: ProductRelease[] = [];
    const categories = ["shirts", "dresses", "outerwear"];
    const base = Date.parse(T(1));
    for (let i = 0; i < 120; i++) {
      const f = purchase({
        productId: `product/tee-${i}`,
        name: `Tee ${String(i).padStart(3, "0")}`,
        sku: `TEE-${String(i).padStart(3, "0")}`,
        category: categories[i % 3],
        now: new Date(base + i * 60_000).toISOString(),
      });
      entitlements.push(f.entitlement);
      orders.push(f.order);
      products.push(f.product);
      releases.push(f.release);
    }
    const library = buildLibrary({ customerId: "cust/ana", entitlements, orders, products, releases, nowIso: T(2) });
    expect(library.entries).toHaveLength(120);

    expect(searchLibrary(library.entries, "tee-042")).toHaveLength(1);
    expect(searchLibrary(library.entries, "  TEE-042  ")).toHaveLength(1);
    expect(searchLibrary(library.entries, "dresses")).toHaveLength(40);
    expect(searchLibrary(library.entries, "no such tee")).toHaveLength(0);

    expect(filterLibrary(library.entries, { categories: ["outerwear"] })).toHaveLength(40);
    expect(filterLibrary(library.entries, { updateStates: ["current"] })).toHaveLength(120);
    expect(filterLibrary(library.entries, { query: "tee", categories: ["shirts"], usableOnly: true })).toHaveLength(40);

    const byPurchased = sortLibrary(library.entries, "purchased-desc");
    expect(byPurchased[0].productId).toBe("product/tee-119");
    expect(byPurchased[119].productId).toBe("product/tee-0");
    expect(sortLibrary(library.entries, "name-asc")[0].name).toBe("Tee 000");

    const organized = organizeLibrary(library.entries, {
      filter: { query: "tee" },
      sort: "purchased-asc",
      limit: 5,
    });
    expect(organized.map((e) => e.productId)).toEqual([
      "product/tee-0", "product/tee-1", "product/tee-2", "product/tee-3", "product/tee-4",
    ]);
    expect(() => organizeLibrary(library.entries, { limit: -1 })).toThrowError(/non-negative/);
    expect(() => organizeLibrary(library.entries, { limit: 1.5 })).toThrowError(/non-negative/);
  });

  it("account changes and cross-customer isolation", () => {
    const ana = purchase({ customerId: "cust/ana", productId: "product/ana-item" });
    const eve = purchase({ customerId: "cust/eve", email: "eve@example.com", productId: "product/eve-item" });
    const input = {
      customerId: "cust/ana",
      entitlements: [ana.entitlement, eve.entitlement],
      orders: [ana.order, eve.order],
      products: [ana.product, eve.product],
      releases: [ana.release, eve.release],
      nowIso: T(5),
    };
    const library = buildLibrary(input);
    expect(library.entries.map((e) => e.productId)).toEqual(["product/ana-item"]);

    // Another customer sees an empty library from the same input.
    expect(buildLibrary({ ...input, customerId: "cust/stranger" }).entries).toEqual([]);

    // Account lifecycle changes never rewrite the library view.
    const customer = createCustomer({ id: "cust/ana", email: "ana@example.com", displayName: "Ana", now: T(1) });
    expect(setAccountStatus(customer, "suspended").status).toBe("suspended");
    expect(buildLibrary(input).entries).toEqual(library.entries);

    // A foreign order snapshot never contributes purchase dates.
    const foreign: Order = { ...ana.order, customerId: "cust/eve" };
    const noOrders = entryOf(buildLibrary({
      customerId: "cust/ana",
      entitlements: [ana.entitlement],
      orders: [foreign],
      products: [ana.product],
      releases: [ana.release],
      nowIso: T(5),
    }), "product/ana-item");
    expect(noOrders.purchasedAt).toBe(ana.entitlement.issuedAt);
  });

  it("favorites: add, remove, toggle, persist, and filter", () => {
    let favorites = createFavorites("cust/ana", T(1));
    expect(favorites.productIds).toEqual([]);

    favorites = addFavorite(favorites, "product/shirt", T(2));
    favorites = addFavorite(favorites, "product/shirt", T(3)); // idempotent
    expect(favorites.productIds).toEqual(["product/shirt"]);
    expect(favorites.updatedAt).toBe(T(2)); // no-op never bumps state

    favorites = addFavorite(favorites, "product/dress", T(4));
    expect(isFavorite(favorites, "product/dress")).toBe(true);
    favorites = removeFavorite(favorites, "product/shirt");
    expect(isFavorite(favorites, "product/shirt")).toBe(false);
    favorites = removeFavorite(favorites, "product/shirt"); // idempotent
    favorites = toggleFavorite(favorites, "product/dress");
    expect(isFavorite(favorites, "product/dress")).toBe(false);
    favorites = toggleFavorite(favorites, "product/dress");
    expect(isFavorite(favorites, "product/dress")).toBe(true);

    expect(serializeFavorites(deserializeFavorites(serializeFavorites(favorites)))).toBe(serializeFavorites(favorites));
    expect(() => addFavorite(favorites, " ")).toThrowError(/non-empty/);
    expect(() => removeFavorite(favorites, "")).toThrowError(/non-empty/);
    expect(() => createFavorites("")).toThrowError(/non-empty/);
    expect(() => deserializeFavorites("not json")).toThrowError(/valid JSON/);
    expect(() => deserializeFavorites(JSON.stringify({ schemaVersion: 99, customerId: "c", productIds: [] })))
      .toThrowError(/shape or schema/);
    expect(() => deserializeFavorites(JSON.stringify({ schemaVersion: 1, customerId: "c", productIds: [1] })))
      .toThrowError(/non-empty strings/);
    // Duplicates collapse on read.
    expect(deserializeFavorites(JSON.stringify({ schemaVersion: 1, customerId: "c", productIds: ["p/a", "p/a"], updatedAt: T(1) })).productIds)
      .toEqual(["p/a"]);

    const a = purchase({ productId: "product/fav-a" });
    const b = purchase({ productId: "product/fav-b" });
    const library = buildLibrary({
      customerId: "cust/ana",
      entitlements: [a.entitlement, b.entitlement],
      products: [a.product, b.product],
      releases: [a.release, b.release],
      nowIso: T(5),
    });
    const onlyA = addFavorite(createFavorites("cust/ana", T(1)), a.product.id);
    expect(filterLibrary(library.entries, { favoritesOnly: true }, onlyA).map((e) => e.productId))
      .toEqual(["product/fav-a"]);
    expect(() => filterLibrary(library.entries, { favoritesOnly: true }))
      .toThrowError(/requires a favorites list/);
  });

  it("search, filter, and sort combine deterministically", () => {
    const shirt = purchase({ productId: "product/alpha-shirt", name: "Alpha Shirt", sku: "ALPHA-1", category: "shirts", now: T(4) });
    const dress = purchase({ productId: "product/beta-dress", name: "Beta Dress", sku: "BETA-1", category: "dresses", now: T(2) });
    const coat = purchase({ productId: "product/gamma-coat", name: "Gamma Coat", sku: "GAMMA-1", category: "outerwear", now: T(3) });
    const input = {
      customerId: "cust/ana",
      entitlements: [shirt.entitlement, dress.entitlement, coat.entitlement],
      orders: [shirt.order, dress.order, coat.order],
      products: [shirt.product, dress.product, coat.product],
      releases: [shirt.release, dress.release, coat.release],
      nowIso: T(5),
    };
    const entries = buildLibrary(input).entries;

    // Search: name, SKU, productId; case-insensitive; trims; empty shows all.
    expect(searchLibrary(entries, "beta").map((e) => e.productId)).toEqual(["product/beta-dress"]);
    expect(searchLibrary(entries, "  alpha-1  ")).toHaveLength(1);
    expect(searchLibrary(entries, "gamma-coat")).toHaveLength(1);
    expect(searchLibrary(entries, "")).toHaveLength(3);

    // Filter: categories, states, runtime validation.
    expect(filterLibrary(entries, { categories: ["shirts", "outerwear"] }).map((e) => e.productId))
      .toEqual(["product/alpha-shirt", "product/gamma-coat"]);
    expect(filterLibrary(entries, { updateStates: ["current"] })).toHaveLength(3);
    expect(() => filterLibrary(entries, { updateStates: ["bogus" as LibraryUpdateState] }))
      .toThrowError(/unknown library update state/);
    expect(() => sortLibrary(entries, "bogus" as LibrarySort)).toThrowError(/unknown library sort/);

    // Sorts: purchase order, name order, deterministic tie-break.
    expect(sortLibrary(entries, "purchased-desc").map((e) => e.productId))
      .toEqual(["product/alpha-shirt", "product/gamma-coat", "product/beta-dress"]);
    expect(sortLibrary(entries, "purchased-asc").map((e) => e.productId))
      .toEqual(["product/beta-dress", "product/gamma-coat", "product/alpha-shirt"]);
    expect(sortLibrary(entries, "name-asc").map((e) => e.name))
      .toEqual(["Alpha Shirt", "Beta Dress", "Gamma Coat"]);
    expect(sortLibrary(entries, "name-desc")[0].name).toBe("Gamma Coat");

    const tieZ = purchase({ productId: "product/zzz-tie", now: T(3) });
    const tieA = purchase({ productId: "product/aaa-tie", now: T(3) });
    const tied = buildLibrary({
      customerId: "cust/ana",
      entitlements: [tieZ.entitlement, tieA.entitlement],
      products: [tieZ.product, tieA.product],
      releases: [tieZ.release, tieA.release],
      nowIso: T(5),
    }).entries;
    expect(sortLibrary(tied, "purchased-desc").map((e) => e.productId))
      .toEqual(["product/aaa-tie", "product/zzz-tie"]);

    // Revoked entry: usableOnly narrows; revoked state filterable.
    const revokedCoat = revokeEntitlement(coat.entitlement, T(5));
    const withRevoked = buildLibrary({ ...input, entitlements: [shirt.entitlement, dress.entitlement, revokedCoat] }).entries;
    expect(filterLibrary(withRevoked, { usableOnly: true })).toHaveLength(2);
    expect(filterLibrary(withRevoked, { updateStates: ["revoked"] })).toHaveLength(1);

    // No downloads yet → recently downloaded empty; purchased limit works.
    expect(recentlyDownloaded(entries)).toEqual([]);
    expect(recentlyPurchased(entries, 2).map((e) => e.productId))
      .toEqual(["product/alpha-shirt", "product/gamma-coat"]);

    // Organization never mutates the input array.
    expect(entries.map((e) => e.productId))
      .toEqual(["product/alpha-shirt", "product/beta-dress", "product/gamma-coat"]);
  });

  it("product detail: release, sizes, formats, license, updates, support", () => {
    const f = purchase({
      productId: "product/detail",
      name: "Detail Shirt",
      supportContact: "support@atelier.example",
      requirements: "A4 printer or plotter",
    });
    const input = {
      customerId: "cust/ana",
      entitlements: [f.entitlement],
      orders: [f.order],
      products: [f.product],
      releases: [f.release],
      nowIso: T(5),
    };
    const detail = libraryProductDetail({ ...input, entitlementId: f.entitlement.id });
    expect(detail).toMatchObject({
      productId: "product/detail",
      name: "Detail Shirt",
      sku: f.product.sku,
      purchasedAt: f.order.createdAt,
      orderId: f.order.id,
      entitlementId: f.entitlement.id,
      variantId: f.variantId,
      purchasedRelease: 1,
      sizes: ["size/s", "size/m"],
      formats: ["pdf"],
      requirements: "A4 printer or plotter",
    });
    expect(detail.variant).toMatchObject({ id: f.variantId, kind: "multi-size" });
    expect(detail.downloads).toHaveLength(1);
    expect(detail.downloads[0].available).toBe(true);
    expect(detail.license).toMatchObject({ type: "commercial-single", version: "1.0" });
    expect(detail.update).toMatchObject({
      state: "current",
      available: false,
      eligible: false,
      policy: "original-only",
      ownedRelease: 1,
    });
    expect(detail.support).toMatchObject({
      contact: "support@atelier.example",
      productId: "product/detail",
      orderId: f.order.id,
      entitlementId: f.entitlement.id,
      variantId: f.variantId,
      releaseId: f.release.id,
      releaseRevision: 1,
    });
    expect(JSON.stringify(detail)).not.toContain("projectId");

    // Ownership boundary: other customers' entitlements are unreachable.
    expect(() => libraryProductDetail({ ...input, customerId: "cust/eve", entitlementId: f.entitlement.id }))
      .toThrowError(/another customer/);
    expect(() => libraryProductDetail({ ...input, entitlementId: "ent/ghost" }))
      .toThrowError(/was not found/);
  });

  it("download: delegates to G20 delivery, never bypassing entitlement checks", () => {
    const f = purchase({ productId: "product/dl" });
    const other = purchase({ productId: "product/dl-other", customerId: "cust/eve", email: "eve@example.com" });
    const v2 = republishAtV2(f);
    const ledger = memoryLedger();
    const limiter = tokenBucketLimiter(5, 60_000);
    const nowIso = T(5);
    const nowMs = Date.parse(nowIso);
    const base = {
      customerId: "cust/ana",
      entitlement: f.entitlement,
      release: f.release,
      artifactId: f.pdfArtifactId,
      ledger,
      limiter,
      nowIso,
      nowMs,
    };

    // Happy path: authorize → fulfill with the checksum-frozen payload.
    const auth = requestLibraryDownload({ ...base, token: "tok/a" });
    expect(auth.filename).toBe(`${f.product.sku}.pdf`);
    const done = fulfillDownload({
      id: "dl/1",
      authorization: auth,
      payload: f.payload,
      expectedChecksum: f.release.artifacts[0].checksum,
      hashHex: sha256,
      nowIso,
      nowMs,
    });
    expect(done.record.status).toBe("completed");
    expect(done.bytes).toBe(f.payload.length);

    // Repeated download: fresh token succeeds; replayed token dies.
    expect(requestLibraryDownload({ ...base, token: "tok/b" }).token).toBe("tok/b");
    expect(() => requestLibraryDownload({ ...base, token: "tok/a" })).toThrowError(/already used/);

    // Wrong customer, wrong product, uncovered revision: all refused.
    expect(() => requestLibraryDownload({ ...base, customerId: "cust/eve", token: "tok/c" }))
      .toThrowError(/another customer/);
    expect(() => requestLibraryDownload({ ...base, release: other.release, artifactId: other.pdfArtifactId, token: "tok/d" }))
      .toThrowError(/another product/);
    expect(() => requestLibraryDownload({ ...base, release: v2.release, artifactId: v2.release.artifacts[0].artifactId, token: "tok/e" }))
      .toThrowError(/does not cover release revision 2/);

    // Rate limiting still enforced through delegation.
    const strict = tokenBucketLimiter(1, 60_000);
    expect(requestLibraryDownload({ ...base, limiter: strict, token: "tok/f" }).token).toBe("tok/f");
    expect(() => requestLibraryDownload({ ...base, limiter: strict, token: "tok/g" }))
      .toThrowError(/rate limit/);
  });
});
