// G19A tests: product identity, states, variants, revisions, persistence.
import { describe, expect, it } from "vitest";
import {
  addArtifact,
  addVariant,
  attachPreview,
  bumpProductRevision,
  createProduct,
  deserializeProduct,
  duplicateProduct,
  fingerprintPayload,
  isPinnedCurrent,
  markArtifact,
  productReady,
  serializeProduct,
  setProductStatus,
  updateProductDetails,
  validateLicense,
  validateProduct,
  type Product,
} from "../../src/product/product.js";

const LICENSE = {
  type: "commercial-single",
  allowsCommercialUse: true,
  allowsModification: true,
  allowsRedistribution: false,
  attributionRequired: false,
  version: "1.0",
};

function garmentRef(overrides: Partial<Product["garment"]> = {}): Product["garment"] {
  return { projectId: "proj/1", garmentId: "garment/shirt", garmentRevision: 3, garmentFingerprint: "fp/abc123", ...overrides };
}

function makeProduct(overrides: Partial<Product> = {}): Product {
  const base = createProduct({
    id: "product/classic-shirt",
    name: "Classic Collared Shirt",
    sku: "CSHIRT-001",
    category: "shirts",
    author: "Atelier",
    garment: garmentRef(),
    sizes: ["size/xs", "size/s", "size/m", "size/l", "size/xl", "size/xxl"],
    formats: ["dxf", "pdf", "svg"],
    license: { ...LICENSE },
    description: "A classic collared shirt pattern.",
    tags: ["shirt", "collar"],
  });
  return { ...base, ...overrides };
}

describe("G19A product model", () => {
  it("creates products with slugs and stable identity", () => {
    const product = makeProduct();
    expect(product.slug).toBe("classic-collared-shirt");
    expect(product.status).toBe("draft");
    expect(product.revision).toBe(1);
    expect(product.schemaVersion).toBe(1);
    expect(() => createProduct({ ...makeProduct(), id: "", name: "x", sku: "s", category: "c", author: "a", garment: garmentRef(), license: { ...LICENSE } })).toThrowError(/id/);
    expect(() => createProduct({ ...makeProduct(), id: "x", name: "!!!", sku: "s", category: "c", author: "a", garment: garmentRef(), license: { ...LICENSE } })).toThrowError(/slug/);
  });

  it("moves through application states with gating", () => {
    let product = makeProduct();
    // Draft without description/previews/license-complete cannot go ready.
    const bare = createProduct({
      id: "p/bare", name: "Bare", sku: "BARE-1", category: "shirts", author: "A",
      garment: garmentRef(), license: { type: "personal", allowsCommercialUse: false, allowsModification: true, allowsRedistribution: false, attributionRequired: false, version: "1.0" },
    });
    expect(() => setProductStatus(bare, "ready")).toThrowError(/not releasable/);
    expect(() => setProductStatus(product, "published")).toThrowError(/cannot move/);
    expect(() => setProductStatus(product, "archived")).not.toThrow();
    product = makeProduct();
    // Complete product: description + preview + current artifacts.
    const withPreview = attachPreview(
      { ...product, artifacts: [{ id: "a/preview", filename: "hero.png", type: "preview", sizeIds: [], format: "png", revision: 1, checksum: "fp/1", generator: "g15", status: "current" }] },
      "a/preview",
      true,
    );
    expect(productReady(withPreview)).toBe(true);
    expect(setProductStatus(withPreview, "ready").status).toBe("ready");
    const published = setProductStatus(setProductStatus(withPreview, "ready"), "published");
    expect(published.status).toBe("published");
    expect(setProductStatus(published, "unpublished").status).toBe("unpublished");
  });

  it("manages variants without duplicating garment data", () => {
    let product = makeProduct();
    const a = addArtifact(product, { filename: "shirt-m.pdf", type: "pdf", sizeIds: ["size/m"], format: "pdf", revision: 1, checksum: "fp/1", generator: "g13" });
    product = a.product;
    const v = addVariant(product, { name: "M only", kind: "single-size", sizeIds: ["size/m"], formats: ["pdf"], artifactIds: [a.id] });
    expect(v.id).toBe("product/classic-shirt/variant/1");
    expect(() => addVariant(v.product, { name: "Bad", kind: "single-size", sizeIds: ["size/m", "size/l"], formats: [], artifactIds: [] })).toThrowError(/exactly one size/);
    expect(() => addVariant(v.product, { name: "Bad", kind: "multi-size", sizeIds: ["size/xxxl"], formats: [], artifactIds: [] })).toThrowError(/outside the product size range/);
    expect(() => addVariant(v.product, { name: "Bad", kind: "custom", sizeIds: [], formats: [], artifactIds: ["ghost"] })).toThrowError(/unknown artifact/);
    expect(() => addArtifact(v.product, { filename: "x", type: "pdf", sizeIds: [], format: "pdf", revision: 99, checksum: "c", generator: "g" })).toThrowError(/revision/);
  });

  it("pins revisions: garment edits never move a release silently", () => {
    let product = makeProduct();
    expect(isPinnedCurrent(product, 3, "fp/abc123")).toBe(true);
    expect(isPinnedCurrent(product, 4, "fp/def456")).toBe(false);
    const bumped = bumpProductRevision(product, { ...garmentRef(), garmentRevision: 4, garmentFingerprint: "fp/def456" });
    expect(bumped.revision).toBe(2);
    expect(bumped.status).toBe("draft");
    expect(isPinnedCurrent(bumped, 4, "fp/def456")).toBe(true);
    expect(() => bumpProductRevision(product, { ...garmentRef(), projectId: "other", garmentRevision: 4, garmentFingerprint: "x" })).toThrowError(/same garment/);
  });

  it("duplicates, archives, and round-trips", () => {
    const product = makeProduct();
    const copy = duplicateProduct(product, "product/copy", "CSHIRT-002");
    expect(copy.id).toBe("product/copy");
    expect(copy.sku).toBe("CSHIRT-002");
    expect(copy.status).toBe("draft");
    expect(copy.garment.garmentFingerprint).toBe(product.garment.garmentFingerprint);
    expect(() => duplicateProduct(product, "product/classic-shirt", "X")).toThrowError(/must differ/);
    expect(serializeProduct(deserializeProduct(serializeProduct(product)))).toBe(serializeProduct(product));
    expect(() => deserializeProduct("nope")).toThrowError(/JSON/);
    const updated = updateProductDetails(product, { name: "Classic Shirt v2", tags: ["a", "b"] });
    expect(updated.slug).toBe("classic-shirt-v2");
    expect(validateLicense({ ...LICENSE }));
    expect(() => validateLicense({ type: "custom", allowsCommercialUse: false, allowsModification: false, allowsRedistribution: false, attributionRequired: false, version: "1.0" }, "p")).toThrowError(/terms/);
  });

  it("fingerprints payloads deterministically", () => {
    expect(fingerprintPayload('{"a":1}')).toBe(fingerprintPayload('{"a":1}'));
    expect(fingerprintPayload('{"a":1}')).not.toBe(fingerprintPayload('{"a":2}'));
  });

  it("marks artifacts and attaches previews explicitly", () => {
    let product = makeProduct();
    const a = addArtifact(product, { filename: "f.pdf", type: "pdf", sizeIds: [], format: "pdf", revision: 1, checksum: "c", generator: "g" });
    product = markArtifact(a.product, a.id, "stale");
    expect(product.artifacts[0].status).toBe("stale");
    expect(() => markArtifact(product, "ghost", "current")).toThrowError(/does not exist/);
    expect(() => attachPreview(product, "ghost")).toThrowError(/does not exist/);
  });
});
