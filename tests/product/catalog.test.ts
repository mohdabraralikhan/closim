// Catalog, previews, licensing tests (G19C/D/E integration surfaces).
import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  addProduct,
  createCatalog,
  deserializeCatalog,
  findProduct,
  productCard,
  removeProduct,
  searchCatalog,
  serializeCatalog,
  updateProduct,
  validateCatalog,
} from "../../src/product/catalog.js";
import { createProduct, type Product } from "../../src/product/product.js";
import {
  createLicense,
  customerSummary,
  isKnownLicenseType,
  licensedArtifacts,
} from "../../src/product/licensing.js";
import {
  generatePreviews,
  markPreviewAsset,
  previewFilename,
  primaryPreview,
  staleAfterGarmentChange,
  type PreviewAsset,
} from "../../src/product/previews.js";

const LICENSE = {
  type: "commercial-single",
  allowsCommercialUse: true,
  allowsModification: true,
  allowsRedistribution: false,
  attributionRequired: false,
  version: "1.0",
};

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function makeProduct(id: string, sku: string, name: string, overrides: Partial<Product> = {}): Product {
  return {
    ...createProduct({
      id, name, sku, category: "shirts", author: "Atelier",
      garment: { projectId: "proj/1", garmentId: "garment/a", garmentRevision: 1, garmentFingerprint: "fp/1" },
      sizes: ["size/s", "size/m"], formats: ["dxf", "pdf"],
      license: { ...LICENSE }, description: `${name} pattern.`,
    }),
    ...overrides,
  };
}

describe("G19C catalog", () => {
  it("adds, updates, removes with identity protection", () => {
    let catalog = createCatalog("cat/main", "Main", ["shirts", "dresses"]);
    const a = makeProduct("product/a", "SKU-A", "Alpha");
    catalog = addProduct(catalog, a);
    expect(() => addProduct(catalog, a)).toThrowError(/already in catalog/);
    expect(() => addProduct(catalog, makeProduct("product/b", "SKU-A", "Beta"))).toThrowError(/SKU/);
    expect(findProduct(catalog, "product/a")?.name).toBe("Alpha");
    catalog = updateProduct(catalog, { ...a, name: "Alpha 2" });
    expect(findProduct(catalog, "product/a")?.name).toBe("Alpha 2");
    expect(() => updateProduct(catalog, { ...a, id: "ghost", sku: "SKU-A" })).toThrowError(/not in catalog/);
    catalog = removeProduct(catalog, "product/a");
    expect(findProduct(catalog, "product/a")).toBeNull();
    expect(() => removeProduct(catalog, "product/a")).toThrowError(/not in catalog/);
  });

  it("searches, filters, and sorts deterministically", () => {
    let catalog = createCatalog("cat", "C");
    const a = { ...makeProduct("product/a", "SKU-A", "Alpha Shirt"), status: "published" as const, tags: ["collar"] };
    const b = { ...makeProduct("product/b", "SKU-B", "Beta Dress"), status: "draft" as const, category: "dresses", tags: ["summer"] };
    catalog = addProduct(addProduct(catalog, a), b);
    expect(searchCatalog(catalog, { query: "alpha" }).map((p) => p.id)).toEqual(["product/a"]);
    expect(searchCatalog(catalog, { query: "sku-b" }).map((p) => p.id)).toEqual(["product/b"]);
    expect(searchCatalog(catalog, { category: "dresses" }).map((p) => p.id)).toEqual(["product/b"]);
    expect(searchCatalog(catalog, { tags: ["collar"] }).map((p) => p.id)).toEqual(["product/a"]);
    expect(searchCatalog(catalog, { status: ["draft"] }).map((p) => p.id)).toEqual(["product/b"]);
    expect(searchCatalog(catalog, { size: "size/m" })).toHaveLength(2);
    expect(searchCatalog(catalog, { format: "svg" })).toHaveLength(0);
    expect(searchCatalog(catalog, {}, "sku").map((p) => p.sku)).toEqual(["SKU-A", "SKU-B"]);
    expect(searchCatalog(catalog, {}, "revision").map((p) => p.id)).toEqual(["product/a", "product/b"]);
  });

  it("scales to large catalogs and audits integrity", () => {
    let catalog = createCatalog("cat", "C");
    for (let i = 0; i < 200; i++) {
      catalog = addProduct(catalog, makeProduct(`product/${i}`, `SKU-${i}`, `Garment ${i}`));
    }
    expect(searchCatalog(catalog, { query: "garment 1" }).length).toBeGreaterThan(0);
    expect(catalog.products).toHaveLength(200);
    const cards = searchCatalog(catalog, {}, "sku").slice(0, 3).map(productCard);
    expect(cards[0]).toMatchObject({ sku: "SKU-0", revision: 1 });
    // Scale products carry no previews: the audit must say exactly that
    // (and nothing about identity, which is clean).
    const issues = validateCatalog(catalog);
    expect(issues).toHaveLength(200);
    expect(new Set(issues.map((i) => i.code))).toEqual(new Set(["missing-preview"]));
    const duped = { ...catalog, products: [...catalog.products, catalog.products[0]] };
    expect(validateCatalog(duped).map((i) => i.code)).toContain("duplicate-id");
    expect(serializeCatalog(deserializeCatalog(serializeCatalog(catalog)))).toBe(serializeCatalog(catalog));
  });

  it("cards stay lightweight (no garment data)", () => {
    const card = productCard(makeProduct("p", "S", "N"));
    expect(JSON.stringify(card)).not.toContain("garment");
    expect(card.thumbnail).toBeNull();
  });
});

describe("G19D previews", () => {
  it("names files deterministically and batches generation", () => {
    expect(previewFilename("classic-shirt", 2, "front")).toBe("classic-shirt-r2-front.png");
    expect(previewFilename("classic-shirt", 2, "front", "size-m", "jpg")).toBe("classic-shirt-r2-front-size-m.jpg");
    const product = makeProduct("product/a", "SKU-A", "Alpha");
    const report = generatePreviews(product, ["front", "back", "three-quarter"], (view) => ({
      filename: previewFilename(product.slug, product.revision, view),
      checksum: sha256(`${product.id}:${view}`),
      widthPx: 1024,
      heightPx: 1024,
    }), { camera: "studio", lighting: "product" });
    expect(report.completed).toEqual(["front", "back", "three-quarter"]);
    expect(report.failed).toEqual([]);
    expect(report.generated).toHaveLength(3);
    expect(report.generated[0].garmentRevision).toBe(1);
    expect(report.generated[0].renderConfig.lighting).toBe("product");
    const failing = generatePreviews(product, ["front", "back"], (view) => {
      if (view === "back") throw new Error("no GL context");
      return { filename: "f.png", checksum: "c", widthPx: 1, heightPx: 1 };
    });
    expect(failing.completed).toEqual(["front"]);
    expect(failing.failed).toEqual([{ view: "back", reason: "no GL context" }]);
  });

  it("marks staleness on garment change, never silently", () => {
    const previews: PreviewAsset[] = [{
      id: "product/a/preview/front", productId: "product/a", productRevision: 1,
      view: "front", garmentRevision: 3, garmentFingerprint: "fp/old",
      renderConfig: {}, filename: "a.png", checksum: "c", status: "current",
    }];
    const stale = staleAfterGarmentChange(previews, 4, "fp/new");
    expect(stale[0].status).toBe("stale");
    expect(previews[0].status).toBe("current"); // input untouched
    const same = staleAfterGarmentChange(previews, 3, "fp/old");
    expect(same[0].status).toBe("current");
    const marked = markPreviewAsset(stale, previews[0].id, "missing");
    expect(marked[0].status).toBe("missing");
    expect(() => markPreviewAsset(stale, "ghost", "current")).toThrowError(/does not exist/);
    expect(primaryPreview(stale)).toBeNull();
  });
});

describe("G19E licensing", () => {
  it("builds licenses and customer summaries", () => {
    expect(isKnownLicenseType("commercial-single")).toBe(true);
    expect(isKnownLicenseType("pirate")).toBe(false);
    const license = createLicense({ type: "educational", supportContact: "help@example.com" });
    expect(license.allowsModification).toBe(true);
    expect(() => createLicense({ type: "custom" })).toThrowError(/terms/);
    const product = makeProduct("product/a", "SKU-A", "Alpha");
    const withFiles = {
      ...product,
      artifacts: [
        { id: "a/1", filename: "shirt.pdf", type: "pdf" as const, sizeIds: [], format: "pdf", revision: 1, checksum: "c1", generator: "g13", status: "current" as const },
        { id: "a/2", filename: "old.dxf", type: "dxf" as const, sizeIds: [], format: "dxf", revision: 1, checksum: "c2", generator: "g13", status: "stale" as const },
      ],
    };
    const summary = customerSummary(withFiles, withFiles.artifacts);
    expect(summary.filesIncluded).toEqual(["shirt.pdf"]); // stale excluded
    expect(summary.license.commercialUse).toBe(true);
    expect(summary.version).toBe("r1");
    expect(licensedArtifacts(withFiles)).toHaveLength(1);
    expect(JSON.stringify(summary)).not.toContain("garmentId");
  });
});
