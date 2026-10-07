// Shared commerce test fixtures (G21B–G21FINAL). Not a test file itself.
import { createHash } from "node:crypto";
import {
  addArtifact,
  addVariant,
  attachPreview,
  createProduct,
  setProductStatus,
  type LicenseInfo,
  type Product,
} from "../../src/product/product.js";
import { createOrder, markOrderPaid, type Order } from "../../src/commerce/orders.js";
import {
  createRelease,
  issueEntitlement,
  type Entitlement,
  type ProductRelease,
  type UpdatePolicy,
} from "../../src/commerce/entitlements.js";

export const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
export const T = (day: number): string => `2026-01-${String(day).padStart(2, "0")}T00:00:00.000Z`;

export interface Fixture {
  product: Product;
  variantId: string;
  pdfArtifactId: string;
  previewArtifactId: string;
  order: Order;
  entitlement: Entitlement;
  release: ProductRelease;
  payload: string;
}

export interface PurchaseOptions {
  customerId?: string;
  email?: string;
  productId?: string;
  name?: string;
  sku?: string;
  category?: string;
  sizes?: string[];
  formats?: string[];
  priceMinor?: number;
  quantity?: number;
  now?: string;
  updatePolicy?: UpdatePolicy;
  expiresAt?: string;
  supportContact?: string;
  requirements?: string;
  license?: Partial<LicenseInfo>;
  author?: string;
}

let seq = 0;

/** Full G19→G20 pipeline: published product → paid order → entitlement → release. */
export function purchase(options: PurchaseOptions = {}): Fixture {
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
    ...options.license,
  };

  let product = createProduct({
    id: productId,
    name,
    sku,
    category,
    author: options.author ?? "Atelier",
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
    revision: 1, checksum: sha256(`preview:${productId}:rev1`), generator: "test-harness",
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
    items: [{ product, variantId: variant.id, quantity: options.quantity ?? 1, unitPriceMinor: priceMinor }],
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
