// G19A — digital garment product model.
//
// A Product is a commercial definition DERIVED from a garment, never the
// garment source document itself. It pins the exact released garment
// revision (number + content fingerprint): later garment edits cannot
// silently change a released product — a new product revision is an
// explicit action.
//
// Product schema is versioned independently (PRODUCT_SCHEMA_VERSION) from
// every garment/pattern schema.

import { PatternCadError } from "../pattern/cad.js";

export const PRODUCT_SCHEMA_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function fnv1a(text: string): string {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export type ProductStatus = "draft" | "ready" | "published" | "unpublished" | "archived";

export const PRODUCT_STATUSES: ProductStatus[] = ["draft", "ready", "published", "unpublished", "archived"];

export interface ProductGarmentRef {
  projectId: string;
  garmentId: string;
  /** Released garment revision number (GarmentProject.metadata.revision). */
  garmentRevision: number;
  /** Content fingerprint of the released garment (see fingerprintGarment). */
  garmentFingerprint: string;
  /** Production payload fingerprint at release (when attached). */
  productionFingerprint?: string;
}

export type VariantKind = "single-size" | "multi-size" | "format-bundle" | "custom";

export interface ProductVariant {
  id: string;
  name: string;
  kind: VariantKind;
  sizeIds: string[];
  formats: string[];
  artifactIds: string[];
}

export type ArtifactType =
  | "dxf" | "svg" | "pdf" | "marker" | "grading" | "spec"
  | "preview" | "native" | "readme";

export type ArtifactStatus = "current" | "stale" | "missing";

export interface ProductArtifact {
  id: string;
  filename: string;
  type: ArtifactType;
  sizeIds: string[];
  format: string;
  /** Product revision this artifact was generated for. */
  revision: number;
  /** Hex checksum of the artifact bytes (compared, never recomputed silently). */
  checksum: string;
  generator: string;
  status: ArtifactStatus;
}

export interface LicenseInfo {
  type: string;
  allowsCommercialUse: boolean;
  allowsModification: boolean;
  allowsRedistribution: boolean;
  attributionRequired: boolean;
  supportContact?: string;
  /** Creator-entered terms (required for custom licenses). */
  terms?: string;
  version: string;
}

export interface Product {
  schemaVersion: typeof PRODUCT_SCHEMA_VERSION;
  id: string;
  name: string;
  slug: string;
  sku: string;
  shortDescription?: string;
  description?: string;
  category: string;
  tags: string[];
  author: string;
  brand?: string;
  status: ProductStatus;
  thumbnailArtifactId?: string;
  previewIds: string[];
  garment: ProductGarmentRef;
  /** Product revision (bumped explicitly on re-release). */
  revision: number;
  sizes: string[];
  formats: string[];
  license: LicenseInfo;
  variants: ProductVariant[];
  artifacts: ProductArtifact[];
  difficulty?: string;
  requirements?: string;
  instructions?: string;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

/** Content fingerprint for garment payloads (garment JSON, production JSON, ...). */
export function fingerprintPayload(canonicalJson: string): string {
  return `fp/${fnv1a(canonicalJson)}`;
}

export function slugify(name: string): string {
  const slug = name.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (!slug) throw new PatternCadError("invalid-document", "product name yields an empty slug");
  return slug;
}

function requireNonEmpty(value: string | undefined, label: string, entityId?: string): string {
  if (!value || !value.trim()) throw new PatternCadError("invalid-document", `product ${label} must be non-empty`, entityId);
  return value;
}

export function createProduct(partial: {
  id: string;
  name: string;
  sku: string;
  category: string;
  author: string;
  garment: ProductGarmentRef;
  sizes?: string[];
  formats?: string[];
  license: LicenseInfo;
  shortDescription?: string;
  description?: string;
  brand?: string;
  tags?: string[];
  difficulty?: string;
  requirements?: string;
  instructions?: string;
  notes?: string;
  now?: string;
}): Product {
  requireNonEmpty(partial.id, "id");
  requireNonEmpty(partial.name, "name", partial.id);
  requireNonEmpty(partial.sku, "SKU", partial.id);
  requireNonEmpty(partial.category, "category", partial.id);
  requireNonEmpty(partial.author, "author", partial.id);
  if (!partial.garment || !partial.garment.projectId || !partial.garment.garmentId ||
    !Number.isInteger(partial.garment.garmentRevision) || partial.garment.garmentRevision < 1 ||
    !partial.garment.garmentFingerprint) {
    throw new PatternCadError("invalid-document", "product needs a pinned garment revision", partial.id);
  }
  validateLicense(partial.license, partial.id);
  const timestamp = partial.now ?? new Date().toISOString();
  return {
    schemaVersion: PRODUCT_SCHEMA_VERSION,
    id: partial.id,
    name: partial.name,
    slug: slugify(partial.name),
    sku: partial.sku,
    ...(partial.shortDescription ? { shortDescription: partial.shortDescription } : {}),
    ...(partial.description ? { description: partial.description } : {}),
    category: partial.category,
    tags: [...(partial.tags ?? [])],
    author: partial.author,
    ...(partial.brand ? { brand: partial.brand } : {}),
    status: "draft",
    previewIds: [],
    garment: clone(partial.garment),
    revision: 1,
    sizes: [...(partial.sizes ?? [])],
    formats: [...(partial.formats ?? [])],
    license: clone(partial.license),
    variants: [],
    artifacts: [],
    ...(partial.difficulty ? { difficulty: partial.difficulty } : {}),
    ...(partial.requirements ? { requirements: partial.requirements } : {}),
    ...(partial.instructions ? { instructions: partial.instructions } : {}),
    ...(partial.notes ? { notes: partial.notes } : {}),
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

export function validateLicense(license: LicenseInfo, entityId?: string): void {
  if (!license || typeof license !== "object" || !license.type || !license.version) {
    throw new PatternCadError("invalid-document", "license needs type and version", entityId);
  }
  if (license.type === "custom" && !license.terms) {
    throw new PatternCadError("invalid-document", "custom licenses require entered terms", entityId);
  }
}

function touch(product: Product, now?: string): Product {
  return { ...product, updatedAt: now ?? new Date().toISOString() };
}

export function updateProductDetails(
  product: Product,
  patch: Partial<Pick<Product, "name" | "shortDescription" | "description" | "category" | "tags" | "brand" | "difficulty" | "requirements" | "instructions" | "notes">>,
  now?: string,
): Product {
  const next = clone(product);
  Object.assign(next, clone(patch));
  if (patch.name) next.slug = slugify(patch.name);
  if (patch.tags) next.tags = [...patch.tags];
  return touch(next, now);
}

export function addVariant(product: Product, variant: Omit<ProductVariant, "id"> & { id?: string }, now?: string): { product: Product; id: string } {
  const next = clone(product);
  const id = variant.id ?? `${product.id}/variant/${next.variants.length + 1}`;
  if (next.variants.some((v) => v.id === id)) {
    throw new PatternCadError("duplicate-id", `variant '${id}' already exists`, id);
  }
  if (variant.kind !== "single-size" && variant.kind !== "multi-size" && variant.kind !== "format-bundle" && variant.kind !== "custom") {
    throw new PatternCadError("invalid-document", `unknown variant kind '${String(variant.kind)}'`, id);
  }
  if (variant.kind === "single-size" && variant.sizeIds.length !== 1) {
    throw new PatternCadError("invalid-document", "single-size variants need exactly one size", id);
  }
  if (variant.sizeIds.some((s) => !product.sizes.includes(s))) {
    throw new PatternCadError("invalid-document", "variant references sizes outside the product size range", id);
  }
  for (const artifactId of variant.artifactIds) {
    if (!next.artifacts.some((a) => a.id === artifactId)) {
      throw new PatternCadError("missing-reference", `variant references unknown artifact '${artifactId}'`, id);
    }
  }
  next.variants.push({ ...clone(variant), id } as ProductVariant);
  return { product: touch(next, now), id };
}

export function addArtifact(product: Product, artifact: Omit<ProductArtifact, "id" | "status"> & { id?: string }, now?: string): { product: Product; id: string } {
  const next = clone(product);
  const id = artifact.id ?? `${product.id}/artifact/${next.artifacts.length + 1}`;
  if (next.artifacts.some((a) => a.id === id)) {
    throw new PatternCadError("duplicate-id", `artifact '${id}' already exists`, id);
  }
  if (!artifact.filename) throw new PatternCadError("invalid-document", "artifact needs a filename (identity stays on id)", id);
  if (artifact.revision !== product.revision) {
    throw new PatternCadError("invalid-document", `artifact revision ${artifact.revision} does not match product revision ${product.revision}`, id);
  }
  if (!artifact.checksum) throw new PatternCadError("invalid-document", "artifact needs a checksum", id);
  next.artifacts.push({ ...clone(artifact), id, status: "current" } as ProductArtifact);
  return { product: touch(next, now), id };
}

export function markArtifact(product: Product, artifactId: string, status: ArtifactStatus, now?: string): Product {
  const next = clone(product);
  const artifact = next.artifacts.find((a) => a.id === artifactId);
  if (!artifact) throw new PatternCadError("missing-reference", `artifact '${artifactId}' does not exist`, artifactId);
  artifact.status = status;
  return touch(next, now);
}

export function attachPreview(product: Product, artifactId: string, thumbnail = false, now?: string): Product {
  const next = clone(product);
  if (!next.artifacts.some((a) => a.id === artifactId)) {
    throw new PatternCadError("missing-reference", `preview artifact '${artifactId}' does not exist`, artifactId);
  }
  if (!next.previewIds.includes(artifactId)) next.previewIds.push(artifactId);
  if (thumbnail) next.thumbnailArtifactId = artifactId;
  return touch(next, now);
}

/** Explicit re-release against a new garment revision (never automatic). */
export function bumpProductRevision(
  product: Product,
  garment: ProductGarmentRef,
  now?: string,
): Product {
  if (garment.projectId !== product.garment.projectId || garment.garmentId !== product.garment.garmentId) {
    throw new PatternCadError("invalid-document", "re-release must reference the same garment", product.id);
  }
  const next = clone(product);
  next.garment = clone(garment);
  next.revision += 1;
  for (const artifact of next.artifacts) artifact.status = "stale";
  next.status = "draft";
  return touch(next, now);
}

/** True when the product still pins the given live garment state. */
export function isPinnedCurrent(product: Product, garmentRevision: number, garmentFingerprint: string): boolean {
  return product.garment.garmentRevision === garmentRevision &&
    product.garment.garmentFingerprint === garmentFingerprint;
}

// ---------------------------------------------------------------------------
// Status transitions + validation
// ---------------------------------------------------------------------------

export type ProductDiagnosticCode =
  | "invalid-document"
  | "unsupported-schema"
  | "duplicate-id"
  | "missing-reference"
  | "missing-preview"
  | "missing-license"
  | "incomplete-metadata"
  | "invalid-transition"
  | "stale-artifact";

export interface ProductDiagnostic {
  code: ProductDiagnosticCode;
  message: string;
  entityId?: string;
}

const TRANSITIONS: Record<ProductStatus, ProductStatus[]> = {
  draft: ["ready", "archived"],
  ready: ["published", "draft", "archived"],
  published: ["unpublished", "archived"],
  unpublished: ["published", "archived"],
  archived: [],
};

export function validateProduct(product: Product): ProductDiagnostic[] {
  const diagnostics: ProductDiagnostic[] = [];
  const fail = (code: ProductDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  if (!product || typeof product !== "object" || !Array.isArray(product.variants) || !Array.isArray(product.artifacts)) {
    return [{ code: "invalid-document", message: "product shape is invalid" }];
  }
  if (product.schemaVersion !== PRODUCT_SCHEMA_VERSION) {
    fail("unsupported-schema", `product schema v${String(product.schemaVersion)} unsupported (current v${PRODUCT_SCHEMA_VERSION})`);
    return diagnostics;
  }
  if (!product.id || !product.name || !product.sku || !product.category || !product.author) {
    fail("invalid-document", "product needs id, name, SKU, category, and author", product.id);
  }
  try {
    validateLicense(product.license, product.id);
  } catch (error) {
    fail("missing-license", error instanceof Error ? error.message : String(error), product.id);
  }
  if (!product.description) fail("incomplete-metadata", "product needs a description", product.id);
  if (product.previewIds.length === 0) fail("missing-preview", "product needs at least one preview", product.id);
  for (const previewId of product.previewIds) {
    if (!product.artifacts.some((a) => a.id === previewId)) {
      fail("missing-reference", `preview '${previewId}' is not a product artifact`, previewId);
    }
  }
  if (product.status === "ready" || product.status === "published") {
    for (const artifact of product.artifacts) {
      if (artifact.status !== "current") {
        fail("stale-artifact", `artifact '${artifact.id}' is ${artifact.status}`, artifact.id);
      }
      if (artifact.revision !== product.revision) {
        fail("stale-artifact", `artifact '${artifact.id}' targets revision ${artifact.revision}, product is at ${product.revision}`, artifact.id);
      }
    }
    for (const variant of product.variants) {
      for (const artifactId of variant.artifactIds) {
        if (!product.artifacts.some((a) => a.id === artifactId)) {
          fail("missing-reference", `variant '${variant.id}' references unknown artifact '${artifactId}'`, variant.id);
        }
      }
    }
  }
  return diagnostics;
}

/** Readiness for Ready/Published states (draft needs nothing beyond shape). */
export function productReady(product: Product): boolean {
  return validateProduct(product).length === 0;
}

export function setProductStatus(product: Product, status: ProductStatus, now?: string): Product {
  if (!TRANSITIONS[product.status].includes(status)) {
    throw new PatternCadError(
      "invalid-document",
      `product status cannot move from '${product.status}' to '${status}'`,
      product.id,
    );
  }
  if ((status === "ready" || status === "published") && !productReady(product)) {
    const first = validateProduct(product)[0];
    throw new PatternCadError("invalid-document", `product is not releasable: ${first.code}: ${first.message}`, product.id);
  }
  const next = clone(product);
  next.status = status;
  return touch(next, now);
}

export function duplicateProduct(product: Product, newId: string, newSku: string, now?: string): Product {
  if (!newId || !newSku) throw new PatternCadError("invalid-document", "duplicate needs a new id and SKU");
  if (newId === product.id) throw new PatternCadError("duplicate-id", "duplicate product id must differ");
  const timestamp = now ?? new Date().toISOString();
  const next = clone(product);
  next.id = newId;
  next.sku = newSku;
  next.slug = `${slugify(product.name)}-copy`;
  next.status = "draft";
  next.createdAt = timestamp;
  next.updatedAt = timestamp;
  return next;
}

// ---------------------------------------------------------------------------
// Persistence (versioned independently from garment schemas)
// ---------------------------------------------------------------------------

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export function serializeProduct(product: Product): string {
  return canonicalJson(product);
}

export function deserializeProduct(serialized: string): Product {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized product is not valid JSON");
  }
  const product = parsed as Product;
  if (!product || typeof product !== "object" || product.schemaVersion !== PRODUCT_SCHEMA_VERSION ||
    !product.id || !Array.isArray(product.variants) || !Array.isArray(product.artifacts)) {
    throw new PatternCadError("invalid-document", "product shape or schema version is invalid");
  }
  return clone(product);
}
