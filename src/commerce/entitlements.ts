// G20A entitlements + product releases.
//
// Entitlement: what a customer may download (customer × product × revision ×
// variant). Issued only against paid orders. Update eligibility is explicit
// policy data, never billing logic.
// Release: an immutable snapshot of a product revision's downloadable files.

import { PatternCadError } from "../pattern/cad.js";
import type { Order } from "./orders.js";
import type { Product, ProductArtifact } from "../product/product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type EntitlementStatus = "active" | "expired" | "revoked";
export type UpdatePolicy = "original-only" | "free-updates" | "paid-updates";

export interface Entitlement {
  id: string;
  customerId: string;
  orderId: string;
  productId: string;
  variantId: string;
  productRevision: number;
  status: EntitlementStatus;
  updatePolicy: UpdatePolicy;
  issuedAt: string;
  expiresAt?: string;
}

export function issueEntitlement(partial: {
  id: string;
  order: Order;
  productId: string;
  variantId: string;
  updatePolicy?: UpdatePolicy;
  expiresAt?: string;
  now?: string;
}): Entitlement {
  const { id, order, productId, variantId } = partial;
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", "entitlement id must be non-empty");
  if (order.status !== "paid") {
    throw new PatternCadError("invalid-document", `entitlement requires a paid order (got '${order.status}')`, order.id);
  }
  const item = order.items.find((i) => i.productId === productId && i.variantId === variantId);
  if (!item) {
    throw new PatternCadError("missing-reference", "order contains no matching product/variant line", order.id);
  }
  return {
    id,
    customerId: order.customerId,
    orderId: order.id,
    productId,
    variantId,
    productRevision: item.productRevision,
    status: "active",
    updatePolicy: partial.updatePolicy ?? "original-only",
    issuedAt: partial.now ?? new Date().toISOString(),
    ...(partial.expiresAt ? { expiresAt: partial.expiresAt } : {}),
  };
}

export function revokeEntitlement(entitlement: Entitlement, now?: string): Entitlement {
  void now;
  if (entitlement.status === "revoked") return clone(entitlement);
  return { ...clone(entitlement), status: "revoked" };
}

/** Expiry is time-relative; callers pass now explicitly (deterministic tests). */
export function isEntitlementUsable(entitlement: Entitlement, nowIso: string): boolean {
  if (entitlement.status !== "active") return false;
  if (entitlement.expiresAt && Date.parse(nowIso) >= Date.parse(entitlement.expiresAt)) return false;
  return true;
}

/** May this entitlement take release `revision` without a new purchase? */
export function updateEligible(entitlement: Entitlement, revision: number): boolean {
  if (revision === entitlement.productRevision) return true;
  if (revision < entitlement.productRevision) return false;
  return entitlement.updatePolicy === "free-updates";
}

// ---------------------------------------------------------------------------
// Releases (immutable)
// ---------------------------------------------------------------------------

export type ReleaseStatus = "active" | "revoked" | "archived";

export interface ReleaseArtifact {
  artifactId: string;
  filename: string;
  checksum: string;
  sizeBytes: number;
  revoked?: boolean;
}

export interface ProductRelease {
  id: string;
  productId: string;
  revision: number;
  artifacts: ReleaseArtifact[];
  status: ReleaseStatus;
  updatePolicy: UpdatePolicy;
  releasedAt: string;
}

/** Freeze a release from a ready/published product (current artifacts only). */
export function createRelease(partial: {
  id: string;
  product: Product;
  artifacts: ProductArtifact[];
  updatePolicy?: UpdatePolicy;
  now?: string;
}): ProductRelease {
  const { id, product } = partial;
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", "release id must be non-empty");
  if (product.status !== "published" && product.status !== "ready") {
    throw new PatternCadError("invalid-document", `release requires a ready/published product (got '${product.status}')`, product.id);
  }
  const artifacts: ReleaseArtifact[] = partial.artifacts.map((a) => {
    if (a.revision !== product.revision) {
      throw new PatternCadError("invalid-document", `artifact '${a.id}' targets revision ${a.revision}, product is at ${product.revision}`, a.id);
    }
    if (a.status !== "current") {
      throw new PatternCadError("invalid-document", `artifact '${a.id}' is ${a.status}, not current`, a.id);
    }
    return { artifactId: a.id, filename: a.filename, checksum: a.checksum, sizeBytes: 0 };
  });
  if (artifacts.length === 0) throw new PatternCadError("invalid-document", "release needs at least one artifact", id);
  return {
    id,
    productId: product.id,
    revision: product.revision,
    artifacts,
    status: "active",
    updatePolicy: partial.updatePolicy ?? "original-only",
    releasedAt: partial.now ?? new Date().toISOString(),
  };
}

export function revokeRelease(release: ProductRelease): ProductRelease {
  return { ...clone(release), status: "revoked" };
}

export function revokeReleaseArtifact(release: ProductRelease, artifactId: string): ProductRelease {
  const next = clone(release);
  const artifact = next.artifacts.find((a) => a.artifactId === artifactId);
  if (!artifact) throw new PatternCadError("missing-reference", `artifact '${artifactId}' not in release`, artifactId);
  artifact.revoked = true;
  return next;
}

/** Resolve a deliverable artifact: member of the release, unrevoked, release active. */
export function resolveReleaseArtifact(release: ProductRelease, artifactId: string): ReleaseArtifact {
  if (release.status !== "active") {
    throw new PatternCadError("invalid-document", `release '${release.id}' is ${release.status}`, release.id);
  }
  const artifact = release.artifacts.find((a) => a.artifactId === artifactId);
  if (!artifact) throw new PatternCadError("missing-reference", `artifact '${artifactId}' not in release`, artifactId);
  if (artifact.revoked) throw new PatternCadError("invalid-document", `artifact '${artifactId}' is revoked`, artifactId);
  return clone(artifact);
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

export function serializeEntitlement(entitlement: Entitlement): string {
  return canonicalJson(entitlement);
}

export function deserializeEntitlement(serialized: string): Entitlement {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized entitlement is not valid JSON");
  }
  const entitlement = parsed as Entitlement;
  if (!entitlement || !entitlement.id || !entitlement.customerId || !entitlement.orderId) {
    throw new PatternCadError("invalid-document", "entitlement shape is invalid");
  }
  return clone(entitlement);
}

export function serializeRelease(release: ProductRelease): string {
  return canonicalJson(release);
}

export function deserializeRelease(serialized: string): ProductRelease {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized release is not valid JSON");
  }
  const release = parsed as ProductRelease;
  if (!release || !release.id || !release.productId || !Array.isArray(release.artifacts)) {
    throw new PatternCadError("invalid-document", "release shape is invalid");
  }
  return clone(release);
}
