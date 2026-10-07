// G21A — customer digital library (account area).
//
// The library is a DERIVED view over entitlements, orders, products, and
// releases. It never grants anything itself: every download it lists still
// authorizes through the G20 delivery boundary (requestLibraryDownload
// delegates to authorizeDownload), so entitlement verification can never be
// bypassed. Entries are scoped to a single customerId — other customers'
// records are unreachable by construction — and expose only customer-facing
// facts: no garment references, no project data, no filesystem paths.
//
// Update-state precedence (mutually exclusive):
//   revoked > download-unavailable > deprecated > update-available > current
// "update-available" is factual (a newer release exists); policy eligibility
// is reported separately as eligibleForUpdate (never conflated).

import { PatternCadError } from "../pattern/cad.js";
import type { Entitlement, ProductRelease, UpdatePolicy } from "./entitlements.js";
import { isEntitlementUsable, updateEligible } from "./entitlements.js";
import type { Order } from "./orders.js";
import type { Product } from "../product/product.js";
import type { DownloadAuthorization, DownloadRecord, RateLimiter, TokenLedger } from "./delivery.js";
import { authorizeDownload } from "./delivery.js";

export const LIBRARY_SCHEMA_VERSION = 1;
export const FAVORITES_SCHEMA_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
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

function requireId(id: string | undefined, label: string): void {
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", `${label} must be non-empty`);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export const LIBRARY_UPDATE_STATES = [
  "current",
  "update-available",
  "deprecated",
  "download-unavailable",
  "revoked",
] as const;

export type LibraryUpdateState = (typeof LIBRARY_UPDATE_STATES)[number];

export type DownloadUnavailableReason =
  | "entitlement-revoked"
  | "entitlement-expired"
  | "release-unavailable"
  | "artifact-revoked";

export interface LibraryDownload {
  artifactId: string;
  filename: string;
  format: string;
  available: boolean;
  unavailableReason?: DownloadUnavailableReason;
  sizeIds?: string[];
}

export interface LibraryLicense {
  type: string;
  commercialUse: boolean;
  modification: boolean;
  redistribution: boolean;
  attribution: boolean;
  version: string;
}

export interface LibraryEntry {
  productId: string;
  name: string;
  sku?: string;
  category?: string;
  thumbnailArtifactId?: string;
  /** Order creation time; falls back to entitlement issue time. */
  purchasedAt: string;
  orderId: string;
  entitlementId: string;
  variantId: string;
  /** Product release pinned at purchase. */
  ownedRelease: number;
  /** Newest active release known for this product (registry or explicit). */
  latestRelease?: number;
  updateState: LibraryUpdateState;
  /** A newer release exists (regardless of policy). */
  updateAvailable: boolean;
  /** The newer release is deliverable under this entitlement's policy. */
  eligibleForUpdate: boolean;
  usable: boolean;
  productMissing?: boolean;
  downloads: LibraryDownload[];
  license?: LibraryLicense;
  lastDownloadedAt?: string;
}

export interface Library {
  schemaVersion: typeof LIBRARY_SCHEMA_VERSION;
  customerId: string;
  entries: LibraryEntry[];
  builtAt: string;
}

// ---------------------------------------------------------------------------
// Update state
// ---------------------------------------------------------------------------

export interface UpdateStateFacts {
  entitlement: Entitlement;
  productMissing: boolean;
  productArchived: boolean;
  releaseMissing: boolean;
  releaseRevoked: boolean;
  availableDownloads: number;
  latestRelease?: number;
  nowIso: string;
}

/**
 * Display state for an owned product. Precedence:
 * revoked > download-unavailable > deprecated > update-available > current.
 * Eligibility (policy) is deliberately NOT part of this state.
 */
export function computeUpdateState(facts: UpdateStateFacts): LibraryUpdateState {
  if (facts.entitlement.status === "revoked") return "revoked";
  if (
    !isEntitlementUsable(facts.entitlement, facts.nowIso) ||
    facts.releaseMissing ||
    facts.releaseRevoked ||
    facts.availableDownloads === 0
  ) {
    return "download-unavailable";
  }
  if (facts.productMissing || facts.productArchived) return "deprecated";
  if (facts.latestRelease !== undefined && facts.latestRelease > facts.entitlement.productRevision) {
    return "update-available";
  }
  return "current";
}

// ---------------------------------------------------------------------------
// Resolution helpers (shared by library list + product detail)
// ---------------------------------------------------------------------------

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot > 0 ? filename.slice(dot + 1).toLowerCase() : "";
}

function buildDownloads(
  release: ProductRelease | undefined,
  entitlement: Entitlement,
  usable: boolean,
  product: Product | undefined,
): LibraryDownload[] {
  if (!release) return [];
  const blocked: DownloadUnavailableReason | undefined =
    entitlement.status === "revoked"
      ? "entitlement-revoked"
      : !usable
        ? "entitlement-expired"
        : release.status !== "active"
          ? "release-unavailable"
          : undefined;
  return release.artifacts.map((artifact) => {
    const reason = blocked ?? (artifact.revoked ? "artifact-revoked" : undefined);
    const source = product?.artifacts.find((a) => a.id === artifact.artifactId);
    const format = source?.format || extensionOf(artifact.filename);
    return {
      artifactId: artifact.artifactId,
      filename: artifact.filename,
      format,
      available: reason === undefined,
      ...(reason ? { unavailableReason: reason } : {}),
      ...(source && source.sizeIds.length > 0 ? { sizeIds: [...source.sizeIds] } : {}),
    };
  });
}

/** Newest active release: explicit registry wins, else max over releases. */
function deriveLatestRelease(
  productId: string,
  releases: ProductRelease[],
  explicit?: Record<string, number>,
): number | undefined {
  const mapped = explicit?.[productId];
  if (mapped !== undefined) {
    if (!Number.isInteger(mapped) || mapped < 1) {
      throw new PatternCadError("invalid-transform", `latest release for '${productId}' must be a positive integer`, productId);
    }
    return mapped;
  }
  let latest: number | undefined;
  for (const release of releases) {
    if (release.productId !== productId || release.status !== "active") continue;
    if (latest === undefined || release.revision > latest) latest = release.revision;
  }
  return latest;
}

function latestCompletedDownload(
  records: DownloadRecord[],
  customerId: string,
  entitlementId: string,
): string | undefined {
  let latest: string | undefined;
  for (const record of records) {
    if (record.customerId !== customerId) continue;
    if (record.entitlementId !== entitlementId) continue;
    if (record.status !== "completed") continue;
    if (latest === undefined || record.timestamp > latest) latest = record.timestamp;
  }
  return latest;
}

function toLibraryLicense(product: Product): LibraryLicense {
  return {
    type: product.license.type,
    commercialUse: product.license.allowsCommercialUse,
    modification: product.license.allowsModification,
    redistribution: product.license.allowsRedistribution,
    attribution: product.license.attributionRequired,
    version: product.license.version,
  };
}

interface OwnershipContext {
  product?: Product;
  release?: ProductRelease;
  order?: Order;
  usable: boolean;
  downloads: LibraryDownload[];
  lastDownloadedAt?: string;
  latestRelease?: number;
  updateAvailable: boolean;
  eligibleForUpdate: boolean;
  updateState: LibraryUpdateState;
}

function resolveOwnership(args: {
  customerId: string;
  entitlement: Entitlement;
  productById: Map<string, Product>;
  releases: ProductRelease[];
  orderById: Map<string, Order>;
  downloadRecords: DownloadRecord[];
  latestReleases?: Record<string, number>;
  nowIso: string;
}): OwnershipContext {
  const { entitlement } = args;
  const product = args.productById.get(entitlement.productId);
  const order = args.orderById.get(entitlement.orderId);
  const release = args.releases.find(
    (r) => r.productId === entitlement.productId && r.revision === entitlement.productRevision,
  );
  const usable = isEntitlementUsable(entitlement, args.nowIso);
  const downloads = buildDownloads(release, entitlement, usable, product);
  const availableDownloads = downloads.filter((d) => d.available).length;
  const latestRelease = deriveLatestRelease(entitlement.productId, args.releases, args.latestReleases);
  const lastDownloadedAt = latestCompletedDownload(args.downloadRecords, args.customerId, entitlement.id);

  let updateAvailable = false;
  let eligibleForUpdate = false;
  if (latestRelease !== undefined && latestRelease > entitlement.productRevision) {
    updateAvailable = true;
    eligibleForUpdate = updateEligible(entitlement, latestRelease);
  }
  const updateState = computeUpdateState({
    entitlement,
    productMissing: !product,
    productArchived: product?.status === "archived",
    releaseMissing: !release,
    releaseRevoked: release !== undefined && release.status !== "active",
    availableDownloads,
    ...(latestRelease !== undefined ? { latestRelease } : {}),
    nowIso: args.nowIso,
  });
  return {
    ...(product ? { product } : {}),
    ...(release ? { release } : {}),
    ...(order ? { order } : {}),
    usable,
    downloads,
    ...(lastDownloadedAt ? { lastDownloadedAt } : {}),
    ...(latestRelease !== undefined ? { latestRelease } : {}),
    updateAvailable,
    eligibleForUpdate,
    updateState,
  };
}

function buildContexts(input: LibraryInput): OwnershipContext[] {
  requireId(input.customerId, "customer id");
  if (!input.nowIso || Number.isNaN(Date.parse(input.nowIso))) {
    throw new PatternCadError("invalid-transform", "library build time must be a valid timestamp");
  }
  if (!Array.isArray(input.entitlements)) {
    throw new PatternCadError("invalid-document", "library needs an entitlements array");
  }
  const orders = input.orders ?? [];
  const products = input.products ?? [];
  const releases = input.releases ?? [];
  const downloadRecords = input.downloads ?? [];
  const productById = new Map(products.map((p) => [p.id, p]));
  // Orders only match their own customer: a mismatched snapshot can never
  // contribute purchase dates to someone else's library.
  const orderById = new Map(
    orders.filter((o) => o.customerId === input.customerId).map((o) => [o.id, o]),
  );
  return input.entitlements
    .filter((e) => {
      requireId(e?.customerId, "entitlement customer id");
      requireId(e.productId, "entitlement product id");
      return e.customerId === input.customerId;
    })
    .map((entitlement) =>
      resolveOwnership({
        customerId: input.customerId,
        entitlement,
        productById,
        releases,
        orderById,
        downloadRecords,
        ...(input.latestReleases ? { latestReleases: input.latestReleases } : {}),
        nowIso: input.nowIso,
      }),
    );
}

// ---------------------------------------------------------------------------
// Library construction
// ---------------------------------------------------------------------------

export interface LibraryInput {
  customerId: string;
  entitlements: Entitlement[];
  orders?: Order[];
  products?: Product[];
  /** Release registry: all known releases (owned and newer), not just this customer's. */
  releases?: ProductRelease[];
  downloads?: DownloadRecord[];
  /** Explicit newest-release override per productId (registry wins when omitted). */
  latestReleases?: Record<string, number>;
  nowIso: string;
}

function byProductId(a: LibraryEntry, b: LibraryEntry): number {
  return a.productId < b.productId ? -1 : a.productId > b.productId ? 1 : 0;
}

/** Build the customer's library view. Deterministic; entries sorted by productId. */
export function buildLibrary(input: LibraryInput): Library {
  const contexts = buildContexts(input);
  const entitlements = input.entitlements.filter((e) => e.customerId === input.customerId);
  const entries = entitlements.map((entitlement, index): LibraryEntry => {
    const ctx = contexts[index];
    const product = ctx.product;
    const thumbnailArtifactId = product?.thumbnailArtifactId ?? product?.previewIds[0];
    return {
      productId: entitlement.productId,
      name: product?.name ?? entitlement.productId,
      ...(product?.sku ? { sku: product.sku } : {}),
      ...(product?.category ? { category: product.category } : {}),
      ...(thumbnailArtifactId ? { thumbnailArtifactId } : {}),
      purchasedAt: ctx.order?.createdAt ?? entitlement.issuedAt,
      orderId: entitlement.orderId,
      entitlementId: entitlement.id,
      variantId: entitlement.variantId,
      ownedRelease: entitlement.productRevision,
      ...(ctx.latestRelease !== undefined ? { latestRelease: ctx.latestRelease } : {}),
      updateState: ctx.updateState,
      updateAvailable: ctx.updateAvailable,
      eligibleForUpdate: ctx.eligibleForUpdate,
      usable: ctx.usable,
      ...(product ? {} : { productMissing: true }),
      downloads: ctx.downloads,
      ...(product ? { license: toLibraryLicense(product) } : {}),
      ...(ctx.lastDownloadedAt ? { lastDownloadedAt: ctx.lastDownloadedAt } : {}),
    };
  });
  entries.sort(byProductId);
  return {
    schemaVersion: LIBRARY_SCHEMA_VERSION,
    customerId: input.customerId,
    entries,
    builtAt: input.nowIso,
  };
}

// ---------------------------------------------------------------------------
// Search / filter / sort / organization
// ---------------------------------------------------------------------------

/** Case-insensitive substring search over name, SKU, category, and productId. */
export function searchLibrary(entries: LibraryEntry[], query: string): LibraryEntry[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [...entries];
  return entries.filter(
    (e) =>
      e.name.toLowerCase().includes(needle) ||
      (e.sku ?? "").toLowerCase().includes(needle) ||
      (e.category ?? "").toLowerCase().includes(needle) ||
      e.productId.toLowerCase().includes(needle),
  );
}

export interface LibraryFilter {
  query?: string;
  updateStates?: LibraryUpdateState[];
  categories?: string[];
  favoritesOnly?: boolean;
  usableOnly?: boolean;
}

export function filterLibrary(
  entries: LibraryEntry[],
  filter: LibraryFilter,
  favorites?: FavoriteList,
): LibraryEntry[] {
  let out = entries;
  if (filter.query !== undefined) out = searchLibrary(out, filter.query);
  if (filter.updateStates) {
    for (const state of filter.updateStates) {
      if (!(LIBRARY_UPDATE_STATES as readonly string[]).includes(state)) {
        throw new PatternCadError("invalid-document", `unknown library update state '${String(state)}'`);
      }
    }
    const wanted = new Set<string>(filter.updateStates);
    out = out.filter((e) => wanted.has(e.updateState));
  }
  if (filter.categories) {
    const wanted = new Set(filter.categories);
    out = out.filter((e) => e.category !== undefined && wanted.has(e.category));
  }
  if (filter.favoritesOnly) {
    if (!favorites) {
      throw new PatternCadError("invalid-document", "favoritesOnly filtering requires a favorites list");
    }
    const wanted = new Set(favorites.productIds);
    out = out.filter((e) => wanted.has(e.productId));
  }
  if (filter.usableOnly) out = out.filter((e) => e.usable);
  return out;
}

export const LIBRARY_SORTS = [
  "purchased-desc",
  "purchased-asc",
  "name-asc",
  "name-desc",
  "downloaded-desc",
] as const;

export type LibrarySort = (typeof LIBRARY_SORTS)[number];

function cmp(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Deterministic sort (ties always break on productId). Returns a new array. */
export function sortLibrary(entries: LibraryEntry[], sort: LibrarySort): LibraryEntry[] {
  const out = [...entries];
  switch (sort) {
    case "purchased-desc":
      out.sort((a, b) => cmp(b.purchasedAt, a.purchasedAt) || byProductId(a, b));
      break;
    case "purchased-asc":
      out.sort((a, b) => cmp(a.purchasedAt, b.purchasedAt) || byProductId(a, b));
      break;
    case "name-asc":
      out.sort((a, b) => cmp(a.name.toLowerCase(), b.name.toLowerCase()) || byProductId(a, b));
      break;
    case "name-desc":
      out.sort((a, b) => cmp(b.name.toLowerCase(), a.name.toLowerCase()) || byProductId(a, b));
      break;
    case "downloaded-desc": {
      // Newest download first; never-downloaded entries sink to the end.
      out.sort((a, b) => {
        const at = a.lastDownloadedAt ?? "";
        const bt = b.lastDownloadedAt ?? "";
        if (at !== bt) return at < bt ? 1 : -1;
        return byProductId(a, b);
      });
      break;
    }
    default:
      throw new PatternCadError("invalid-document", `unknown library sort '${String(sort)}'`);
  }
  return out;
}

function applyLimit(entries: LibraryEntry[], limit?: number): LibraryEntry[] {
  if (limit === undefined) return entries;
  if (!Number.isInteger(limit) || limit < 0) {
    throw new PatternCadError("invalid-transform", "library limit must be a non-negative integer");
  }
  return entries.slice(0, limit);
}

export function recentlyPurchased(entries: LibraryEntry[], limit?: number): LibraryEntry[] {
  return applyLimit(sortLibrary(entries, "purchased-desc"), limit);
}

export function recentlyDownloaded(entries: LibraryEntry[], limit?: number): LibraryEntry[] {
  const downloaded = entries.filter((e) => e.lastDownloadedAt !== undefined);
  return applyLimit(sortLibrary(downloaded, "downloaded-desc"), limit);
}

export interface LibraryOptions {
  filter?: LibraryFilter;
  favorites?: FavoriteList;
  /** Default sort: purchased-desc. */
  sort?: LibrarySort;
  limit?: number;
}

/** One-call organization: filter → sort → limit. */
export function organizeLibrary(entries: LibraryEntry[], options: LibraryOptions = {}): LibraryEntry[] {
  let out = options.filter ? filterLibrary(entries, options.filter, options.favorites) : [...entries];
  out = sortLibrary(out, options.sort ?? "purchased-desc");
  return applyLimit(out, options.limit);
}

// ---------------------------------------------------------------------------
// Favorites (customer-owned state, persisted)
// ---------------------------------------------------------------------------

export interface FavoriteList {
  schemaVersion: typeof FAVORITES_SCHEMA_VERSION;
  customerId: string;
  productIds: string[];
  updatedAt: string;
}

export function createFavorites(customerId: string, now?: string): FavoriteList {
  requireId(customerId, "customer id");
  return {
    schemaVersion: FAVORITES_SCHEMA_VERSION,
    customerId,
    productIds: [],
    updatedAt: now ?? new Date().toISOString(),
  };
}

/** Idempotent: adding an already-favorited product is a no-op. */
export function addFavorite(list: FavoriteList, productId: string, now?: string): FavoriteList {
  requireId(productId, "product id");
  if (list.productIds.includes(productId)) return clone(list);
  return {
    ...clone(list),
    productIds: [...list.productIds, productId],
    updatedAt: now ?? new Date().toISOString(),
  };
}

/** Idempotent: removing an absent product is a no-op. */
export function removeFavorite(list: FavoriteList, productId: string): FavoriteList {
  requireId(productId, "product id");
  if (!list.productIds.includes(productId)) return clone(list);
  const next = clone(list);
  next.productIds = next.productIds.filter((id) => id !== productId);
  return next;
}

export function isFavorite(list: FavoriteList, productId: string): boolean {
  return list.productIds.includes(productId);
}

export function toggleFavorite(list: FavoriteList, productId: string, now?: string): FavoriteList {
  return isFavorite(list, productId) ? removeFavorite(list, productId) : addFavorite(list, productId, now);
}

// ---------------------------------------------------------------------------
// Product detail (owned product)
// ---------------------------------------------------------------------------

export interface LibraryProductDetail {
  productId: string;
  name: string;
  sku?: string;
  productMissing?: boolean;
  purchasedAt: string;
  orderId: string;
  entitlementId: string;
  variantId: string;
  /** Release pinned at purchase. */
  purchasedRelease: number;
  latestRelease?: number;
  variant?: { id: string; name: string; kind: string };
  sizes: string[];
  formats: string[];
  downloads: LibraryDownload[];
  license?: LibraryLicense;
  requirements?: string;
  update: {
    state: LibraryUpdateState;
    available: boolean;
    eligible: boolean;
    policy: UpdatePolicy;
    ownedRelease: number;
    latestRelease?: number;
  };
  support: {
    contact?: string;
    productId: string;
    orderId: string;
    entitlementId: string;
    variantId: string;
    releaseId?: string;
    releaseRevision: number;
  };
}

/** Full detail for one owned product. Refuses entitlements of other customers. */
export function libraryProductDetail(input: LibraryInput & { entitlementId: string }): LibraryProductDetail {
  requireId(input.entitlementId, "entitlement id");
  const entitlement = input.entitlements.find((e) => e.id === input.entitlementId);
  if (!entitlement) {
    throw new PatternCadError("missing-reference", `entitlement '${input.entitlementId}' was not found`, input.entitlementId);
  }
  if (entitlement.customerId !== input.customerId) {
    throw new PatternCadError("invalid-document", "entitlement belongs to another customer", entitlement.id);
  }
  const context = buildContexts({ ...input, entitlements: [entitlement] })[0];
  const product = context.product;
  const variant = product?.variants.find((v) => v.id === entitlement.variantId);
  const sizes = variant ? [...variant.sizeIds] : [...(product?.sizes ?? [])];
  const formats = variant ? [...variant.formats] : [...(product?.formats ?? [])];
  return {
    productId: entitlement.productId,
    name: product?.name ?? entitlement.productId,
    ...(product?.sku ? { sku: product.sku } : {}),
    ...(product ? {} : { productMissing: true }),
    purchasedAt: context.order?.createdAt ?? entitlement.issuedAt,
    orderId: entitlement.orderId,
    entitlementId: entitlement.id,
    variantId: entitlement.variantId,
    purchasedRelease: entitlement.productRevision,
    ...(context.latestRelease !== undefined ? { latestRelease: context.latestRelease } : {}),
    ...(variant ? { variant: { id: variant.id, name: variant.name, kind: variant.kind } } : {}),
    sizes,
    formats,
    downloads: context.downloads,
    ...(product ? { license: toLibraryLicense(product) } : {}),
    ...(product?.requirements ? { requirements: product.requirements } : {}),
    update: {
      state: context.updateState,
      available: context.updateAvailable,
      eligible: context.eligibleForUpdate,
      policy: entitlement.updatePolicy,
      ownedRelease: entitlement.productRevision,
      ...(context.latestRelease !== undefined ? { latestRelease: context.latestRelease } : {}),
    },
    support: {
      ...(product?.license.supportContact ? { contact: product.license.supportContact } : {}),
      productId: entitlement.productId,
      orderId: entitlement.orderId,
      entitlementId: entitlement.id,
      variantId: entitlement.variantId,
      ...(context.release ? { releaseId: context.release.id } : {}),
      releaseRevision: entitlement.productRevision,
    },
  };
}

// ---------------------------------------------------------------------------
// Download (delegates to the G20 delivery boundary — never bypassed)
// ---------------------------------------------------------------------------

/**
 * The library's single download entry point. It enforces library scoping
 * (customer, product, covered release) and then delegates all entitlement,
 * artifact, single-use-token, expiry, and rate-limit verification to the G20
 * authorizeDownload boundary — this module never grants access itself.
 */
export function requestLibraryDownload(input: {
  customerId: string;
  entitlement: Entitlement;
  release: ProductRelease;
  artifactId: string;
  token: string;
  ledger: TokenLedger;
  limiter: RateLimiter;
  nowIso: string;
  nowMs: number;
  ttlMs?: number;
}): DownloadAuthorization {
  requireId(input.customerId, "customer id");
  if (input.entitlement.customerId !== input.customerId) {
    throw new PatternCadError("invalid-document", "entitlement belongs to another customer", input.entitlement.id);
  }
  if (input.release.productId !== input.entitlement.productId) {
    throw new PatternCadError("invalid-document", `release '${input.release.id}' belongs to another product`, input.release.id);
  }
  if (!updateEligible(input.entitlement, input.release.revision)) {
    throw new PatternCadError(
      "invalid-document",
      `entitlement '${input.entitlement.id}' does not cover release revision ${input.release.revision}`,
      input.release.id,
    );
  }
  return authorizeDownload(input);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

export function serializeLibrary(library: Library): string {
  return canonicalJson(library);
}

export function deserializeLibrary(serialized: string): Library {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized library is not valid JSON");
  }
  const library = parsed as Library;
  if (
    !library ||
    library.schemaVersion !== LIBRARY_SCHEMA_VERSION ||
    !library.customerId ||
    !library.builtAt ||
    !Array.isArray(library.entries)
  ) {
    throw new PatternCadError("invalid-document", "library shape or schema version is invalid");
  }
  return clone(library);
}

export function serializeFavorites(list: FavoriteList): string {
  return canonicalJson(list);
}

export function deserializeFavorites(serialized: string): FavoriteList {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized favorites are not valid JSON");
  }
  const list = parsed as FavoriteList;
  if (!list || list.schemaVersion !== FAVORITES_SCHEMA_VERSION || !list.customerId || !Array.isArray(list.productIds)) {
    throw new PatternCadError("invalid-document", "favorites shape or schema version is invalid");
  }
  const seen = new Set<string>();
  const productIds: string[] = [];
  for (const id of list.productIds) {
    if (typeof id !== "string" || !id) {
      throw new PatternCadError("invalid-document", "favorites product ids must be non-empty strings");
    }
    if (seen.has(id)) continue;
    seen.add(id);
    productIds.push(id);
  }
  return { ...clone(list), productIds };
}
