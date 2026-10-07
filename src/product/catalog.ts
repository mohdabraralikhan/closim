// G19C (integration surface) — commercial product catalog.
//
// A headless library over stored products: configurable taxonomy, search,
// filters, sorting, and lightweight product cards. Cards carry only stored
// metadata — browsing never loads garment or simulation state.

import { PatternCadError } from "../pattern/cad.js";
import {
  validateProduct,
  type Product,
  type ProductStatus,
} from "./product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export interface Catalog {
  id: string;
  name: string;
  /** Configurable taxonomy (never hard-coded in code paths). */
  categories: string[];
  products: Product[];
}

export function createCatalog(id: string, name: string, categories: string[] = []): Catalog {
  if (!id || !name) throw new PatternCadError("invalid-document", "catalog needs id and name");
  return { id, name, categories: [...categories], products: [] };
}

export function addProduct(catalog: Catalog, product: Product): Catalog {
  if (catalog.products.some((p) => p.id === product.id)) {
    throw new PatternCadError("duplicate-id", `product '${product.id}' already in catalog`, product.id);
  }
  if (catalog.products.some((p) => p.sku === product.sku)) {
    throw new PatternCadError("duplicate-id", `SKU '${product.sku}' already in catalog`, product.id);
  }
  const next = clone(catalog);
  next.products.push(clone(product));
  return next;
}

export function removeProduct(catalog: Catalog, productId: string): Catalog {
  if (!catalog.products.some((p) => p.id === productId)) {
    throw new PatternCadError("missing-reference", `product '${productId}' not in catalog`, productId);
  }
  const next = clone(catalog);
  next.products = next.products.filter((p) => p.id !== productId);
  return next;
}

export function updateProduct(catalog: Catalog, product: Product): Catalog {
  const next = clone(catalog);
  const index = next.products.findIndex((p) => p.id === product.id);
  if (index < 0) throw new PatternCadError("missing-reference", `product '${product.id}' not in catalog`, product.id);
  if (next.products.some((p) => p.id !== product.id && p.sku === product.sku)) {
    throw new PatternCadError("duplicate-id", `SKU '${product.sku}' already in catalog`, product.id);
  }
  next.products[index] = clone(product);
  return next;
}

export function findProduct(catalog: Catalog, productId: string): Product | null {
  return clone(catalog.products.find((p) => p.id === productId) ?? null);
}

export interface ProductCard {
  id: string;
  name: string;
  sku: string;
  status: ProductStatus;
  sizes: string[];
  formats: string[];
  revision: number;
  updatedAt: string;
  category: string;
  thumbnail: string | null;
}

/** Lightweight card: stored fields only, no garment/simulation loading. */
export function productCard(product: Product): ProductCard {
  return {
    id: product.id,
    name: product.name,
    sku: product.sku,
    status: product.status,
    sizes: [...product.sizes],
    formats: [...product.formats],
    revision: product.revision,
    updatedAt: product.updatedAt,
    category: product.category,
    thumbnail: product.thumbnailArtifactId ?? null,
  };
}

export interface SearchFilter {
  query?: string;
  category?: string;
  tags?: string[];
  status?: ProductStatus[];
  size?: string;
  format?: string;
}

export type SortKey = "name" | "sku" | "updated" | "revision";

/** Deterministic search: filters in fixed order, stable sort with id tie-break. */
export function searchCatalog(catalog: Catalog, filter: SearchFilter = {}, sort: SortKey = "name"): Product[] {
  const q = filter.query?.toLowerCase();
  let out = catalog.products.filter((p) => {
    if (q && !(p.name.toLowerCase().includes(q) || p.sku.toLowerCase().includes(q))) return false;
    if (filter.category && p.category !== filter.category) return false;
    if (filter.tags && !filter.tags.every((t) => p.tags.includes(t))) return false;
    if (filter.status && !filter.status.includes(p.status)) return false;
    if (filter.size && !p.sizes.includes(filter.size)) return false;
    if (filter.format && !p.formats.includes(filter.format)) return false;
    return true;
  });
  const key: Record<SortKey, (p: Product) => string | number> = {
    name: (p) => p.name.toLowerCase(),
    sku: (p) => p.sku.toLowerCase(),
    updated: (p) => p.updatedAt,
    revision: (p) => p.revision,
  };
  const get = key[sort];
  out = [...out].sort((a, b) => {
    const va = get(a), vb = get(b);
    if (va < vb) return -1;
    if (va > vb) return 1;
    return a.id < b.id ? -1 : 1;
  });
  return clone(out);
}

export interface CatalogIntegrityIssue {
  code: "duplicate-id" | "duplicate-sku" | "invalid-product" | "missing-preview";
  message: string;
  entityId?: string;
}

/** Catalog-wide audit: identity collisions, invalid members, dangling previews. */
export function validateCatalog(catalog: Catalog): CatalogIntegrityIssue[] {
  const issues: CatalogIntegrityIssue[] = [];
  const ids = new Set<string>();
  const skus = new Set<string>();
  for (const product of catalog.products) {
    if (ids.has(product.id)) issues.push({ code: "duplicate-id", message: `duplicate product id '${product.id}'`, entityId: product.id });
    ids.add(product.id);
    if (skus.has(product.sku)) issues.push({ code: "duplicate-sku", message: `duplicate SKU '${product.sku}'`, entityId: product.id });
    skus.add(product.sku);
    for (const diagnostic of validateProduct(product)) {
      if (diagnostic.code === "missing-preview") {
        issues.push({ code: "missing-preview", message: diagnostic.message, entityId: product.id });
      } else if (diagnostic.code === "invalid-document" || diagnostic.code === "unsupported-schema") {
        issues.push({ code: "invalid-product", message: diagnostic.message, entityId: product.id });
      }
    }
  }
  return issues;
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

export function serializeCatalog(catalog: Catalog): string {
  return canonicalJson(catalog);
}

export function deserializeCatalog(serialized: string): Catalog {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized catalog is not valid JSON");
  }
  const catalog = parsed as Catalog;
  if (!catalog || typeof catalog !== "object" || !catalog.id || !Array.isArray(catalog.products)) {
    throw new PatternCadError("invalid-document", "catalog shape is invalid");
  }
  return clone(catalog);
}
