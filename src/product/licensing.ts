// G19E (integration surface) — licensing metadata and customer summary.
//
// License *metadata* only: types, permissions, creator-entered terms.
// No payments, accounts, enforcement, or DRM anywhere in this module.

import { PatternCadError } from "../pattern/cad.js";
import type { Product, ProductArtifact } from "./product.js";
import { validateLicense } from "./product.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export const LICENSE_TYPES = [
  "personal",
  "commercial-single",
  "commercial-extended",
  "educational",
  "custom",
] as const;

export type LicenseType = (typeof LICENSE_TYPES)[number];

export function isKnownLicenseType(type: string): boolean {
  return (LICENSE_TYPES as readonly string[]).includes(type);
}

export interface LicenseMetadata {
  type: string;
  allowsCommercialUse: boolean;
  allowsModification: boolean;
  allowsRedistribution: boolean;
  attributionRequired: boolean;
  supportContact?: string;
  terms?: string;
  version: string;
}

export function createLicense(partial: {
  type: string;
  allowsCommercialUse?: boolean;
  allowsModification?: boolean;
  allowsRedistribution?: boolean;
  attributionRequired?: boolean;
  supportContact?: string;
  terms?: string;
  version?: string;
}): LicenseMetadata {
  if (!partial.type) throw new PatternCadError("invalid-document", "license needs a type");
  const license: LicenseMetadata = {
    type: partial.type,
    allowsCommercialUse: partial.allowsCommercialUse ?? false,
    allowsModification: partial.allowsModification ?? true,
    allowsRedistribution: partial.allowsRedistribution ?? false,
    attributionRequired: partial.attributionRequired ?? false,
    ...(partial.supportContact ? { supportContact: partial.supportContact } : {}),
    ...(partial.terms ? { terms: partial.terms } : {}),
    version: partial.version ?? "1.0",
  };
  validateLicense(license);
  return license;
}

export interface CustomerSummary {
  product: string;
  sku: string;
  revision: number;
  status: string;
  sizes: string[];
  formats: string[];
  filesIncluded: string[];
  license: {
    type: string;
    commercialUse: boolean;
    modification: boolean;
    redistribution: boolean;
    attribution: boolean;
  };
  requirements?: string;
  version: string;
}

/** Structured customer-facing summary (no garment internals leak). */
export function customerSummary(product: Product, artifacts: ProductArtifact[]): CustomerSummary {
  return {
    product: product.name,
    sku: product.sku,
    revision: product.revision,
    status: product.status,
    sizes: [...product.sizes],
    formats: [...product.formats],
    filesIncluded: artifacts
      .filter((a) => a.status === "current")
      .map((a) => a.filename)
      .sort(),
    license: {
      type: product.license.type,
      commercialUse: product.license.allowsCommercialUse,
      modification: product.license.allowsModification,
      redistribution: product.license.allowsRedistribution,
      attribution: product.license.attributionRequired,
    },
    ...(product.requirements ? { requirements: product.requirements } : {}),
    version: `r${product.revision}`,
  };
}

export function licensedArtifacts(product: Product): ProductArtifact[] {
  return clone(product.artifacts.filter((a) => a.status === "current"));
}
