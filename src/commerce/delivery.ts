// G20A secure delivery: entitlement-gated, single-use, checksum-verified.
//
// Delivery never exposes storage locations: callers receive an opaque
// authorization token bound to (customer, entitlement, release, artifact).
// Tokens are single-use nonces (replay fails), expiry-bounded, and checked
// against entitlement usability, release/artifact revocation, and payload
// checksums before any byte is described as deliverable.
// Rate limiting is an injected abstraction with a token-bucket default.

import { PatternCadError } from "../pattern/cad.js";
import { isEntitlementUsable, resolveReleaseArtifact, updateEligible, type Entitlement, type ProductRelease } from "./entitlements.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export interface DownloadAuthorization {
  token: string;
  customerId: string;
  entitlementId: string;
  releaseId: string;
  artifactId: string;
  filename: string;
  issuedAt: string;
  expiresAt: string;
}

export interface DownloadRecord {
  id: string;
  customerId: string;
  entitlementId: string;
  artifactId: string;
  releaseId: string;
  timestamp: string;
  status: "completed" | "failed";
  bytesDelivered: number;
}

/** Single-use nonce ledger (production: persistent store; tests: memory). */
export interface TokenLedger {
  has(token: string): boolean;
  consume(token: string): void;
}

export function memoryLedger(): TokenLedger {
  const used = new Set<string>();
  return {
    has: (token) => used.has(token),
    consume: (token) => {
      used.add(token);
    },
  };
}

export interface RateLimiter {
  /** True when the request may proceed (and is counted). */
  check(customerId: string, nowMs: number): boolean;
}

/** Token bucket: capacity requests per windowMs, per customer. */
export function tokenBucketLimiter(capacity: number, windowMs: number): RateLimiter {
  if (!Number.isInteger(capacity) || capacity < 1 || !(windowMs > 0)) {
    throw new PatternCadError("invalid-transform", "rate limiter needs capacity >= 1 and positive window");
  }
  const buckets = new Map<string, number[]>();
  return {
    check(customerId: string, nowMs: number): boolean {
      const window = (buckets.get(customerId) ?? []).filter((t) => nowMs - t < windowMs);
      if (window.length >= capacity) {
        buckets.set(customerId, window);
        return false;
      }
      window.push(nowMs);
      buckets.set(customerId, window);
      return true;
    },
  };
}

export function authorizeDownload(partial: {
  token: string;
  customerId: string;
  entitlement: Entitlement;
  release: ProductRelease;
  artifactId: string;
  ledger: TokenLedger;
  limiter: RateLimiter;
  nowIso: string;
  nowMs: number;
  ttlMs?: number;
}): DownloadAuthorization {
  const { token, customerId, entitlement, release, artifactId, ledger, limiter, nowIso, nowMs } = partial;
  if (!token) throw new PatternCadError("invalid-document", "download token must be non-empty");
  if (ledger.has(token)) {
    throw new PatternCadError("invalid-document", "download authorization was already used", entitlement.id);
  }
  if (entitlement.customerId !== customerId) {
    throw new PatternCadError("invalid-document", "entitlement belongs to another customer", entitlement.id);
  }
  if (!isEntitlementUsable(entitlement, nowIso)) {
    throw new PatternCadError("invalid-document", `entitlement '${entitlement.id}' is not usable`, entitlement.id);
  }
  // Release must belong to the entitled product and be covered by the
  // entitlement's revision + update policy (a forged/foreign release never
  // delivers, and eligibility cannot be bypassed by pointing at a newer one).
  if (release.productId !== entitlement.productId) {
    throw new PatternCadError("invalid-document", `release '${release.id}' belongs to another product`, release.id);
  }
  if (!updateEligible(entitlement, release.revision)) {
    throw new PatternCadError("invalid-document", `entitlement '${entitlement.id}' does not cover release revision ${release.revision}`, release.id);
  }
  const artifact = resolveReleaseArtifact(release, artifactId);
  if (!limiter.check(customerId, nowMs)) {
    throw new PatternCadError("invalid-document", "download rate limit exceeded", entitlement.id);
  }
  ledger.consume(token);
  const ttlMs = partial.ttlMs ?? 5 * 60 * 1000;
  return {
    token,
    customerId,
    entitlementId: entitlement.id,
    releaseId: release.id,
    artifactId: artifact.artifactId,
    filename: artifact.filename,
    issuedAt: nowIso,
    expiresAt: new Date(nowMs + ttlMs).toISOString(),
  };
}

/**
 * Fulfill an authorization against a payload: re-validates expiry and
 * checksum, then records the download. Returns the record (bytes only flow
 * to the caller through their own storage adapter).
 */
export function fulfillDownload(partial: {
  id: string;
  authorization: DownloadAuthorization;
  payload: string;
  expectedChecksum: string;
  hashHex: (payload: string) => string;
  nowIso: string;
  nowMs: number;
}): { record: DownloadRecord; bytes: number } {
  const { id, authorization, payload, expectedChecksum, hashHex, nowIso, nowMs } = partial;
  if (Date.parse(nowIso) > Date.parse(authorization.expiresAt)) {
    return {
      record: {
        id, customerId: authorization.customerId, entitlementId: authorization.entitlementId,
        artifactId: authorization.artifactId, releaseId: authorization.releaseId,
        timestamp: nowIso, status: "failed", bytesDelivered: 0,
      },
      bytes: 0,
    };
  }
  if (hashHex(payload) !== expectedChecksum) {
    return {
      record: {
        id, customerId: authorization.customerId, entitlementId: authorization.entitlementId,
        artifactId: authorization.artifactId, releaseId: authorization.releaseId,
        timestamp: nowIso, status: "failed", bytesDelivered: 0,
      },
      bytes: 0,
    };
  }
  void nowMs;
  return {
    record: {
      id, customerId: authorization.customerId, entitlementId: authorization.entitlementId,
      artifactId: authorization.artifactId, releaseId: authorization.releaseId,
      timestamp: nowIso, status: "completed", bytesDelivered: payload.length,
    },
    bytes: payload.length,
  };
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

export function serializeAuthorization(authorization: DownloadAuthorization): string {
  return canonicalJson(authorization);
}
