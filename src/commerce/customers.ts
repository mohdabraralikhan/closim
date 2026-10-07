// G20A customers: minimal identity, auth boundary, account lifecycle.
//
// Privacy rule: only an id, email/contact reference, display name, status,
// and timestamps. No passwords here — authentication is a provider boundary
// (AuthProvider), never custom cryptography. Deleting an account marks it
// deleted; financial records keep their snapshot references (see orders).

import { PatternCadError } from "../pattern/cad.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export type AccountStatus = "active" | "suspended" | "deleted";

export interface Customer {
  id: string;
  email: string;
  displayName: string;
  status: AccountStatus;
  createdAt: string;
}

export interface AuthRecord {
  customerId: string;
  /** External provider name ("test-stub", "oidc", ...). Never a password. */
  provider: string;
  /** Provider-side subject/account id. */
  subject: string;
}

/** Authentication boundary: verify a token against an external provider. */
export interface AuthProvider {
  name: string;
  verify(token: string): { customerId: string } | null;
}

function requireId(id: string, label: string): void {
  if (!id || !id.trim()) throw new PatternCadError("invalid-document", `${label} must be non-empty`);
}

function requireEmail(email: string): void {
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw new PatternCadError("invalid-document", "customer email is not a valid address");
  }
}

export function createCustomer(partial: {
  id: string;
  email: string;
  displayName: string;
  now?: string;
}): Customer {
  requireId(partial.id, "customer id");
  requireEmail(partial.email);
  if (!partial.displayName || !partial.displayName.trim()) {
    throw new PatternCadError("invalid-document", "customer display name must be non-empty");
  }
  return {
    id: partial.id,
    email: partial.email,
    displayName: partial.displayName,
    status: "active",
    createdAt: partial.now ?? new Date().toISOString(),
  };
}

export function setAccountStatus(customer: Customer, status: AccountStatus): Customer {
  if (status !== "active" && status !== "suspended" && status !== "deleted") {
    throw new PatternCadError("invalid-document", `unknown account status '${status}'`, customer.id);
  }
  return { ...clone(customer), status };
}

/** Authenticate a token; returns the customer record or throws. */
export function authenticate(
  customers: Customer[],
  provider: AuthProvider,
  token: string,
): Customer {
  if (!token) throw new PatternCadError("invalid-document", "authentication token is missing");
  const verified = provider.verify(token);
  if (!verified) throw new PatternCadError("invalid-document", "authentication failed: unknown token");
  const customer = customers.find((c) => c.id === verified.customerId);
  if (!customer) throw new PatternCadError("missing-reference", "authenticated customer does not exist", verified.customerId);
  if (customer.status !== "active") {
    throw new PatternCadError("invalid-document", `account '${customer.id}' is ${customer.status}`, customer.id);
  }
  return clone(customer);
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

export function serializeCustomer(customer: Customer): string {
  return canonicalJson(customer);
}

export function deserializeCustomer(serialized: string): Customer {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized customer is not valid JSON");
  }
  const customer = parsed as Customer;
  if (!customer || !customer.id || !customer.email || !customer.createdAt) {
    throw new PatternCadError("invalid-document", "customer shape is invalid");
  }
  return clone(customer);
}
