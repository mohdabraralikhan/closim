export const REVISION_LEDGER_VERSION = 1;

export type RevisionStatus = "Draft" | "In Review" | "Approved" | "Released" | "Obsolete";

export interface ProductionArtifactRef {
  id: string;
  kind: "pattern" | "grading" | "bom" | "marker" | "tech-pack" | "export" | "cut-plan" | "production-package" | string;
  fingerprint: string;
  revisionId: string;
}

export interface ProductionRevision {
  id: string;
  number: number;
  parentRevisionId: string | null;
  author: string;
  timestamp: string;
  changeSummary: string;
  approvalState: string;
  sourceProjectVersion: string;
  status: RevisionStatus;
  sourceFingerprint: string;
  /** Optional canonical source-project snapshot for a self-contained rollback. */
  sourceSnapshot?: string;
  artifacts: ProductionArtifactRef[];
}

export interface RevisionEvent {
  id: string;
  revisionId: string;
  kind: "created" | "branched" | "status-changed" | "artifact-recorded" | "rollback";
  timestamp: string;
  author: string;
  summary: string;
  data: Record<string, string>;
}

export interface RevisionLedger {
  schemaVersion: typeof REVISION_LEDGER_VERSION;
  garmentId: string;
  revisions: ProductionRevision[];
  history: RevisionEvent[];
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return `{${Object.keys(obj).sort().map((key) => `${JSON.stringify(key)}:${canonical(obj[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function pushEvent(ledger: RevisionLedger, event: RevisionEvent): RevisionLedger {
  return { ...ledger, history: [...ledger.history, event] };
}

export function createRevisionLedger(garmentId: string): RevisionLedger {
  if (!garmentId.trim()) throw new Error("revision ledger needs a garment id");
  return { schemaVersion: REVISION_LEDGER_VERSION, garmentId, revisions: [], history: [] };
}

export function validateRevisionLedger(ledger: RevisionLedger): string[] {
  if (!ledger || ledger.schemaVersion !== REVISION_LEDGER_VERSION || !ledger.garmentId ||
    !Array.isArray(ledger.revisions) || !Array.isArray(ledger.history)) return ["ledger shape or schema is invalid"];
  const errors: string[] = [];
  const ids = new Set<string>();
  const statuses: RevisionStatus[] = ["Draft", "In Review", "Approved", "Released", "Obsolete"];
  for (const revision of ledger.revisions) {
    if (!revision.id || ids.has(revision.id)) errors.push(`revision id '${revision.id}' is empty or duplicated`);
    ids.add(revision.id);
    if (!Number.isSafeInteger(revision.number) || revision.number < 1 || !revision.author ||
      !revision.timestamp || !revision.changeSummary || !revision.sourceFingerprint) {
      errors.push(`revision '${revision.id}' has invalid metadata`);
    }
    if (!statuses.includes(revision.status)) errors.push(`revision '${revision.id}' has unknown status '${revision.status}'`);
    if (revision.sourceSnapshot !== undefined) {
      try {
        JSON.parse(revision.sourceSnapshot);
      } catch {
        errors.push(`revision '${revision.id}' source snapshot is not valid JSON`);
      }
    }
    const artifactIds = new Set<string>();
    for (const artifact of revision.artifacts ?? []) {
      if (!artifact.id || artifactIds.has(artifact.id) || !artifact.kind || !artifact.fingerprint ||
        artifact.revisionId !== revision.id) {
        errors.push(`revision '${revision.id}' has an invalid or duplicate artifact reference`);
      }
      artifactIds.add(artifact.id);
    }
  }
  for (const revision of ledger.revisions) {
    if (revision.parentRevisionId && !ids.has(revision.parentRevisionId)) {
      errors.push(`revision '${revision.id}' has missing parent '${revision.parentRevisionId}'`);
    }
    const seenParents = new Set<string>([revision.id]);
    let parentId = revision.parentRevisionId;
    while (parentId) {
      if (seenParents.has(parentId)) {
        errors.push(`revision '${revision.id}' contains a parent cycle`);
        break;
      }
      seenParents.add(parentId);
      parentId = ledger.revisions.find((entry) => entry.id === parentId)?.parentRevisionId ?? null;
    }
  }
  const eventIds = new Set<string>();
  for (const event of ledger.history) {
    if (!event.id || eventIds.has(event.id) || !ids.has(event.revisionId) ||
      !event.timestamp || !event.author || !event.summary || !event.data || typeof event.data !== "object") {
      errors.push(`revision history contains an invalid or duplicated event '${event.id}'`);
    }
    eventIds.add(event.id);
  }
  return errors;
}

export function createRevision(input: {
  ledger: RevisionLedger;
  id: string;
  author: string;
  timestamp: string;
  changeSummary: string;
  sourceProjectVersion: string;
  sourceFingerprint: string;
  sourceSnapshot?: string;
  parentRevisionId?: string | null;
  status?: RevisionStatus;
}): RevisionLedger {
  if (!input.id.trim() || !input.author.trim() || !input.timestamp.trim() ||
    !input.changeSummary.trim() || !input.sourceProjectVersion.trim() || !input.sourceFingerprint.trim()) {
    throw new Error("revision requires an id, author, timestamp, summary, source version, and source fingerprint");
  }
  if (input.sourceSnapshot !== undefined) {
    try {
      JSON.parse(input.sourceSnapshot);
    } catch {
      throw new Error("revision source snapshot must be valid JSON");
    }
  }
  const ledger = clone(input.ledger);
  if (ledger.revisions.some((revision) => revision.id === input.id)) throw new Error(`revision '${input.id}' already exists`);
  const parentId = input.parentRevisionId === undefined
    ? (ledger.revisions.at(-1)?.id ?? null)
    : input.parentRevisionId;
  if (parentId && !ledger.revisions.some((revision) => revision.id === parentId)) {
    throw new Error(`parent revision '${parentId}' does not exist`);
  }
  const revision: ProductionRevision = {
    id: input.id,
    number: ledger.revisions.length ? Math.max(...ledger.revisions.map((entry) => entry.number)) + 1 : 1,
    parentRevisionId: parentId,
    author: input.author,
    timestamp: input.timestamp,
    changeSummary: input.changeSummary,
    approvalState: "unconfigured",
    sourceProjectVersion: input.sourceProjectVersion,
    status: input.status ?? "Draft",
    sourceFingerprint: input.sourceFingerprint,
    ...(input.sourceSnapshot !== undefined ? { sourceSnapshot: input.sourceSnapshot } : {}),
    artifacts: [],
  };
  ledger.revisions.push(revision);
  return pushEvent(ledger, {
    id: `${input.id}/event/created`, revisionId: input.id, kind: parentId ? "branched" : "created",
    timestamp: input.timestamp, author: input.author, summary: input.changeSummary,
    data: { parentRevisionId: parentId ?? "" },
  });
}

export function setRevisionStatus(
  ledger: RevisionLedger,
  revisionId: string,
  status: RevisionStatus,
  options: { timestamp: string; author: string; approvalState?: string; allowedTransitions?: Record<string, string[]> },
): RevisionLedger {
  const next = clone(ledger);
  const revision = next.revisions.find((entry) => entry.id === revisionId);
  if (!revision) throw new Error(`revision '${revisionId}' does not exist`);
  if (options.allowedTransitions && !options.allowedTransitions[revision.status]?.includes(status)) {
    throw new Error(`transition '${revision.status}' to '${status}' is not configured`);
  }
  const oldStatus = revision.status;
  revision.status = status;
  if (options.approvalState !== undefined) revision.approvalState = options.approvalState;
  return pushEvent(next, {
    id: `${revisionId}/event/${next.history.length + 1}`, revisionId, kind: "status-changed",
    timestamp: options.timestamp, author: options.author, summary: `Status changed from ${oldStatus} to ${status}`,
    data: { from: oldStatus, to: status, approvalState: revision.approvalState },
  });
}

export function recordRevisionArtifact(
  ledger: RevisionLedger,
  revisionId: string,
  artifact: Omit<ProductionArtifactRef, "revisionId">,
  options: { timestamp: string; author: string },
): RevisionLedger {
  const next = clone(ledger);
  const revision = next.revisions.find((entry) => entry.id === revisionId);
  if (!revision) throw new Error(`revision '${revisionId}' does not exist`);
  if (revision.status === "Obsolete") throw new Error(`cannot modify artifacts on obsolete revision '${revisionId}'`);
  if (revision.status === "Released" &&
    (artifact.kind !== "production-package" || revision.artifacts.some((entry) => entry.id === artifact.id))) {
    throw new Error(`released revision '${revisionId}' only accepts a new production-package trace artifact`);
  }
  const ref = { ...clone(artifact), revisionId };
  const existing = revision.artifacts.findIndex((entry) => entry.id === ref.id);
  if (existing >= 0) revision.artifacts[existing] = ref;
  else revision.artifacts.push(ref);
  return pushEvent(next, {
    id: `${revisionId}/event/${next.history.length + 1}`, revisionId, kind: "artifact-recorded",
    timestamp: options.timestamp, author: options.author, summary: `${ref.kind} artifact recorded`,
    data: { artifactId: ref.id, kind: ref.kind, fingerprint: ref.fingerprint },
  });
}

export function rollbackRevision(
  ledger: RevisionLedger,
  revisionId: string,
  input: { id: string; author: string; timestamp: string; sourceProjectVersion: string },
): RevisionLedger {
  const source = ledger.revisions.find((revision) => revision.id === revisionId);
  if (!source) throw new Error(`revision '${revisionId}' does not exist`);
  const branched = createRevision({
    ledger,
    id: input.id,
    author: input.author,
    timestamp: input.timestamp,
    changeSummary: `Rollback to revision ${source.number}`,
    sourceProjectVersion: input.sourceProjectVersion,
    sourceFingerprint: source.sourceFingerprint,
    ...(source.sourceSnapshot !== undefined ? { sourceSnapshot: source.sourceSnapshot } : {}),
    parentRevisionId: source.id,
  });
  const newRevision = branched.revisions[branched.revisions.length - 1];
  newRevision.artifacts = clone(source.artifacts).map((artifact) => ({ ...artifact, revisionId: newRevision.id }));
  return pushEvent(branched, {
    id: `${input.id}/event/rollback`, revisionId: input.id, kind: "rollback",
    timestamp: input.timestamp, author: input.author, summary: `Rollback derived from revision ${source.number}`,
    data: { sourceRevisionId: source.id },
  });
}

export function serializeRevisionLedger(ledger: RevisionLedger): string {
  const errors = validateRevisionLedger(ledger);
  if (errors.length) throw new Error(`invalid revision ledger: ${errors[0]}`);
  return canonical(ledger);
}

export function deserializeRevisionLedger(serialized: string): RevisionLedger {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("serialized revision ledger is not valid JSON");
  }
  const ledger = parsed as RevisionLedger;
  const errors = validateRevisionLedger(ledger);
  if (errors.length) throw new Error(`invalid revision ledger: ${errors[0]}`);
  return clone(ledger);
}
