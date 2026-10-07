// G17A store: filesystem persistence with atomicity and locking.
//
// Crash safety comes from ordering, not luck:
//   1. validate the project in memory (invalid projects never touch disk),
//   2. write the full payload to a temp file in the same directory,
//   3. fsync-equivalent flush through the adapter, then atomic rename.
//
// A crashed save leaves either the old project.json (rename never ran) or
// a complete new one (rename is atomic) — never a half-written file.
// Lockfiles detect concurrent sessions; stale locks expire by TTL.
//
// Browser-safe: all I/O goes through the injected FileSystem interface.
// Node and in-memory implementations live outside src/ (tests, app shell).

import { PatternCadError } from "../pattern/cad.js";
import {
  deserializeProject,
  fingerprintProject,
  serializeProject,
  validateProject,
  type AppProject,
} from "./project.js";

export interface FileSystem {
  readTextFile(path: string): string;
  /** Must write atomically where the platform supports it (tmp + rename). */
  writeTextFileAtomic(path: string, content: string): void;
  exists(path: string): boolean;
  mkdirp(path: string): void;
  remove(path: string): void;
}

export const PROJECT_FILENAME = "project.json";
export const LOCK_FILENAME = ".closim-lock";
export const AUTOSAVE_FILENAME = "autosave.json";
export const CACHE_DIRNAME = "cache";
export const EXPORTS_DIRNAME = "exports";
/** Lock considered stale (crashed holder) after this many milliseconds. */
export const LOCK_TTL_MS = 5 * 60 * 1000;

export interface ProjectLock {
  sessionId: string;
  projectId: string;
  timestamp: string;
}

function joinPath(dir: string, file: string): string {
  return dir.endsWith("/") || dir.endsWith("\\") ? `${dir}${file}` : `${dir}/${file}`;
}

export function projectFilePath(dir: string): string {
  return joinPath(dir, PROJECT_FILENAME);
}

export function lockFilePath(dir: string): string {
  return joinPath(dir, LOCK_FILENAME);
}

export function autosaveFilePath(dir: string): string {
  return joinPath(joinPath(dir, CACHE_DIRNAME), AUTOSAVE_FILENAME);
}

function parseLock(content: string): ProjectLock | null {
  try {
    const lock = JSON.parse(content) as ProjectLock;
    if (!lock || typeof lock.sessionId !== "string" || typeof lock.timestamp !== "string") return null;
    return lock;
  } catch {
    return null;
  }
}

/** Who holds the lock, if anyone live. Stale locks are reported as absent. */
export function lockHolder(fs: FileSystem, dir: string, nowMs?: number): ProjectLock | null {
  const path = lockFilePath(dir);
  if (!fs.exists(path)) return null;
  const lock = parseLock(fs.readTextFile(path));
  if (!lock) return null;
  const now = nowMs ?? Date.now();
  const age = now - Date.parse(lock.timestamp);
  if (!Number.isFinite(age) || age > LOCK_TTL_MS) return null;
  return lock;
}

export function acquireLock(fs: FileSystem, dir: string, sessionId: string, projectId: string, now?: string): void {
  const holder = lockHolder(fs, dir);
  if (holder && holder.sessionId !== sessionId) {
    throw new PatternCadError(
      "invalid-document",
      `project is locked by session '${holder.sessionId}' (since ${holder.timestamp})`,
    );
  }
  fs.mkdirp(dir);
  fs.writeTextFileAtomic(lockFilePath(dir), JSON.stringify({
    sessionId, projectId, timestamp: now ?? new Date().toISOString(),
  }));
}

export function releaseLock(fs: FileSystem, dir: string, sessionId: string): void {
  const path = lockFilePath(dir);
  if (!fs.exists(path)) return;
  const holder = parseLock(fs.readTextFile(path));
  if (holder && holder.sessionId !== sessionId) {
    throw new PatternCadError("invalid-document", `lock belongs to session '${holder.sessionId}', not '${sessionId}'`);
  }
  fs.remove(path);
}

// ---------------------------------------------------------------------------
// Save / open (atomic). Validation always precedes any write.
// ---------------------------------------------------------------------------

export interface SaveResult {
  path: string;
  fingerprint: string;
}

export function saveProject(
  fs: FileSystem, dir: string, project: AppProject, sessionId: string, now?: string,
): SaveResult {
  const diagnostics = validateProject(project);
  if (diagnostics.length > 0) {
    throw new PatternCadError("invalid-document", `refusing to save invalid project: ${diagnostics[0].code}: ${diagnostics[0].message}`);
  }
  acquireLock(fs, dir, sessionId, project.id, now);
  fs.mkdirp(joinPath(dir, CACHE_DIRNAME));
  fs.mkdirp(joinPath(dir, EXPORTS_DIRNAME));
  const path = projectFilePath(dir);
  fs.writeTextFileAtomic(path, serializeProject(project));
  return { path, fingerprint: fingerprintProject(project) };
}

export interface OpenResult {
  project: AppProject;
  lockedBy: ProjectLock | null;
}

export function openProject(fs: FileSystem, dir: string): OpenResult {
  const path = projectFilePath(dir);
  if (!fs.exists(path)) {
    throw new PatternCadError("invalid-document", `no project at '${path}'`);
  }
  const project = deserializeProject(fs.readTextFile(path));
  return { project, lockedBy: lockHolder(fs, dir) };
}

export function saveProjectAs(
  fs: FileSystem, dir: string, project: AppProject, sessionId: string, newId?: string, now?: string,
): SaveResult {
  const renamed = newId && newId !== project.id
    ? { ...JSON.parse(JSON.stringify(project)) as AppProject, id: newId }
    : project;
  return saveProject(fs, dir, renamed, sessionId, now);
}

// ---------------------------------------------------------------------------
// Autosave + recovery (mechanics; scheduling policy belongs to G17D)
// ---------------------------------------------------------------------------

export interface AutosaveRecord {
  projectId: string;
  fingerprint: string;
  savedAt: string;
  payload: string;
}

/** Write a recoverable copy under cache/ (never touches project.json). */
export function writeAutosave(fs: FileSystem, dir: string, project: AppProject, now?: string): AutosaveRecord {
  const diagnostics = validateProject(project);
  if (diagnostics.length > 0) {
    throw new PatternCadError("invalid-document", "refusing to autosave an invalid project");
  }
  fs.mkdirp(joinPath(dir, CACHE_DIRNAME));
  const record: AutosaveRecord = {
    projectId: project.id,
    fingerprint: fingerprintProject(project),
    savedAt: now ?? new Date().toISOString(),
    payload: serializeProject(project),
  };
  fs.writeTextFileAtomic(autosaveFilePath(dir), JSON.stringify(record));
  return record;
}

export function readAutosave(fs: FileSystem, dir: string): AutosaveRecord | null {
  const path = autosaveFilePath(dir);
  if (!fs.exists(path)) return null;
  try {
    const record = JSON.parse(fs.readTextFile(path)) as AutosaveRecord;
    if (!record || typeof record.payload !== "string" || typeof record.fingerprint !== "string") return null;
    return record;
  } catch {
    return null;
  }
}

export type RecoveryDecision = "no-recovery" | "autosave-newer" | "autosave-corrupt" | "project-corrupt";

/**
 * Compare authoritative state against the autosave copy. Returns what a
 * recovery workflow should offer; it never writes anything itself.
 */
export function assessRecovery(fs: FileSystem, dir: string): { decision: RecoveryDecision; record: AutosaveRecord | null } {
  const path = projectFilePath(dir);
  let projectOk = false;
  let projectFingerprint = "";
  try {
    if (fs.exists(path)) {
      projectFingerprint = fingerprintProject(deserializeProject(fs.readTextFile(path)));
      projectOk = true;
    }
  } catch {
    projectOk = false;
  }
  const autoPath = autosaveFilePath(dir);
  const autoExists = fs.exists(autoPath);
  const record = readAutosave(fs, dir);
  let payloadOk = false;
  if (record) {
    try {
      const payload = deserializeProject(record.payload);
      payloadOk = fingerprintProject(payload) === record.fingerprint;
    } catch {
      payloadOk = false;
    }
  }
  if (!projectOk) {
    // Authoritative copy broken: recoverable only from a valid autosave.
    if (record && payloadOk) return { decision: "autosave-newer", record };
    return { decision: "project-corrupt", record: null };
  }
  if (autoExists && (!record || !payloadOk)) {
    // Project is fine, but a corrupt autosave lingers: flag for cleanup.
    return { decision: "autosave-corrupt", record: null };
  }
  if (!record || !payloadOk) return { decision: "no-recovery", record: null };
  if (projectFingerprint !== record.fingerprint) return { decision: "autosave-newer", record };
  return { decision: "no-recovery", record };
}

// ---------------------------------------------------------------------------
// Recent projects (caller-owned persistence path)
// ---------------------------------------------------------------------------

export interface RecentEntry {
  dir: string;
  projectId: string;
  name: string;
  lastOpened: string;
}

export function recordRecent(entries: RecentEntry[], entry: Omit<RecentEntry, "lastOpened">, now?: string, limit = 10): RecentEntry[] {
  const stamped: RecentEntry = { ...entry, lastOpened: now ?? new Date().toISOString() };
  const rest = entries.filter((e) => e.dir !== entry.dir);
  return [stamped, ...rest].slice(0, Math.max(1, limit));
}

export function serializeRecent(entries: RecentEntry[]): string {
  return JSON.stringify(entries);
}

export function deserializeRecent(serialized: string): RecentEntry[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "recent list is not valid JSON");
  }
  if (!Array.isArray(parsed)) throw new PatternCadError("invalid-document", "recent list shape is invalid");
  return parsed as RecentEntry[];
}
