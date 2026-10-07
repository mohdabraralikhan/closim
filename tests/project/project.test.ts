// G17A tests: project model, atomic store, locks, recovery, migrations.
import { describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  acquireLock,
  assessRecovery,
  autosaveFilePath,
  lockHolder,
  openProject,
  projectFilePath,
  readAutosave,
  recordRecent,
  deserializeRecent,
  releaseLock,
  saveProject,
  saveProjectAs,
  serializeRecent,
  writeAutosave,
  type FileSystem,
} from "../../src/project/store.js";
import {
  addGarment,
  APP_VERSION,
  createProject,
  deserializeProject,
  duplicateProject,
  fingerprintProject,
  ProjectMigrator,
  removeGarment,
  renameProject,
  serializeProject,
  setProjectSetting,
  validateProject,
  type AppProject,
} from "../../src/project/project.js";
import { buildTshirtProject } from "../../src/garment/tshirt.js";

const nodeFs: FileSystem = {
  readTextFile: (p) => fs.readFileSync(p, "utf8"),
  writeTextFileAtomic: (p, content) => {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    const tmp = `${p}.tmp-${process.pid}`;
    fs.writeFileSync(tmp, content, "utf8");
    fs.renameSync(tmp, p);
  },
  exists: (p) => fs.existsSync(p),
  mkdirp: (p) => fs.mkdirSync(p, { recursive: true }),
  remove: (p) => fs.rmSync(p, { force: true }),
};

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "closim-g17-"));
}

function garmentEntry(id = "garment/shirt", name = "Shirt") {
  const { project } = buildTshirtProject();
  return { id, name, garment: project };
}

describe("G17A project model", () => {
  it("creates, renames, and validates projects", () => {
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "Shirt", fixed);
    expect(project.metadata.applicationVersion).toBe(APP_VERSION);
    expect(validateProject(project)).toEqual([]);
    project = addGarment(project, garmentEntry(), fixed);
    expect(project.garments).toHaveLength(1);
    project = renameProject(project, "Renamed", fixed);
    expect(project.name).toBe("Renamed");
    expect(() => addGarment(project, garmentEntry())).toThrowError(/already in project/);
    expect(() => renameProject(project, "")).toThrowError(/non-empty/);
    const removed = removeGarment(project, "garment/shirt", fixed);
    expect(removed.garments).toEqual([]);
    expect(() => removeGarment(removed, "garment/shirt")).toThrowError(/not in project/);
  });

  it("rejects invalid garments and duplicate ids", () => {
    let project = createProject("proj/1", "P");
    const bad = garmentEntry();
    (bad.garment as { id: string }).id = "";
    expect(() => addGarment(project, bad)).toThrowError(/invalid/);
    project = addGarment(project, garmentEntry("g/1"));
    project = addGarment(project, garmentEntry("g/2"));
    expect(validateProject(project)).toEqual([]);
    const hacked = { ...project, garments: [project.garments[0], project.garments[0]] };
    expect(validateProject(hacked).map((d) => d.code)).toContain("duplicate-id");
    expect(validateProject({} as AppProject).map((d) => d.code)).toContain("invalid-document");
    expect(validateProject({ ...project, schemaVersion: 99 } as unknown as AppProject).map((d) => d.code))
      .toContain("unsupported-schema");
    expect(() => deserializeProject("nope")).toThrowError(/JSON/);
  });

  it("duplicates with fresh identity and fingerprints content only", () => {
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    const copy = duplicateProject(project, "proj/2", fixed);
    expect(copy.id).toBe("proj/2");
    expect(copy.garments.map((g) => g.id)).toEqual(["garment/shirt"]);
    expect(fingerprintProject(copy)).toBe(fingerprintProject(project));
    const renamed = renameProject(project, "Other", fixed);
    expect(fingerprintProject(renamed)).not.toBe(fingerprintProject(project));
    expect(serializeProject(deserializeProject(serializeProject(project)))).toBe(serializeProject(project));
    expect(() => duplicateProject(project, "proj/1")).toThrowError(/must differ/);
    const withSetting = setProjectSetting(project, "units", "mm", fixed);
    expect(withSetting.settings.units).toBe("mm");
  });

  it("migrates older schemas through a registered chain", () => {
    const migrator = new ProjectMigrator();
    migrator.register({
      from: 0,
      to: 1,
      migrate: (document: unknown) => {
        const doc = document as Record<string, unknown>;
        return {
          schemaVersion: 1,
          id: doc["id"],
          name: doc["name"],
          metadata: { createdAt: "1970-01-01T00:00:00.000Z", updatedAt: "1970-01-01T00:00:00.000Z", applicationVersion: APP_VERSION },
          garments: doc["garments"] ?? [],
          settings: doc["settings"] ?? {},
        };
      },
    });
    const v0 = { schemaVersion: 0, id: "proj/old", name: "Old", garments: [], settings: {} };
    const migrated = migrator.migrateToLatest(v0);
    expect(migrated.schemaVersion).toBe(1);
    expect(validateProject(migrated)).toEqual([]);
    expect(() => migrator.migrateToLatest({ schemaVersion: 9, id: "x" })).toThrowError(/newer/);
    expect(() => new ProjectMigrator().migrateToLatest(v0)).toThrowError(/no migration/);
    expect(() => migrator.register({ from: 0, to: 1, migrate: (d) => d })).toThrowError(/already registered/);
    expect(() => migrator.register({ from: 0, to: 2, migrate: (d) => d })).toThrowError(/exactly one/);
    expect(() => migrator.migrateToLatest({})).toThrowError(/schemaVersion/);
  });
});

describe("G17A atomic store", () => {
  it("saves and reopens byte-identical projects", () => {
    const dir = tempDir();
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    const saved = saveProject(nodeFs, dir, project, "session/a", fixed);
    expect(saved.path).toBe(projectFilePath(dir));
    expect(nodeFs.exists(projectFilePath(dir))).toBe(true);
    // Repeated save is byte-identical (deterministic serialization).
    const again = saveProject(nodeFs, dir, project, "session/a", fixed);
    expect(nodeFs.readTextFile(saved.path)).toBe(nodeFs.readTextFile(again.path));
    const opened = openProject(nodeFs, dir);
    expect(opened.project).toEqual(project);
    releaseLock(nodeFs, dir, "session/a");
    expect(lockHolder(nodeFs, dir)).toBeNull();
    expect(() => openProject(nodeFs, tempDir())).toThrowError(/no project/);
  });

  it("never writes invalid projects and survives interrupted saves", () => {
    const dir = tempDir();
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    saveProject(nodeFs, dir, project, "session/a", fixed);
    const goodBytes = nodeFs.readTextFile(projectFilePath(dir));
    expect(() => saveProject(nodeFs, dir, { ...project, garments: "broken" } as unknown as AppProject, "session/a"))
      .toThrowError(/invalid project/);
    expect(nodeFs.readTextFile(projectFilePath(dir))).toBe(goodBytes);
    // Crash mid-write: fake FS truncates then throws; authoritative file untouched.
    const writes: string[] = [];
    const crashing: FileSystem = {
      ...nodeFs,
      writeTextFileAtomic: (p, content) => {
        writes.push(p);
        if (p === projectFilePath(dir)) throw new Error("simulated power loss");
        nodeFs.writeTextFileAtomic(p, content);
      },
    };
    expect(() => saveProject(crashing, dir, renameProject(project, "New", fixed), "session/a", fixed))
      .toThrowError(/power loss/);
    expect(nodeFs.readTextFile(projectFilePath(dir))).toBe(goodBytes);
    expect(openProject(nodeFs, dir).project.name).toBe("P");
  });

  it("Save As writes a renamed copy elsewhere", () => {
    const src = tempDir();
    const dst = tempDir();
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    saveProject(nodeFs, src, project, "session/a", fixed);
    const result = saveProjectAs(nodeFs, dst, project, "session/a", "proj/2", fixed);
    expect(openProject(nodeFs, dst).project.id).toBe("proj/2");
    expect(openProject(nodeFs, src).project.id).toBe("proj/1");
    expect(result.fingerprint).toBe(fingerprintProject({ ...project, id: "proj/2" }));
  });

  it("detects concurrent sessions and expires stale locks", () => {
    const dir = tempDir();
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    // Live timestamps here: fixed test dates would age the lock past its TTL.
    saveProject(nodeFs, dir, project, "session/a");
    expect(lockHolder(nodeFs, dir)?.sessionId).toBe("session/a");
    expect(() => saveProject(nodeFs, dir, project, "session/b")).toThrowError(/locked by session 'session\/a'/);
    expect(() => releaseLock(nodeFs, dir, "session/b")).toThrowError(/belongs to session/);
    // Stale lock (older than TTL) no longer blocks.
    const lockPath = projectFilePath(dir).replace("project.json", ".closim-lock");
    fs.writeFileSync(lockPath, JSON.stringify({ sessionId: "dead", projectId: "proj/1", timestamp: "2000-01-01T00:00:00.000Z" }));
    expect(lockHolder(nodeFs, dir)).toBeNull();
    saveProject(nodeFs, dir, project, "session/b");
    expect(lockHolder(nodeFs, dir)?.sessionId).toBe("session/b");
    releaseLock(nodeFs, dir, "session/b");
  });

  it("tracks recent projects with bounded history", () => {
    let entries = recordRecent([], { dir: "/a", projectId: "p1", name: "A" }, "2026-01-01T00:00:00.000Z");
    entries = recordRecent(entries, { dir: "/b", projectId: "p2", name: "B" }, "2026-01-02T00:00:00.000Z");
    expect(entries.map((e) => e.dir)).toEqual(["/b", "/a"]);
    entries = recordRecent(entries, { dir: "/a", projectId: "p1", name: "A2" }, "2026-01-03T00:00:00.000Z");
    expect(entries.map((e) => e.dir)).toEqual(["/a", "/b"]);
    expect(entries[0].name).toBe("A2");
    expect(deserializeRecent(serializeRecent(entries))).toEqual(entries);
    expect(() => deserializeRecent("nope")).toThrowError(/JSON/);
  });
});

describe("G17A autosave and recovery", () => {
  it("autosaves beside the authoritative file and assesses recovery", () => {
    const dir = tempDir();
    const t0 = "2026-01-01T00:00:00.000Z";
    const t1 = "2026-01-01T00:01:00.000Z";
    let project = createProject("proj/1", "P", t0);
    project = addGarment(project, garmentEntry(), t0);
    saveProject(nodeFs, dir, project, "session/a", t0);
    expect(assessRecovery(nodeFs, dir).decision).toBe("no-recovery");
    // Edit + autosave (no save): recovery offers the newer copy.
    const edited = renameProject(project, "Edited", t1);
    const record = writeAutosave(nodeFs, dir, edited, t1);
    expect(record.fingerprint).toBe(fingerprintProject(edited));
    expect(nodeFs.exists(autosaveFilePath(dir))).toBe(true);
    expect(nodeFs.readTextFile(projectFilePath(dir))).not.toContain("Edited");
    const assessment = assessRecovery(nodeFs, dir);
    expect(assessment.decision).toBe("autosave-newer");
    expect(readAutosave(nodeFs, dir)?.projectId).toBe("proj/1");
    // Crash with no project at all: autosave is the only copy.
    const empty = tempDir();
    fs.mkdirSync(path.join(empty, "cache"), { recursive: true });
    fs.copyFileSync(autosaveFilePath(dir), autosaveFilePath(empty));
    expect(assessRecovery(nodeFs, empty).decision).toBe("autosave-newer");
  });

  it("detects corruption on either side", () => {
    const dir = tempDir();
    const fixed = "2026-01-01T00:00:00.000Z";
    let project = createProject("proj/1", "P", fixed);
    project = addGarment(project, garmentEntry(), fixed);
    saveProject(nodeFs, dir, project, "session/a", fixed);
    writeAutosave(nodeFs, dir, project, fixed);
    fs.writeFileSync(projectFilePath(dir), "{corrupt");
    expect(assessRecovery(nodeFs, dir).decision).toBe("autosave-newer");
    expect(() => openProject(nodeFs, dir)).toThrowError(/JSON/);
    fs.writeFileSync(autosaveFilePath(dir), "{corrupt");
    expect(readAutosave(nodeFs, dir)).toBeNull();
    // Both sides corrupt: the authoritative failure leads.
    expect(assessRecovery(nodeFs, dir).decision).toBe("project-corrupt");
    // Project healthy but autosave garbage: flagged for cleanup, nothing to recover.
    const dir2 = tempDir();
    saveProject(nodeFs, dir2, project, "session/a", fixed);
    fs.writeFileSync(autosaveFilePath(dir2), "{corrupt");
    expect(assessRecovery(nodeFs, dir2).decision).toBe("autosave-corrupt");
    expect(() => writeAutosave(nodeFs, dir, { ...project, garments: "x" } as unknown as AppProject))
      .toThrowError(/invalid project/);
  });
});
