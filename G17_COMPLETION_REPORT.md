# G17 — Commercial Product Infrastructure (G17A + FINAL)

G16 and below pre-existed. This round adds `src/project/` + `tests/project/`
(12 tests). No solver, FEM, collision, GPU, grading, CAD, marker, render,
view, or construction file was modified.

## Interfaces reused

- G8 `GarmentProject`/`validateGarmentProject` (embedded per entry);
  G9A `CadSession` history for pattern-level undo/redo in the acceptance
  flow (app-wide command history itself is the G17B track);
  G11A production builders + G11C readiness + G11E export package;
  G12 grading fixtures/model/validation; G14 marker workspace/model;
  G15 materials/presentation/package; G16 dart/component derivation;
  G8 assembly/solver for simulation steps.

## Files added

- `src/project/project.ts` — G17A model: `AppProject` (id, name, metadata
  with app version, garment entries bundling garment + construction +
  components + grading + production + marker + materials + presentation +
  settings + export refs), ops (add/remove/rename/duplicate/settings),
  validation (schema, duplicates, garment validity), content fingerprint
  (identity + timestamps excluded), canonical serialization, and a
  `ProjectMigrator` framework (single-step chained migrations, newer-than-
  app rejection, missing-migration errors).
- `src/project/store.ts` — atomic save (validate → tmp + rename; invalid
  projects and crashed writes never touch `project.json`), session locks
  with TTL expiry (never silently overwrites another session), open/Save-As,
  autosave-to-cache + non-writing `assessRecovery` (no-recovery /
  autosave-newer / autosave-corrupt / project-corrupt), recent-list helpers.
  All I/O behind an injected `FileSystem` (browser-safe; node + crashing
  fakes live in tests).
- `src/project/index.ts` — barrel.
- Tests (12): `project` (11: model/validation/fingerprint/migrations,
  atomicity, crash survival, locks + expiry, Save As, recent, autosave and
  all four recovery verdicts) and `g17-workflow` (1 × 21-step commercial
  acceptance session).

## Verification

- Scope: `tests/project` **12/12 pass**.
- Broad gate (project/construction/cad/pattern/garment/grading/marker/
  render/view): **65 files / 709 tests pass**, 0 failures.
- `npx tsc --noEmit`: exactly one error, in another worker's in-flight
  `tests/cad/g13-adversarial.test.ts` (implicit `any`; runtime green).
  Nothing in this scope.
- Acceptance (`g17-workflow`): create/open → pattern edit → dart + collar
  derivation → material assignment → grading extended S/M/L→XXL → NaN-free
  simulation → render representation + saved scene/captures → save →
  session-history edit/undo/redo → save (fingerprint moves) → READY
  production export written to `exports/` → marker optimized (all placed)
  → save → close/reopen with byte-equal fingerprint → full integrity
  verification (pattern, seams, features, components, grading, materials,
  avatar, sim settings, scene, production meta, marker, export refs +
  files on disk) → garment edit re-simulated with changed positions →
  autosave-without-save → restart assesses `autosave-newer` → restore +
  save reproduces the autosaved fingerprint.
- Performance load is trivially small (whole acceptance ~0.5 s; save/load
  are single small-file writes; no UI thread exists here to block).

## Defects found and fixed (own scope)

1. Fingerprint covered the project id (duplicates never matched) — identity
   and timestamps now excluded by documented design.
2. Recovery verdict precedence (both-corrupt must lead with the
   authoritative failure; healthy-project + garbage autosave flagged for
   cleanup rather than hidden).
3. Fixed test timestamps instantly aged locks past TTL — lock tests use
   live timestamps.
4. Derived panels (e.g. collar bands) need placements in the garment or
   their seams fail assembly — added alongside the seam.
5. Centered grainlines can fall in dart cutouts — deterministic sideways
   nudge helper in tests (validation itself is correct to refuse).

## Assumptions / limits

- Pattern undo/redo runs on the existing session history; the app-wide
  command bus, asset library, autosave scheduling, and preferences UI are
  the sibling G17B–G17E tracks and are deliberately not duplicated here.
- Lock TTL is crash-tolerance, not distributed consensus; NFS-style
  filesystems are out of scope.
- No marketplace/payments/accounts/cloud/AI/try-on/social features.
