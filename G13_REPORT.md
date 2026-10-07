# G13 — Commercial Export & File Compatibility: Delivery Report

Tracks implemented in this pass: **G13A (Export IR), G13B (DXF), G13C (SVG/PDF),
G13D (Package), G13E (Adversarial QA), G13 FINAL (acceptance)**.

G11E's minimal export (JSON + faceted DXF, single size) remains in
`src/cad/export.ts` for compatibility; the canonical G13 path is the new
`src/cad/export-ir.ts` (IR v2) plus its adapters. Both are public; they are
separate modules with separate version constants.

## Architecture

```
Native garment document (PatternDocument + ProductionSet + Seam[] + G12 grading)
      │
      ▼
production gate (exportGate: strict | allow-warnings; INVALID never passes)
      │
      ▼
Export IR v2  (src/cad/export-ir.ts)        — canonical metres, deterministic
      │
 ┌────┼───────────────┬────────────────┬───────────────────┐
 ▼    ▼               ▼                ▼                   ▼
DXF   SVG             PDF              grade rules         package
(dxf-export) (svg-export) (pdf-export) (grade-rules-export) (export-package)
```

Key invariants, enforced by tests:

- **The IR is derived.** `buildExportIR` never mutates the native document; the
  IR is rebuilt from source and only compared (never re-imported for editing).
- **Exact geometry in the IR.** Boundaries are stored as edge records
  (line endpoints; arc centre/radius/start-angle/sweep) — never flattened.
  DXF emits real `ARC` entities; SVG emits `A` path commands; only PDF
  flattens, at a declared 0.1 mm sagitta bound, and says so in every export's
  warnings. Derived allowance rings are sampled polylines by construction
  (they are offsets, not authored curves) and are documented as such.
- **One canonical unit.** Metres internally; each adapter converts explicitly
  (`convertM`) and declares the emitted unit structurally: DXF `$INSUNITS`
  (4=mm, 5=cm, 1=in), SVG `width="…mm"` + `data-units`, PDF Info keywords.
  Undeclared/unknown units throw (`requireExportUnits`) — no guessing.
- **Gate before files.** `exportGate` runs G11C readiness; errors always block;
  warnings block by default and pass only under explicit `allow-warnings`,
  remaining visible in the IR, the validation manifest, and the README.
- **Determinism.** No timestamps (the manifest's `generatedAt` is caller-
  supplied and omitted by default), no randomness, sorted JSON keys; repeated
  exports are byte-identical (asserted for IR JSON, DXF, SVG, PDF, and the
  whole package).

## G13A — Export IR v2 (`src/cad/export-ir.ts`)

Represents: style metadata, garment metadata, panel metadata (material,
section, mirror pair), cut quantity, cut boundary, sewing boundary, seam
allowance, notches, grainlines (+ cross-grain), fold lines, drill marks,
internal construction lines, labels, annotations, dimensions, grading
information (sizes, base size, per-size rule deltas), units, readiness.
Every object keeps its stable source ID (`panelId`, `segmentId`,
`production/...` IDs, `size/...` IDs). IR schema version is **2**, independent
of the native document's `schemaVersion: 1`.

## G13B — DXF (`dxf-profile.ts`, `dxf-export.ts`, `dxf-validate.ts`)

Profiles are declared contracts (`DXF_PROFILES`), never interchangeable:

| Profile | Layers | Units | Multi-size | Unsupported (warned) |
| --- | --- | --- | --- | --- |
| `generic-r12` | named (CUT, SEW, ALLOWANCE, …) | mm ($INSUNITS 4) | yes | dimensions* |
| `aama-style` | numeric (1 cut, 8 internal, 10 notch, 11 drill, 13 note, 14 sew, 16 grain) | mm | yes | fold, construction |
| `astm-oriented` | numeric (same family) | cm ($INSUNITS 5) | yes | fold, construction, dimension |

\* generic emits dimension TEXT on the DIM layer; the others declare what they
cannot carry and warn — nothing is dropped silently.

**Compatibility matrix (machine-checked in `g13-adversarial.test.ts`):**

| Format/profile | Supported | Unsupported | Unit behavior | Grading behavior | Round-trip result | Known limitations |
| --- | --- | --- | --- | --- | --- | --- |
| DXF generic-r12 | boundaries, sewing, allowance, holes, notches, grain, folds, drills, internals, labels, dimensions, construction | — | mm default; mm/cm/m/in selectable; `$INSUNITS` set | sizes as piece-name text; deltas in separate file | boundary vertices match IR exactly (0.0002 unit tol, layout-aware) | R12 subset: LINE/ARC/CIRCLE/TEXT only; no SPLINE/MTEXT/HATCH/paper space |
| DXF aama-style | as above minus dimensions | fold, construction (warned) | mm; `$INSUNITS=4` | sizes as piece-name text; deltas in separate file | same | numeric layer conventions are community-documented; **not certified** against AAMA docs |
| DXF astm-oriented | as above | fold, construction, dimensions (warned) | cm; `$INSUNITS=5` | sizes as piece-name text; grade tables are a separate artifact | same | **not certified** ASTM D6673 output; ASTM explicitly does not cover cutter instructions or marker laying/spreading — this file is neither |
| SVG | all production entities; true arcs | — | user unit = 1 mm; `width/height` in mm; `data-units` | single-size geometry; sizes listed per piece | path geometry matches IR exactly | none for pattern use |
| PDF full-scale | all production entities | arcs flattened at 0.1 mm sagitta (declared) | pt (72/25.4 per mm), page sized to content + margins | single size | page extent matches layout | "never fit-to-page"; large garments need the tiled mode or a larger preset |
| PDF tiled | as above + registration marks, calibration square, page numbers, neighbour indicators | arcs flattened (declared) | as above | single size | page grid deterministic | manual assembly required |
| Grade rules JSON/CSV | sizes, measurements, per-size deltas | — | canonical metres; measurements as authored | full fidelity | byte-stable | none |

**No profile claims certification.** The compliance note ("not certified …")
is embedded in every DXF export's warnings; profiles were built from
community-documented layer conventions because the AAMA/ASTM standard texts
are not available in this environment.

Post-write validation (`validateDxfOutput`) re-parses the generated file and
checks: structural integrity (sections/ENDSEC/EOF), `$INSUNITS` vs profile,
closed-piece count by endpoint connectivity, entity counts per type, required
markings (notch ticks, grainlines, drills, labels), grading size labels, and
— given the writer's layout — **every boundary vertex within 0.0002 emitted
units**. Unsupported IR features are reported, not hidden.

## G13C — SVG + PDF (`svg-export.ts`, `pdf-export.ts`)

- SVG: 1 user unit = 1 mm contract, real `mm` dimensions, `data-true-scale`,
  all markings classed (`cut`, `sew`, `allow`, `notch`, `grain`, `fold`,
  `drill`, `internal`, `constr`, labels). True arcs via `A` commands.
- PDF: dependency-free deterministic PDF 1.4. Configurable page table
  (A4–A0 presets + custom `PageSize`), margins, overlap, registration marks,
  page numbering, neighbour-tile indicators, a **calibration square** (default
  50 mm, exact to 0.05 pt in tests), and per-page "SCALE 100% · 1 unit = 1 mm"
  text plus units/scale in the Info dictionary. Full-scale pages are sized to
  content + margins — fit-to-page is impossible by construction.
- Tiling is deterministic: `cols = ceil((layoutW − overlap)/step)`, row-major
  page order; tests reconstruct the expected page count and verify per-page
  calibration squares and page numbers.

## G13D — Package builder (`export-package.ts`)

One action (`exportCommercialPackage`) → validated, self-describing tree:

```
<slug>/
  README.md                      (sizes, formats, units, printing rules, per-file checksums)
  preview/preview.svg            (lightweight preview)
  patterns/<name>.dxf|.svg|.pdf  (deterministic names: style[.sizes].revN[.label].ext)
  grading/grade-rules.json       (canonical JSON; per-size deltas)
  grading/size-chart.csv         (measurements per size)
  metadata/manifest.json         (product, revision, generatedAt?, app, units, size set,
                                  piece count, cut quantity total, formats, validation status,
                                  source document version)
  validation/validation-manifest.json  (state, counts, panel/seam measurements, diagnostics)
```

- Names are slugged, deterministic, and collision-safe (numeric suffixes when
  two logical names slug identically — asserted).
- Checksums: SHA-256 per file; verified against the bytes in tests.
- Reproducibility: same source + same settings → byte-identical files except
  the explicitly designated `generatedAt` (manifest only, omitted by default).
- No filesystem paths leak into any artifact (asserted against `C:\`,
  `/Users/`, `/home/`, `AppData`, `Desktop`).
- Refuses to write anything unless the gate passes (`INVALID` always refused;
  `WARNINGS` only with explicit `allow-warnings`, and then they stay visible).

## G13E — Adversarial QA (tests/cad/g13-adversarial.test.ts)

Attacked and verified: unit matrix (mm/cm/in/points, ambiguous units throw),
semantic entity preservation through every adapter, 2/6/12-size sets,
asymmetric deltas, concave L-pieces, self-intersecting geometry (gate
refusal), tiny (20 mm) and huge (1000 m) pieces, Unicode labels (DXF
sanitized deterministically; package/IR keep UTF-8), mirrored pieces,
size-set round trips, and the declared-unsupported matrix.

## Verification

| Check | Result |
| --- | --- |
| `tsc -p tsconfig.json --noEmit` (my modules) | clean |
| `vitest run tests/cad/g13-*.test.ts` | **56/56 pass** (6 files) |
| Full repo suite (`npx vitest run`) | **132 files / 1020 tests pass**, 0 failures |
| Acceptance metrics (workflow test) | gate 10 ms · export 18 ms · package 20 ms · 10 files · 75 KB package · 264 DXF entities · 18 tiled pages |

G13 test breakdown: IR 13, DXF 12, SVG/PDF 11, package 7, adversarial 12,
final workflow 1.

## Known limitations (explicit, not hidden)

1. **No standards certification.** AAMA/ASTM profiles use documented layer
   conventions but were not validated against the purchased standard texts;
   every DXF export repeats this caveat. Do not label these files "ASTM
   compliant".
2. **DXF grade tables are a separate artifact.** ASTM D6673 covers grade-rule
   exchange, but the in-file grade-table encoding was not validated against
   the standard; this repo therefore exports deltas as `grade-rules.json` /
   `size-chart.csv` and states so in the DXF warnings.
3. **PDF flattens arcs** (0.1 mm sagitta bound, declared). DXF/SVG keep exact
   arcs.
4. **Round-trip importers are internal.** The DXF parser validates our own
   R12 subset; it is not a general DXF reader, so third-party DXF files are
   out of scope.
5. **Evaluated-only grading anchors** (G12A limitation) are not materialized
   as split topology; the IR carries the reported positions as-is.
6. Package files are an in-memory tree; writing them to disk (or zip) is left
   to the caller/environment.

## Definition of done

A completed, validated garment can be exported as a professional digital
pattern package — DXF (single/multi-size, profile-declared), true-scale SVG,
full-scale and tiled PDF with calibration, and separate grade-rule data — from
one canonical, gated, deterministic export representation, with every
unsupported feature and unvalidated claim explicitly reported.
