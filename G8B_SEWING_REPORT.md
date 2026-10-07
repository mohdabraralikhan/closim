# G8B Sewing and Construction

## Scope

G8B adds an explicit construction layer over G8A CAD boundary entity IDs. Seams reference a panel ID, boundary-loop ID, and ordered segment IDs. Their references remain independent from the triangulated simulation mesh.

## Implementation

- `src/garment/sewing.ts` defines seam sides, seam metadata, stitch pairs, validation diagnostics, and an assembly graph.
- Seam lengths are measured in transformed pattern space. Lines use their exact endpoint length; circular arcs use deterministic 128-segment sampling.
- Stitch correspondence uses normalized arc length on each side, so the sides can differ in length or segmentation. An explicit `reversed` flag changes correspondence direction without rewriting authored CAD IDs.
- Validation detects missing panels, loops, and segments, segments outside their referenced loop, broken/non-contiguous chains, zero-length seams, duplicate IDs/definitions, same-side joins, and invalid stitch counts.
- `buildAssemblyGraph` produces resolved stitch-pair records for G8C. It does not alter solver topology or add physics constraints by itself.

## Verification

- `npx.cmd tsc --noEmit` — passed.
- `npx.cmd vitest run tests/pattern/ tests/garment/ --reporter=basic` — passed, **4 files / 43 tests**.
- The G8A pre-G8B full suite passed **72 files / 360 tests**. G8B has not yet had a full-suite run; reserve the full regression run for vertical-slice integration.

## Integration boundary

The stitch pairs are deterministic 2D pattern-space correspondence. G8C still needs to map them through panel placement to the assembled rest mesh and provide the resulting connectivity/state to the existing solver interface. No solver, FEM, contact, or collision implementation was changed here.

The first sellable asset milestone remains open until G8C/G8D/G8E complete the real garment fitting, repeatable 2D edit/rebuild/simulation loop, and deliverable project/export path.
