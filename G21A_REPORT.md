# G21A Completion Report — Customer Digital Library

## Summary

G21A builds the customer account area: a persistent, searchable library of every
legitimately purchased garment product, with organization (search / filter / sort /
favorites / recents), full owned-product detail, and a download entry point that
delegates — never bypasses — the G20 entitlement/delivery boundary.

## Deliverables

### `src/commerce/library.ts` (new)

| Area | API |
|---|---|
| Library build | `buildLibrary(input) → Library` (customerId-scoped, deterministic, sorted by productId) |
| Update state | `computeUpdateState(facts)` — precedence: `revoked > download-unavailable > deprecated > update-available > current`; eligibility reported separately (`updateAvailable` vs `eligibleForUpdate` via G20 `updateEligible`) |
| Product detail | `libraryProductDetail({…, entitlementId})` — purchased release, sizes/formats (variant-scoped), downloads, license, update block, support block (ids + contact only) |
| Download | `requestLibraryDownload(...)` — library scoping checks, then delegates to G20 `authorizeDownload` (ownership, usability, release/artifact, single-use token, rate limit) |
| Organization | `searchLibrary`, `filterLibrary` (states/categories/query/favoritesOnly/usableOnly), `sortLibrary` (5 deterministic sorts, productId tiebreak), `organizeLibrary`, `recentlyPurchased`, `recentlyDownloaded` |
| Favorites | `createFavorites` / `addFavorite` / `removeFavorite` / `toggleFavorite` / `isFavorite` (idempotent, timestamp-stable no-ops) |
| Persistence | `serializeLibrary` / `deserializeLibrary`, `serializeFavorites` / `deserializeFavorites` (canonical JSON, versioned, dedup on read) |

Entry display fields: thumbnail, name, sku, category, purchase date, owned release,
latest release, available downloads (per-artifact availability + reason), license,
update state/eligibility, last-downloaded timestamp.

### `src/commerce/delivery.ts` (hardened — G20 security fix)

`authorizeDownload` now also refuses:
- a release belonging to **another product** than the entitlement's, and
- a release revision **not covered** by the entitlement's revision + update policy
  (`updateEligible`) — forged/foreign releases never deliver, and update
  eligibility cannot be bypassed by pointing at a newer release.

This closes the G21F "wrong release" / "update eligibility cannot be forged"
attack surface at the delivery boundary itself. All 13 G20 tests still pass.

### `src/commerce/index.ts`

Added `export * from "./library.js"`.

### `tests/commerce/g21a-library.test.ts` (13 tests, all required scenarios)

1. **empty library** — empty build, safe organization, serialization + input validation
2. **display fields** — full entry shape, license, downloads, thumbnails, deterministic rebuild, no `projectId`/`garmentId`/`proj/` leak
3. **revoked + expired entitlements** — `revoked` vs `download-unavailable` states, blocked downloads, delivery refusal
4. **updated product** — `update-available` state, `original-only` ⇒ not eligible / `free-updates` ⇒ eligible, explicit registry override, revoked-newest fallback, registry value validation
5. **deleted product** — record retained (`productMissing`, id fallback), owned release still delivers (`deprecated`), no product + no release ⇒ `download-unavailable`, archived ⇒ `deprecated`
6. **missing preview** — no thumbnail, library renders
7. **repeated downloads** — latest *own completed* success tracked (failed/foreign records ignored), recents ordering, entry invariant
8. **large library** — 120 entries: search (case-insensitive/trim), category/state filters, sorts, organized limit, limit validation
9. **account changes + isolation** — suspended account doesn't rewrite the view; other customers' entitlements invisible; foreign order snapshot never contributes purchase dates
10. **favorites** — idempotent add/remove/toggle, timestamp stability, round-trip, validation, `favoritesOnly` filter
11. **search/filter/sort determinism** — all sorts, tie-breaks, runtime validation of unknown states/sorts, no input mutation
12. **product detail** — release/sizes/formats/license/updates/support assertions, ownership boundary, unknown entitlement refusal
13. **download delegation** — authorize→fulfill happy path, repeat + replay, wrong customer/product/revision refusals, rate limit through delegation

## Verification

- `npx vitest run tests/commerce` → **26/26 green** (13 G21A + 13 G20)
- `npx tsc --noEmit` → **0 errors** in `src/commerce/`, `tests/commerce/` (pre-existing baseline errors remain only in other tracks' files: `tests/cad/g13-adversarial.test.ts`, `tests/construction/g16-workflow.test.ts`, `tests/production/workflow.test.ts`)
- Broad gate (`vitest run` minus webgpu): **106 files / 936 tests — all passed**

## Design Notes

- **Derived, never authoritative**: the library reads entitlements/orders/products/releases and grants nothing; every download flows through `authorizeDownload`.
- **Privacy by construction**: entries scoped to one `customerId`; order snapshots from other customers are ignored; no garment/project/paths in any customer-facing record; `libraryProductDetail` refuses foreign entitlements.
- **Update state ≠ eligibility**: state is factual (newer release exists); policy eligibility is a separate field (G21C composes on top of both).
- **Precedence documented and tested**: `revoked > download-unavailable > deprecated > update-available > current`.

## Cross-track observation (for G21C)

`bumpProductRevision` (G19) leaves old artifacts `stale` at the old revision, and
`validateProduct` requires **all** artifacts current + revision-matched for
Ready/Published — so the public API currently has **no path to republish a
product at revision 2**, and `createRelease` then cannot freeze a rev-2 release
(product is stuck in draft). G21A's re-release fixtures therefore build the
completed state through the validating `deserializeProduct` + real
`createRelease`. **G21C (release system) should add the republish transition**
(e.g. prune/retire stale artifacts on re-release) so `Release 1.0 → 1.1` works
end-to-end for the G21 FINAL release test. `product.ts` was intentionally left
untouched by this track to avoid concurrent-edit conflicts.
