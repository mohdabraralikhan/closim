# G21 Handoff — What's Left (written end of Day 1, ~12h in, G8 → G21)

Status snapshot written after G21B landed. **Next session picks up at G21C.**
G21 FINAL (integration test + final report) is the last step — per the owner,
"finish that and that will be the end of implementation."

---

## 1. Verified state right now (do not redo)

| Track | Deliverable | Tests |
|---|---|---|
| G20 commerce backend | `src/commerce/{customers,orders,entitlements,payments,delivery,storefront}.ts`, `G20_COMPLETION_REPORT.md` | 13/13 green (`commerce.test.ts` 12, `g20-workflow.test.ts` 1) |
| G21A customer library | `src/commerce/library.ts`, hardened `delivery.ts`, `G21A_REPORT.md` | 13/13 green (`g21a-library.test.ts`) |
| G21B receipts/licensing | `src/commerce/receipts.ts`, `tests/commerce/fixtures.ts` (shared fixtures), barrel export added | 8/8 green (`g21b-receipts.test.ts`) |

- **Full commerce suite: 4 files / 34 tests, all passing.**
- **`npx tsc --noEmit`: zero errors in any `src/commerce/**` or `tests/commerce/**` file.**
- Last known broad gate (before G21B): 106 files / 936 tests all passed.

### G21B specifics (done)
- `orderHistory`, `findOrder` (identical error for miss + foreign access — enumeration-safe),
  `createReceipt`, `createLicenseDocument`, `serializeReceipt`, `serializeLicenseDocument`.
- Update-state precedence (G21A): `revoked > download-unavailable > deprecated > update-available > current`;
  `eligibleForUpdate` (G20 `updateEligible`) is separate and factual.
- License docs are frozen/immutable: renaming + bumping the product later leaves issued
  receipts/licenses byte-identical (tested).
- `PatternCadError` codes must come from `PatternDiagnosticCode` in `src/pattern/cad.ts`
  (`invalid-document`, `missing-reference`, …). **`missing-license` is NOT valid** — use
  `"invalid-document"` with a message containing `license missing`.

---

## 2. Remaining work (in order)

### G21C — Releases + republish path  ← START HERE
The G19 gap (documented in `G21A_REPORT.md`): `bumpProductRevision` leaves stale rev-1
artifacts; `validateProduct` requires all artifacts current+rev-matched for `ready`, so
there is currently **no public path to republish** (`Release 1.0 → 1.1`).

1. Additive helper in `src/product/product.ts` (do NOT change existing function behavior):
   ```ts
   pruneSupersededArtifacts(product, now) // drop stale artifacts; remap variant.artifactIds
   // by filename (PatternCadError "missing-reference" if unreplaced);
   // filter dangling previewIds; clone-on-write.
   ```
2. New `src/commerce/releases.ts`:
   - `ReleaseEntry` / `ReleaseCatalog`, `version`, `changeSummary`,
     `affectedFiles` (name/size/format), `compatible|breaking|replacement` compatibility,
     `UpdatePlan = none|free|paid|included` → `planToPolicy(plan, UpdatePolicy)`,
     `updateOffer(...)` returning eligibility + reason,
     `releaseHistory` / `latest` / revoked-release handling.
3. Export from `src/commerce/index.ts`.
4. Tests: real `Release 1.0 → 1.1` using `pruneSupersededArtifacts`, old purchase stays
   valid, policy honored (free vs paid vs included), new/removed/changed files traceable,
   immutable rev-2 artifacts, previews updated.

### G21D — Support (`src/commerce/support.ts`)
- Tickets with configurable categories; auto-context pulled from entitlement/order/product/
  release/variant; status machine `open → in-progress → waiting → resolved → closed`
  with reopen rules; duplicate-ticket refusal; attachment refs; privacy (foreign-customer
  access throws same-shape error as miss).
- Tests incl. scenario from FINAL: "PDF is missing a size" ticket auto-attaches the
  right product/release context.

### G21E — Engagement (`src/commerce/engagement.ts`)
- Purchase-gated reviews → pending moderation → approved → **public view strips
  `customerId`**; verified designation recomputed at view time (entitlement still valid);
  rating averaging; wishlist CRUD; notifications with dedupe.
- Tests: non-purchaser rejected, duplicate review refused, manipulated rating bounds
  rejected, unmoderated review never public.

### G21F — Adversarial (`tests/commerce/g21f-adversarial.test.ts`)
Read-only attacks across all tracks: cross-customer orders/products/library/receipts/tickets,
wrong release/variant download, expired/revoked/replay downloads, update-eligibility
forgery, non-purchaser/duplicate/manipulated reviews, attachment abuse, receipt/license
tampering. Assert critical invariants (no credential material in outputs, no enumeration).

### G21 FINAL — Integration (the end of implementation)
- `tests/commerce/g21-final.test.ts`:
  1. Full acceptance flow: purchase → library → download → license → re-download →
     update 1.0 → 1.1 (via G21C republish) → download new release.
  2. Support auto-attach scenario ("PDF is missing a size").
  3. Release test: old purchase valid, update policy works, old files traceable,
     new artifacts immutable, previews updated.
- Run the **broad gate** (below) and record exact totals/failures.
- Write `G21_FINAL_REPORT.md` (format: like `G20_COMPLETION_REPORT.md`/`G21A_REPORT.md`).

---

## 3. Hard rules (violating these breaks other agents' work)

- **Never modify solver/FEM/collision/GPU internals.** Solver untouched through all commerce work.
- **Never touch other agents' baseline-error files** (pre-existing tsc errors, not ours):
  - `src/cad/index.ts` (duplicate re-exports of `export.js` members)
  - `tests/cad/g13-adversarial.test.ts` (~L170)
  - `tests/construction/g16-workflow.test.ts` (~L505/521)
  - `tests/production/workflow.test.ts` (~L373–377)
  - `tests/view/pick.test.ts` (call sites pass 12 args, signature has 11)
- **Do NOT implement** (explicitly out of scope): multi-vendor marketplace, seller
  onboarding/payouts, advanced advertising, subscriptions, social network, AI recommendations.
- Commerce imports **only product records**, never garment/solver engine.
- Money = integer minor units + explicit currency. No hardcoded secrets (HMAC/sha256 only).
- Clone-on-write; canonical JSON (sorted keys, -0 normalized); explicit `now`/`nowIso`
  params; runtime validation of array/id inputs with clear messages.
- Do not rewrite `tests/commerce/fixtures.ts` inline fixtures in `g21a` — g21a keeps its own;
  **B–F and FINAL tests use `fixtures.ts`** (`purchase({...})` returns
  `{ product, variantId, pdfArtifactId, previewArtifactId, order, entitlement, release, payload }`).

## 4. Commands (PowerShell; no `head`/`grep` — use `Select-String`)

```powershell
npx vitest run tests/commerce --reporter=basic          # 34/34 expected now
npx vitest run tests/commerce/g21c-releases.test.ts --reporter=basic
npx tsc --noEmit 2>&1 | Select-String "commerce"        # expect no output
# Broad gate (full regression; use the excludes — webgpu/node_modules are not runnable here):
npx vitest run --exclude "**/webgpu/**" --exclude "**/node_modules/**" --reporter=basic
```

## 5. Key files

- `src/commerce/`: `customers, orders, entitlements, payments, delivery, storefront, library, receipts, index`
- `src/product/product.ts`: G19 model — add `pruneSupersededArtifacts` here (G21C)
- `src/pattern/cad.ts`: `PatternCadError` + `PatternDiagnosticCode` union (L109+)
- `tests/commerce/`: `commerce.test.ts`, `g20-workflow.test.ts`, `g21a-library.test.ts`,
  `g21b-receipts.test.ts`, `fixtures.ts`
- Reports so far: `G20_COMPLETION_REPORT.md`, `G21A_REPORT.md` → pending: `G21_FINAL_REPORT.md`
