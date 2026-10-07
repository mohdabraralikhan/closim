# G20 Completion Report — Commerce Backend

## Summary

G20 adds the commerce layer: customers, orders, payments/webhooks, entitlements,
immutable product releases, secure delivery, and storefront data. It sits strictly
on top of the G19 product records and never imports the garment/solver engine.

## Deliverables

### `src/commerce/`

| Module | Responsibility |
|---|---|
| `customers.ts` | Customer accounts, provider-bound `authenticate`, account status, serialization |
| `orders.ts` | Order/order-item model, integer-minor-unit pricing, state machine `pending → paid/failed/canceled → refunded`, revision pinning, serialization |
| `entitlements.ts` | Paid-order-only `issueEntitlement`, expiry/revocation/update policy, immutable `ProductRelease`, `createRelease`/`revokeReleaseArtifact`/`resolveReleaseArtifact`, serializers |
| `payments.ts` | Payment records, `PaymentProvider` interface, HMAC-signed `WebhookEvent`, `applyWebhook` (idempotent, seen-event set, amount/checkout mismatch guards, refund-before-confirmation held), `StubPaymentProvider` |
| `delivery.ts` | `authorizeDownload` (single-use token ledger + TTL + rate-limit bucket + customer/release/artifact binding), `fulfillDownload` (sha256 checksum verification, completed/failed records), `memoryLedger`, `tokenBucketLimiter` |
| `storefront.ts` | `productPage` (purchasable/unavailable reason, files, previews), `resolveVariant`, `quoteVariant`, `purchaseState` (available/owned/discontinued), `customerLibrary` |
| `index.ts` | Barrel |

### Tests (`tests/commerce/`)

- **`commerce.test.ts`** (12 tests): authentication + suspended accounts; order pricing/state-machine/bad-transition guards; webhook confirm/forgery/replay/mismatch/out-of-order/refund-before-confirm; entitlement issue/expiry/revocation/update policy; release immutability + revocation; download authorization attacks (wrong customer, revoked, expired, rate-limit, replay) and checksum-verified fulfillment; storefront paging/variant/quote/library.
- **`g20-workflow.test.ts`** (1 test, end-to-end): garment → published product → storefront page/variant/quote → customer → pending order → checkout → *untrusted* browser claim yields nothing → signed webhook → paid order → entitlement → secure download (checksum verified) → purchase library → re-download → new product revision (`original-only` policy pins old entitlement, `free-updates` eligible) → signed refund → order refunded + entitlement revoked + download refused.

## Verification

- `npx tsc --noEmit`: **0 errors** in `src/commerce/` and `tests/commerce/`
- `npx vitest run tests/commerce`: **13/13 green** (2 files)
- Broad gate (`vitest run` minus webgpu): **922–923/923 tests passed**; the 1–2 intermittent failures were in other agents' in-flight files (`tests/construction/g16-workflow.test.ts`, `tests/production/workflow.test.ts`) and pass in isolation — not touched by this work.

## Architecture Notes

- Commerce imports product records only (`src/product/`), never garment, FEM, or solver code (isolation assertion in workflow test).
- All money is integer minor units with explicit currency; all timestamps explicit ISO/`nowMs`.
- Immutable updates throughout (clone-on-write), deterministic serialization (sorted keys) for round-trip tests.
- Checksums via `node:crypto` sha256; webhook auth via HMAC-SHA256 — no hardcoded secrets.
- Solver/FEM untouched (hard rule respected).
