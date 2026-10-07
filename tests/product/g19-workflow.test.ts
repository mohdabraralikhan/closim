// G19 FINAL — commercial product acceptance.
//
// Two-panel work top -> released revision -> product (TOP-001) with SKU,
// XS–XXL sizes, DXF/SVG/native/grading/marker/README bundles, generated
// previews, license metadata -> validate -> ready -> published -> catalog.
// Then: garment edit cannot move the release (explicit re-release only),
// artifacts regenerate with new checksums, everything persists.
import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { movePoint, serializePatternDocument } from "../../src/pattern/cad.js";
import { createGarmentProject, serializeGarmentProject } from "../../src/garment/project.js";
import { makeCapsuleAvatar } from "../../src/garment/avatar.js";
import {
  addAllowance,
  addCutLine,
  addGrainline,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { exportProductionPackage } from "../../src/cad/export.js";
import { renderTechSheet } from "../../src/cad/techsheet.js";
import { createFabric, createCutPlan, addCutItem, defaultNestingConstraint } from "../../src/marker/model.js";
import { MarkerWorkspace } from "../../src/marker/workspace.js";
import {
  addCoreSizes,
  buildGradingFixture,
  buildSideSeam,
} from "../grading/fixtures.js";
import { addSize, createSize } from "../../src/grading/model.js";
import { serializeGradingDocument } from "../../src/grading/serialize.js";
import { validateGradingDocument } from "../../src/grading/validate.js";
import {
  addArtifact,
  addVariant,
  attachPreview,
  bumpProductRevision,
  createProduct,
  deserializeProduct,
  fingerprintPayload,
  isPinnedCurrent,
  markArtifact,
  serializeProduct,
  setProductStatus,
} from "../../src/product/product.js";
import {
  addProduct,
  createCatalog,
  productCard,
  searchCatalog,
  serializeCatalog,
  deserializeCatalog,
} from "../../src/product/catalog.js";
import { createLicense, customerSummary } from "../../src/product/licensing.js";
import { generatePreviews, previewFilename } from "../../src/product/previews.js";

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

const COTTON = {
  arealDensityKgM2: 0.15,
  thickness: 0.001,
  stretchWarp: 20000,
  stretchWeft: 20000,
  stretchCoupling: 0,
  shear: 5000,
  bendWarp: 1e-5,
  bendWeft: 1e-5,
  damping: 0.001,
};

describe("G19 final integration — commercial product", () => {
  it("releases, publishes, catalogs, and version-protects a garment product", () => {
    // Garment: two-panel top from the grading master + side seam.
    const g = buildGradingFixture();
    let grading = addCoreSizes(g.doc);
    for (const size of [
      { id: "size/xs", label: "XS", displayName: "Extra Small" },
      { id: "size/xl", label: "XL", displayName: "Extra Large" },
      { id: "size/xxl", label: "XXL", displayName: "Extra Extra Large" },
    ]) {
      grading = addSize(grading, createSize(size));
    }
    expect(validateGradingDocument(grading)).toEqual([]);
    const sizeIds = grading.sizeSet.sizes.map((s) => s.id);
    expect(sizeIds).toEqual(["size/s", "size/m", "size/l", "size/xs", "size/xl", "size/xxl"]);

    const avatar = makeCapsuleAvatar({ radiusM: 0.15, cylinderLengthM: 0.5, center: [0.2, 1.0, 0] });
    const garment = createGarmentProject("garment/top", "Work Top", g.document, {
      seams: [buildSideSeam(g)],
      placements: [
        { panelId: g.ids.frontPanel, translation: [0, 1, 0.2], yawRad: 0 },
        { panelId: g.ids.backPanel, translation: [0.8, 1, -0.2], yawRad: Math.PI },
      ],
      avatar,
      materials: { cotton: COTTON, "default-material": COTTON },
    });
    const garmentFingerprint = fingerprintPayload(serializeGarmentProject(garment));

    // Production engineering to READY (allowances, grainlines, metadata).
    let prod = createProductionSet();
    for (const panel of garment.pattern.panels) {
      const loop = panel.boundaryLoops.find((l) => l.role === "outer")!;
      prod = addAllowance(prod, panel.id, loop.id, 0.01).set;
      const grain = centeredGrainline(garment.pattern, panel.id);
      prod = addGrainline(prod, panel.id, grain.from, grain.to).set;
      prod = setPanelMeta(prod, { panelId: panel.id, cutQuantity: 1 });
      prod = addCutLine(prod, panel.id, loop.id, "sewing").set;
    }
    const pkg = exportProductionPackage(garment.pattern, garment.seams, prod, { garmentName: "Work Top" });
    expect(pkg.readiness.state).toBe("READY_FOR_EXPORT");

    // Marker artifacts from a real nesting run.
    let plan = createCutPlan("cut/top", "Top order", grading.id);
    const i1 = addCutItem(plan, { panelId: g.ids.frontPanel, sizeId: "size/m", quantity: 1, mirror: "allowed" });
    plan = i1.plan;
    const fabric = createFabric({ id: "fabric/cotton", name: "Cotton", widthM: 1.5, lengthM: 10, materialType: "cotton" });
    const workspace = MarkerWorkspace.open(grading, plan, fabric, defaultNestingConstraint(), "Top order", "marker/top", () => 0);
    const nest = workspace.optimize([0]);
    expect(nest.best.result.unplaced).toEqual([]);

    // Payloads (deterministic) + checksums.
    const payloads: Record<string, string> = {
      "top.dxf": pkg.dxf.dxf,
      "top.tech.svg": renderTechSheet(garment.pattern, prod, { title: "Work Top" }).svg,
      "top.native.json": serializeGarmentProject(garment),
      "top.grading.json": serializeGradingDocument(grading),
      "top.marker.json": JSON.stringify(nest.best.result.placements),
      "README.txt": "Work Top pattern.\nSizes: XS-XXL.\nFormats: DXF, SVG, native.\n",
    };

    // Released revision -> product.
    let product = createProduct({
      id: "product/work-top",
      name: "Classic Work Top",
      sku: "TOP-001",
      category: "shirts",
      author: "Atelier",
      garment: { projectId: "proj/1", garmentId: garment.id, garmentRevision: 1, garmentFingerprint },
      sizes: sizeIds,
      formats: ["dxf", "svg", "native", "grading", "marker", "readme"],
      license: createLicense({ type: "commercial-single", allowsCommercialUse: true, supportContact: "support@example.com", version: "1.0" }),
      description: "A classic two-panel work top with graded sizes XS-XXL.",
      tags: ["top", "workwear"],
    });
    const artifactKinds: Array<[string, "dxf" | "svg" | "native" | "grading" | "marker" | "readme", string[]]> = [
      ["top.dxf", "dxf", sizeIds],
      ["top.tech.svg", "svg", []],
      ["top.native.json", "native", []],
      ["top.grading.json", "grading", sizeIds],
      ["top.marker.json", "marker", ["size/m"]],
      ["README.txt", "readme", []],
    ];
    for (const [filename, type, sizes] of artifactKinds) {
      const added = addArtifact(product, {
        filename, type, sizeIds: sizes, format: filename.split(".").pop()!,
        revision: 1, checksum: sha256(payloads[filename]), generator: "test-harness",
      });
      product = added.product;
    }

    // Previews from deterministic render stubs (real G15 path injects pixels).
    const views = ["front", "back", "three-quarter"] as const;
    const batch = generatePreviews(product, [...views], (view) => {
      const svg = renderTechSheet(garment.pattern, prod, { title: `Work Top ${view}` }).svg;
      const filename = previewFilename(product.slug, product.revision, view);
      return { filename, checksum: sha256(svg), widthPx: 1024, heightPx: 1024 };
    }, { lighting: "product" });
    expect(batch.failed).toEqual([]);
    for (const generated of batch.generated) {
      const added = addArtifact(product, {
        id: generated.id, filename: generated.filename, type: "preview",
        sizeIds: [], format: "png", revision: 1, checksum: generated.checksum, generator: "g15-stub",
      });
      product = added.product;
      product = attachPreview(product, generated.id, generated.view === "front");
    }
    expect(product.thumbnailArtifactId).toBe("product/work-top/preview/front");

    // Variants: full multi-size bundle + single-size M.
    const allIds = product.artifacts.map((a) => a.id);
    const full = addVariant(product, {
      name: "Full XS-XXL bundle", kind: "multi-size",
      sizeIds, formats: ["dxf", "svg", "native"], artifactIds: allIds,
    });
    product = full.product;
    const single = addVariant(product, {
      name: "Size M only", kind: "single-size", sizeIds: ["size/m"],
      formats: ["dxf"], artifactIds: product.artifacts.filter((a) => a.type === "dxf").map((a) => a.id),
    });
    product = single.product;

    // Validate -> ready -> published.
    product = setProductStatus(product, "ready");
    product = setProductStatus(product, "published");
    expect(product.status).toBe("published");

    // Catalog: detail, search, card.
    let catalog = createCatalog("catalog/main", "Main", ["shirts", "dresses", "trousers"]);
    catalog = addProduct(catalog, product);
    expect(searchCatalog(catalog, { query: "TOP-001" })).toHaveLength(1);
    expect(searchCatalog(catalog, { status: ["published"], format: "dxf" })).toHaveLength(1);
    expect(searchCatalog(catalog, { category: "dresses" })).toHaveLength(0);
    const card = productCard(product);
    expect(card).toMatchObject({ sku: "TOP-001", status: "published", revision: 1 });
    const summary = customerSummary(product, product.artifacts);
    expect(summary.filesIncluded).toContain("top.dxf");
    expect(summary.license.commercialUse).toBe(true);
    expect(summary.sizes).toHaveLength(6);

    // Version safety: garment edit cannot move the release.
    const edited = movePoint(garment.pattern, g.ids.frontPanel, g.ids.points.A, [0, 0.05]);
    const editedFingerprint = fingerprintPayload(serializeGarmentProject({ ...garment, pattern: edited }));
    expect(editedFingerprint).not.toBe(garmentFingerprint);
    expect(isPinnedCurrent(product, 1, editedFingerprint)).toBe(false);
    expect(isPinnedCurrent(product, 1, garmentFingerprint)).toBe(true);
    // Artifacts still verify against their recorded checksums.
    for (const artifact of product.artifacts) {
      if (artifact.type === "preview") continue;
      const name = artifact.filename;
      const payload = name === "top.marker.json"
        ? JSON.stringify(nest.best.result.placements)
        : payloads[name as keyof typeof payloads];
      expect(sha256(payload as string)).toBe(artifact.checksum);
    }

    // Explicit re-release: revision bumps, artifacts go stale, regeneration renews.
    product = bumpProductRevision(product, {
      projectId: "proj/1", garmentId: garment.id, garmentRevision: 2, garmentFingerprint: editedFingerprint,
    });
    expect(product.revision).toBe(2);
    expect(product.status).toBe("draft");
    expect(product.artifacts.every((a) => a.status === "stale")).toBe(true);
    const renewed = markArtifact(product, product.artifacts[0].id, "current");
    expect(renewed.artifacts[0].status).toBe("current");

    // Persistence: product + catalog round-trip byte-identically.
    expect(serializeProduct(deserializeProduct(serializeProduct(product)))).toBe(serializeProduct(product));
    expect(serializeCatalog(deserializeCatalog(serializeCatalog(catalog)))).toBe(serializeCatalog(catalog));
  });
});
