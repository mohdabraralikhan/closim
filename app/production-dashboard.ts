import {
  createProductionDependencyGraph,
  buildManufacturingReports,
} from "../src/production/reporting.js";
import {
  serializeProductionSpecification,
  type ProductionSpecification,
} from "../src/production/specification-model.js";
import { generateTechPack, serializeTechPack } from "../src/production/specification.js";
import { serializePatternDocument } from "../src/pattern/cad.js";
import type { GarmentProject } from "../src/garment/project.js";

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "garment";
}

function fingerprint(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `ui/${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function makeSpecification(project: GarmentProject, styleNumber: string): ProductionSpecification {
  const materials = Object.keys(project.materials).sort().map((id) => ({
    id,
    name: id,
    category: "uncategorized",
  }));
  return {
    schemaVersion: 1,
    id: `specification/${project.id}`,
    garmentId: project.id,
    garmentName: project.metadata.name,
    styleNumber,
    revision: String(project.metadata.revision),
    sizeRange: ["base"],
    materials,
    colorways: [],
    panels: project.pattern.panels.map((panel) => ({
      panelId: panel.id,
      ...(panel.materialId ? { materialId: panel.materialId } : {}),
    })),
    construction: project.seams.map((seam) => ({
      id: seam.id,
      kind: "seam",
      title: seam.id,
      data: {
        panelA: seam.sideA.panelId,
        panelB: seam.sideB.panelId,
        segmentIdsA: seam.sideA.segmentIds.join(","),
        segmentIdsB: seam.sideB.segmentIds.join(","),
        ...(seam.stitchCount !== undefined ? { stitchCount: seam.stitchCount } : {}),
      },
    })),
    measurements: project.pattern.panels.map((panel) => ({
      id: `measurement/panel-width/${panel.id}`,
      name: `${panel.name} width`,
      source: { kind: "panel-width", panelId: panel.id },
      unit: "m",
      targetsBySize: {},
    })),
    productionNotes: project.metadata.description ? [project.metadata.description] : [],
    finishingNotes: [],
  };
}

function appendText(parent: HTMLElement, tag: string, text: string, className?: string): HTMLElement {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

export function mountProductionDashboard(
  root: HTMLElement,
  toggle: HTMLButtonElement,
  getProject: () => GarmentProject | null,
): () => void {
  let currentProjectId = "";
  let styleNumber = "";
  root.hidden = true;
  toggle.addEventListener("click", () => {
    root.hidden = !root.hidden;
    toggle.setAttribute("aria-expanded", String(!root.hidden));
  });

  const refresh = (): void => {
    root.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = "Manufacturing readiness";
    root.appendChild(heading);
    const project = getProject();
    if (!project) {
      appendText(root, "p", "Load a garment to inspect its production readiness.");
      return;
    }
    if (project.id !== currentProjectId) {
      currentProjectId = project.id;
      styleNumber = slug(project.id).toUpperCase();
    }

    const styleLabel = document.createElement("label");
    styleLabel.textContent = "Style number ";
    const styleInput = document.createElement("input");
    styleInput.value = styleNumber;
    styleInput.setAttribute("aria-label", "Style number");
    styleInput.addEventListener("input", () => { styleNumber = styleInput.value; });
    styleInput.addEventListener("change", refresh);
    styleLabel.appendChild(styleInput);
    root.appendChild(styleLabel);

    try {
      const specification = makeSpecification(project, styleNumber.trim());
      const techPack = generateTechPack(project, specification);
      const patternFingerprint = fingerprint(serializePatternDocument(project.pattern));
      const materialsFingerprint = fingerprint(JSON.stringify(
        Object.entries(project.materials).sort(([a], [b]) => a.localeCompare(b)),
      ));
      const specificationFingerprint = fingerprint(serializeProductionSpecification(specification));
      const artifacts = createProductionDependencyGraph({
        pattern: { captured: patternFingerprint, current: patternFingerprint },
        materials: { captured: materialsFingerprint, current: materialsFingerprint },
        specification: { captured: specificationFingerprint, current: specificationFingerprint },
        techPack: { captured: techPack.sourceFingerprint, current: techPack.sourceFingerprint },
      });
      const { dashboard, reports } = buildManufacturingReports({
        garmentId: project.id,
        garmentName: project.metadata.name,
        garmentRevision: project.metadata.revision,
        pattern: project.pattern,
        specification,
        techPack,
        artifacts,
      });
      appendText(root, "p", `${dashboard.garmentName} · ${dashboard.styleNumber} · garment r${project.metadata.revision}`);
      appendText(root, "p", `${reports.pattern.pieceCount} pattern panels · ${project.seams.length} seams · base size`);
      appendText(root, "p", `${dashboard.materials.length} assigned physical materials; BOM, grading, markers and production run not yet recorded.`);
      const status = appendText(root, "p", `Readiness: ${dashboard.validation.status.toUpperCase()} · export: ${dashboard.exportStatus}`);
      status.setAttribute("role", "status");

      const artifactsList = document.createElement("ul");
      for (const artifact of dashboard.artifacts) {
        appendText(artifactsList, "li", `${artifact.kind}: ${artifact.state}`);
      }
      root.appendChild(artifactsList);

      const measurements = document.createElement("table");
      const header = document.createElement("tr");
      appendText(header, "th", "Measurement");
      appendText(header, "th", "Size");
      appendText(header, "th", "Measured");
      appendText(header, "th", "Status");
      measurements.appendChild(header);
      for (const measurement of reports.pattern.measurements) {
        const tr = document.createElement("tr");
        appendText(tr, "td", measurement.name);
        appendText(tr, "td", measurement.sizeId);
        appendText(tr, "td", measurement.measuredM === undefined ? "—" : `${measurement.measuredM.toFixed(3)} m`);
        appendText(tr, "td", measurement.status);
        measurements.appendChild(tr);
      }
      root.appendChild(measurements);

      const diagnostics = [...dashboard.validation.errors, ...dashboard.validation.warnings];
      if (diagnostics.length) {
        const list = document.createElement("ul");
        for (const diagnostic of diagnostics) appendText(list, "li", diagnostic);
        root.appendChild(list);
      }
      const preview = document.createElement("details");
      appendText(preview, "summary", "Technical drawing preview");
      const drawing = techPack.drawings[0];
      if (drawing) {
        const image = document.createElement("img");
        image.alt = `Technical drawing for ${drawing.sizeId}`;
        image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(drawing.svg)}`;
        preview.appendChild(image);
      }
      root.appendChild(preview);

      const download = document.createElement("button");
      download.textContent = "Download derived tech pack";
      download.addEventListener("click", () => {
        const bytes = serializeTechPack(techPack);
        const url = URL.createObjectURL(new Blob([bytes], { type: "application/json" }));
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `${slug(specification.styleNumber)}-r${slug(specification.revision)}-tech-pack.json`;
        anchor.click();
        URL.revokeObjectURL(url);
      });
      root.appendChild(download);
    } catch (error) {
      appendText(root, "p", error instanceof Error ? error.message : String(error), "production-error");
    }
  };

  refresh();
  return refresh;
}
