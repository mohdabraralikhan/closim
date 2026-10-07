// G14 — true-scale marker SVG.
//
// Renders a nested marker as an SVG with 1 user unit == 1 millimetre (same
// contract as the G13C pattern SVG): the frame is the fabric, pieces are
// drawn at their placed positions, labels carry piece id + rotation +
// mirror flag, and the header reports the efficiency numbers. Deterministic:
// identical markers produce identical bytes. Marker y-up is flipped to SVG
// y-down exactly like the pattern exporter.

import type { Vec2 } from "./geom.js";
import { convertM, requireExportUnits, type ExportUnits } from "./export-ir.js";
import type { Marker } from "./marker.js";

export interface MarkerSvgOptions {
  units?: ExportUnits;
  title?: string;
}

export interface MarkerSvgResult {
  svg: string;
  warnings: string[];
  widthUnits: number;
  heightUnits: number;
}

function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmt(n: number): string {
  const r = Math.round(n * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
}

/** Render the marker to a true-scale SVG string. */
export function exportMarkerSVG(marker: Marker, opts: MarkerSvgOptions = {}): MarkerSvgResult {
  const units = requireExportUnits(opts.units ?? "mm", "marker SVG");
  const u = (metres: number): number => convertM(metres, units);
  const warnings = [
    "Marker SVG is true scale at 100% (1 user unit = 1 mm); never rescale before printing without a calibration check",
    ...marker.warnings,
  ];

  const margin = 5; // mm paper margin around the frame
  const header = 12; // mm header band for the title/efficiency line
  const widthU = u(marker.widthM) + margin * 2;
  const heightU = u(marker.lengthM) + margin * 2 + header;

  const X = (x: number): number => margin + u(x);
  const Y = (y: number): number => margin + header + u(marker.lengthM - y);

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(widthU)}mm" height="${fmt(heightU)}mm" viewBox="0 0 ${fmt(widthU)} ${fmt(heightU)}" font-family="monospace">`,
  );
  parts.push(
    `<metadata data-format="closim-marker-svg" data-units="mm" data-true-scale="1" ` +
    `data-style="${esc(marker.style.garmentName)}" data-revision="${marker.style.revision}" ` +
    `data-efficiency="${fmt(marker.efficiency.efficiency * 100)}" data-app="closim G14"/>`,
  );
  parts.push(
    `<style>.frame{fill:#fdfdfd;stroke:#111;stroke-width:0.5}.piece{fill:#eef2ff;fill-opacity:0.35;stroke:#1e293b;stroke-width:0.4}` +
    `.plabel{fill:#1e293b;font-size:2.6px}.head{fill:#111;font-size:3.4px;font-weight:bold}.sub{fill:#374151;font-size:2.6px}` +
    `.declined{fill:#b91c1c;font-size:2.8px}</style>`,
  );

  // Header: title + efficiency line.
  const title = opts.title ?? marker.style.garmentName;
  parts.push(
    `<text class="head" x="${fmt(margin)}" y="${fmt(margin + 3.4)}">${esc(title)} — marker ${fmt(u(marker.widthM))} x ${fmt(u(marker.lengthM))} mm</text>`,
  );
  parts.push(
    `<text class="sub" x="${fmt(margin)}" y="${fmt(margin + 7.6)}">efficiency ${fmt(marker.efficiency.efficiency * 100)}% | pieces ${marker.placements.length}` +
    ` | piece area ${fmt(u(marker.efficiency.pieceAreaM2) / 1000)} m2 | waste ${fmt(u(marker.efficiency.wasteM2) / 1000)} m2 | warp +X | ${marker.sizes.length > 0 ? esc(marker.sizes.join(", ")) : "single size"}</text>`,
  );
  if (marker.declined.length > 0) {
    parts.push(
      `<text class="declined" x="${fmt(margin)}" y="${fmt(margin + 11)}">DECLINED: ${esc(marker.declined.map((d) => `${d.pieceId} (${d.reason})`).join("; "))}</text>`,
    );
  }

  // Frame.
  parts.push(
    `<rect class="frame" x="${fmt(X(0))}" y="${fmt(Y(marker.lengthM))}" width="${fmt(u(marker.widthM))}" height="${fmt(u(marker.lengthM))}"/>`,
  );

  // Pieces.
  const fpById = new Map(marker.footprints.map((f) => [f.pieceId, f.ring]));
  for (const p of marker.placements) {
    const ring = fpById.get(p.pieceId);
    if (!ring) continue; // cannot happen for a consistent marker; skip defensively
    const d = ring.map((q, i) => `${i === 0 ? "M" : "L"} ${fmt(X(q[0]))} ${fmt(Y(q[1]))}`).join(" ") + " Z";
    parts.push(`<path class="piece" data-piece="${esc(p.pieceId)}" data-rotation="${p.rotation}" data-mirror="${p.mirrored ? "1" : "0"}" d="${d}"/>`);
    // Label at the footprint bbox centroid.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const q of ring as Vec2[]) {
      if (q[0] < minX) minX = q[0];
      if (q[1] < minY) minY = q[1];
      if (q[0] > maxX) maxX = q[0];
      if (q[1] > maxY) maxY = q[1];
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const suffix = `${p.mirrored ? " M" : ""}${p.rotation !== 0 ? ` R${p.rotation}` : ""}`;
    parts.push(
      `<text class="plabel" text-anchor="middle" x="${fmt(X(cx))}" y="${fmt(Y(cy))}">${esc(p.pieceId + suffix)}</text>`,
    );
  }

  parts.push("</svg>");
  return { svg: parts.join("\n"), warnings, widthUnits: widthU, heightUnits: heightU };
}
