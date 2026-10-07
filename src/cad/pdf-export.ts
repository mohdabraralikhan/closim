// G13C — PDF pattern export (full-scale + tiled home-printer output).
//
// Emits a dependency-free, deterministic PDF 1.4: one page per layout page,
// vector content streams, a configurable page-size table, tiling with
// registration/corner marks, page numbers, neighbouring-page indicators,
// a measurable calibration square, and per-page scale text.
//
// Scale safety: full-scale mode never fits-to-page; each page carries
// "SCALE 100% (1 unit = 1 mm)" text, and every tiled page carries a
// calibration square that must measure exactly `calibrationMm` when printed.
// The Info dictionary embeds units and intended scale for machine checks.
//
// Geometry: boundaries, allowances, notches, grainlines, folds, drills,
// internals, labels. Arcs are flattened at a declared sagitta (0.1 mm) —
// the only approximated geometry, reported in every export's warnings.

import type { Vec2 } from "./geom.js";
import {
  convertM,
  requireExportUnits,
  type ExportIR,
  type ExportPanelIR,
  type ExportUnits,
} from "./export-ir.js";

// ---------------------------------------------------------------------------
// Page-size table (configurable; nothing about paper is assumed)
// ---------------------------------------------------------------------------

export interface PageSize {
  id: string;
  /** Width in mm (portrait). */
  widthMm: number;
  /** Height in mm (portrait). */
  heightMm: number;
}

/** ISO A-series presets; consumers may pass their own table. */
export const PAGE_PRESETS: Record<string, PageSize> = {
  A4: { id: "A4", widthMm: 210, heightMm: 297 },
  A3: { id: "A3", widthMm: 297, heightMm: 420 },
  A2: { id: "A2", widthMm: 420, heightMm: 594 },
  A1: { id: "A1", widthMm: 594, heightMm: 841 },
  A0: { id: "A0", widthMm: 841, heightMm: 1189 },
};

// ---------------------------------------------------------------------------
// Geometry flattening (sagitta-bounded, deterministic)
// ---------------------------------------------------------------------------

type Ring = readonly { a: Vec2; b: Vec2; kind: string; center?: Vec2; radiusM?: number; a0Rad?: number; sweepRad?: number }[];

function sampleRing(ring: Ring, sagittaMm: number): Vec2[] {
  const pts: Vec2[] = [];
  for (const e of ring) {
    if (e.kind === "arc" && e.center && e.radiusM !== undefined && e.a0Rad !== undefined && e.sweepRad !== undefined) {
      const rMm = e.radiusM * 1000;
      const step = 2 * Math.acos(Math.max(-1, Math.min(1, 1 - Math.min(sagittaMm, rMm) / rMm)));
      const n = Math.max(2, Math.min(720, Math.ceil(Math.abs(e.sweepRad) / step)));
      for (let i = 0; i < n; i++) {
        const ang = e.a0Rad + e.sweepRad * (i / n);
        pts.push([e.center[0] + e.radiusM * Math.cos(ang), e.center[1] + e.radiusM * Math.sin(ang)]);
      }
    } else {
      pts.push(e.a);
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// PDF primitives
// ---------------------------------------------------------------------------

const PT_PER_MM = 72 / 25.4;

function pt(mm: number): number {
  return Math.round(mm * PT_PER_MM * 1000) / 1000;
}

function escPdf(text: string): string {
  return text.replace(/[\\()]/g, (m) => `\\${m}`).replace(/[^\x20-\x7E]/g, "?");
}

function pdfPath(pts: Array<[number, number]>, close: boolean): string {
  if (pts.length === 0) return "";
  let d = `${pt(pts[0][0])} ${pt(pts[0][1])} m\n`;
  for (let i = 1; i < pts.length; i++) {
    d += `${pt(pts[i][0])} ${pt(pts[i][1])} l\n`;
  }
  if (close) d += "h\n";
  return d;
}

function stroke(d: string, widthMm: number): string {
  return `${d}${pt(widthMm)} w S\n`;
}

function pdfText(x: number, y: number, sizeMm: number, text: string): string {
  return `BT /F1 ${pt(sizeMm).toFixed(2)} Tf ${pt(x).toFixed(2)} ${pt(y).toFixed(2)} Td (${escPdf(text)}) Tj ET\n`;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function panelBBox(panel: ExportPanelIR): { min: Vec2; max: Vec2 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const scan = (ring: Ring): void => {
    for (const e of ring) {
      for (const p of [e.a, e.b]) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      if (e.kind === "arc" && e.center && e.radiusM !== undefined) {
        const c = e.center, r = e.radiusM;
        if (c[0] - r < minX) minX = c[0] - r;
        if (c[0] + r > maxX) maxX = c[0] + r;
        if (c[1] - r < minY) minY = c[1] - r;
        if (c[1] + r > maxY) maxY = c[1] + r;
      }
    }
  };
  scan(panel.sewingEdges);
  if (panel.allowanceEdges) scan(panel.allowanceEdges);
  if (panel.cutEdges) scan(panel.cutEdges);
  for (const h of panel.holes) scan(h);
  for (const g of panel.grainlines) scan([{ a: g.from, b: g.to, kind: "line" }]);
  for (const f of panel.folds) scan([{ a: f.a, b: f.b, kind: "line" }]);
  if (!Number.isFinite(minX)) return { min: [0, 0], max: [0, 0] };
  return { min: [minX, minY], max: [maxX, maxY] };
}

interface PlacedPiece {
  panel: ExportPanelIR;
  ox: number; // layout offset in metres (pattern space)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export interface PdfExportOptions {
  /** "A4" | "A3" | ... preset id, or an explicit PageSize. */
  page: string | PageSize;
  /** Page margin in mm. */
  marginMm: number;
  /** Full-scale single page (large format) vs tiled home printing. */
  mode: "full-scale" | "tiled";
  /** Tile overlap in mm (tiled mode). Default 10. */
  overlapMm?: number;
  /** Corner registration marks on tiled pages. Default true. */
  registrationMarks?: boolean;
  /** Calibration square side in mm on every tiled page. Default 50. */
  calibrationMm?: number;
  /** Include page numbering + neighbour indicators. Default true. */
  pageNumbers?: boolean;
  /** Declared units for metadata text. */
  units?: ExportUnits;
}

export interface PdfExportResult {
  pdf: Uint8Array;
  warnings: string[];
  pageCount: number;
  pageSizesMm: Array<{ id: string; widthMm: number; heightMm: number }>;
  units: ExportUnits;
  mode: "full-scale" | "tiled";
}

/**
 * Export the IR as PDF. Full-scale mode produces one true-scale page sized
 * exactly to layout + margins (declared in pageSizesMm); tiled mode
 * deterministically slices the layout into `page`-sized tiles with overlap,
 * registration marks, calibration squares, and page furniture.
 */
export function exportIRToPDF(ir: ExportIR, opts: PdfExportOptions): PdfExportResult {
  const units = requireExportUnits(opts.units ?? "mm", "PDF export");
  const page = typeof opts.page === "string"
    ? (PAGE_PRESETS[opts.page] ?? (() => { throw new Error(`unknown page preset '${String(opts.page)}'`); })())
    : opts.page;
  if (!Number.isFinite(opts.marginMm) || opts.marginMm < 0) {
    throw new Error("PDF marginMm must be a finite non-negative number");
  }
  const marginMm = opts.marginMm;
  const tiled = opts.mode === "tiled";
  const overlapMm = opts.overlapMm ?? 10;
  const calibrationMm = opts.calibrationMm ?? 50;
  const regMarks = opts.registrationMarks ?? true;
  const pageNumbers = opts.pageNumbers ?? true;
  const gapM = 0.1;
  const warnings: string[] = [];
  warnings.push("PDF geometry flattened at 0.1 mm sagitta (declared); DXF/SVG carry exact arcs");
  warnings.push(tiled
    ? `tiled print at 100%: every page carries a ${calibrationMm} mm calibration square — measure before cutting`
    : `full-scale PDF: print at 100% on the declared page size (${page.id}); never allow fit-to-page`);

  // Layout: pieces left-to-right; layout mm-space is (x, up) with y flipped.
  const u = (metres: number): number => convertM(metres, units);
  let cursorX = 0;
  const pieces: PlacedPiece[] = [];
  let minY = Infinity, maxY = -Infinity;
  for (const panel of ir.panels) {
    const bb = panelBBox(panel);
    pieces.push({ panel, ox: cursorX - bb.min[0] });
    cursorX += bb.max[0] - bb.min[0] + gapM;
    minY = Math.min(minY, bb.min[1]);
    maxY = Math.max(maxY, bb.max[1]);
  }
  if (pieces.length === 0) throw new Error("PDF export needs at least one panel");
  const layoutWMm = u(cursorX - gapM);
  const layoutHMm = u(maxY - minY);

  const pageSizesMm: Array<{ id: string; widthMm: number; heightMm: number }> = [];
  const pageContents: string[] = [];

  const emitPage = (spec: { widthMm: number; heightMm: number; originXMm: number; originYMm: number; cols: number; rows: number; col: number; row: number; pageIndex: number }): string => {
    const ops: string[] = [];
    const headerBand = tiled ? 6 : 5;
    const contentTop = spec.heightMm - marginMm - headerBand; // y of layout top
    // Pattern-space (mm, y-up) -> page-space (pt, y-up).
    const conv = (pM: [number, number]): [number, number] => {
      const xmm = u(pM[0]) - spec.originXMm + marginMm;
      const ymm = contentTop - (u(pM[1]) - u(maxY)) + spec.originYMm * -1;
      return [xmm, ymm];
    };
    const convM = (xM: number, yM: number): [number, number] => conv([xM, yM]);
    void convM;
    const flat = (ring: Ring): Vec2[] => sampleRing(ring, 0.1);
    const emitFlat = (ring: Ring, w: number, close = true): void => {
      const pts = flat(ring).map((p) => conv([p[0], p[1]]));
      ops.push(stroke(pdfPath(pts, close), w));
    };
    for (const piece of pieces) {
      const panel = piece.panel;
      const shift = (p: Vec2): [number, number] => [p[0] + piece.ox, p[1]];
      const cut = (panel.cutEdges ?? panel.sewingEdges).map((e) => ({ ...e, a: shift(e.a), b: shift(e.b) }) as typeof e);
      emitFlat(cut, 0.4);
      if (panel.cutEdges && panel.cutSource === "allowance") {
        emitFlat(panel.sewingEdges.map((e) => ({ ...e, a: shift(e.a), b: shift(e.b) }) as typeof e), 0.25);
      }
      if (panel.allowanceEdges && !(panel.cutEdges && panel.cutSource === "allowance")) {
        emitFlat(panel.allowanceEdges.map((e) => ({ ...e, a: shift(e.a), b: shift(e.b) }) as typeof e), 0.3);
      }
      for (const hole of panel.holes) emitFlat(hole.map((e) => ({ ...e, a: shift(e.a), b: shift(e.b) }) as typeof e), 0.4);
      for (const notch of panel.notches) {
        for (const t of notch.ticks) {
          ops.push(stroke(pdfPath([conv(shift(t.from)), conv(shift(t.to))], false), 0.35));
        }
      }
      for (const grain of panel.grainlines) {
        ops.push(stroke(pdfPath([conv(shift(grain.from)), conv(shift(grain.to))], false), 0.35));
      }
      for (const fold of panel.folds) {
        ops.push(stroke(pdfPath([conv(shift(fold.a)), conv(shift(fold.b))], false), 0.3));
      }
      for (const drill of panel.drills) {
        const r = u(drill.radiusM);
        const c = conv(shift(drill.pos));
        if (drill.mark === "cross") {
          ops.push(stroke(pdfPath([[c[0] - r * PT_PER_MM, c[1]], [c[0] + r * PT_PER_MM, c[1]]], false), 0.3));
          ops.push(stroke(pdfPath([[c[0], c[1] - r * PT_PER_MM], [c[0], c[1] + r * PT_PER_MM]], false), 0.3));
        } else {
          const k = 16;
          const circ: Array<[number, number]> = [];
          for (let i = 0; i <= k; i++) {
            const a = (i / k) * Math.PI * 2;
            circ.push([c[0] + r * PT_PER_MM * Math.cos(a), c[1] + r * PT_PER_MM * Math.sin(a)]);
          }
          ops.push(stroke(pdfPath(circ, false), 0.3));
        }
      }
      for (const internal of panel.internals) {
        const pts = internal.edges.map((e) => conv(shift(e.a)));
        pts.push(conv(shift(internal.edges[internal.edges.length - 1].b)));
        ops.push(stroke(pdfPath(pts, false), 0.25));
      }
      for (const region of panel.labelRegions) {
        const a = conv(shift(region.min)), b = conv(shift(region.max));
        ops.push(stroke(pdfPath([a, [b[0], a[1]], b, [a[0], b[1]]], true), 0.2));
        Object.entries(region.fields).forEach(([k, v], i) => {
          ops.push(pdfText(a[0] + 2, a[1] - 4 - i * 3.2, 2.4, `${k}: ${v}`));
        });
      }
      for (const ann of panel.annotations) {
        const p = conv(shift(ann.pos));
        ops.push(pdfText(p[0] + 1.5, p[1], 2.2, ann.note));
      }
      const bb = panelBBox(panel);
      const labelPos = conv(shift([bb.min[0], bb.max[1]]));
      ops.push(pdfText(labelPos[0], labelPos[1] + 1.5, 3, `${panel.name} ×${panel.cutQuantity}`));
    }
    // Page furniture.
    if (tiled) {
      if (regMarks) {
        const m = marginMm, L = 8;
        const mark = (x: number, y: number, dx: number, dy: number): void => {
          ops.push(stroke(pdfPath([[x, y], [x + dx, y + dy]], false), 0.2));
        };
        mark(m, m, L, 0); mark(m, m, 0, L);
        mark(spec.widthMm - m, m, -L, 0); mark(spec.widthMm - m, m, 0, L);
        mark(m, spec.heightMm - m, L, 0); mark(m, spec.heightMm - m, 0, -L);
        mark(spec.widthMm - m, spec.heightMm - m, -L, 0); mark(spec.widthMm - m, spec.heightMm - m, 0, -L);
      }
      const cx = spec.widthMm - marginMm - calibrationMm - 4;
      const cy = marginMm + 4;
      ops.push(stroke(pdfPath([[cx, cy], [cx + calibrationMm, cy], [cx + calibrationMm, cy + calibrationMm], [cx, cy + calibrationMm]], true), 0.4));
      ops.push(pdfText(cx, cy + calibrationMm + 2.5, 2.2, `calibration: exactly ${calibrationMm} mm at 100%`));
      ops.push(pdfText(marginMm, spec.heightMm - marginMm - 3, 2.4, `SCALE 100% · 1 unit = 1 ${units} · tile ${spec.col + 1}/${spec.cols} col, ${spec.row + 1}/${spec.rows} row`));
      if (pageNumbers) {
        ops.push(pdfText(marginMm, marginMm + 2, 2.4, `page ${spec.pageIndex}`));
        if (spec.col + 1 < spec.cols) ops.push(pdfText(spec.widthMm - marginMm - 34, marginMm + 2, 2.2, `next -> c${spec.col + 2} r${spec.row + 1}`));
        if (spec.col > 0) ops.push(pdfText(marginMm + 14, marginMm + 2, 2.2, `<- prev c${spec.col} r${spec.row + 1}`));
        if (spec.row + 1 < spec.rows) ops.push(pdfText(spec.widthMm - marginMm - 40, marginMm + 5.2, 2.2, `next row v r${spec.row + 2} c${spec.col + 1}`));
      }
    } else {
      ops.push(pdfText(marginMm, spec.heightMm - marginMm - 3, 2.4, `SCALE 100% · 1 unit = 1 ${units} · full-scale ${page.id}`));
    }
    return ops.join("");
  };

  if (!tiled) {
    const w = layoutWMm + marginMm * 2;
    const h = layoutHMm + marginMm * 2 + 8;
    pageSizesMm.push({ id: page.id, widthMm: w, heightMm: h });
    pageContents.push(emitPage({ widthMm: w, heightMm: h, originXMm: 0, originYMm: 0, cols: 1, rows: 1, col: 0, row: 0, pageIndex: 1 }));
  } else {
    const printableW = page.widthMm - marginMm * 2;
    const printableH = page.heightMm - marginMm * 2 - 6;
    const stepX = printableW - overlapMm;
    const stepY = printableH - overlapMm;
    const cols = Math.max(1, Math.ceil((layoutWMm - overlapMm) / stepX));
    const rows = Math.max(1, Math.ceil((layoutHMm - overlapMm) / stepY));
    let pageIndex = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        pageIndex++;
        pageSizesMm.push({ id: page.id, widthMm: page.widthMm, heightMm: page.heightMm });
        pageContents.push(emitPage({
          widthMm: page.widthMm, heightMm: page.heightMm,
          originXMm: col * stepX, originYMm: row * stepY,
          cols, rows, col, row, pageIndex,
        }));
      }
    }
  }

  // Serialize the PDF (object 1 Catalog, 2 Info, 3 Pages, 4 Font, 5+ pages/content).
  const objects: string[] = [];
  const fontId = 4;
  objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
  const pageIds: number[] = [];
  let nextId = 5;
  for (let i = 0; i < pageContents.length; i++) {
    const pageId = nextId++;
    const contentId = nextId++;
    pageIds.push(pageId);
    const size = pageSizesMm[i];
    objects.push(
      `<< /Type /Page /Parent 3 0 R /MediaBox [0 0 ${pt(size.widthMm)} ${pt(size.heightMm)}] /Resources << /Font << /F1 ${fontId} 0 R >> >> /Contents ${contentId} 0 R >>`,
    );
    const stream = pageContents[i];
    objects.push(`<< /Length ${stream.length} >>\nstream\n${stream}endstream`);
  }
  const n = 3 + objects.length;
  let out = "%PDF-1.4\n%\xB1\xB2\xB3\xB4\n";
  const offsets: number[] = [];
  const bodies = new Map<number, string>();
  bodies.set(1, "<< /Type /Catalog /Pages 3 0 R >>");
  bodies.set(2, `<< /Title (${escPdf(ir.style.garmentName)}) /Producer (closim G13) /Keywords (units=${units}; scale=100%; mode=${opts.mode}; pages=${pageContents.length}) >>`);
  bodies.set(3, `<< /Type /Pages /Count ${pageIds.length} /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] >>`);
  for (let i = 0; i < objects.length; i++) bodies.set(4 + i, objects[i]);
  for (let id = 1; id <= n; id++) {
    offsets[id] = out.length;
    out += `${id} 0 obj\n${bodies.get(id) ?? "<< >>"}\nendobj\n`;
  }
  const xrefStart = out.length;
  out += `xref\n0 ${n + 1}\n0000000000 65535 f \n`;
  for (let id = 1; id <= n; id++) {
    out += `${String(offsets[id]).padStart(10, "0")} 00000 n \n`;
  }
  out += `trailer\n<< /Size ${n + 1} /Root 1 0 R /Info 2 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`;
  const bytes = new Uint8Array(out.length);
  for (let i = 0; i < out.length; i++) bytes[i] = out.charCodeAt(i) & 0xff;
  return { pdf: bytes, warnings, pageCount: pageContents.length, pageSizesMm, units, mode: opts.mode };
}
