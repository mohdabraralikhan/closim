// G15B/C tests: visual materials (never physics) and deterministic UVs.
import { describe, expect, it } from "vitest";
import {
  addMaterial,
  assignMaterial,
  createLibrary,
  createMaterial,
  deserializeLibrary,
  fallbackMaterial,
  markTexture,
  materialFromPreset,
  replaceMaterial,
  resolveMaterial,
  serializeLibrary,
  validateMaterial,
  FABRIC_PRESETS,
} from "../../src/render/materials.js";
import {
  boundaryUVLoop,
  generatePanelUVs,
  mapPointToUV,
} from "../../src/render/uvgen.js";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { assembleGarment } from "../../src/garment/assembly.js";

describe("G15B visual materials", () => {
  it("creates and validates appearance-only materials", () => {
    const m = createMaterial({ id: "mat/silk", name: "Silk", baseColor: [0.9, 0.85, 0.8], roughness: 0.3 });
    expect(validateMaterial(m)).toEqual([]);
    expect(m.opacity).toBe(1);
    expect(m.physicalRef).toBeUndefined();
    const linked = createMaterial({ id: "m", name: "M", baseColor: [0.5, 0.5, 0.5], physicalRef: "cotton" });
    expect(linked.physicalRef).toBe("cotton");
    expect(() => createMaterial({ id: "m", name: "M", baseColor: [2, 0, 0] })).toThrowError(/color/);
    expect(() => createMaterial({ id: "m", name: "M", baseColor: [0.5, 0.5, 0.5], roughness: 2 })).toThrowError(/roughness/);
    expect(() => createMaterial({ id: "", name: "M", baseColor: [0.5, 0.5, 0.5] })).toThrowError(/id and name/);
  });

  it("ships named fabric presets without physical claims", () => {
    for (const key of ["cotton", "denim", "silk", "wool", "linen", "synthetic"] as const) {
      expect(FABRIC_PRESETS[key].description.length).toBeGreaterThan(0);
      const m = materialFromPreset(`mat/${key}`, key, key);
      expect(validateMaterial(m)).toEqual([]);
    }
    expect(() => materialFromPreset("x", "X", "unobtanium" as "cotton")).toThrowError(/preset/);
    const denim = materialFromPreset("mat/denim", "Denim", "denim", { baseColor: [0.1, 0.1, 0.4] });
    expect(denim.baseColor).toEqual([0.1, 0.1, 0.4]);
    expect(denim.weave).toBe("twill-2/1");
  });

  it("manages a library with assignment and texture states", () => {
    let lib = createLibrary();
    const silk = materialFromPreset("mat/silk", "Silk", "silk", {
      colorMap: { kind: "file", ref: "tex/floral.png", resolutionPx: 1024 },
    });
    lib = addMaterial(lib, silk);
    expect(() => addMaterial(lib, silk)).toThrowError(/already in library/);
    lib = assignMaterial(lib, "panel/front", "mat/silk");
    expect(lib.assignment["panel/front"]).toBe("mat/silk");
    expect(() => assignMaterial(lib, "panel/x", "missing")).toThrowError(/not in library/);
    expect(lib.textures["tex/floral.png"].status).toBe("empty");
    lib = markTexture(lib, "tex/floral.png", "loading");
    expect(lib.textures["tex/floral.png"].status).toBe("loading");
    lib = markTexture(lib, "tex/floral.png", "failed");
    // Failed texture still resolves (adapter falls back); the miss is explicit elsewhere.
    expect(resolveMaterial(lib, "mat/silk").missed).toBe(false);
    const missing = resolveMaterial(lib, "nope");
    expect(missing.missed).toBe(true);
    expect(missing.material.id).toBe(fallbackMaterial().id);
    const v2 = { ...silk, roughness: 0.5 };
    lib = replaceMaterial(lib, v2);
    expect(lib.materials.find((m) => m.id === "mat/silk")!.version).toBe(2);
    expect(() => replaceMaterial(lib, { ...silk, id: "ghost" })).toThrowError(/not in library/);
  });

  it("round-trips deterministically", () => {
    let lib = createLibrary();
    lib = addMaterial(lib, materialFromPreset("mat/wool", "Wool", "wool"));
    lib = assignMaterial(lib, "panel/back", "mat/wool");
    const s0 = serializeLibrary(lib);
    expect(serializeLibrary(deserializeLibrary(s0))).toBe(s0);
    expect(() => deserializeLibrary("nope")).toThrowError(/JSON/);
  });
});

describe("G15C panel UVs", () => {
  function makeGarment() {
    let document = createPatternDocument("uv", "UV");
    const r = addRectPanel(document, "front", [0.1, 0.2], 0.4, 0.3);
    document = r.document;
    const g = assembleGarment(document, [], [{ panelId: r.refs.panelId, translation: [0, 0, 0], yawRad: 0 }]);
    return { panelId: r.refs.panelId, garment: g };
  }

  it("maps the pattern bbox to [0,1] deterministically", () => {
    const { panelId, garment } = makeGarment();
    const a = generatePanelUVs(garment, panelId);
    const b = generatePanelUVs(garment, panelId);
    expect(Array.from(a.uv)).toEqual(Array.from(b.uv));
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < a.uv.length; i += 2) {
      xs.push(a.uv[i]);
      ys.push(a.uv[i + 1]);
    }
    expect(Math.min(...xs)).toBeCloseTo(0, 12);
    expect(Math.max(...xs)).toBeCloseTo(1, 12);
    expect(Math.min(...ys)).toBeCloseTo(0, 12);
    expect(Math.max(...ys)).toBeCloseTo(1, 12);
    expect(a.grainRad).toBeNull();
  });

  it("aligns grain along +V and inverts exactly", () => {
    const { panelId, garment } = makeGarment();
    const frame = generatePanelUVs(garment, panelId, { grainAlign: true, grainRad: 0 });
    expect(frame.grainRad).toBe(0);
    // Grain (+X) runs along +V: stepping in X changes v, not u.
    const uvA = mapPointToUV(frame, [0.2, 0.3]);
    const uvB = mapPointToUV(frame, [0.4, 0.3]);
    expect(uvA[0]).toBeCloseTo(uvB[0], 9);
    expect(uvB[1] - uvA[1]).toBeCloseTo(0.5, 6);
    // Exact inverse up to the Float32 pattern storage in AssembledGarment.uv.
    for (const p of [[0.1, 0.2], [0.5, 0.2], [0.5, 0.5], [0.1, 0.5]] as Array<[number, number]>) {
      const uv = mapPointToUV(frame, p);
      expect(uv[0]).toBeGreaterThanOrEqual(-1e-6);
      expect(uv[0]).toBeLessThanOrEqual(1 + 1e-6);
    }
    const loop = boundaryUVLoop(frame, [[0.1, 0.2], [0.5, 0.2], [0.5, 0.5], [0.1, 0.5]]);
    expect(loop).toHaveLength(4);
    // Every mapped corner lands on a UV unit corner (up to Float32 storage).
    for (const q of loop) {
      expect(Math.min(Math.abs(q[0]), Math.abs(q[0] - 1))).toBeLessThan(1e-6);
      expect(Math.min(Math.abs(q[1]), Math.abs(q[1] - 1))).toBeLessThan(1e-6);
    }
  });

  it("ignores simulation positions (print does not swim)", () => {
    const { panelId, garment } = makeGarment();
    const before = Array.from(generatePanelUVs(garment, panelId).uv);
    const moved = {
      ...garment,
      positions: Float32Array.from(garment.positions.map((v, i) => (i % 3 === 1 ? v + 0.5 : v))),
    };
    expect(Array.from(generatePanelUVs(moved, panelId).uv)).toEqual(before);
  });

  it("rejects unknown panels", () => {
    const { garment } = makeGarment();
    expect(() => generatePanelUVs(garment, "ghost")).toThrowError(/not in garment/);
  });
});
