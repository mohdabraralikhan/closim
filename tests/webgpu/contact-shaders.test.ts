// G2 shader structural tests: twin-region identity (WGSL has no imports, so
// closest/cubic cores are textually duplicated), barrier floor binding (D4),
// compaction overflow/failure plumbing, and no-penalty / no-subgroup guards.
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const shaderDir = join(here, "..", "..", "src", "backend", "webgpu", "shaders");
const read = (f: string): string => readFileSync(join(shaderDir, f), "utf8");

function twinRegion(src: string, name: string): string {
  const begin = `// @twin ${name}-begin`;
  const end = `// @twin ${name}-end`;
  const i = src.indexOf(begin);
  const j = src.indexOf(end);
  expect(i, `${name} begin`).toBeGreaterThanOrEqual(0);
  expect(j, `${name} end`).toBeGreaterThan(i);
  return src.slice(i + begin.length, j);
}

describe("G2 contact shaders", () => {
  it("twin vt-core is identical in closest-vt and ccd-vt", () => {
    expect(twinRegion(read("ccd-vt.wgsl"), "vt-core")).toBe(
      twinRegion(read("closest-vt.wgsl"), "vt-core"),
    );
  });

  it("twin ee-core is identical in closest-ee and ccd-ee", () => {
    expect(twinRegion(read("ccd-ee.wgsl"), "ee-core")).toBe(
      twinRegion(read("closest-ee.wgsl"), "ee-core"),
    );
  });

  it("twin cubic-core is identical in ccd-vt and ccd-ee", () => {
    expect(twinRegion(read("ccd-vt.wgsl"), "cubic-core")).toBe(
      twinRegion(read("ccd-ee.wgsl"), "cubic-core"),
    );
  });

  it("barrier floor branch subtracts prm.z (D4)", () => {
    const src = read("barrier-gradient.wgsl");
    expect(src).toContain("position[id.x].y - prm.z");
  });

  it("compact shader has explicit overflow + failure plumbing", () => {
    const src = read("contact-compact.wgsl");
    expect(src).toContain("atomicMax(&contactOverflow[0], 1u)");
    expect(src).toContain("atomicMax(&contactFail[0], 1u)");
    expect(src).toContain("expand_pairs");
    expect(src).toContain("compact_pairs_ee");
    expect(src).toContain("compact_floor");
    // kind encodings match the barrier consumer (0 VT, 1 EE, 2 floor)
    expect(src).toContain("vec4f(0.0, 1.0, 0.0, 2.0)");
  });

  it("G2 shaders add no penalty forces and no subgroup dependency", () => {
    for (const f of ["closest-vt.wgsl", "closest-ee.wgsl", "ccd-vt.wgsl", "ccd-ee.wgsl", "contact-compact.wgsl"]) {
      const src = read(f);
      expect(src, f).not.toContain("kPenalty");
      expect(src, f).not.toContain("penaltyStiffness");
      expect(src, f).not.toContain("subgroup");
    }
  });
});
