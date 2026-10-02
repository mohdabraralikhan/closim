// WGSL source consistency: shaders/*.wgsl must contain the documented entry
// points and the frozen physics conventions (S01 = 2*G*E01, no subgroups).
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { SHADER_ENTRY_POINTS } from "../../src/backend/webgpu/gpu-pipelines.js";

const here = dirname(fileURLToPath(import.meta.url));
const shaderDir = join(here, "..", "..", "src", "backend", "webgpu", "shaders");

describe("webgpu WGSL sources", () => {
  it("all registered shaders exist on disk with matching entry points", () => {
    const files = readdirSync(shaderDir);
    for (const [name, eps] of Object.entries(SHADER_ENTRY_POINTS)) {
      const file = `${name}.wgsl`;
      expect(files, name).toContain(file);
      const src = readFileSync(join(shaderDir, file), "utf8");
      for (const ep of eps) expect(src, `${name}:${ep}`).toContain(`fn ${ep}`);
      expect(src, `${name} workgroup_size`).toMatch(/@workgroup_size\(\d+\)/);
    }
  });

  it("membrane kernel preserves the S01 = 2*G*E01 convention", () => {
    const src = readFileSync(join(shaderDir, "membrane-gradient.wgsl"), "utf8");
    expect(src).toContain("2.0 * matG * E01");
    expect(src).not.toContain("4.0 * matG * E01");
  });

  it("no G0 kernel requires subgroups", () => {
    const files = readdirSync(shaderDir).filter((f) => f.endsWith(".wgsl"));
    for (const f of files) {
      const src = readFileSync(join(shaderDir, f), "utf8");
      expect(src, f).not.toContain("subgroupBallot");
      expect(src, f).not.toContain("subgroupShuffle");
      expect(src, f).not.toContain("enable subgroups");
    }
  });

  it("barrier kernel documents frozen-projection (no penalty force)", () => {
    const src = readFileSync(join(shaderDir, "barrier-gradient.wgsl"), "utf8");
    expect(src.toLowerCase()).toContain("frozen");
    // No penalty-force implementation: reject penalty stiffness uniforms.
    expect(src).not.toContain("kPenalty");
    expect(src).not.toContain("penaltyStiffness");
  });
});
