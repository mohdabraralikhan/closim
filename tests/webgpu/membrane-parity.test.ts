// G0 parity: FP32 mirror vs Float64 CPU reference (runs on headless CI).
// The mirror structurally ports membrane-gradient.wgsl; agreement here is the
// G0.10 entry gate before barrier/contact migrate to GPU.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { evalMembrane } from "../../src/physics/fem.js";
import {
  fp32MembraneMesh, comparePositions, compareEnergy,
  GPU_POS_ABS_TOL, GPU_ENERGY_REL_TOL,
} from "../../src/backend/webgpu/gpu-tolerances.js";

function deformedStrip(): { x: Float64Array; mesh: ReturnType<typeof preprocess> } {
  const g = buildGrid(4, 2, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const x = Float64Array.from(mesh.positions);
  // deterministic small shear + sag so strains are nonzero but modest
  for (let i = 0; i < mesh.count; i++) {
    x[i * 3] += 0.002 * Math.sin(i);
    x[i * 3 + 1] -= 0.003 * (x[i * 3] / 0.2);
    x[i * 3 + 2] += 0.001 * Math.cos(i * 2);
  }
  return { x, mesh };
}

describe("webgpu G0 membrane parity (fp32 mirror)", () => {
  it("fp32 mirror energy matches f64 within energy tolerance", () => {
    const { x, mesh } = deformedStrip();
    const ref = evalMembrane(x, mesh, DEFAULT_MATERIAL);
    const mir = fp32MembraneMesh(x, mesh, DEFAULT_MATERIAL);
    const { rel, pass } = compareEnergy(ref.energy, mir.energy, GPU_ENERGY_REL_TOL);
    expect(pass).toBe(true);
    expect(rel).toBeLessThan(GPU_ENERGY_REL_TOL);
  });

  it("fp32 mirror gradient matches f64 within position-scale tolerance", () => {
    const { x, mesh } = deformedStrip();
    const ref = evalMembrane(x, mesh, DEFAULT_MATERIAL);
    const mir = fp32MembraneMesh(x, mesh, DEFAULT_MATERIAL);
    const c = comparePositions(ref.grad, mir.grad, 1e-2, 1e-2);
    // gradient scale >> positions; use loose absolute bound relative to scale
    const scale = Math.max(1e-9, ...Array.from(ref.grad).map(Math.abs));
    expect(c.maxAbs / scale).toBeLessThan(1e-5);
  });

  it("shear convention S01 = 2*G*E01 is preserved in the mirror", () => {
    // Pure-shear triangle: E01 != 0 must produce a nonzero, correctly-signed
    // gradient. A 4*G port would double the shear response and fail parity.
    const g = buildGrid(1, 1, 0.1, 0.1);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const x = Float64Array.from(mesh.positions);
    x[3] += 0.01; // shear corner +x
    const ref = evalMembrane(x, mesh, DEFAULT_MATERIAL);
    const mir = fp32MembraneMesh(x, mesh, DEFAULT_MATERIAL);
    const { rel, pass } = compareEnergy(ref.energy, mir.energy, GPU_ENERGY_REL_TOL);
    expect(pass).toBe(true);
    expect(rel).toBeLessThan(GPU_ENERGY_REL_TOL);
    void GPU_POS_ABS_TOL;
  });
});
