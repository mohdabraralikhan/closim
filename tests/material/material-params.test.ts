// G7A parameters, calibration, serialization, clone independence, SI units.
// All CPU, no device, no meshes needed except one parity cross-check.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import {
  describeParameters,
  FRICTION_OWNERSHIP_NOTE,
  MaterialValidationError,
  OrthotropicStVKMaterial,
  type PhysicalMaterialParams,
} from "../../src/physics/material-model.js";

function validParams(): PhysicalMaterialParams {
  return {
    inertia: { arealDensityKgM2: 0.15 },
    thicknessM: 0.001,
    membrane: { warpPa: 20000, weftPa: 20000, couplingPa: 0, shearPa: 5000 },
    bending: { warpNm: 1e-5, weftNm: 1e-5 },
    dampingRatio: 0.001,
  };
}

function tinyMesh(): ReturnType<typeof preprocess> {
  const g = buildGrid(2, 1, 0.04, 0.02);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

describe("G7A SI units", () => {
  it("parameter table carries exact SI units; friction is contact-owned", () => {
    expect(describeParameters()).toEqual([
      { path: "inertia.arealDensityKgM2", unit: "kg/m^2" },
      { path: "thicknessM", unit: "m" },
      { path: "membrane.warpPa", unit: "Pa" },
      { path: "membrane.weftPa", unit: "Pa" },
      { path: "membrane.couplingPa", unit: "Pa" },
      { path: "membrane.shearPa", unit: "Pa" },
      { path: "bending.warpNm", unit: "N*m" },
      { path: "bending.weftNm", unit: "N*m" },
      { path: "dampingRatio", unit: "dimensionless" },
    ]);
    expect(FRICTION_OWNERSHIP_NOTE).toContain("contact");
  });
});

describe("G7A invalid parameter rejection", () => {
  const cases: Array<{ name: string; mutate: (p: PhysicalMaterialParams) => void; code: string }> = [
    { name: "zero thickness", mutate: (p) => { p.thicknessM = 0; }, code: "non-positive-thickness" },
    { name: "negative thickness", mutate: (p) => { p.thicknessM = -1e-4; }, code: "non-positive-thickness" },
    { name: "NaN thickness", mutate: (p) => { p.thicknessM = NaN; }, code: "non-finite" },
    { name: "infinite stiffness", mutate: (p) => { p.membrane.warpPa = Infinity; }, code: "non-finite" },
    { name: "zero density", mutate: (p) => { p.inertia.arealDensityKgM2 = 0; }, code: "non-positive-density" },
    { name: "negative density", mutate: (p) => { p.inertia.arealDensityKgM2 = -0.2; }, code: "non-positive-density" },
    { name: "negative warp", mutate: (p) => { p.membrane.warpPa = -1; }, code: "negative-stiffness" },
    { name: "negative shear", mutate: (p) => { p.membrane.shearPa = -50; }, code: "negative-stiffness" },
    { name: "negative bending", mutate: (p) => { p.bending.warpNm = -1e-9; }, code: "negative-stiffness" },
    {
      name: "indefinite coupling", mutate: (p) => {
        p.membrane.warpPa = 1e3; p.membrane.weftPa = 1e3; p.membrane.couplingPa = 5000;
      }, code: "indefinite-coupling",
    },
    { name: "negative damping", mutate: (p) => { p.dampingRatio = -0.1; }, code: "damping-range" },
    { name: "damping above 1", mutate: (p) => { p.dampingRatio = 1.5; }, code: "damping-range" },
    { name: "NaN damping", mutate: (p) => { p.dampingRatio = NaN; }, code: "non-finite" },
  ];
  for (const c of cases) {
    it(`rejects ${c.name}`, () => {
      const p = validParams();
      c.mutate(p);
      let err: unknown = null;
      try {
        new OrthotropicStVKMaterial(p);
      } catch (e) {
        err = e;
      }
      expect(err instanceof MaterialValidationError).toBe(true);
      expect((err as MaterialValidationError).code).toBe(c.code);
    });
  }

  it("rejects malformed shapes and zero-stiffness edge stays valid", () => {
    expect(() => new OrthotropicStVKMaterial({} as unknown as PhysicalMaterialParams))
      .toThrow(MaterialValidationError);
    expect(() => new OrthotropicStVKMaterial(null as unknown as PhysicalMaterialParams))
      .toThrow(MaterialValidationError);
    // Fully compliant membrane (all-zero stiffness) is degenerate but valid.
    const zero = validParams();
    zero.membrane.warpPa = 0; zero.membrane.weftPa = 0;
    zero.membrane.couplingPa = 0; zero.membrane.shearPa = 0;
    expect(() => new OrthotropicStVKMaterial(zero)).not.toThrow();
    // Boundary PSD (equality) is allowed.
    const edge = validParams();
    edge.membrane.warpPa = 4e4; edge.membrane.weftPa = 1e4;
    edge.membrane.couplingPa = Math.sqrt(4e4 * 1e4);
    expect(() => new OrthotropicStVKMaterial(edge)).not.toThrow();
  });

  it("rejects unphysical legacy values at the bridge (legacy path untouched)", () => {
    expect(() => OrthotropicStVKMaterial.fromLegacy({ ...DEFAULT_MATERIAL })).not.toThrow();
    expect(() => OrthotropicStVKMaterial.fromLegacy({ ...DEFAULT_MATERIAL, thickness: -0.5 }))
      .toThrow(MaterialValidationError);
  });
});

describe("G7A calibration representation (metadata only)", () => {
  it("stores elastic + provenance metadata; reserves hysteresis/plasticity as none", () => {
    const m = new OrthotropicStVKMaterial(validParams(), {
      elastic: {
        dataset: "Fabric-101",
        specimenId: "cotton-poplin-03",
        protocol: "uniaxial-warp/weft + 45deg-shear",
        measuredAt: "2026-03-14",
        moduliPa: { warp: 21000, weft: 19000, coupling: 500, shear: 5200 },
      },
      hysteresis: { model: "none", notes: "recoverable loop area recorded off-model" },
      plasticity: { model: "none" },
      provenanceNotes: "lab-conditioned 20C/65%RH",
    });
    expect(m.calibration?.elastic.dataset).toBe("Fabric-101");
    expect(m.calibration?.plasticity?.model).toBe("none");
    // Round-trips through JSON with the metadata intact.
    const back = OrthotropicStVKMaterial.fromJSON(JSON.stringify(m.toJSON()));
    expect(back.calibration).toEqual(m.calibration);
  });

  it("requires an elastic dataset; rejects implemented dissipative models", () => {
    expect(() => new OrthotropicStVKMaterial(validParams(), { elastic: { dataset: "" } }))
      .toThrow(MaterialValidationError);
    expect(() => new OrthotropicStVKMaterial(validParams(), {
      elastic: { dataset: "Fabric-101" },
      plasticity: { model: "j2-plastic" } as unknown as { model: "none" },
    })).toThrowError(/not-implemented/);
    expect(() => new OrthotropicStVKMaterial(validParams(), {
      elastic: { dataset: "Fabric-101" },
      hysteresis: { model: "prandtl" } as unknown as { model: "none" },
    })).toThrowError(/not-implemented/);
  });
});

describe("G7A serialization round-trip", () => {
  it("JSON string and object forms both restore bitwise-identical behavior", () => {
    const mesh = tinyMesh();
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] *= 1.1;
    const m = new OrthotropicStVKMaterial(validParams(), {
      elastic: { dataset: "Fabric-101", specimenId: "s-1" },
    });
    for (const restored of [
      OrthotropicStVKMaterial.fromJSON(JSON.stringify(m.toJSON())),
      OrthotropicStVKMaterial.fromJSON(m.toJSON()),
    ]) {
      expect(restored.toLegacy()).toEqual(m.toLegacy());
      expect(restored.energy(x, mesh)).toBe(m.energy(x, mesh));
      const a = restored.gradient(x, mesh).grad;
      const b = m.gradient(x, mesh).grad;
      let d = 0;
      for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
      expect(d).toBe(0);
    }
    // Version + kind tags are load-bearing.
    expect(m.toJSON().kind).toBe("orthotropic-stvk");
    expect(m.toJSON().version).toBe(1);
    expect(() => OrthotropicStVKMaterial.fromJSON({ ...m.toJSON(), kind: "neo-hookean" }))
      .toThrowError(/unknown-kind/);
    expect(() => OrthotropicStVKMaterial.fromJSON({ ...m.toJSON(), version: 999 }))
      .toThrowError(/version/);
    expect(() => OrthotropicStVKMaterial.fromJSON("not json"))
      .toThrow();
  });

  it("models without calibration round-trip with calibration undefined", () => {
    const m = new OrthotropicStVKMaterial(validParams());
    expect(m.calibration).toBeUndefined();
    const back = OrthotropicStVKMaterial.fromJSON(JSON.stringify(m.toJSON()));
    expect(back.calibration).toBeUndefined();
    expect(back.toLegacy()).toEqual(m.toLegacy());
  });
});

describe("G7A clone independence", () => {
  it("clone shares values but no references; instances are immutable snapshots", () => {
    const mesh = tinyMesh();
    const x = Float64Array.from(mesh.positions);
    const m = new OrthotropicStVKMaterial(validParams(), { elastic: { dataset: "d" } });
    const c = m.clone();
    expect(c).not.toBe(m);
    expect(c.params).not.toBe(m.params);
    expect(c.calibration).not.toBe(m.calibration);
    expect(c.toLegacy()).toEqual(m.toLegacy());
    expect(c.energy(x, mesh)).toBe(m.energy(x, mesh));
    // A derived instance with altered params leaves the original untouched.
    const altered = new OrthotropicStVKMaterial({
      ...validParams(), thicknessM: 0.002,
    });
    expect(altered.energy(x, mesh)).not.toBe(m.energy(x, mesh));
    expect(m.toLegacy().thickness).toBe(0.001);
    // Params are frozen against silent mutation.
    expect(Object.isFrozen(m.params)).toBe(true);
  });
});
