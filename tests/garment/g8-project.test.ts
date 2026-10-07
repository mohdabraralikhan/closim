import { describe, expect, it } from "vitest";
import { movePoint } from "../../src/pattern/cad.js";
import {
  applyPatternEdit,
  createGarmentProject,
  deserializeGarmentProject,
  rebuildGarment,
  serializeGarmentProject,
  validateGarmentProject,
  type GarmentProject,
} from "../../src/garment/project.js";
import { buildTshirtProject } from "../../src/garment/tshirt.js";

function tshirt(): GarmentProject {
  return buildTshirtProject().project;
}

describe("G8D garment project format", () => {
  it("validates the T-shirt project with no diagnostics", () => {
    const result = validateGarmentProject(tshirt());
    expect(result).toEqual({ valid: true, diagnostics: [] });
  });

  it("round-trips byte-identically through serialize/deserialize", () => {
    const project = tshirt();
    const once = serializeGarmentProject(project);
    const twice = serializeGarmentProject(deserializeGarmentProject(once));
    expect(twice).toBe(once);
    // Key order is canonical (sorted): schemaVersion is not first.
    expect(once.indexOf('"avatar"')).toBeLessThan(once.indexOf('"schemaVersion"'));
  });

  it("rejects unknown schema versions", () => {
    const project = tshirt();
    const bad = { ...project, schemaVersion: 999 };
    const result = validateGarmentProject(bad as unknown as GarmentProject);
    expect(result.valid).toBe(false);
    expect(result.diagnostics.map((d) => d.code)).toContain("unsupported-schema");
  });

  it("rejects missing materials, bad placements, and bad simulation config", () => {
    const project = tshirt();
    const noMat = { ...project, materials: {} };
    expect(validateGarmentProject(noMat).diagnostics.map((d) => d.code)).toContain("missing-material");

    const badPlacement = {
      ...project,
      placements: [...project.placements, { panelId: "nope", translation: [0, 0, 0] as [number, number, number], yawRad: 0 }],
    };
    expect(validateGarmentProject(badPlacement).diagnostics.map((d) => d.code)).toContain("invalid-placement");

    const badSim = { ...project, simulation: { ...project.simulation, dt: -1 } };
    expect(validateGarmentProject(badSim).diagnostics.map((d) => d.code)).toContain("invalid-simulation-config");

    const badContact = {
      ...project,
      simulation: { ...project.simulation, contact: { ...project.simulation.contact, dMinM: 1 } },
    };
    expect(validateGarmentProject(badContact).diagnostics.map((d) => d.code)).toContain("invalid-simulation-config");
  });

  it("bumps revision on pattern edits and rebuilds changed geometry", () => {
    const project = tshirt();
    const before = rebuildGarment(project);
    const beforePositions = Array.from(before.assembled.positions);
    // Widen the front panel by moving one corner point +0.02 m in x.
    const frontPanelId = project.pattern.panels[0].id;
    const targetPoint = project.pattern.points.find((p) => p.panelId === frontPanelId)!;
    const edited = applyPatternEdit(project, (pattern) =>
      movePoint(pattern, frontPanelId, targetPoint.id, [targetPoint.x + 0.02, targetPoint.y]),
    );
    expect(edited.metadata.revision).toBe(2);
    expect(edited.id).toBe(project.id);
    const after = rebuildGarment(edited);
    expect(Array.from(after.assembled.positions)).not.toEqual(beforePositions);
    // Seams and placements survive the edit untouched.
    expect(after.assembled.weldPairs).toHaveLength(before.assembled.weldPairs.length);
    expect(after.assembled.panelRanges.map((r) => r.panelId)).toEqual(
      before.assembled.panelRanges.map((r) => r.panelId),
    );
  });

  it("rejects pattern edits that break the project", () => {
    const project = tshirt();
    expect(() =>
      applyPatternEdit(project, (pattern) => ({
        ...pattern,
        points: [],
      })),
    ).toThrow(/invalid project/);
  });

  it("creates minimal projects with sensible simulation defaults", () => {
    const base = tshirt();
    const minimal = createGarmentProject("garment/min", "Minimal", base.pattern);
    expect(minimal.simulation.dt).toBeCloseTo(1 / 60);
    expect(minimal.avatar).toBeNull();
    expect(minimal.metadata.revision).toBe(1);
    expect(() => createGarmentProject("", "No id", base.pattern)).toThrow(/non-empty/);
  });
});
