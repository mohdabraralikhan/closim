import { describe, expect, it } from "vitest";
import {
  decimateMesh,
  keepLargestComponent,
  makeMeshAvatar,
  meshStats,
  parseAvatarOBJ,
} from "../../src/garment/avatar-mesh.js";
import { validateAvatarSpec } from "../../src/garment/avatar.js";

const CUBE = `# unit cube, y-up, metres
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
`;

describe("OBJ avatar loading", () => {
  it("parses quads via fan triangulation over welded vertices", () => {
    const mesh = parseAvatarOBJ(CUBE);
    expect(meshStats(mesh.positions, mesh.indices)).toMatchObject({
      vertexCount: 8,
      triCount: 12,
      degenerateTris: 0,
    });
    const stats = meshStats(mesh.positions, mesh.indices);
    expect(stats.min).toEqual([0, 0, 0]);
    expect(stats.max).toEqual([1, 1, 1]);
  });

  it("applies unit scale", () => {
    const mesh = parseAvatarOBJ(CUBE, { unitScale: 0.01 });
    expect(meshStats(mesh.positions, mesh.indices).max).toEqual([0.01, 0.01, 0.01]);
  });

  it("supports slash face formats and negative indices", () => {
    const mesh = parseAvatarOBJ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\nf -3 -2 -1\n");
    expect(meshStats(mesh.positions, mesh.indices).triCount).toBe(2);
  });

  it("rejects empty and malformed input", () => {
    expect(() => parseAvatarOBJ("# nothing here\n")).toThrow(/no faces/);
    expect(() => parseAvatarOBJ("v 0 0 0\nf 1 2 3\n")).toThrow();
  });
});

describe("avatar mesh cleanup", () => {
  // 5x5 vertex grid in the unit square (32 tris) — dense enough to decimate.
  const gridOBJ = (() => {
    const lines: string[] = [];
    for (let j = 0; j < 5; j++) {
      for (let i = 0; i < 5; i++) lines.push(`v ${i / 4} 0 ${j / 4}`);
    }
    const id = (i: number, j: number) => j * 5 + i + 1;
    for (let j = 0; j < 4; j++) {
      for (let i = 0; i < 4; i++) lines.push(`f ${id(i, j)} ${id(i + 1, j)} ${id(i + 1, j + 1)} ${id(i, j + 1)}`);
    }
    return `${lines.join("\n")}\n`;
  })();

  it("keeps the largest connected component", () => {
    // Cube plus one detached triangle.
    const extra = "v 9 9 9\nv 10 9 9\nv 9 10 9\nf 9 10 11\n";
    const mesh = parseAvatarOBJ(`${CUBE}${extra}`);
    expect(meshStats(mesh.positions, mesh.indices).triCount).toBe(13);
    const kept = keepLargestComponent(mesh.positions, mesh.indices);
    expect(meshStats(kept.positions, kept.indices).triCount).toBe(12);
  });

  it("decimates deterministically and preserves bounds", () => {
    const mesh = parseAvatarOBJ(gridOBJ);
    expect(meshStats(mesh.positions, mesh.indices).triCount).toBe(32);
    const a = decimateMesh(mesh.positions, mesh.indices, { cellM: 0.3 });
    const b = decimateMesh(mesh.positions, mesh.indices, { cellM: 0.3 });
    expect(a).toEqual(b);
    const stats = meshStats(a.positions, a.indices);
    expect(stats.min).toEqual([0, 0, 0]);
    expect(stats.max).toEqual([1, 0, 1]);
    expect(stats.triCount).toBeLessThan(32);
    expect(() => decimateMesh(mesh.positions, mesh.indices, { cellM: 0 })).toThrow();
  });
});

describe("mesh avatar spec", () => {
  it("builds a validated spec with a collision proxy", () => {
    const grid = (() => {
      const lines: string[] = [];
      for (let j = 0; j < 5; j++) {
        for (let i = 0; i < 5; i++) lines.push(`v ${i / 4} ${j / 4} 0`);
      }
      const id = (i: number, j: number) => j * 5 + i + 1;
      for (let j = 0; j < 4; j++) {
        for (let i = 0; i < 4; i++) lines.push(`f ${id(i, j)} ${id(i + 1, j)} ${id(i + 1, j + 1)} ${id(i, j + 1)}`);
      }
      return parseAvatarOBJ(`${lines.join("\n")}\n`);
    })();
    const spec = makeMeshAvatar(grid, "grid", { proxyCellM: 0.3 });
    validateAvatarSpec(spec);
    expect(spec.collision).toBeDefined();
    expect(spec.collision!.indices.length).toBeLessThan(spec.indices.length);
    expect(spec.id).toBe("avatar/grid");
  });
});
