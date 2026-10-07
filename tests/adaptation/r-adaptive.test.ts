import { describe, expect, it } from "vitest";
import { adaptRMesh } from "../../src/adaptation/r-adaptive.js";

function grid(width = 7, height = 7, spacing = 0.1) {
  const uv = new Float64Array(width * height * 2);
  const xyz = new Float64Array(width * height * 3);
  const indices: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      uv[2 * i] = x * spacing;
      uv[2 * i + 1] = y * spacing;
      xyz[3 * i] = x * spacing;
      xyz[3 * i + 2] = y * spacing;
    }
  }
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = y * width + x, b = a + 1, c = a + width, d = c + 1;
      indices.push(a, b, d, a, d, c);
    }
  }
  return { uv, xyz, indices: Uint32Array.from(indices), width, height };
}

function incidentMeanEdge(result: ReturnType<typeof adaptRMesh>, vertex: number) {
  const lengths = new Map<string, number>();
  const { indices, positions } = result;
  for (let t = 0; t < indices.length; t += 3) {
    for (const [a, b] of [[indices[t], indices[t + 1]], [indices[t + 1], indices[t + 2]], [indices[t + 2], indices[t]]] as const) {
      if (a !== vertex && b !== vertex) continue;
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      lengths.set(key, Math.hypot(positions[2 * a] - positions[2 * b], positions[2 * a + 1] - positions[2 * b + 1]));
    }
  }
  return [...lengths.values()].reduce((sum, value) => sum + value, 0) / lengths.size;
}

describe("G7B fixed-connectivity r-adaptation", () => {
  it("leaves smooth planar cloth unchanged when no feature asks for refinement", () => {
    const mesh = grid();
    const result = adaptRMesh({ referencePositions: mesh.uv, indices: mesh.indices, deformedPositions: mesh.xyz });
    expect(result.accepted).toBe(true);
    expect(result.iterations).toBe(0);
    expect(result.positions).toEqual(mesh.uv);
    expect(result.indices).toBe(mesh.indices);
    expect(Math.max(...result.curvature)).toBe(0);
  });

  it("detects a fold and lowers target spacing near its curvature ridge", () => {
    const mesh = grid();
    const creaseX = (mesh.width - 1) / 2;
    for (let i = 0; i < mesh.width * mesh.height; i++) {
      const x = mesh.uv[2 * i];
      mesh.xyz[3 * i + 1] = 0.08 * Math.abs(x - creaseX * 0.1);
    }
    const result = adaptRMesh({ referencePositions: mesh.uv, indices: mesh.indices, deformedPositions: mesh.xyz });
    const ridge = (mesh.height >> 1) * mesh.width + creaseX;
    const flat = (mesh.height >> 1) * mesh.width + 1;
    expect(result.curvature[ridge]).toBeGreaterThan(result.curvature[flat]);
    expect(result.targetResolution[ridge]).toBeLessThan(result.targetResolution[flat]);
    expect(result.accepted).toBe(true);
  });

  it("concentrates target resolution and relocation around a contact region", () => {
    const mesh = grid();
    const contact = new Float64Array(mesh.width * mesh.height);
    const center = (mesh.height >> 1) * mesh.width + (mesh.width >> 1);
    contact[center] = 5;
    const result = adaptRMesh({ referencePositions: mesh.uv, indices: mesh.indices, contact });
    expect(result.accepted).toBe(true);
    expect(result.targetResolution[center]).toBeLessThan(result.targetResolution[center - 1]);
    expect(incidentMeanEdge(result, center)).toBeLessThan(incidentMeanEdge({ ...result, positions: mesh.uv }, center));
  });

  it("rejects degenerate input and prevents inverted, collapsed, or poor-quality output", () => {
    const mesh = grid();
    const degenerate = Float64Array.from(mesh.uv);
    degenerate[2 * 1] = degenerate[0];
    degenerate[2 * 1 + 1] = degenerate[1];
    const rejected = adaptRMesh({ referencePositions: degenerate, indices: mesh.indices });
    expect(rejected.accepted).toBe(false);
    expect(rejected.reason).toMatch(/zero-area|collapsed-edge/);

    const contact = new Float64Array(mesh.width * mesh.height).fill(30);
    const result = adaptRMesh({ referencePositions: mesh.uv, indices: mesh.indices, contact }, {
      iterations: 8,
      minimumResolutionScale: 0.25,
    });
    expect(result.accepted).toBe(true);
    expect(result.maxMeanRatioAfter).toBeLessThanOrEqual(4);
    expect(result.positions.every(Number.isFinite)).toBe(true);
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const [a, b, c] = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
      const area2 = (result.positions[2 * b] - result.positions[2 * a]) * (result.positions[2 * c + 1] - result.positions[2 * a + 1]) -
        (result.positions[2 * b + 1] - result.positions[2 * a + 1]) * (result.positions[2 * c] - result.positions[2 * a]);
      expect(area2).toBeGreaterThan(0);
    }
  });

  it("keeps inferred boundaries and explicit pins exactly fixed", () => {
    const mesh = grid();
    const pin = (mesh.height >> 1) * mesh.width + 2;
    const pinned = new Uint8Array(mesh.width * mesh.height);
    pinned[pin] = 1;
    const contact = new Float64Array(mesh.width * mesh.height).fill(3);
    const result = adaptRMesh({ referencePositions: mesh.uv, indices: mesh.indices, pinned, contact });
    for (let y = 0; y < mesh.height; y++) {
      for (let x = 0; x < mesh.width; x++) {
        const i = y * mesh.width + x;
        if (x === 0 || y === 0 || x === mesh.width - 1 || y === mesh.height - 1 || i === pin) {
          expect(result.fixed[i]).toBe(1);
          expect(result.positions[2 * i]).toBe(mesh.uv[2 * i]);
          expect(result.positions[2 * i + 1]).toBe(mesh.uv[2 * i + 1]);
        }
      }
    }
  });

  it("is deterministic for identical mesh, indicators, and options", () => {
    const mesh = grid();
    const contact = new Float64Array(mesh.width * mesh.height);
    contact[(mesh.height >> 1) * mesh.width + (mesh.width >> 1)] = 2;
    const input = { referencePositions: mesh.uv, indices: mesh.indices, contact };
    const options = { iterations: 4, relaxation: 0.3 };
    const first = adaptRMesh(input, options);
    const second = adaptRMesh(input, options);
    expect(first.positions).toEqual(second.positions);
    expect(first.targetResolution).toEqual(second.targetResolution);
    expect(first.iterations).toBe(second.iterations);
    expect(first.qualityEnergyAfter).toBe(second.qualityEnergyAfter);
  });
});
