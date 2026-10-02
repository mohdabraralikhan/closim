// G4B incidence maps: structural validity + gather bit-exactness vs direct
// scatter (element-ordered CSR lists reproduce scatter sums bit-for-bit).
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import {
  buildMembraneIncidence, buildHingeIncidence, gatherAdd,
} from "../src/mesh/incidence.js";
import { evalMembrane, evalInternal } from "../src/physics/fem.js";
import { evalMembraneHvp, triangleHvp } from "../src/physics/membrane-hvp.js";
import { triangleEnergyGradient } from "../src/physics/membrane.js";
import { addHingeGradient } from "../src/physics/bending.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";

function fanMesh(arms: number) {
  // center + ring fan (center valence = arms, ring valence 3, boundary ring)
  const positions = new Float32Array((arms + 1) * 3);
  const uv = new Float32Array((arms + 1) * 2);
  const indices = new Uint32Array(arms * 3);
  for (let k = 0; k < arms; k++) {
    const a = (2 * Math.PI * k) / arms;
    positions[(k + 1) * 3] = Math.cos(a) * 0.05;
    positions[(k + 1) * 3 + 1] = 0;
    positions[(k + 1) * 3 + 2] = Math.sin(a) * 0.05;
    uv[(k + 1) * 2] = Math.cos(a);
    uv[(k + 1) * 2 + 1] = Math.sin(a);
    indices[k * 3] = 0;
    indices[k * 3 + 1] = k + 1;
    indices[k * 3 + 2] = ((k + 1) % arms) + 1;
  }
  return preprocess(positions, uv, indices, 0.15);
}

describe("G4B incidence maps", () => {
  it("membrane map: exact 3m records, corners in range, tri-ordered lists", () => {
    const g = buildGrid(9, 7, 0.16, 0.12);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const map = buildMembraneIncidence(mesh.indices, mesh.triCount, mesh.count);
    expect(map.offsets.length).toBe(mesh.count + 1);
    expect(map.offsets[mesh.count]).toBe(3 * mesh.triCount);
    expect(map.ids.length).toBe(3 * mesh.triCount);
    // offsets monotone, corners valid, per-vertex tri ids ascending
    for (let v = 0; v < mesh.count; v++) {
      expect(map.offsets[v + 1]).toBeGreaterThanOrEqual(map.offsets[v]);
      let prev = -1;
      for (let k = map.offsets[v]; k < map.offsets[v + 1]; k++) {
        expect(map.corners[k]).toBeLessThanOrEqual(2);
        expect(map.ids[k]).toBeGreaterThanOrEqual(prev); // tri-ordered
        prev = map.ids[k];
        // corner consistency: indices[t*3+c] === v
        expect(mesh.indices[map.ids[k] * 3 + map.corners[k]]).toBe(v);
      }
    }
    // every vertex incident to >= 1 triangle (connected grid)
    for (let v = 0; v < mesh.count; v++) {
      expect(map.offsets[v + 1]).toBeGreaterThan(map.offsets[v]);
    }
  });

  it("membrane gather is bit-exact vs direct scatter (gradient + HVP)", () => {
    const g = buildGrid(5, 4, 0.1, 0.08);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += 0.002 * Math.sin(i * 3.7);
    const p = new Float64Array(n * 3);
    for (let i = 0; i < p.length; i++) p[i] = Math.sin(i * 12.9898);
    const map = buildMembraneIncidence(mesh.indices, mesh.triCount, n);
    // per-triangle contributions, then gather
    const elem = new Float64Array(mesh.triCount * 9);
    for (let t = 0; t < mesh.triCount; t++) {
      const at = (vv: number): number[] => [x[vv * 3], x[vv * 3 + 1], x[vv * 3 + 2]];
      const r = triangleEnergyGradient(
        at(mesh.indices[t * 3]), at(mesh.indices[t * 3 + 1]), at(mesh.indices[t * 3 + 2]),
        [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]],
        mesh.areas[t], DEFAULT_MATERIAL,
      );
      elem.set(r.grad, t * 9);
    }
    const gathered = new Float64Array(n * 3);
    gatherAdd(gathered, elem, 9, map);
    const direct = evalMembrane(x, mesh, DEFAULT_MATERIAL).grad;
    for (let i = 0; i < n * 3; i++) expect(gathered[i]).toBe(direct[i]);
    // HVP contributions gather bit-exact too
    const helem = new Float64Array(mesh.triCount * 9);
    for (let t = 0; t < mesh.triCount; t++) {
      const at = (vv: number): number[] => [x[vv * 3], x[vv * 3 + 1], x[vv * 3 + 2]];
      const ap = (vv: number): number[] => [p[vv * 3], p[vv * 3 + 1], p[vv * 3 + 2]];
      const r = triangleHvp(
        at(mesh.indices[t * 3]), at(mesh.indices[t * 3 + 1]), at(mesh.indices[t * 3 + 2]),
        ap(mesh.indices[t * 3]), ap(mesh.indices[t * 3 + 1]), ap(mesh.indices[t * 3 + 2]),
        [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]],
        mesh.areas[t], DEFAULT_MATERIAL,
      );
      helem.set(r.out, t * 9);
    }
    const gatheredH = new Float64Array(n * 3);
    gatherAdd(gatheredH, helem, 9, map);
    const directH = evalMembraneHvp(x, p, mesh, DEFAULT_MATERIAL);
    for (let i = 0; i < n * 3; i++) expect(gatheredH[i]).toBe(directH[i]);
  });

  it("hinge map: 4h records, fan center valence exact, gather bit-exact", () => {
    const mesh = fanMesh(12);
    const hmap = buildHingeIncidence(mesh.hinges, mesh.hinges.length, mesh.count);
    expect(hmap.offsets[mesh.count]).toBe(4 * mesh.hinges.length);
    // fan center (v0) incident to every hinge exactly once
    const centerRecs = hmap.offsets[1] - hmap.offsets[0];
    expect(centerRecs).toBe(mesh.hinges.length);
    // per-hinge: exactly 4 records across the map (multiset over ids)
    const perHinge = new Array(mesh.hinges.length).fill(0);
    for (let k = 0; k < hmap.ids.length; k++) {
      perHinge[hmap.ids[k]]++;
      expect(hmap.corners[k]).toBeLessThanOrEqual(3);
      const h = mesh.hinges[hmap.ids[k]];
      expect([h.v0, h.v1, h.v2, h.v3][hmap.corners[k]]).toBe(
        (() => { // find which vertex this record belongs to
          for (let v = 0; v < mesh.count; v++) {
            if (k >= hmap.offsets[v] && k < hmap.offsets[v + 1]) return v;
          }
          return -1;
        })(),
      );
    }
    for (const c of perHinge) expect(c).toBe(4);
    // hinge gather bit-exact vs isolated per-hinge gradients
    const n = mesh.count;
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += 0.001 * Math.sin(i * 1.3);
    const helem = new Float64Array(mesh.hinges.length * 12);
    const kb = 0.5 * (DEFAULT_MATERIAL.bendWarp + DEFAULT_MATERIAL.bendWeft);
    for (let hh = 0; hh < mesh.hinges.length; hh++) {
      const h = mesh.hinges[hh];
      const solo = new Float64Array(n * 3);
      addHingeGradient(x, solo, h.v0, h.v1, h.v2, h.v3, h.restAngle, h.edgeLen, h.areaSum, kb);
      // extract this hinge's 4 corner contributions
      const vs = [h.v0, h.v1, h.v2, h.v3];
      for (let c = 0; c < 4; c++) {
        helem[hh * 12 + c * 3] = solo[vs[c] * 3];
        helem[hh * 12 + c * 3 + 1] = solo[vs[c] * 3 + 1];
        helem[hh * 12 + c * 3 + 2] = solo[vs[c] * 3 + 2];
      }
    }
    const gathered = new Float64Array(n * 3);
    gatherAdd(gathered, helem, 12, hmap);
    // direct hinge-only accumulation (same per-hinge call sequence as
    // evalInternal; full-minus-membrane would inject cancellation rounding)
    const directH = new Float64Array(n * 3);
    for (let hh = 0; hh < mesh.hinges.length; hh++) {
      const h = mesh.hinges[hh];
      addHingeGradient(x, directH, h.v0, h.v1, h.v2, h.v3, h.restAngle, h.edgeLen, h.areaSum, kb);
    }
    for (let i = 0; i < n * 3; i++) expect(gathered[i]).toBe(directH[i]);
  });

  it("grids of several sizes: incidence totals exact", () => {
    for (const [nx, ny] of [[2, 2], [3, 5], [6, 6], [9, 7]] as Array<[number, number]>) {
      const g = buildGrid(nx, ny, 0.02 * nx, 0.02 * ny);
      const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
      const mmap = buildMembraneIncidence(mesh.indices, mesh.triCount, mesh.count);
      expect(mmap.offsets[mesh.count]).toBe(3 * mesh.triCount);
      const hmap = buildHingeIncidence(mesh.hinges, mesh.hinges.length, mesh.count);
      expect(hmap.offsets[mesh.count]).toBe(4 * mesh.hinges.length);
    }
  });
});
