import { describe, expect, it } from "vitest";
import {
  attachCollisionObject, horizontalPlaneHeight, interpolateTransform,
  preprocessCollisionMesh, sampleMotion, toCollisionObject, toCollisionObjectMotion,
  type AvatarCollider, type MotionProvider, type Transform,
} from "../src/collision/avatar-collider.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";

const tri = {
  kind: "triangle-mesh" as const,
  positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
  indices: [0, 1, 2],
};

class LinearMotion implements MotionProvider {
  constructor(private readonly start: Transform, private readonly end: Transform, private readonly t0: number, private readonly t1: number) {}
  sample(timeSeconds: number): Transform {
    return interpolateTransform(this.start, this.end, (timeSeconds - this.t0) / (this.t1 - this.t0));
  }
}

function collider(overrides: Partial<AvatarCollider> = {}): AvatarCollider {
  return {
    id: "body-collider",
    bodyPart: "torso",
    geometry: tri,
    thicknessM: 0.004,
    ...overrides,
  };
}

describe("G7E avatar collider layer", () => {
  it("adapts a static triangle mesh into the existing static collision buffers", () => {
    const mesh = toCollisionObject(collider(), 0);
    expect(mesh.positions).toEqual(new Float32Array(tri.positions));
    expect(mesh.indices).toEqual(new Uint32Array(tri.indices));
    expect(mesh.triangleBodyParts).toEqual(["torso"]);
  });

  it("samples translation at t0/t1 and adapts both endpoint poses", () => {
    const motion = new LinearMotion(
      { translation: [0, 0, 0], rotation: [0, 0, 0, 1] },
      { translation: [2, -1, 3], rotation: [0, 0, 0, 1] }, 0, 2,
    );
    const sampled = sampleMotion(motion, 0, 2);
    expect(sampled.start.translation).toEqual([0, 0, 0]);
    expect(sampled.end.translation).toEqual([2, -1, 3]);
    const objects = toCollisionObjectMotion(collider({ motion }), 0, 2);
    expect(objects.start.positions[0]).toBe(0);
    expect(objects.end.positions[0]).toBe(2);
    expect(objects.end.positions[1]).toBe(-1);
  });

  it("applies rigid rotation to collider geometry", () => {
    const quarterTurn: Transform = { translation: [0, 0, 0], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] };
    const mesh = toCollisionObject(collider({ transform: quarterTurn }), 0);
    expect(mesh.positions[0]).toBeCloseTo(0);
    expect(mesh.positions[1]).toBeCloseTo(0);
    expect(mesh.positions[3]).toBeCloseTo(0);
    expect(mesh.positions[4]).toBeCloseTo(1);
  });

  it("flattens compound primitives with stable per-body-part triangle labels", () => {
    const compound: AvatarCollider = collider({
      geometry: {
        kind: "compound",
        children: [
          { geometry: { kind: "sphere", radiusM: 0.1, segments: 8, rings: 4 }, bodyPart: "head" },
          { geometry: { kind: "box", halfExtentsM: [0.2, 0.1, 0.1] }, bodyPart: "chest", transform: { translation: [0, -0.3, 0], rotation: [0, 0, 0, 1] } },
          { geometry: { kind: "capsule", radiusM: 0.05, cylinderLengthM: 0.3, segments: 8, hemisphereRings: 2 }, bodyPart: "arm" },
        ],
      },
    });
    const mesh = toCollisionObject(compound, 0);
    expect(mesh.indices.length).toBeGreaterThan(0);
    expect(new Set(mesh.triangleBodyParts)).toEqual(new Set(["head", "chest", "arm"]));
    expect(mesh.positions.length / 3).toBeGreaterThan(8);
  });

  it("preserves thickness metadata and attaches buffers via the public adapter boundary", () => {
    const mesh = toCollisionObject(collider({ thicknessM: 0.0125 }), 0);
    expect(mesh.thicknessM).toBe(0.0125);
    let attached: [Float32Array, Uint32Array] | null = null;
    attachCollisionObject({ setStaticMesh: (p, i) => { attached = [p, i]; } }, mesh);
    expect(attached?.[0]).toBe(mesh.positions);
    expect(attached?.[1]).toBe(mesh.indices);
  });

  it("interpolates transforms deterministically along the shortest quaternion arc", () => {
    const a: Transform = { translation: [0, 2, 0], rotation: [0, 0, 0, 1] };
    const b: Transform = { translation: [4, 0, 2], rotation: [0, 0, 1, 0] };
    const first = interpolateTransform(a, b, 0.25);
    const second = interpolateTransform(a, b, 0.25);
    expect(first).toEqual(second);
    expect(first.translation).toEqual([1, 1.5, 0.5]);
    expect(Math.hypot(...first.rotation)).toBeCloseTo(1);
    const antipodal = interpolateTransform(a, { ...b, rotation: [0, 0, -1, 0] }, 0.25);
    expect(antipodal).toEqual(first);
  });

  it("rejects invalid geometry, dimensions, transforms, and thickness", () => {
    expect(() => toCollisionObject(collider({ geometry: { kind: "sphere", radiusM: -1 } }), 0)).toThrow(/positive/);
    expect(() => toCollisionObject(collider({ geometry: { ...tri, indices: [0, 1, 7] } }), 0)).toThrow(/index/);
    expect(() => toCollisionObject(collider({ transform: { translation: [0, 0, 0], rotation: [0, 0, 0, 0] } }), 0)).toThrow(/nonzero/);
    expect(() => toCollisionObject(collider({ thicknessM: -0.1 }), 0)).toThrow(/nonnegative/);
    expect(() => toCollisionObject(collider({ geometry: { kind: "compound", children: [] } }), 0)).toThrow(/children/);
    expect(() => preprocessCollisionMesh({ kind: "triangle-mesh", positions: tri.positions, indices: [0, 1, 8] }, "part")).toThrow(/index/);
  });

  it("matches a horizontal static plane's distance with the existing floor representation", () => {
    const plane = toCollisionObject(collider({
      geometry: {
        kind: "triangle-mesh",
        positions: [-1, 0.25, -1, 1, 0.25, -1, 1, 0.25, 1, -1, 0.25, 1],
        indices: [0, 1, 2, 0, 2, 3],
      },
    }), 0);
    const y = horizontalPlaneHeight(plane);
    const cloth = new Float64Array([0, 0.251, 0, 0.1, 0.251, 0, 0, 0.251, 0.1]);
    const floor = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, new Uint32Array([0, 1, 2]));
    floor.setFloor(y);
    floor.beginStep(cloth);
    floor.updateActiveSet(cloth);
    const meshContacts = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, new Uint32Array([0, 1, 2]));
    meshContacts.setStaticMesh(plane.positions, plane.indices);
    meshContacts.beginStep(cloth);
    meshContacts.updateActiveSet(cloth);
    expect(floor.diag.minDistance).toBeCloseTo(0.001, 6);
    expect(meshContacts.diag.minDistance).toBeCloseTo(floor.diag.minDistance, 6);
    expect(() => horizontalPlaneHeight(toCollisionObject(collider(), 0))).toThrow(/horizontal plane/);
  });
});
