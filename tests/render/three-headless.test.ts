import { describe, expect, it } from "vitest";
import * as THREE from "three";

describe("three headless import", () => {
  it("constructs scene-graph objects without a GL context", () => {
    const scene = new THREE.Scene();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
    geo.setIndex([0, 1, 2]);
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: 0xff0000, roughness: 0.8 });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    expect(scene.children).toHaveLength(1);
    expect((geo.getAttribute("normal") as THREE.BufferAttribute).count).toBe(3);
    geo.dispose();
    (mat as THREE.Material).dispose();
  });
});
