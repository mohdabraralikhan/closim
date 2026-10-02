// Swept-AABB helpers over SoA position arrays (length 3n).

export interface Aabb {
  minX: number; minY: number; minZ: number;
  maxX: number; maxY: number; maxZ: number;
}

/** Swept bounds of triangle (i0,i1,i2) between x0 and x1, expanded by pad. */
export function sweptTriAabb(
  x0: ArrayLike<number>, x1: ArrayLike<number>,
  i0: number, i1: number, i2: number, pad: number,
): Aabb {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const ids = [i0, i1, i2];
  for (const v of ids) {
    for (const x of [x0, x1]) {
      const px = x[v * 3], py = x[v * 3 + 1], pz = x[v * 3 + 2];
      if (px < minX) minX = px;
      if (py < minY) minY = py;
      if (pz < minZ) minZ = pz;
      if (px > maxX) maxX = px;
      if (py > maxY) maxY = py;
      if (pz > maxZ) maxZ = pz;
    }
  }
  return {
    minX: minX - pad, minY: minY - pad, minZ: minZ - pad,
    maxX: maxX + pad, maxY: maxY + pad, maxZ: maxZ + pad,
  };
}

export function overlaps(a: Aabb, b: Aabb): boolean {
  return (
    a.minX <= b.maxX && a.maxX >= b.minX &&
    a.minY <= b.maxY && a.maxY >= b.minY &&
    a.minZ <= b.maxZ && a.maxZ >= b.minZ
  );
}
