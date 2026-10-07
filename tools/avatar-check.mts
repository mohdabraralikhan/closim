// Validate a downloaded body OBJ through the repo avatar pipeline:
// parse -> stats -> largest component -> decimation sweep -> spec + proxy.
// Usage: npx tsx tools/avatar-check.mts ./assets/closim-avatar-female-168.obj
import { readFileSync } from "node:fs";
import {
  decimateMesh,
  keepLargestComponent,
  makeMeshAvatar,
  meshStats,
  parseAvatarOBJ,
} from "../src/garment/avatar-mesh.js";
import { validateAvatarSpec } from "../src/garment/avatar.js";

const file = process.argv[2];
if (!file) {
  console.error("usage: npx tsx tools/avatar-check.mts <avatar.obj>");
  process.exit(1);
}
const raw = parseAvatarOBJ(readFileSync(file, "utf8"));
console.log("parsed:", JSON.stringify(meshStats(raw.positions, raw.indices)));
const main = keepLargestComponent(raw.positions, raw.indices);
console.log("largest:", JSON.stringify(meshStats(main.positions, main.indices)));
for (const cellM of [0.01, 0.02, 0.03, 0.05]) {
  try {
    const proxy = decimateMesh(main.positions, main.indices, { cellM });
    console.log(`cell ${cellM}:`, JSON.stringify(meshStats(proxy.positions, proxy.indices)));
  } catch (error) {
    console.log(`cell ${cellM}: FAILED (${error instanceof Error ? error.message : String(error)})`);
  }
}
const spec = makeMeshAvatar(main, "baked", { proxyCellM: 0.02 });
validateAvatarSpec(spec);
console.log(
  `spec ok: render tris=${spec.indices.length / 3}, ` +
    `proxy tris=${spec.collision!.indices.length / 3}`,
);
