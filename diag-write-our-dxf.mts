import { writeFileSync } from "node:fs";
import { exportIRToDXFProfile } from "./src/cad/dxf-export.js";
import { buildExportIR2 } from "./src/cad/index.js";
import { engineeredGarment } from "./tests/cad/g13-fixtures.js";
import { tmpdir } from "node:os";
import { join } from "node:path";

const f = engineeredGarment();
const ir = buildExportIR2(f.document, f.seams, f.set, { garmentName: "interop check" });
const dxf = exportIRToDXFProfile(ir, { profile: "aama-style", units: "mm" });
const p = join(tmpdir(), "closim-ours.dxf");
writeFileSync(p, dxf.dxf);
console.log(`path=${p} bytes=${dxf.dxf.length}`);
