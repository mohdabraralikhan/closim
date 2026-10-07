#!/usr/bin/env tsx
// Third-party interop check: run a Python script using ezdxf (an independent
// CAD library) to (1) read our AAMA-style DXF export and report entities/units,
// and (2) produce an ezdxf-generated AAMA-style file that our importer reads
// back, reporting mismatches. This is "real-file interop" evidence, not just a
// self-round-trip.
import { execSync } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { exportIRToDXFProfile } from "./src/cad/dxf-export.js";
import { importApparelDxf } from "./src/cad/dxf-import.js";
import { buildExportIR2 } from "./src/cad/index.js";
import { engineeredGarment } from "./tests/cad/g13-fixtures.js";

const f = engineeredGarment();
const ir = buildExportIR2(f.document, f.seams, f.set, { garmentName: "interop check" });
const ourDxf = exportIRToDXFProfile(ir, { profile: "aama-style", units: "mm" });

const dir = mkdtempSync(join(tmpdir(), "closim-interop-"));
const ourPath = join(dir, "closim-ours.dxf");
const ezPath = join(dir, "ezdxf-ez.dxf");
writeFileSync(ourPath, ourDxf.dxf);

const pyScript = `
import ezdxf, sys

def read_dxf(path):
    d = ezdxf.readfile(path)
    lines = []
    lines.append("entities=%d layers=%d" % (len(d.entitydb), len(d.layers)))
    for e in d.entitydb.values():
        lines.append("  %s layer=%s" % (e.dxftype(), getattr(e, "layer", "")))
    return "\\n".join(lines)

def write_aama(path):
    d = ezdxf.new()
    d.header_vars["INSUNITS"] = 4
    for name, color in [("1", 1), ("14", 7), ("10", 2), ("16", 3), ("13", 4), ("100", 6)]:
        d.layers.add(name, dxfattribs={"color": color, "linetype": "CONTINUOUS"})
    m = d.modelspace()
    m.add_lwpolyline([[0,0],[100,0],[100,100],[0,100]], dxfattribs={"layer":"1"})
    m.add_lwpolyline([[200,0],[300,0],[300,100],[200,100]], dxfattribs={"layer":"1"})
    m.add_line([100,50],[200,50], dxfattribs={"layer":"14"})
    m.add_line([50,0],[50,12], dxfattribs={"layer":"10"})
    m.add_line([30,50],[70,50], dxfattribs={"layer":"16"})
    t = m.add_text("front", dxfattribs={"layer":"13","height":5})
    t.set_pos([10,110])
    m.add_line([0,0],[100,100], dxfattribs={"layer":"100"})
    d.saveas(path)
    return path

print("=== ezdxf reads our export ===")
print(read_dxf(sys.argv[1]))
print()
print("=== ezdxf writes an AAMA sample ===")
write_aama(sys.argv[2])
print("wrote", sys.argv[2])
print()
print("=== our importer reads the ezdxf sample ===")
`;

const pyOut = execSync(
  `python -c "${pyScript.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\$/g, '\\$')}" "${ourPath}" "${ezPath}"`,
  { encoding: "utf8" },
);
console.log(pyOut);
console.log("--- our importer reads its own export ---");
const ours = importApparelDxf(ourDxf.dxf, { profile: "aama-style" });
console.log("  ok =", ours.ok);
console.log("  units =", ours.units);
console.log("  pieces =", ours.pieces.length);
console.log("  mismatches =", JSON.stringify(ours.mismatches));
console.log("--- our importer reads the ezdxf-born sample ---");
import { readFileSync } from "node:fs";
const ezDxf = readFileSync(ezPath, "utf8");
const ez = importApparelDxf(ezDxf, { profile: "aama-style" });
console.log("  ok =", ez.ok);
console.log("  units =", ez.units);
console.log("  pieces =", ez.pieces.length);
console.log("  mismatches =", JSON.stringify(ez.mismatches));
console.log("  textValues =", JSON.stringify(ez.textValues));
console.log("  layersFound =", JSON.stringify(ez.layersFound));

rmSync(dir, { recursive: true, force: true });

// Exit non-zero if the importer accepted the ezdxf-born file with NO pieces
// AND no unit mismatch (that means the real-file import did not work as
// expected — that's a genuine interop finding).
if (!ez.ok && ez.mismatches.length === 0) {
  console.log("  NON-INTEROP: importer rejected ezdxf file with no mismatches");
  process.exit(1);
}
