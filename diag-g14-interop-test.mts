import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { importApparelDxf } from "./src/cad/dxf-import.js";

async function main() {
  // Direct ezdxf write via python subprocess.
  const ez_dir = join(tmpdir(), "closim-interop");
  const ez_path = join(ez_dir, "ezdxf-ez.dxf");
  // write via python inline
  const ez_script = `
import ezdxf, sys, os
os.makedirs(sys.argv[1], exist_ok=True)
d = ezdxf.new()
hv = d.header.hdrvars.get("$INSUNITS")
try:
    hv.value = 4
except Exception as exc:
    print("HVERR", file=sys.stderr)
d.saveas(os.path.join(sys.argv[1], "ezdxf-ez.dxf"))
`;
  const ez_write_out = require("child_process").execSync(`python -c "${ez_script.replace(/"/g, '\\"').replace(/`/g, '\\`').replace(/\$/g, '\\$')}" "${ez_dir}"`, { encoding: "utf8" });
  // read it back via ezdxf inline
  const ez_read_out = require("child_process").execSync(`python -c "
import ezdxf, sys
d = ezdxf.readfile(sys.argv[1])
lines = ['entities=%d layers=%d' % (len(d.entitydb), len(d.layers))]
for e in d.entitydb.values():
    lines.append('  %s layer=%s' % (e.dxftype(), getattr(e, 'layer', '')))
print(chr(10).join(lines))
" "${ez_path}"`, { encoding: "utf8" });
  console.log("=== B. ezdxf writes an AAMA-style sample ===");
  console.log("  wrote", ez_path);
  console.log(ez_read_out);

  const ez_text = readFileSync(ez_path, "utf8");
  const ez_report = importApparelDxf(ez_text, { profile: "aama-style" });
  console.log("=== C. our importer reads the ezdxf-born sample ===");
  console.log(JSON.stringify({
    ok: ez_report.ok,
    units: ez_report.units,
    pieces: ez_report.pieces.length,
    mismatches: ez_report.mismatches,
    layersFound: ez_report.layersFound,
  }, null, 2));

  if (!ez_report.ok && !ez_report.mismatches) {
    console.error("NON-INTEROP: importer rejected ezdxf file with no mismatch report");
    process.exit(1);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
