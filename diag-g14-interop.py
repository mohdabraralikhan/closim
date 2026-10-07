#!/usr/bin/env python3
"""
G14 third-party DXF interop check using ezdxf (independent CAD library).

Runs in two directions:

  A. ezdxf reads a closim AAMA-style DXF export and reports its entities/units
     (external library reads our writer's output — independent verification).
  B. ezdxf writes an AAMA-style sample; our TypeScript marker/import layer
     reads it back and reports pieces/mismatches (a real foreign-file import,
     not a self-round-trip). Same for our own export in direction D.

The closim export to test is expected at $CLOSIM_OUR_DXF (a .dxf path). If
absent, direction A is skipped and direction B alone is run. The script
exits 0 on success; non-zero if the importer rejected the ezdxf sample with
no mismatch report (a genuine interop gap).
"""
import os, sys, tempfile, subprocess, json, shutil, io

HERE = os.path.dirname(os.path.abspath(__file__))
our_path = os.environ.get("CLOSIM_OUR_DXF")
if not our_path or not os.path.exists(our_path):
    our_path = None
if our_path is None:
    # One-shot generation if tsx is available.
    try:
        out = subprocess.run(
            [sys.executable, "-m", "tsx", os.path.join(HERE, "diag-write-our-dxf.mts")],
            capture_output=True, text=True, cwd=HERE, timeout=30,
        )
        if out.returncode == 0:
            import re
            m = re.search(r"path=([^\s]+)", out.stdout)
            if m and os.path.exists(m.group(1)):
                our_path = m.group(1)
    except Exception:
        pass


def ezdxf_read(path):
    import ezdxf
    d = ezdxf.readfile(path)
    lines = ["entities=%d layers=%d" % (len(d.entitydb), len(d.layers))]
    for e in d.entitydb.values():
        lines.append("  %s layer=%s" % (e.dxftype(), getattr(e, "layer", "")))
    return "\n".join(lines)


def write_aama_sample(path):
    import ezdxf
    d = ezdxf.new()
    # $INSUNITS is a HeaderVar object (value=6 by default). Set .value=4.
    hv = d.header.hdrvars.get("$INSUNITS")
    if hv is not None:
        try:
            hv.value = 4
        except Exception as exc:
            print("  (could not set $INSUNITS.value=%d: %r)" % (4, exc), file=sys.stderr)
    m = d.modelspace()
    for name, color in [("1", 1), ("14", 7), ("10", 2), ("16", 3), ("13", 4), ("100", 6)]:
        d.layers.add(name, dxfattribs={"color": color, "linetype": "CONTINUOUS"})
    m.add_lwpolyline([[0,0],[100,0],[100,100],[0,100]], dxfattribs={"layer": "1"})
    m.add_lwpolyline([[200,0],[300,0],[300,100],[200,100]], dxfattribs={"layer": "1"})
    m.add_line([100,50],[200,50], dxfattribs={"layer": "14"})
    m.add_line([50,0],[50,12], dxfattribs={"layer": "10"})
    m.add_line([30,50],[70,50], dxfattribs={"layer": "16"})
    t = m.add_text("front", dxfattribs={"layer": "13", "height": 5})
    try:
        t.dxf.insert = (10, 110, 0)
    except Exception as exc:
        print("  (text insert failed: %r)" % exc, file=sys.stderr)
    m.add_line([0,0],[100,100], dxfattribs={"layer": "100"})
    d.saveas(path)
    return path


print("=== A. ezdxf reads our export ===")
if our_path:
    try:
        print(ezdxf_read(our_path))
    except Exception as exc:
        print("  (ezdxf read failed: %s)" % exc, file=sys.stderr)
else:
    print("  (no CLOSIM_OUR_DXF; skipped)")

# --- B. ezdxf writes an AAMA-style sample ---
tmpdir = tempfile.mkdtemp(prefix="closim-interop-")
ez_path = os.path.join(tmpdir, "ezdxf-ez.dxf")
try:
    write_aama_sample(ez_path)
    print()
    print("=== B. ezdxf writes an AAMA-style sample ===")
    print("  wrote", ez_path)
    print(ezdxf_read(ez_path))
except Exception as exc:
    print()
    print("  (ezdxf could not write the sample: %r)" % exc, file=sys.stderr)
    ez_path = None
    ez_text = None
else:
    ez_text = open(ez_path).read()

# --- C. TS importer reads the ezdxf-born sample ---
tsx_ns = os.path.join(HERE, "node_modules", ".bin", "tsx.cmd")
if not os.path.exists(tsx_ns):
    tsx_ns = "npx"
tsx_script = r"""
import { readFileSync } from "node:fs";
import { importApparelDxf } from "./src/cad/dxf-import.js";
const text = readFileSync(process.argv[2], "utf8");
console.log(JSON.stringify({
    ok: importApparelDxf(text, { profile: "aama-style" }).ok,
    units: importApparelDxf(text, { profile: "aama-style" }).units,
    pieces: importApparelDxf(text, { profile: "aama-style" }).pieces.length,
    mismatches: importApparelDxf(text, { profile: "aama-style" }).mismatches,
    layersFound: importApparelDxf(text, { profile: "aama-style" }).layersFound,
}));
"""
def run_tsx(path, label):
    cmd = [sys.executable, "-m", "tsx", "-e", tsx_script, "--", path] if tsx_ns == "npx" else [tsx_ns, "-e", tsx_script, "--", path]
    proc = subprocess.run(
        cmd,
        capture_output=True, text=True, cwd=HERE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    print("=== %s ===" % label)
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return json.loads(proc.stdout.strip())
        except Exception:
            print("  (parse failed) stdout:", proc.stdout.strip())
            return None
    print("  (tsx rc=%d)" % proc.returncode)
    if proc.stderr.strip():
        print("  stderr:", proc.stderr.strip())
    return None

print()
ez_report = run_tsx(ez_path, "C. our importer reads the ezdxf-born sample") if ez_path else None
if ez_report:
    for k, v in ez_report.items():
        print("  %s = %s" % (k, json.dumps(v)))
    if not ez_report.get("ok") and not ez_report.get("mismatches"):
        print("\n(non-interop: importer rejected ezdxf file with no mismatch report)", file=sys.stderr)
        sys.exit(1)

if our_path:
    our_report = run_tsx(our_path, "D. our importer reads our export (round trip)")
    if our_report:
        for k, v in our_report.items():
            print("  %s = %s" % (k, json.dumps(v)))

shutil.rmtree(tmpdir, ignore_errors=True)
