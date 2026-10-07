"""Closim T4 job: build + fast CPU tests + Anny avatar bake.

Runs on a fresh Colab GPU VM (see README in this folder). Designed to run
via `colab run --gpu T4 tools/colab/closim_t4_job.py -- --repo <url>` or
`colab exec -f` on a persistent session. Every stage prints [STAGE]/[OK]/[FAIL]
markers; artifacts land in --out-dir for `colab download`.

Stages:
  1. env      - nvidia-smi, torch/cuda sanity
  2. node     - install Node 20 via nodesource if `node` is missing
  3. repo     - git clone (or pull) the closim repo at --branch
  4. build    - npm ci + `npm run build` (tsc, must be clean)
  5. tests    - fast CPU vitest subset (no GPU/adversarial/heavy sim files)
  6. avatar   - pip install anny + generate adult female A-pose OBJ + stats
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

FAST_TESTS = [
    "tests/smoke.test.ts",
    "tests/garment/avatar-mesh.test.ts",
    "tests/garment/sewing.test.ts",
    "tests/production/workflow.test.ts",
    "tests/cad/g13-adversarial.test.ts",
    "tests/construction/g16-workflow.test.ts",
]


def sh(cmd, cwd=None, check=True):
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    sys.stderr.write(r.stderr[-4000:])
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def stage_env():
    print("[STAGE] env", flush=True)
    sh(["nvidia-smi", "-L"])
    import torch

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}", flush=True)


def stage_node():
    print("[STAGE] node", flush=True)
    if shutil.which("node") and shutil.which("npm"):
        sh(["node", "--version"])
        return
    sh(["bash", "-lc",
        "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && "
        "sudo apt-get install -y nodejs"])
    sh(["node", "--version"])


def stage_repo(repo, branch, workdir):
    print("[STAGE] repo", flush=True)
    if os.path.isdir(os.path.join(workdir, ".git")):
        sh(["git", "-C", workdir, "fetch", "origin"])
        sh(["git", "-C", workdir, "checkout", branch])
        sh(["git", "-C", workdir, "pull", "--ff-only"])
    else:
        os.makedirs(workdir, exist_ok=True)
        sh(["git", "clone", "--branch", branch, "--depth", "1", repo, workdir])
    sh(["git", "-C", workdir, "log", "--oneline", "-3"])


def stage_build(workdir):
    print("[STAGE] build", flush=True)
    try:
        sh(["npm", "ci", "--no-audit", "--no-fund"], cwd=workdir)
    except RuntimeError:
        print("npm ci failed, falling back to npm install", flush=True)
        sh(["npm", "install", "--no-audit", "--no-fund"], cwd=workdir)
    sh(["npm", "run", "build"], cwd=workdir)
    print("[OK] tsc clean", flush=True)


def stage_tests(workdir):
    print("[STAGE] tests", flush=True)
    missing = [t for t in FAST_TESTS if not os.path.exists(os.path.join(workdir, t))]
    if missing:
        raise RuntimeError(f"test files missing: {missing}")
    sh(["npx", "vitest", "run", *FAST_TESTS], cwd=workdir)
    print("[OK] fast CPU tests pass", flush=True)


AVATAR_SCRIPT = r"""
import json
import torch
import anny
import trimesh

print("anny model labels:")
model_probe = anny.Anny()
print("phenotypes:", list(model_probe.phenotype_labels))
print("bones:", model_probe.bone_count)

# Adult female ~1.68m: map interpretable phenotype sliders heuristically.
# Unknown keys fall back to neutral 0.5.
phen = {key: 0.5 for key in model_probe.phenotype_labels}
def set_like(*needles, value):
    for key in phen:
        kl = key.lower()
        if all(n in kl for n in needles):
            phen[key] = value

set_like("gender", value=0.0)   # 0 = female end (verify in labels printout)
set_like("female", value=1.0)
set_like("male", value=0.0)
set_like("age", value=0.45)     # adult, not elderly
set_like("baby", value=0.0)
set_like("child", value=0.0)
set_like("height", value=0.55)
set_like("weight", value=0.45)
set_like("muscle", value=0.45)
print("phenotype_kwargs:", json.dumps(phen, indent=None))

model = anny.Anny().to(dtype=torch.float32)
pose = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)
out = model(pose_parameters=pose, phenotype_kwargs=phen)
verts = out["vertices"].squeeze(dim=0).detach().cpu().numpy()
faces = model.faces
print(f"raw verts={verts.shape} faces={faces.shape}")
print(f"raw bounds min={verts.min(axis=0).tolist()} max={verts.max(axis=0).tolist()}")

# Unit auto-detect: Anny/MakeHuman heritage may be m, dm, or cm.
height = float(verts[:, 1].max() - verts[:, 1].min())
scale = 1.0
if height > 50:
    scale = 0.01
elif height > 5:
    scale = 0.1
verts = verts * scale
print(f"height_raw={height:.3f} scale={scale} height_m={height * scale:.3f}")

# Feet on y=0, centered XZ.
verts[:, 1] -= verts[:, 1].min()
verts[:, 0] -= (verts[:, 0].min() + verts[:, 0].max()) / 2
verts[:, 2] -= (verts[:, 2].min() + verts[:, 2].max()) / 2

mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export(OBJ_OUT)
stats = {
    "verts": int(len(verts)),
    "tris": int(len(faces)),
    "min": [float(v) for v in verts.min(axis=0)],
    "max": [float(v) for v in verts.max(axis=0)],
    "height_m": float(verts[:, 1].max()),
    "watertight": bool(mesh.is_watertight),
    "phenotype": phen,
}
with open(STATS_OUT, "w") as f:
    json.dump(stats, f, indent=2)
print("avatar stats:", json.dumps(stats, indent=2))
"""


def stage_avatar(out_dir):
    print("[STAGE] avatar", flush=True)
    sh([sys.executable, "-m", "pip", "install", "--quiet", "anny"])
    gen_path = os.path.join(out_dir, "_gen_avatar.py")
    obj_path = os.path.join(out_dir, "closim-avatar-female-168.obj")
    stats_path = os.path.join(out_dir, "closim-avatar-female-168.stats.json")
    with open(gen_path, "w") as f:
        f.write(AVATAR_SCRIPT.replace("OBJ_OUT", repr(obj_path)).replace("STATS_OUT", repr(stats_path)))
    env = dict(os.environ)
    env.setdefault("ANNY_CACHE_DIR", os.path.join(out_dir, ".anny-cache"))
    r = subprocess.run([sys.executable, gen_path], capture_output=True, text=True, env=env)
    sys.stdout.write(r.stdout[-6000:])
    sys.stderr.write(r.stderr[-6000:])
    if r.returncode != 0:
        raise RuntimeError("avatar generation failed")
    with open(stats_path) as f:
        stats = json.load(f)
    h = stats["height_m"]
    if not (1.4 <= h <= 2.0):
        raise RuntimeError(f"avatar height {h:.2f}m outside sane range; check pose/units")
    print(f"[OK] avatar baked: {stats['verts']} verts, {stats['tris']} tris, height {h:.3f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="closim git URL")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--workdir", default="/content/closim")
    ap.add_argument("--out-dir", default="/content/closim-out")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-avatar", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    failed = []
    for name, fn in [
        ("env", stage_env),
        ("node", lambda: stage_node()),
        ("repo", lambda: stage_repo(args.repo, args.branch, args.workdir)),
        ("build", lambda: stage_build(args.workdir)),
        ("tests", lambda: stage_tests(args.workdir)),
        ("avatar", lambda: stage_avatar(args.out_dir)),
    ]:
        if name == "tests" and args.skip_tests:
            print("[SKIP] tests", flush=True)
            continue
        if name == "avatar" and args.skip_avatar:
            print("[SKIP] avatar", flush=True)
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - keep the job going, report at end
            print(f"[FAIL] {name}: {e}", flush=True)
            failed.append(name)
            if name in ("env", "repo"):
                break
    print("=" * 60, flush=True)
    if failed:
        print(f"JOB DONE WITH FAILURES: {failed}", flush=True)
        sys.exit(2)
    print("JOB DONE: all stages ok", flush=True)


if __name__ == "__main__":
    main()
