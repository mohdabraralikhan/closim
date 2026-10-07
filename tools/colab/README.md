# Closim on Colab (T4)

Build + fast CPU tests + Anny (Apache-2.0) avatar bake on a Colab GPU VM,
driven by your local agent through
[colab-mcp](https://github.com/googlecolab/colab-mcp)
(a plain `colab run`-style CLI is Linux/macOS-only, so the notebook +
agent flow below is the Windows-safe path).

## Setup (once)

1. Open `tools/colab/closim_t4_notebook.ipynb` in Colab (File → Upload
   notebook) on a **T4 GPU** runtime.
2. Attach your MCP-capable agent to that browser session with colab-mcp.
   With `uv` installed:
   ```json
   "mcpServers": {
     "colab-mcp": {
       "command": "uvx",
       "args": ["git+https://github.com/googlecolab/colab-mcp"],
       "timeout": 30000
     }
   }
   ```
   Supported clients: Gemini CLI, Claude Code, Windsurf (needs
   `notifications/tools/list_changed` support, running locally).

## What the notebook does (top to bottom, idempotent)

1. **env** — `nvidia-smi`, torch/CUDA sanity (torch is preinstalled on GPU images).
2. **node** — installs Node 20 via nodesource if missing.
3. **repo** — clones (or pulls) `https://github.com/mohdabraralikhan/closim.git`
   at `master` into `/content/closim`.
4. **build** — `npm ci` + `npm run build` (tsc must be clean).
5. **tests** — fast CPU vitest subset only (no GPU/adversarial/heavy sim files).
6. **avatar probe** — `pip install anny`, prints phenotype labels + bone count.
7. **avatar bake** — adult-female bind pose, unit auto-detect, feet on y=0,
   centered XZ, height sanity gate (1.4–2.0 m), exports OBJ + stats JSON to
   `/content/closim-out/`.

Headless alternative: `tools/colab/closim_t4_job.py` runs the same stages
from a single script (for CLIs that support `exec -f` on a session).

## Get the avatar back

Download `/content/closim-out/closim-avatar-female-168.obj` (+ stats JSON)
into the repo's `assets/` folder, then validate locally through the repo
avatar pipeline (parse → largest component → decimation sweep → proxy):

```bash
npx tsx tools/avatar-check.mts ./assets/closim-avatar-female-168.obj
```

The repo side is ready: `src/garment/avatar-mesh.ts` (OBJ loader, cleanup,
decimation, `collision` proxy on `AvatarSpec`) with `collision`-aware
`rebuildGarment`, covered by `tests/garment/avatar-mesh.test.ts`.
