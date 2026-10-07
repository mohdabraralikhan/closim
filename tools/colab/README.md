# Closim on Colab (T4)

Runs the closim build + fast CPU tests and bakes an open-source Anny
(Apache-2.0) avatar OBJ on a Colab GPU VM, driven from your terminal with
[colab-cli](https://github.com/googlecolab/google-colab-cli)
(Linux/macOS; on Windows use WSL or the `termios` stub documented below).

## One-time setup

```bash
pip install google-colab-cli   # or: uv tool install google-colab-cli
colab sessions                 # browser OAuth on first run, then cached
gh auth login                  # for pushing the repo
```

Windows note: upstream doesn't support Windows (`import termios` fails).
A minimal stub unblocks the non-interactive commands (`new`, `exec`,
`run`, `install`, `download`, `stop`); interactive `console`/`repl` still
need a Unix TTY — use WSL for those:

```powershell
# PowerShell: create the stub once, then always set PYTHONPATH first
mkdir $env:TEMP\colab-stub
"TCSANOW = 0`nTCSADRAIN = 1`nTCSAFLUSH = 2`nECHO = 8`nICANON = 2`nVMIN = 6`nVTIME = 5`ndef tcgetattr(fd): raise OSError('no TTY on Windows')`ndef tcsetattr(fd, when, attrs): raise OSError('no TTY on Windows')`n" | Out-File $env:TEMP\colab-stub\termios.py -Encoding ascii
$env:PYTHONPATH = "$env:TEMP\colab-stub"
colab --help
```

## Push the repo, then run the whole job on a T4

```bash
# from the closim checkout
git push origin master   # after gh auth login + remote add (first time only)

# ephemeral T4: provision, run, teardown in one command
colab run --gpu T4 tools/colab/closim_t4_job.py -- \
  --repo https://github.com/<you>/closim.git
```

Or on a persistent session (keeps the VM for inspection):

```bash
colab new -s closim --gpu T4
colab exec -s closim -f tools/colab/closim_t4_job.py -- --repo https://github.com/<you>/closim.git
```

## Get the avatar back

The job writes `/content/closim-out/closim-avatar-female-168.obj` (+ stats
JSON) on the VM:

```bash
colab download closim-avatar-female-168.obj ./assets/closim-avatar-female-168.obj
colab stop -s closim
```

Then validate locally with the repo pipeline:

```bash
npx tsx tools/avatar-check.mts ./assets/closim-avatar-female-168.obj
```

## What the job does

1. **env** — `nvidia-smi`, torch/CUDA sanity (torch is preinstalled on GPU images).
2. **node** — installs Node 20 via nodesource if missing.
3. **repo** — clones (or pulls) closim at `--branch` (default `master`).
4. **build** — `npm ci` + `npm run build` (tsc must be clean).
5. **tests** — fast CPU vitest subset only (no GPU/adversarial/heavy sim files).
6. **avatar** — `pip install anny`, generates an adult-female bind-pose
   mesh, auto-detects units, grounds feet at y=0, centers XZ, exports OBJ.

Flags: `--skip-tests`, `--skip-avatar`, `--workdir`, `--out-dir`.
Exit code 2 with `[FAIL] <stage>` markers if any stage fails (env/repo
failures abort the job).
