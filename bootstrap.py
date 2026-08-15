#!/usr/bin/env python3
"""
ShelfSense-Q repo bootstrap.

Run ONCE, on the laptop, inside an empty folder:

    mkdir shelfsense-q && cd shelfsense-q
    python bootstrap.py

It creates the folder tree and every starter file. It does NOT touch git or
the network — it only writes files, so it is safe to inspect first and safe
to re-run (existing files are skipped, never overwritten).
"""

from pathlib import Path

ROOT = Path(".")

# --------------------------------------------------------------------------
# Directory tree
# --------------------------------------------------------------------------

DIRS = [
    "configs",
    "data/splits",            # split definitions (committed, small text)
    "data/raw",               # SKU-110K lands here (gitignored)
    "data/calibration",       # INT8 calibration images (gitignored)
    "src/data",
    "src/train",
    "src/eval",
    "src/export",
    "src/qualcomm",
    "src/openvocab",
    "src/pipeline",
    "app",
    "results/baselines",      # committed: small text/csv/json only
    "results/quantization",
    "results/logs",
    "scripts",
    "notebooks",
    "docs",
]

# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

FILES = {}

# ---------------------------------------------------------------- gitignore
FILES[".gitignore"] = """\
# ============================================================
# SECRETS — never commit. Check this section first, always.
# ============================================================
.env
.env.*
!.env.example
*.token
*_token*
secrets/
.qai_hub/
~/.qai_hub/

# ============================================================
# ENVIRONMENTS
# ============================================================
.venv*/
venv/
env/
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/

# ============================================================
# DATA — SKU-110K is ~13.6 GB. Never in git.
# Split definition files in data/splits/ ARE committed (see below).
# ============================================================
data/raw/
data/calibration/
data/interim/
*.tar
*.tar.gz
*.zip

# ...but DO commit the split definitions. They are small text files
# and they are what makes your evaluation reproducible.
!data/splits/
!data/splits/**

# ============================================================
# MODEL ARTIFACTS — too big for git. Use HF Hub or W&B artifacts.
# ============================================================
*.pth
*.pt
*.pt2
*.onnx
*.engine
*.plan
*.tflite
*.dlc
*.bin
*.xml
checkpoints/
outputs/
exports/

# ============================================================
# TOOLING CACHES
# ============================================================
wandb/
runs/
.cache/
*.log
!results/logs/*.md

# ============================================================
# OS / EDITOR
# ============================================================
.DS_Store
Thumbs.db
desktop.ini
.vscode/settings.json
.idea/
"""

# ---------------------------------------------------------------- env example
FILES[".env.example"] = """\
# Copy this file to .env and fill it in.
# .env is gitignored. NEVER commit the real one.
#
#   cp .env.example .env

# Which machine is this? -> laptop | workstation
# Controls which configs/machine.*.yaml gets loaded.
MACHINE=laptop

# Qualcomm AI Hub API token.
# Get it at: workbench.aihub.qualcomm.com -> Account -> Settings -> API Token
# NOTE: `qai-hub configure` also stores this in ~/.qai_hub/client.ini
#       That file is outside the repo, so it is safe, but do not copy it in.
QAI_HUB_API_TOKEN=

# Weights & Biases
WANDB_API_KEY=
WANDB_PROJECT=shelfsense-q

# Hugging Face (for weight handoff between machines + Spaces demo)
HF_TOKEN=
HF_REPO=your-username/shelfsense-q-weights
"""

# ---------------------------------------------------------------- CLAUDE.md
FILES["CLAUDE.md"] = """\
# ShelfSense-Q — project context for Claude Code

Read this fully at the start of every session.

## What this project is

Retail shelf detection on Qualcomm NPUs. RF-DETR fine-tuned on SKU-110K,
exported to ONNX, quantized to w8a8, compiled and profiled on real Qualcomm
hardware via AI Hub Workbench's cloud device farm.

**The headline result:** Qualcomm publishes RF-DETR latency in FLOAT precision
only, across ~20 chipsets. Nobody has published the INT8 accuracy/latency
trade-off. That gap is the contribution. Everything else supports it.

## Hard deadline

The RTX 5070 workstation is unavailable after **2026-09-15**.
All training, TensorRT, and CUDA-dependent work must be committed by
**2026-09-13**. Never propose a plan that assumes CUDA after that date.
AI Hub and OpenVINO work survives the cliff and can run on the laptop.

## Two machines, two environments — never mix them

| Env | Machine | Contents |
|---|---|---|
| `.venv-train` | workstation only | torch cu128 (Blackwell sm_120), rfdetr, TensorRT |
| `.venv-qai` | both machines | qai-hub, qai-hub-models |

`qai-hub-models` pins its own torch and WILL overwrite a cu128 build,
silently breaking sm_120. Never `pip install` a qai package into `.venv-train`.
The two environments communicate only through `.onnx` files on disk.

## HARD RULES — violating any of these invalidates the project

1. **Calibration data comes from the TRAIN split only.**
   Never val, never test. Calibrating INT8 on evaluation images contaminates
   the headline number and produces no error message. If you are ever unsure
   which split an image came from, stop and check `data/splits/`.

2. **Splits are fixed and committed.** Use `data/splits/{train,val,test}.txt`.
   Never re-split, never shuffle with a random seed at runtime.

3. **Never invent a benchmark number.** If it has not been measured, write
   `TODO(measure)`. Placeholder numbers that look plausible are how portfolio
   projects end up lying.

4. **Never commit secrets.** No tokens in code, configs, notebooks, or commit
   messages. Everything sensitive goes in `.env`.

5. **FP32 and INT8 must be evaluated through identical preprocessing.**
   If the resize, normalization, or letterbox differs by even a little, the
   measured "INT8 cost" is a bug, not a finding. RF-DETR v1.9.1 release notes
   flag exactly this class of drift between exported models and `predict()`.

6. **Log every compile/quantize failure to FAILURES.md** with its QNN error
   code and what changed. These are project assets, not embarrassments.

## Stack notes

- `rfdetr` is pure PyTorch. Do NOT introduce mmcv / OpenMMLab — no sm_120 wheels.
- Python 3.10 <= version < 3.14 (qai-hub-models requirement).
- QNN is now called **QAIRT**. Older docs say QNN.
- `submit_compile_and_quantize_jobs` was REMOVED. Use separate jobs.
- TorchScript compile is deprecated. Export ONNX.
- Check whether the installed version wants `hub.submit_*` (module-level) or
  `client = hub.Client()` then `client.submit_*`. Both appear in the docs.

## Validation anchor

Qualcomm publishes RF-DETR-base 560x560 float at **79.9 ms QNN_DLC on QCS9075**.
Our float compile should land near this. If it doesn't, it is a bug in our
setup, not a discovery. See `results/baselines/`.

## Working style

- Small, reviewable diffs. No large rewrites without asking.
- Explain reasoning BEFORE writing code for anything architectural.
- State uncertainty explicitly, especially about API signatures that changed.
- After any non-trivial decision, remind me to add a line to DECISIONS.md.
"""

# ---------------------------------------------------------------- HANDOFF
FILES["HANDOFF.md"] = """\
# Handoff

Single source of truth for "where am I and what's next".
Update this **before you stop working on a machine**, every time.
It is the thing that prevents losing a day to confusion on Monday morning.

---

## Current state

- **Last active machine:** laptop
- **Last updated:** 2026-08-15
- **Branch:** main

## Done

- [ ] nothing yet

## Next task

Bootstrap the repo and push to GitHub.

## Blockers

None.

## Notes for the other machine

Nothing yet.

---

### The relay rule

Only ONE machine is active at a time. Before leaving a machine:

```
git add -A
git commit -m "wip: <what you did>"
git push
```

Before starting on a machine:

```
git pull --rebase
```

If you forget the pull, you get a conflict and lose 20 minutes. If you forget
the push, you lose the work entirely until you're back at that machine.
"""

# ---------------------------------------------------------------- DECISIONS
FILES["DECISIONS.md"] = """\
# Decisions

One line every time a non-trivial choice gets made. Five minutes a day.

By week five this file is simultaneously:
- your interview preparation,
- the outline of your technical writeup,
- the "engineering trade-offs" section of the README.

Format: `- DATE · WHAT you chose. WHY. What you gave up.`

---

- 2026-08-15 · Qualcomm AI Hub instead of a Jetson Orin Nano.
  No hardware purchase, real silicon in a cloud device farm, and — decisive —
  it keeps working after the RTX 5070 goes away on Sep 15. Gave up: the
  CUDA/TensorRT/DeepStream keyword cluster on the edge device itself, though
  TensorRT is still covered on the 5070 before the cliff.

- 2026-08-15 · Two isolated virtual environments rather than one.
  qai-hub-models pins its own torch and would overwrite the cu128 Blackwell
  build, silently breaking sm_120. Cost: an ONNX file on disk as the handoff
  between them, instead of a single in-process pipeline.

- 2026-08-15 · Target QCS9075 (Dragonwing IQ-9075) as the primary device.
  RF-DETR is not listed as supported on the RB3 Gen 2 Vision Kit (QCS6490),
  which was the more obvious IoT board. IQ-9075 also fits the narrative better:
  industrial edge-AI box, dual HTP, 16x 1080p60 decode for multi-camera.
"""

# ---------------------------------------------------------------- FAILURES
FILES["FAILURES.md"] = """\
# Failure atlas

Every compile, quantize, export, or profile failure — what broke, the error
code, what changed, whether it worked.

This is the highest-value-per-hour artifact in the project. It costs nothing
but honesty, and it is the section of the README that separates someone who
ran a tutorial from someone who debugged a deployment.

Template:

```
## YYYY-MM-DD — short title

**Context:** what I was trying to do
**Error:** exact message + code
**Hypothesis:** what I thought was wrong
**Fix:** what I actually changed
**Outcome:** worked / didn't / partially
**Lesson:** the transferable bit
```

Useful reference — common QAIRT/QNN error meanings:

- **3110** — input/output types not supported by the QNN backend. Usually an
  unquantized tensor passed through a quantized graph. Fix: quantize or
  re-quantize the model.
- **Unsupported rank** — most ops are <=5D; some do 5D; none do >5D.
- **Unsupported type** — common with int32 layers.

---

*(no entries yet)*
"""

# ---------------------------------------------------------------- SETUP
FILES["docs/SETUP.md"] = """\
# Setup — both machines

## 0. Prerequisites

- Python **3.10 <= v < 3.14**  (`python --version`)
- git
- A Qualcomm ID with AI Hub Workbench access

## 1. Clone

```bash
git clone https://github.com/<you>/shelfsense-q.git
cd shelfsense-q
```

## 2. Secrets

```bash
cp .env.example .env
```

Open `.env` and fill it in. Set `MACHINE=laptop` or `MACHINE=workstation`.

`.env` is gitignored. Verify before your first commit:

```bash
git check-ignore -v .env
```

That must print a matching rule. If it prints nothing, **stop** — the file is
not ignored and you are one `git add -A` away from publishing your token.

## 3. Environments

### `.venv-qai` — both machines, no GPU needed

```bash
python -m venv .venv-qai
source .venv-qai/bin/activate          # Windows: .venv-qai\\Scripts\\activate
pip install -r requirements-qai.txt
qai-hub configure --api_token <YOUR_TOKEN>
```

Verify:

```bash
python src/qualcomm/list_devices.py
```

### `.venv-train` — WORKSTATION ONLY

```bash
python -m venv .venv-train
source .venv-train/bin/activate        # Windows: .venv-train\\Scripts\\activate
pip install -r requirements-train.txt
```

Verify Blackwell / sm_120:

```bash
python scripts/check_cuda.py
```

Must report `available=True` and `capability=(12, 0)`.

> **Never** install a `qai-*` package into `.venv-train`, and never install
> `torch` into `.venv-qai` by hand. If you do, delete the env and rebuild it.

## 4. Data — workstation only

SKU-110K is ~13.6 GB: 11,743 images, ~1.73M boxes, ~147 objects/image,
official split 8,219 train / 588 val / 2,936 test.

License: **academic / non-commercial**. This must be stated in the README.

Download into `data/raw/` (gitignored). Then generate the split files:

```bash
python src/data/make_splits.py
```

That writes `data/splits/{train,val,test}.txt`, which ARE committed. Those
files are what make every downstream number reproducible — and what stops
calibration images leaking in from the evaluation splits.
"""

# ---------------------------------------------------------------- WORKFLOW
FILES["docs/TWO_MACHINE_WORKFLOW.md"] = """\
# Two-machine workflow

## The split

| | Laptop (home) | Workstation (work, RTX 5070) |
|---|---|---|
| Available | permanently | **until 2026-09-15** |
| Envs | `.venv-qai` | `.venv-train` + `.venv-qai` |
| Owns | AI Hub jobs, OpenVINO, demo, docs, writeup | training, dataset, TensorRT, ONNX export |
| GPU | Iris Xe (OpenVINO only) | RTX 5070 12 GB, sm_120 |

AI Hub needs **no local GPU** — jobs run on Qualcomm silicon in the cloud.
That is why all Qualcomm benchmarking can happen on the laptop, and why it
survives the September cliff.

## What crosses between machines, and how

| Artifact | Size | Transport |
|---|---|---|
| Code, configs, docs | KB | **git** |
| Split definitions | KB | **git** (`data/splits/`) |
| Results tables, job URLs | KB | **git** (`results/`) |
| SKU-110K dataset | 13.6 GB | **never moves** — lives on workstation only |
| Trained checkpoint | ~100-400 MB | HF Hub private repo, or W&B artifact |
| Exported ONNX | ~30-120 MB | HF Hub private repo — this is the key handoff |
| Compiled DLC / TFLite | varies | download from AI Hub job page on demand |
| Secrets | — | **never moves** — each machine has its own `.env` |

Do **not** use Git LFS for weights. GitHub's free tier is 1 GB storage and
1 GB bandwidth per month; you will iterate past that and hit a wall mid-sprint.

## The relay rule

One machine is active at a time. There is no concurrent editing.

**Before leaving a machine:**
1. Update `HANDOFF.md` — last machine, what's done, what's next, blockers
2. `git add -A && git commit -m "..." && git push`

**Before starting on a machine:**
1. `git pull --rebase`
2. Read `HANDOFF.md`

Helper scripts do both:

```bash
python scripts/handoff.py start     # pull + show HANDOFF.md
python scripts/handoff.py stop      # remind, then commit + push
```

## The critical handoff (week 3)

This is the one that matters, and it must happen before Sep 13:

```
WORKSTATION                              LAPTOP
  train RF-DETR on SKU-110K
  export to ONNX
  upload ONNX -> HF Hub  ─────────────▶  download ONNX
  commit split files + configs ───────▶  git pull
                                          AI Hub: compile / quantize / profile
                                          commit results  ◀────────────────────
```

After that transfer, **the laptop can finish the entire project alone.**
Getting to this point early is the single most important scheduling goal.

## Anti-loss checklist

- [ ] `.env` is gitignored on both machines (`git check-ignore -v .env`)
- [ ] `HANDOFF.md` updated before every machine switch
- [ ] Trained weights exist somewhere other than the workstation disk
- [ ] Every AI Hub job URL recorded in `results/` — they are shareable proof
- [ ] Before Sep 13: ONNX export uploaded and verified downloadable
"""

# ---------------------------------------------------------------- README
FILES["README.md"] = """\
# ShelfSense-Q

Real-time retail shelf analytics deployed to Qualcomm NPUs.
RF-DETR fine-tuned on densely packed shelves, quantized to INT8, and profiled
on real Qualcomm silicon across five hardware targets.

> **Status:** in development. All numbers marked `TODO(measure)` are
> unmeasured placeholders and must not be cited.

## Why

Qualcomm publishes RF-DETR inference latency across ~20 chipsets — in float
precision only. The INT8 accuracy/latency trade-off is unpublished. This
project measures it, on a dense-detection workload (~147 objects per image),
and reports the failure cases as well as the wins.

## Results

TODO(measure)

## Licensing

- **Models:** RF-DETR is Apache-2.0. Commercially usable.
- **Dataset:** SKU-110K is provided for academic / non-commercial use only.

Because the dataset is research-only, this repository is a **research
demonstration**, not a commercially deployable system. The model weights and
toolchain would permit commercial use; the training data does not.

## Reproducing

See [docs/SETUP.md](docs/SETUP.md).

## Engineering decisions

See [DECISIONS.md](DECISIONS.md) and [FAILURES.md](FAILURES.md).
"""

# ---------------------------------------------------------------- requirements
FILES["requirements-qai.txt"] = """\
# .venv-qai  — BOTH machines. No GPU required.
# Do NOT add torch here by hand; qai-hub-models manages its own.

qai-hub
qai-hub-models
python-dotenv
pyyaml
pandas
tabulate
"""

FILES["requirements-train.txt"] = """\
# .venv-train  — WORKSTATION ONLY.
#
# Install torch FIRST, from the CUDA 12.8+ index, or Blackwell (sm_120) will
# not work:
#
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   pip install -r requirements-train.txt
#
# Do NOT add qai-hub packages here. They pin their own torch.

rfdetr[onnx]
supervision
opencv-python
pillow
numpy
pandas
pyyaml
python-dotenv
wandb
onnx
onnxruntime
huggingface-hub
"""

# ---------------------------------------------------------------- configs
FILES["configs/machine.laptop.yaml"] = """\
machine: laptop
role: qualcomm-and-docs

paths:
  data_raw: null            # dataset does not live here
  splits: data/splits
  results: results
  exports: exports

compute:
  cuda: false
  openvino_device: GPU      # Intel Iris Xe; use CPU to compare

notes: >
  Permanent machine. Owns all Qualcomm AI Hub work (no local GPU needed),
  OpenVINO benchmarking, the Gradio demo, and all documentation.
  Must be able to finish the project alone after 2026-09-15.
"""

FILES["configs/machine.workstation.yaml"] = """\
machine: workstation
role: training-and-export

available_until: 2026-09-15
hard_commit_deadline: 2026-09-13

paths:
  data_raw: data/raw/SKU110K
  splits: data/splits
  results: results
  exports: exports

compute:
  cuda: true
  gpu: RTX 5070
  vram_gb: 12
  compute_capability: "12.0"   # sm_120, Blackwell

training:
  # Start small. Get the loop green, then scale if time allows.
  model: RFDETRNano
  resolution: 384
  batch_size: 4
  grad_accum: 4                # effective batch 16 within 12 GB

notes: >
  Temporary machine. Everything CUDA-dependent must be finished and pushed
  before the deadline above. Front-load ruthlessly.
"""

FILES["configs/experiment.yaml"] = """\
# Experiment defaults. Machine-specific overrides live in configs/machine.*.yaml

dataset:
  name: SKU-110K
  license: academic-non-commercial
  num_classes: 1              # single "object" class, dense packing
  official_split: {train: 8219, val: 588, test: 2936}

quantization:
  precision: w8a8             # try w8a16 if attention layers degrade badly
  calibration:
    source_split: train       # HARD RULE — never val, never test
    sizes: [32, 128, 512]     # ablation
    stratify_by: shelf_density

qualcomm:
  primary_device: "Dragonwing IQ-9075 EVK"   # verify exact string via list_devices.py
  comparison_devices:
    - "QCS8550 (Proxy)"
    - "Snapdragon X Elite CRD"
    - "Snapdragon 8 Gen 3"
  runtimes: [qnn_dlc, onnx, tflite]

baselines_published_by_qualcomm:
  note: "RF-DETR base 560x560, float, NPU. Our float compile should land near these."
  qcs9075_qnn_dlc_ms: 79.9
  qcs8550_qnn_dlc_ms: 67.1
  snapdragon_x_elite_qnn_dlc_ms: 66.1
  snapdragon_8gen3_qnn_dlc_ms: 48.4
"""

# ---------------------------------------------------------------- scripts
FILES["scripts/check_cuda.py"] = '''\
"""Blackwell / sm_120 smoke test. Run on the workstation, in .venv-train."""

import sys

try:
    import torch
except ImportError:
    sys.exit("torch not installed. Are you in .venv-train?")

print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit(
        "\\nFAIL: no CUDA.\\n"
        "Reinstall from the CUDA 12.8+ index:\\n"
        "  pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu128"
    )

cap = torch.cuda.get_device_capability()
print(f"device         : {torch.cuda.get_device_name(0)}")
print(f"capability     : {cap}")
print(f"vram           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if cap[0] < 12:
    print(
        "\\nWARNING: expected (12, 0) for RTX 5070 Blackwell.\\n"
        "A lower capability means torch is not seeing the card correctly."
    )
else:
    # Actually exercise the GPU — availability alone is not proof.
    x = torch.randn(1000, 1000, device="cuda")
    _ = (x @ x).sum().item()
    torch.cuda.synchronize()
    print("\\nOK: matmul on sm_120 succeeded.")
'''

FILES["src/qualcomm/list_devices.py"] = '''\
"""
List every device available in the Qualcomm AI Hub device farm.

Run on either machine, in .venv-qai, after `qai-hub configure`.

Device name strings are exact and case-sensitive. Copy them from THIS output
into configs/experiment.yaml — never type them from memory or from a blog post.
"""

import sys

try:
    import qai_hub as hub
except ImportError:
    sys.exit("qai_hub not installed. Are you in .venv-qai?")

try:
    devices = hub.get_devices()
except Exception as e:
    sys.exit(
        f"Could not reach AI Hub: {e}\\n\\n"
        "Check your token:  qai-hub configure --api_token <TOKEN>\\n"
        "Get one at: workbench.aihub.qualcomm.com -> Account -> Settings"
    )

print(f"{len(devices)} devices available\\n")

INTERESTING = ("9075", "8550", "X Elite", "8 Gen 3", "8275", "RB3", "Vision Kit")

print("--- likely relevant to this project ---")
for d in devices:
    if any(k.lower() in d.name.lower() for k in INTERESTING):
        print(f"  {d.name}")

print("\\n--- all devices ---")
for d in devices:
    print(f"  {d.name}")
'''

FILES["scripts/handoff.py"] = '''\
"""
Machine handoff helper. Enforces the relay rule.

    python scripts/handoff.py start    # pull, then show HANDOFF.md
    python scripts/handoff.py stop     # check secrets, commit, push
"""

import subprocess
import sys
from pathlib import Path


def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def start():
    run(["git", "pull", "--rebase"])
    print("\\n" + "=" * 60)
    print(Path("HANDOFF.md").read_text(encoding="utf-8"))
    print("=" * 60)


def stop():
    # Guard: never let .env reach the index.
    staged = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    if any(".env" in line and ".env.example" not in line for line in staged.splitlines()):
        sys.exit(
            "REFUSING TO COMMIT: .env appears in git status.\\n"
            "It must be ignored. Check .gitignore, then run:\\n"
            "  git rm --cached .env"
        )

    print("Did you update HANDOFF.md? (last machine / done / next / blockers)")
    if input("y to continue: ").strip().lower() != "y":
        sys.exit("Update HANDOFF.md first.")

    msg = input("Commit message: ").strip() or "wip"
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", msg], check=False)
    run(["git", "push"])
    print("\\nPushed. Safe to switch machines.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "stop"):
        sys.exit(__doc__)
    (start if sys.argv[1] == "start" else stop)()
'''

# ---------------------------------------------------------------- results stub
FILES["results/baselines/00_reference_float.md"] = """\
# Reference float baselines

Two things live here:

1. **Qualcomm's published numbers** — the anchor we validate against.
2. **Our own reproduction** — proof our toolchain is wired correctly.

If (2) lands near (1), everything downstream is trustworthy. If it doesn't,
we have a bug and must fix it before measuring anything else.

## Published by Qualcomm

RF-DETR base, 560x560, float, NPU. Source: huggingface.co/qualcomm/RF-DETR

| Chipset | QNN_DLC (ms) | ONNX (ms) | TFLite (ms) |
|---|---|---|---|
| Snapdragon X2 Elite | 28.7 | - | - |
| Snapdragon 8 Elite Gen 5 | 31.1 | 30.6 | 36.8 |
| Snapdragon 8 Gen 3 | 48.4 | 49.5 | 57.5 |
| Snapdragon X Elite | 66.1 | - | - |
| QCS8550 (Proxy) | 67.1 | 67.8 | 80.6 |
| QCS9075 / IQ-9075 | 79.9 | 91.5 | 92.0 |
| QCS8275 / IQ-8275 | 151.1 | - | 172.4 |

## Our reproduction

| Date | Device string | Checkpoint | Res | Runtime | Measured ms | Job URL |
|---|---|---|---|---|---|---|
| TODO(measure) | | | | | | |

**Record the job URL every time.** They are shareable — a reader can click
through to Qualcomm's own page and verify the number themselves. Almost
nobody does this, and it turns "trust my table" into "check my work."

### Open question to resolve first

The AI Hub model page lists the default checkpoint as **RF-DETR-small,
512x512, 28.5M params**. The Hugging Face card lists **RF-DETR-base, 560x560,
29.0M params**. These disagree. Determine which the export script actually
uses before recording any number, because the entire comparison depends on
knowing what was benchmarked.
"""

FILES["data/splits/README.md"] = """\
# Split definitions

These files are **committed to git**. They are small, and they are what makes
every number in this project reproducible.

- `train.txt` — 8,219 images. **The only legal source of calibration data.**
- `val.txt` — 588 images. Threshold tuning, model selection.
- `test.txt` — 2,936 images. Final numbers only. Touch as rarely as possible.

## Why this matters

INT8 post-training quantization calibrates on real images. If those images
come from val or test, the quantized model has effectively seen the evaluation
set, and the reported INT8 mAP is contaminated.

There is no error message for this. The number just comes out flattering.

Generate with `python src/data/make_splits.py`, which reads SKU-110K's
official split and writes these files. Never re-split.
"""

# package inits
for pkg in ["data", "train", "eval", "export", "qualcomm", "openvocab", "pipeline"]:
    FILES[f"src/{pkg}/__init__.py"] = ""
FILES["src/__init__.py"] = ""


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def main():
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

    # keep otherwise-empty committed dirs alive
    for d in ["results/quantization", "results/logs", "notebooks", "exports"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        keep = Path(d) / ".gitkeep"
        if not keep.exists():
            keep.write_text("")

    created, skipped = [], []
    for path, content in FILES.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            skipped.append(path)
            continue
        p.write_text(content, encoding="utf-8")
        created.append(path)

    print(f"Created {len(created)} files.")
    if skipped:
        print(f"Skipped {len(skipped)} existing: {', '.join(skipped)}")

    print("""
Next:

  1. Verify the secret guard BEFORE any git command:
       cp .env.example .env
       git init
       git add -A
       git status            <-- .env must NOT appear in this list

     If .env appears, stop and fix .gitignore.

  2. git commit -m "chore: project scaffold"
  3. Create an EMPTY repo on GitHub (no README, no .gitignore)
  4. git remote add origin https://github.com/<you>/shelfsense-q.git
     git branch -M main
     git push -u origin main
""")


if __name__ == "__main__":
    main()
