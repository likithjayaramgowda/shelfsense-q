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
