# Handoff

Single source of truth for "where am I and what's next".
Update this **before you stop working on a machine**, every time.
It is the thing that prevents losing a day to confusion on Monday morning.

---

## Current state

- **Last active machine:** laptop
- **Last updated:** 2026-08-16
- **Branch:** main

## Done

- [x] Repo scaffold pushed
- [x] AI Hub pipeline validated end-to-end (mobilenet_v2, QNN_DLC, Snapdragon X Elite CRD)
- [x] RF-DETR-base float sweep complete across 5 devices (QNN_DLC) and
      validated against Qualcomm's published figures — all within 13.2%,
      three within 5%. See results/baselines/00_reference_float.md.

## Next task

**Monday, on the workstation.** CUDA/TensorRT window closes 2026-09-13 —
don't burn this session on anything that could run on the laptop instead.

1. `git pull --rebase`
2. Clone/sync repo on the workstation if not already present
3. Set up `.venv-train` (torch cu128, Blackwell sm_120) — do NOT install
   any `qai-hub*` package into this env
4. Verify sm_120 is actually being used (`torch.cuda.get_device_capability()`
   or equivalent) before trusting any training run
5. Begin SKU-110K download (this is the training-data prerequisite for the
   RF-DETR fine-tune; check `data/splits/` conventions before touching it —
   train/val/test are fixed and committed, no re-splitting)

## Blockers

None.

## Notes for the other machine

Float baseline is done and validated — INT8 quantization is unblocked
whenever there's laptop time, but SKU-110K download / fine-tuning must
happen on the workstation before the 2026-09-13 CUDA cliff. Peak-memory
figures in the baseline table are flagged unreliable — see the caveat in
results/baselines/00_reference_float.md before using them anywhere.

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
