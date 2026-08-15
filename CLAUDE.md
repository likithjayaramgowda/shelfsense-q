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
