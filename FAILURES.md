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

## 2026-08-15 — AI Hub Workbench version skew (QAIRT + serialization + torch)

**Context:** Bootstrapping `.venv-qai` on the laptop and running the first
`qai-hub-models` export, to validate the AI Hub pipeline end-to-end.
**Error:**
1. `ValueError: QAIRT version 2.43 is not supported by AI Hub Workbench.`
   Available: 2.45 (default), 2.47, 2.48 (latest). qai-hub-models 0.48.0
   from PyPI still targeted 2.43.
2. After upgrading to qai-hub-models 0.60.0: `RuntimeError: This model does
   serialization using the pt2 format, which requires torch>=2.9`. Env had
   torch 2.8.0.
3. `pip install --upgrade torch` installed 2.13.0, but qai-hub-models 0.60.0
   requires torch<=2.11.0 and torchvision<=0.26.0.
**Hypothesis:** The qai-hub-models client library trails the AI Hub managed
service's supported QAIRT versions. Separately, the TorchScript deprecation
landed as a torch>=2.9 requirement (.pt2 / ExportedProgram format), and a
bare `--upgrade` overshot the window that qai-hub-models 0.60.0 actually
supports.
**Fix:** Pinned torch==2.11.0, torchvision==0.26.0 (the valid window for
qai-hub-models 0.60.0), and pinned `--qairt_version=2.45` explicitly on
compile/profile options.
**Outcome:** worked
**Lesson:** Never do a bare `pip install --upgrade` in `.venv-qai`. Freeze
exact versions in requirements-qai.txt and derive the valid torch window
from qai-hub-models' own constraints, not from "latest".

## 2026-08-15 — Runtime/flag mismatch: QAIRT version not applicable to selected runtime

**Context:** Same export run, after resolving the version-skew issues above.
**Error:** `qai_hub.client.UserError: QAIRT SDK version is not applicable
to selected runtime.` The CLI swallowed the real cause until
`QAIHM_CLI_VERBOSE_EXCEPTIONS=1` was set.
**Hypothesis:** `--qairt_version` only applies to QNN-family runtimes, but
the export command's default target runtime was TFLite.
**Fix:** Added `--target-runtime qnn_dlc` to the export command.
**Outcome:** worked — this was the one that actually blocked the pipeline
**Lesson:** Set `QAIHM_CLI_VERBOSE_EXCEPTIONS=1` early; the default CLI
error message here was misleading. Always pass `--target-runtime`
explicitly rather than relying on the CLI default.

## 2026-08-15 — Deprecated CLI invocation form (minor)

**Context:** Same session, before landing on the working command shape.
**Error:** none (deprecation notice) —
`python -m qai_hub_models.models.X.export` is deprecated in favour of
`qai-hub-models export X`.
**Hypothesis:** n/a
**Fix:** Switched to the `qai-hub-models export <model>` form.
**Outcome:** worked
**Lesson:** Prefer the new CLI entry point going forward.

## 2026-08-16 — CLI default variant is `small`, not `base`

AI Hub's model page lists small@512 as default; the HF card documents
base@560 and publishes 79.9ms for it. The CLI actually pulls
rf-detr-small.pth unless --variant is given. Comparing a default export
against the HF card's number silently compares two different models.
Always state variant + resolution alongside any latency figure.

## 2026-08-16 — `large` variant exists but is undocumented on AI Hub

Validator reports supported variants: nano, small, medium, base, large.
Qualcomm's model page lists only nano/small/medium/base.
