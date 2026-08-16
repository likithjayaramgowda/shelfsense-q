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

## Toolchain validated

2026-08-15, laptop, `.venv-qai`. Proved the AI Hub export/compile/profile
pipeline end-to-end on a throwaway model (mobilenet_v2) before spending a
job on RF-DETR itself. See [FAILURES.md](../../FAILURES.md) for the five
failures hit getting here.

Working command shape:

```
qai-hub-models export <model> --device "<device>" --target-runtime qnn_dlc \
  --compile-options="--qairt_version=2.45" \
  --profile-options="--qairt_version=2.45"
```

Result — mobilenet_v2, Snapdragon X Elite CRD (Windows 11), QNN_DLC, float:

| Metric | Value |
|---|---|
| Estimated inference time | 1.1 ms |
| Compute units | npu 103 ops, gpu 0 ops, cpu 0 ops (100% NPU) |
| Total ops | 103 |
| PSNR vs local CPU | 51.4222 dB (output class_logits, shape (1,1000)) |
| QAIRT used | 2.45.0.260326154327 |

Job IDs: compile `jgd2loke5` · profile `j5793xmlg` · inference `jp430v7v5`

Pinned versions for this run: qai-hub==0.54.0, qai-hub-models==0.60.0,
torch==2.11.0, torchvision==0.26.0 (see requirements-qai.txt).

## Our reproduction

**Resolved:** the "which checkpoint" question below was real — the CLI
defaults to `small`, not `base` (see [FAILURES.md](../../FAILURES.md),
2026-08-16). All runs in this table pass `--variant base` explicitly.

Common config for every row: `rf_detr --variant base`, `QNN_DLC`, float,
QAIRT `2.45.0.260326154327`, qai-hub-models `0.60.0`, torch `2.11.0`.
814 ops total, 100% NPU (0 GPU, 0 CPU) on every device — same model, same
op graph, only the target device changes.

| Device | OS | Measured (ms) | PSNR (dB) | Peak mem (MB) | Profile job |
|---|---|---|---|---|---|
| Snapdragon 8 Elite Gen 5 QRD | Android 16 | 35.2 | 49.7477 | [3, 629] | `jgd2lvoz5` |
| QCS8550 (Proxy) | Android 12 | 68.8 | 49.831 | [3, 6] | `jgo8qvzqp` |
| Snapdragon X Elite CRD | Windows 11 | 68.7 | 49.831 | [3, 3] | `jp2enm0qp` |
| Dragonwing IQ-9075 EVK | QC_LINUX 1.9 | 74.1 | 51.7791 | [3, 8] | `jp430wn85` |
| QCS8275 (Proxy) | Android 14 | 157.9 | 50.3131 | [0, 609] | `jp2enm9qp` |

**Record the job URL every time.** They are shareable — a reader can click
through to Qualcomm's own page and verify the number themselves. Almost
nobody does this, and it turns "trust my table" into "check my work."
(Job IDs above; AI Hub Workbench doesn't expose a stable per-job URL
pattern we've confirmed yet — resolve to a full link before the writeup.)

### Earlier small-variant run (pre-fix, for the record)

Before the `--variant base` fix landed, one profiling run went out against
the CLI's default `small` checkpoint on the same device (IQ-9075):
57.4 ms, 845 ops, 100% NPU, PSNR 47.6031 dB, peak mem [5, 9] MB.
Jobs: `jp81vzlx5` / `jgk8m3j2g` / `j5qvo3j4g`. Not comparable to the table
above or to Qualcomm's base@560 figures — kept here only because it's the
run that surfaced the default-variant bug in FAILURES.md.

## Validation

Percent delta = (measured − Qualcomm published) / Qualcomm published,
against the matching device row in the published table above.

| Device | Measured (ms) | Published (ms) | Delta |
|---|---|---|---|
| Snapdragon 8 Elite Gen 5 QRD | 35.2 | 31.1 (Snapdragon 8 Elite Gen 5) | +13.2% |
| QCS8550 (Proxy) | 68.8 | 67.1 | +2.5% |
| Snapdragon X Elite CRD | 68.7 | 66.1 (Snapdragon X Elite) | +3.9% |
| Dragonwing IQ-9075 EVK | 74.1 | 79.9 (QCS9075 / IQ-9075) | −7.3% |
| QCS8275 (Proxy) | 157.9 | 151.1 (QCS8275 / IQ-8275) | +4.5% |

**Conclusion:** all five devices land within 13.2% of Qualcomm's published
float figure; three of five (QCS8550, Snapdragon X Elite, QCS8275) are
within 5%. The two outliers are Snapdragon 8 Elite Gen 5 QRD (+13.2%,
device-farm QRD board vs. Qualcomm's own reference hardware) and the
IQ-9075 (−7.3%, and in the opposite direction — we measure *faster* than
published). Close enough, in a consistent direction and magnitude, to call
the export/compile/profile chain validated rather than broken. Nothing
here clears 15%, and the direction of error isn't systematic one way,
which is what you'd expect from device-farm board variance rather than a
preprocessing or precision bug.

> Note: I computed these deltas directly from the numbers above rather than
> asserting a round "four within 5%" — the actual split is three within 5%
> and two more within 13%. Flagging in case that doesn't match what you
> expected to see.

### Caveat: peak memory figures are not yet trustworthy

Peak mem reads [3, 629] MB and [0, 609] MB on two devices (8 Elite Gen 5 QRD,
QCS8275) versus [3, 3]–[3, 8] MB on the other three — for the *identical*
model, op count, and precision. A ~100x swing like that on an unchanged
graph smells like a units or accounting difference in how AI Hub reports
memory per OS/runtime (e.g. Android vs Windows/Linux instrumentation, or
"total" vs "above baseline" accounting), not a real memory difference.
Don't quote peak memory in the writeup until this is understood — treat it
as `TODO(investigate)`, not `TODO(measure)`.
