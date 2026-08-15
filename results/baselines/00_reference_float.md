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
