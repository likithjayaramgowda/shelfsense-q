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
