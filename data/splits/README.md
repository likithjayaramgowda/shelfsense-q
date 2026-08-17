# Split definitions

These files are **committed to git**. They are small, and they are what makes
every number in this project reproducible.

- `train.txt` — 8,185 images. **The only legal source of calibration data.**
- `val.txt` — 584 images. Threshold tuning, model selection.
- `test.txt` — 2,920 images. Final numbers only. Touch as rarely as possible.

These counts are **not** the canonical published SKU-110K split
(8,219 / 588 / 2,936, 11,743 total). The archive available for this project
is a Kaggle repackaging ("SKU110K_fixed", YOLO-format labels) that is 54
images short of the official release. Verified: images/labels are internally
consistent per split (no partial/corrupt extraction on our end), and the
total bounding-box count (1,723,135) lands within 0.4% of the published
~1.73M reference, so the gap looks like a small number of images the
repackager's own pipeline dropped, not a broken split. Accepted as this
project's ground truth on 2026-08-17 — see DECISIONS.md and
`src/data/_sku110k_common.py`.

## Why this matters

INT8 post-training quantization calibrates on real images. If those images
come from val or test, the quantized model has effectively seen the evaluation
set, and the reported INT8 mAP is contaminated.

There is no error message for this. The number just comes out flattering.

Generate with `python src/data/setup_dataset.py` (extracts the archive) then
`python src/data/make_splits.py`, which reads SKU-110K's official split
(directory placement in the extracted archive) and writes these files.
Never re-split.
