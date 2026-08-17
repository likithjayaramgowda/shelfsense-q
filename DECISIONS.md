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

- 2026-08-17 · Accepted the downloaded SKU-110K archive's actual counts
  (8,185 / 584 / 2,920 = 11,689 images) as the project's split ground truth,
  instead of the canonical published split (8,219 / 588 / 2,936 = 11,743).
  The archive in Downloads is a Kaggle repackaging ("SKU110K_fixed",
  YOLO-format per-image labels, not the original CVPR19 CSV release) that is
  54 images short of the official release; images and labels are internally
  consistent per split, and total boxes (1,723,135) land within 0.4% of the
  published ~1.73M, so this reads as a handful of dropped images rather than
  a broken split. Gave up: exact parity with Qualcomm's published sample
  counts — acceptable since our INT8 vs FLOAT comparison is internally
  consistent (same images, both precisions) regardless of the small overall
  shortfall. `make_splits.py` still fails loudly if the archive's counts
  ever change. See `src/data/_sku110k_common.py` and
  `data/splits/README.md`.

- 2026-08-17 · No COCO conversion needed for RF-DETR training — rfdetr 1.9.3
  has native YOLO-label support (rfdetr.datasets.yolo), so the "class cx cy
  w h" .txt files our Kaggle repackaging ships work as-is. What WAS needed:
  a directory-layout fix. rfdetr's YOLO loader expects split-first nesting
  (`<root>/train/images/`, `<root>/train/labels/`); our archive ships
  type-first (`images/train/`, `labels/train/`). Rather than pointing rfdetr
  at data/raw/ directly (which would make its own directory listing the de
  facto split — exactly what the split-file hard rule forbids), wrote
  `src/data/build_yolo_dataset.py`: hardlinks (zero extra disk) the files
  named in data/splits/{train,val,test}.txt into data/interim/yolo/ in the
  layout rfdetr expects, plus a data.yaml. train.py reads only that
  materialized view, never data/raw/. Gave up: nothing — hardlinks make this
  free to rerun, and it's a cleaner mechanism than a COCO conversion would
  have been (one less format in the pipeline).

- 2026-08-17 · `rfdetr[train,loggers]` extras (pytorch-lightning + logger
  backends) had to be installed into `.venv-train`; the base `rfdetr[onnx]`
  install doesn't include them, and rfdetr 1.9.x trains through a PyTorch
  Lightning `Trainer` internally. Confirmed this didn't touch the cu128
  Blackwell torch build (`torch==2.11.0+cu128`, sm_120, unchanged after
  install). Updated `requirements-train.txt` accordingly so a fresh
  `.venv-train` setup doesn't hit the same `ModuleNotFoundError` blind.

- 2026-08-17 · `src/train/train.py`'s `--smoke-test` bypasses the high-level
  `RFDETR.train()` convenience API and calls `RFDETRModelModule` /
  `RFDETRDataModule` / `build_trainer` directly (all public exports of
  `rfdetr.training`), passing `max_steps=N` straight through to the
  underlying Lightning `Trainer`. `RFDETR.train(**kwargs)` validates its
  kwargs strictly against `TrainConfig`'s pydantic fields and has no
  `max_steps` field — there's no way to get an exact step count through the
  public convenience method. `build_trainer`'s own docstring documents
  `**trainer_kwargs` passthrough for exactly this (`fast_dev_run=2` is its
  own example), so this stays within intended, documented extension points
  rather than reaching into anything private. Verified against actual
  rfdetr 1.9.3 source before writing the call. Confirmed working: smoke
  test ran 3 real optimizer steps on the RTX 5070, peak VRAM 2.02 GB
  (16.8% of the 12 GB budget).

- 2026-08-17 · Added `train.py --max-steps N` (distinct from `--smoke-test`)
  for a capped-but-real run that still produces a loadable checkpoint.
  Investigated the checkpoint format first: rfdetr's own epoch-boundary
  `ModelCheckpoint` callbacks (`last.ckpt`, `checkpoint_{epoch}.pth`) never
  fire within 200 steps on the full 8185-image train split (~511 steps is
  one epoch), and only `BestModelCallback`-written files carry the
  `args`/`model_name`/`model_config` keys `RFDETR.from_checkpoint()`
  requires — a plain PTL checkpoint doesn't. `BestModelCallback.on_fit_end`
  unconditionally writes `last_ema.pth`/`checkpoint_best_ema.pth` when EMA
  is enabled (the default), regardless of epoch completion, so capping via
  `max_steps` alone (no dataset shrinking, no forced validation) is
  sufficient. Confirmed: `checkpoint_best_total.pth` produced after 200
  steps, loads cleanly via `RFDETR.from_checkpoint()`.

- 2026-08-17 · Found and fixed a real crash, not a training-logic bug: the
  first `--max-steps 200` attempt died with an unhandled
  `UnicodeEncodeError` during PyTorch Lightning's pre-training sanity
  validation pass, before any of the 200 steps ran. Root cause: rfdetr's
  `COCOEvalCallback` prints metrics via Rich, whose table headers include
  an em-dash; Windows' default console codepage (cp1252) can't encode it.
  Fixed at the source — `sys.stdout.reconfigure(encoding="utf-8",
  errors="replace")` at the top of both `train.py` and `to_onnx.py`, before
  any rfdetr/Lightning imports — rather than working around it by disabling
  the sanity check, since the same crash would recur on any real training
  run once validation legitimately runs. Note for future scripts on this
  workstation: any script that imports rfdetr's training/eval stack and
  might print Unicode should carry the same guard.

- 2026-08-17 · Exported the 200-step throwaway checkpoint to ONNX
  (`src/export/to_onnx.py`) and verified PyTorch vs ONNX Runtime parity on
  a fixed input before trusting it: dets max abs diff 1.928e-04, labels
  max abs diff 7.343e-05, both comfortably under a 1e-3 tolerance — PASS.
  Compared against a stock (COCO, 90-class) RFDETRNano export at the same
  opset (17): input shape and opset are identical; the only structural
  difference is the `labels` output's last dimension — 2 for our 1-class
  fine-tune, 91 for stock. That's `num_classes + 1` in both cases (a
  background/no-object slot the classification head keeps despite
  `ia_bce_loss=True`), not `num_classes` alone — confirmed by reading the
  actual ONNX graph rather than assumed from the loss-function config, since
  the two disagreed with my first (wrong) guess. This +1 is a real, load-
  bearing fact for the QNN/AI-Hub compile step later: the exported labels
  tensor width must be read from the graph, never hardcoded as num_classes.
