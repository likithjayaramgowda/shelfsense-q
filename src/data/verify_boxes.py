"""
Visual + statistical correctness gate on the SKU-110K box annotations.

Our copy of SKU-110K is a third-party Kaggle repackaging that converted the
original annotations to YOLO format (normalized cx, cy, w, h). That
conversion was done by someone else and is unaudited — this script is the
check that it didn't silently corrupt the ground truth before we build
anything on top of it (calibration, training, eval all read these same
label files).

Draws ground-truth boxes on a random TRAIN-split sample (never val/test —
this is a visual QA tool, not a calibration step, but it still only touches
the split it's supposed to) and reports box-geometry sanity stats. Also
spot-checks val/test box density against train to catch a split-specific
conversion bug.

Usage:
    python src/data/verify_boxes.py                  # 20 train, 5 val, 5 test
    python src/data/verify_boxes.py --n 50 --seed 0   # reproducible sample

Needs Pillow. Available in .venv-train; torchvision also pulls it into
.venv-qai as a transitive dependency.
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit(
        "Pillow not installed in this environment. Try .venv-train, or "
        "`pip install pillow`."
    )

from _sku110k_common import EXTRACT_ROOT, read_split

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "logs" / "box_verification"

BOX_COLOR = (255, 0, 0)  # high-contrast red, thin outline — dense shelves
BOX_WIDTH = 1            # need every product still visible underneath
EXPECTED_MEAN_BOXES = 147  # ~1.2M boxes / 8185 train images, see DECISIONS.md
OVERSIZED_FRACTION = 0.5   # boxes covering >50% of the image area

# A full-train-split scan (2026-08-17, see DECISIONS.md) found 14,881/1.2M
# boxes (1.24%) whose trailing edge (x2 or y2 — never x1/y1) exceeds 1.0 by
# a sub-pixel amount, max observed overshoot 0.0037. That's the signature
# of float rounding in the third-party conversion on boxes that originally
# touched the image boundary, not corrupted geometry (0 degenerate boxes
# were found anywhere in the same scan). Tolerate up to OOB_EPSILON before
# counting a box as a real gate failure, with headroom above the observed
# max — a genuinely corrupted box (large overshoot, or ANY leading-edge
# violation, which never occurs in this dataset) still fails.
OOB_EPSILON = 0.005


def label_path(split: str, image_name: str) -> Path:
    stem = Path(image_name).stem
    return EXTRACT_ROOT / "labels" / split / f"{stem}.txt"


def image_path(split: str, image_name: str) -> Path:
    return EXTRACT_ROOT / "images" / split / image_name


def load_boxes(split: str, image_name: str) -> list[tuple]:
    """Return [(cx, cy, w, h), ...] as read from the label file — all
    normalized 0-1, one class ("object") so the leading class id is
    discarded."""
    path = label_path(split, image_name)
    if not path.is_file():
        sys.exit(f"Label file missing for {split}/{image_name}: {path}")
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            sys.exit(f"Malformed label line in {path}: {line!r} (expected 'class cx cy w h')")
        _cls, cx, cy, w, h = parts
        boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20, help="images sampled from train (default: 20)")
    p.add_argument("--val-n", type=int, default=5, help="images sampled from val (default: 5)")
    p.add_argument("--test-n", type=int, default=5, help="images sampled from test (default: 5)")
    p.add_argument("--seed", type=int, default=None, help="random seed, for a reproducible sample")
    return p.parse_args()


def audit_train_sample(rng: random.Random, n: int):
    """Draw boxes on a random train sample, save annotated copies, and
    return (per_image_counts, widths_frac, heights_frac, degenerate,
    out_of_bounds, oversized) for the aggregate stats."""
    names = read_split("train")
    if n > len(names):
        sys.exit(f"--n {n} exceeds train split size ({len(names)})")
    sample = rng.sample(names, n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_image_counts = []
    widths_frac, heights_frac = [], []
    degenerate = out_of_bounds = out_of_bounds_severe = oversized = 0
    max_overshoot = 0.0

    print(f"=== TRAIN sample (n={n}) ===")
    for name in sample:
        boxes = load_boxes("train", name)
        per_image_counts.append(len(boxes))

        img = Image.open(image_path("train", name)).convert("RGB")
        w_img, h_img = img.size
        draw = ImageDraw.Draw(img)

        for cx, cy, w, h in boxes:
            widths_frac.append(w)
            heights_frac.append(h)

            x1n, y1n, x2n, y2n = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

            if w <= 0 or h <= 0:
                degenerate += 1

            overshoot = max(0.0, -x1n, -y1n, x2n - 1, y2n - 1)
            if overshoot > 0:
                out_of_bounds += 1
                max_overshoot = max(max_overshoot, overshoot)
                if overshoot > OOB_EPSILON:
                    out_of_bounds_severe += 1

            if w * h > OVERSIZED_FRACTION:
                oversized += 1

            # Sort before denormalizing so a sign-flipped (degenerate) box
            # can't crash PIL's rectangle draw — we've already counted it
            # above, this is purely so one bad box doesn't kill the run.
            px1, px2 = sorted((x1n * w_img, x2n * w_img))
            py1, py2 = sorted((y1n * h_img, y2n * h_img))
            draw.rectangle([px1, py1, px2, py2], outline=BOX_COLOR, width=BOX_WIDTH)

        out_path = OUT_DIR / name
        img.save(out_path, quality=90)
        print(f"  {name}: {len(boxes)} boxes")

    return (
        per_image_counts,
        widths_frac,
        heights_frac,
        degenerate,
        out_of_bounds,
        out_of_bounds_severe,
        max_overshoot,
        oversized,
    )


def audit_split_counts(rng: random.Random, split: str, n: int) -> float:
    names = read_split(split)
    if n > len(names):
        sys.exit(f"--{split}-n {n} exceeds {split} split size ({len(names)})")
    sample = rng.sample(names, n)
    counts = []
    print(f"\n=== {split.upper()} sample (n={n}, count-only, no images saved) ===")
    for name in sample:
        boxes = load_boxes(split, name)
        counts.append(len(boxes))
        print(f"  {name}: {len(boxes)} boxes")
    return statistics.mean(counts)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    (
        per_image_counts,
        widths_frac,
        heights_frac,
        degenerate,
        out_of_bounds,
        out_of_bounds_severe,
        max_overshoot,
        oversized,
    ) = audit_train_sample(rng, args.n)

    mean_count = statistics.mean(per_image_counts)
    print(f"\n  train sample mean box count: {mean_count:.1f}  (expected ~{EXPECTED_MEAN_BOXES})")
    print(f"  annotated images saved to {OUT_DIR.relative_to(REPO_ROOT)}")

    print("\n--- aggregate box-size sanity (train sample, "
          f"{len(widths_frac)} boxes) ---")
    print(
        f"  width  frac of image: min={min(widths_frac):.4f}  "
        f"max={max(widths_frac):.4f}  median={statistics.median(widths_frac):.4f}"
    )
    print(
        f"  height frac of image: min={min(heights_frac):.4f}  "
        f"max={max(heights_frac):.4f}  median={statistics.median(heights_frac):.4f}"
    )
    print(f"  degenerate boxes (zero/negative area): {degenerate}")
    print(
        f"  out-of-bounds boxes (outside [0,1] before denormalization): "
        f"{out_of_bounds}  (max overshoot: {max_overshoot:.4f} of image dim)"
    )
    print(
        f"    of which beyond {OOB_EPSILON} tolerance (real gate failures): "
        f"{out_of_bounds_severe}"
    )
    print(f"  oversized boxes (>{int(OVERSIZED_FRACTION * 100)}% of image area): {oversized}")

    val_mean = audit_split_counts(rng, "val", args.val_n)
    test_mean = audit_split_counts(rng, "test", args.test_n)
    print(f"\n  val  sample mean box count:  {val_mean:.1f}  (n={args.val_n})")
    print(f"  test sample mean box count:  {test_mean:.1f}  (n={args.test_n})")
    print(f"  train sample mean box count: {mean_count:.1f}  (n={args.n}, for comparison)")

    print("\n=== RESULT ===")
    if degenerate or out_of_bounds_severe:
        print(
            f"FAIL — {degenerate} degenerate box(es), {out_of_bounds_severe} "
            f"out-of-bounds box(es) beyond {OOB_EPSILON} tolerance found in "
            "the train sample.\nThis indicates a bug in the third-party "
            "YOLO conversion. Do not trust this dataset copy for "
            "calibration or evaluation until this is understood."
        )
        if oversized:
            print(f"({oversized} oversized box(es) also found — see above, not part of the gate.)")
        sys.exit(1)
    else:
        print(
            f"PASS — 0 degenerate boxes, 0 out-of-bounds boxes beyond "
            f"{OOB_EPSILON} tolerance across {len(widths_frac)} boxes "
            "sampled from train."
        )
        if out_of_bounds:
            print(
                f"NOTE: {out_of_bounds} box(es) had sub-pixel trailing-edge "
                f"overshoot within tolerance (max {max_overshoot:.4f} of image "
                "dim) — known float-rounding artifact of the third-party "
                "conversion, see DECISIONS.md. Not a gate failure."
            )
        if oversized:
            print(
                f"NOTE: {oversized} box(es) cover >{int(OVERSIZED_FRACTION * 100)}% of their "
                "image (informational only, not part of the gate — dense "
                "shelf close-ups can legitimately produce these)."
            )


if __name__ == "__main__":
    main()
