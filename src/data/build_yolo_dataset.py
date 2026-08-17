"""
Materialize an rfdetr-compatible YOLO dataset view from data/splits/*.txt.

WHY THIS SCRIPT EXISTS (read before touching src/train/train.py):

rfdetr 1.9.3 has native YOLO-label support (rfdetr.datasets.yolo) — the
per-image "class cx cy w h" .txt files our Kaggle repackaging ships need
NO conversion to COCO annotation JSON. That part of the original concern
turned out to be unfounded.

But two real problems stop us pointing rfdetr straight at
data/raw/SKU110K_fixed/:

1. Directory nesting mismatch. rfdetr's YOLO loader
   (rfdetr.datasets.yolo._resolve_yolo_split_dirs / is_valid_yolo_dataset)
   expects split-first layout: <root>/train/images/, <root>/train/labels/
   (the Ultralytics/Roboflow convention). Our archive ships type-first:
   images/train/, labels/train/. rfdetr does not discover it as-is.

2. rfdetr's YOLO loader lists every file in a directory
   (_list_yolo_image_paths) — it has no concept of "here is my exact file
   list," it just globs a folder. Pointing it at data/raw/ directly would
   make the directory listing BE the split, which is exactly what
   CLAUDE.md's hard rule forbids: "Never read the raw directory listing
   directly — the split files are the source of truth."

So this script builds:

    data/interim/yolo/{train,val,test}/images/*.jpg
    data/interim/yolo/{train,val,test}/labels/*.txt
    data/interim/yolo/data.yaml

containing ONLY the files named in data/splits/{split}.txt, in the layout
rfdetr expects. It uses hardlinks, not copies — same file on disk, a
second directory entry — so this costs no extra space and reruns in
seconds even though data/raw/ is 13.6 GB. data/interim/ is gitignored.

Rerun this after any change to data/raw/ or data/splits/. train.py reads
data/interim/yolo/, never data/raw/ directly, so this script is the only
place split membership can leak into what the model actually sees.
"""

import os
import shutil
import sys
from pathlib import Path

from _sku110k_common import EXTRACT_ROOT, EXPECTED_COUNTS, REPO_ROOT, read_split

YOLO_ROOT = REPO_ROOT / "data" / "interim" / "yolo"
SPLITS = ("train", "val", "test")
NUM_CLASSES = 1
CLASS_NAMES = ["object"]

_warned_copy_fallback = False


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src -> dst, falling back to a copy if hardlinking isn't
    possible (e.g. src/dst on different volumes). Skips if dst already
    exists (idempotent, cheap to check — this is the common case on rerun)."""
    global _warned_copy_fallback
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError as exc:
        if not _warned_copy_fallback:
            print(
                f"  NOTE: hardlink failed ({exc}); falling back to copying "
                "instead (uses extra disk space). This should only happen "
                "if data/raw/ and data/interim/ are on different volumes."
            )
            _warned_copy_fallback = True
        shutil.copy2(src, dst)


def already_materialized(split: str, names: list[str]) -> bool:
    img_dir = YOLO_ROOT / split / "images"
    lbl_dir = YOLO_ROOT / split / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return False
    return (
        len(list(img_dir.iterdir())) == len(names)
        and len(list(lbl_dir.iterdir())) == len(names)
    )


def materialize_split(split: str) -> int:
    names = read_split(split)

    if already_materialized(split, names):
        print(f"{split}: already materialized ({len(names)} images), skipping.")
        return len(names)

    img_dir = YOLO_ROOT / split / "images"
    lbl_dir = YOLO_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    src_img_dir = EXTRACT_ROOT / "images" / split
    src_lbl_dir = EXTRACT_ROOT / "labels" / split

    for name in names:
        stem = Path(name).stem
        _link_or_copy(src_img_dir / name, img_dir / name)
        _link_or_copy(src_lbl_dir / f"{stem}.txt", lbl_dir / f"{stem}.txt")

    print(f"{split}: materialized {len(names)} images -> {img_dir.relative_to(REPO_ROOT)}")
    return len(names)


def write_data_yaml() -> None:
    yaml_path = YOLO_ROOT / "data.yaml"
    # Explicit path/train/val/test keys, even though our layout also
    # satisfies rfdetr's bare filesystem-convention fallback — being
    # explicit here means this file stays correct even if that fallback
    # convention changes in a future rfdetr release.
    content = (
        f"path: {YOLO_ROOT.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES!r}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"wrote {yaml_path.relative_to(REPO_ROOT)}")


def main() -> None:
    if not EXTRACT_ROOT.is_dir():
        sys.exit(f"{EXTRACT_ROOT} not found. Run src/data/setup_dataset.py first.")

    counts = {}
    for split in SPLITS:
        counts[split] = materialize_split(split)

    write_data_yaml()

    mismatches = [
        f"  {s}: expected {EXPECTED_COUNTS[s]}, materialized {counts[s]}"
        for s in SPLITS
        if counts[s] != EXPECTED_COUNTS[s]
    ]
    if mismatches:
        sys.exit(
            "Materialized counts don't match data/splits/ — something is "
            "inconsistent between data/raw/ and data/splits/:\n" + "\n".join(mismatches)
        )

    print(f"\nDone. YOLO dataset view ready at {YOLO_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
