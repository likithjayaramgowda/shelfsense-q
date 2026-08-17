"""
Generate data/splits/{train,val,test}.txt from SKU-110K's official split.

The official split is expressed by directory placement in the extracted
archive: images/{train,val,test}/, labels/{train,val,test}/. SKU-110K's
image filenames are natively prefixed train_/val_/test_ — this directory
layout IS the dataset's official split, not something we're inventing.
This script never re-splits and never shuffles; it only reads what's
already there.

Run src/data/setup_dataset.py first.

HARD RULE (CLAUDE.md): train.txt is the ONLY legal source of calibration
data for INT8 quantization. Never val, never test. If you're ever unsure
which split an image came from, these files are the answer — check here,
don't guess from a filename pattern.

Counts asserted below are 8,185 / 584 / 2,920, not the canonical SKU-110K
8,219 / 588 / 2,936. See _sku110k_common.py and DECISIONS.md (2026-08-17):
this archive is a Kaggle mirror that is 54 images short of the official
release, accepted as this project's ground truth.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

from _sku110k_common import EXPECTED_COUNTS, EXTRACT_ROOT, SPLITS_DIR

SPLITS = ("train", "val", "test")


def list_split_images(split: str) -> list[str]:
    img_dir = EXTRACT_ROOT / "images" / split
    if not img_dir.is_dir():
        sys.exit(
            f"{img_dir} does not exist.\n"
            "Run src/data/setup_dataset.py first to extract the archive."
        )
    names = sorted(p.name for p in img_dir.glob("*.jpg"))
    if not names:
        sys.exit(f"{img_dir} exists but contains no .jpg files.")
    return names


def check_disjoint(splits: dict) -> None:
    sets = {s: set(v) for s, v in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {f"{a}/{b}": sets[a] & sets[b] for a, b in pairs}
    if any(overlaps.values()):
        detail = "\n".join(
            f"  {pair}: {len(names)} shared filenames" for pair, names in overlaps.items() if names
        )
        sys.exit(
            "Splits are NOT disjoint. This must never happen — it would "
            "mean an image can leak between calibration/train and eval.\n"
            f"{detail}"
        )


def check_counts(splits: dict) -> None:
    mismatches = [
        f"  {split}: expected {expected}, found {len(splits[split])}"
        for split, expected in EXPECTED_COUNTS.items()
        if len(splits[split]) != expected
    ]
    if mismatches:
        sys.exit(
            "Split count mismatch — refusing to write split files.\n"
            + "\n".join(mismatches)
            + "\n\nSee DECISIONS.md (2026-08-17) for the accepted baseline "
            "counts (8185/584/2920) and why they differ from the canonical "
            "SKU-110K release (8219/588/2936). If counts don't match either "
            "set of numbers, the extraction in data/raw/ is broken — rerun "
            "src/data/setup_dataset.py."
        )


def write_splits(splits: dict) -> dict:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(EXTRACT_ROOT),
        "splits": {},
    }
    for split, names in splits.items():
        out_path = SPLITS_DIR / f"{split}.txt"
        text = "\n".join(names) + "\n"
        out_path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        manifest["splits"][split] = {
            "count": len(names),
            "sha256": digest,
            "file": out_path.name,
        }
        print(f"wrote {out_path}  ({len(names)} images, sha256={digest[:16]}...)")
    return manifest


def main() -> None:
    splits = {s: list_split_images(s) for s in SPLITS}

    check_disjoint(splits)
    check_counts(splits)

    manifest = write_splits(splits)
    manifest_path = SPLITS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")

    total = sum(len(v) for v in splits.values())
    print(f"\nDone. {total} images across {len(SPLITS)} splits. Disjointness verified. Counts match accepted baseline.")


if __name__ == "__main__":
    main()
