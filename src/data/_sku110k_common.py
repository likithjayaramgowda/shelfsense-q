"""
Shared constants for setup_dataset.py and make_splits.py.

Single source of truth for the split counts, so the two scripts can never
silently disagree about the one number that matters most in this project:
how many images are in each split.

Background (see DECISIONS.md, 2026-08-17):
The archive available for this project is a Kaggle repackaging of SKU-110K
("SKU110K_fixed", YOLO-format labels), not the original CVPR19 CSV release.
Its own data_kaggle.yaml documents the canonical official counts as
8219/588/2936 (11,743 total) — but the archive as downloaded only actually
contains 8185/584/2920 (11,689 total), 54 images short. Images and labels
are internally consistent within the archive (not a corrupt download on
our end); the mirror itself is short those 54 images, most likely due to
corrupt/unreadable source JPEGs dropped during the repackager's own
conversion pipeline (not independently confirmed).

Decision: accept this archive's actual counts as the project's ground
truth rather than blocking on sourcing the missing 54 images. If a future
re-download ever produces different counts, both scripts will fail loudly
rather than silently accept a different number.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
EXTRACT_ROOT = RAW_DIR / "SKU110K_fixed"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Ground truth for THIS archive, confirmed by direct inspection.
EXPECTED_COUNTS = {"train": 8185, "val": 584, "test": 2920}
EXPECTED_TOTAL_IMAGES = sum(EXPECTED_COUNTS.values())  # 11,689

# Canonical published SKU-110K figures — reference only, never asserted
# against. See docstring above for why this archive falls short of them.
CANONICAL_TOTAL_IMAGES = 11_743
CANONICAL_SPLIT_COUNTS = {"train": 8219, "val": 588, "test": 2936}
CANONICAL_TOTAL_BOXES = 1_730_000  # ~1.73M, per project brief
