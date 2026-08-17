"""
Set up SKU-110K from a manually-downloaded archive.

Locates the archive in ~/Downloads (never downloads it — that's a manual
step), extracts it to data/raw/ (gitignored), and verifies the result.
Idempotent: if data/raw/ already holds a complete extraction, this skips
straight to the verification report.

Run in either venv — stdlib only, no torch/qai-hub required.

The archive available for this project is a Kaggle repackaging of
SKU-110K ("SKU110K_fixed") with per-image YOLO-format label files, not the
original CVPR19 per-split CSV release. See _sku110k_common.py and
DECISIONS.md (2026-08-17) for why the verified counts here are
8,185 / 584 / 2,920 rather than the canonical 8,219 / 588 / 2,936.
"""

import sys
import tarfile
import time
import zipfile
from pathlib import Path

from _sku110k_common import (
    CANONICAL_TOTAL_BOXES,
    CANONICAL_TOTAL_IMAGES,
    EXPECTED_COUNTS,
    EXPECTED_TOTAL_IMAGES,
    EXTRACT_ROOT,
    RAW_DIR,
)

DOWNLOADS = Path.home() / "Downloads"
SPLITS = ("train", "val", "test")

# Any .zip/.tar.gz whose name mentions SKU-110K directly is an instant
# match. Anything else has to be over 5 GB (this archive is ~13-14 GB) AND
# contain a SKU110K_fixed/ path near the start of the archive before we'll
# consider it a candidate — catches generic Kaggle download names like
# "archive.zip".
NAME_HINTS = ("sku110k", "sku-110k", "sku_110k")
SIZE_FLOOR_BYTES = 5 * 1024**3


def find_archive() -> Path:
    if not DOWNLOADS.is_dir():
        sys.exit(f"Downloads folder not found at {DOWNLOADS}")

    all_files = sorted(p for p in DOWNLOADS.iterdir() if p.is_file())
    candidates = []
    for p in all_files:
        suffix_ok = p.suffix.lower() == ".zip" or p.name.lower().endswith(".tar.gz")
        if not suffix_ok:
            continue
        if any(h in p.name.lower() for h in NAME_HINTS):
            candidates.append(p)
        elif p.stat().st_size > SIZE_FLOOR_BYTES and _looks_like_sku110k(p):
            candidates.append(p)

    if not candidates:
        listing = "\n".join(
            f"  {p.name} ({p.stat().st_size / 1e9:.2f} GB)" for p in all_files
        )
        sys.exit(
            f"Could not find a SKU-110K archive in {DOWNLOADS}.\n"
            "Looked for .zip/.tar.gz filenames mentioning SKU-110K, and for "
            "any archive over 5 GB containing a SKU110K_fixed/ path (covers "
            "generic Kaggle download names like 'archive.zip').\n\n"
            f"Directory listing of {DOWNLOADS}:\n{listing if listing else '  (empty)'}\n\n"
            "This script never downloads the dataset itself — download it "
            "manually and place the archive in Downloads, then re-run."
        )

    if len(candidates) > 1:
        listing = "\n".join(f"  {p.name}" for p in candidates)
        sys.exit(
            f"Found multiple candidate SKU-110K archives in {DOWNLOADS}, "
            f"refusing to guess which one to use:\n{listing}\n\n"
            "Remove or rename the one you don't want."
        )

    return candidates[0]


def _looks_like_sku110k(path: Path) -> bool:
    """Peek at the first entries of an archive without extracting it."""
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()[:20]
        else:
            names = []
            with tarfile.open(path, "r|gz") as tf:
                for member in tf:
                    names.append(member.name)
                    if len(names) >= 20:
                        break
        return any("SKU110K" in n for n in names)
    except Exception:
        return False


def already_extracted() -> bool:
    """Cheap check: do the expected per-split directories already hold the
    expected file counts? Used for idempotency — never re-extracts 13 GB
    unnecessarily."""
    counts = scan_extraction()
    if counts is None:
        return False
    return all(
        counts[s]["images"] == EXPECTED_COUNTS[s]
        and counts[s]["labels"] == EXPECTED_COUNTS[s]
        for s in SPLITS
    )


def scan_extraction():
    """Return {split: {"images": n, "labels": n}}, or None if the
    extraction root doesn't exist at all."""
    images_root = EXTRACT_ROOT / "images"
    labels_root = EXTRACT_ROOT / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        return None
    counts = {}
    for split in SPLITS:
        img_dir = images_root / split
        lbl_dir = labels_root / split
        n_images = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else 0
        counts[split] = {"images": n_images, "labels": n_labels}
    return counts


def extract(archive: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    size_gb = archive.stat().st_size / 1e9
    print(f"Extracting {archive.name} ({size_gb:.1f} GB) to {RAW_DIR} ...")
    start = time.time()
    if archive.suffix.lower() == ".zip":
        _extract_zip(archive)
    else:
        _extract_targz(archive)
    print(f"Extraction finished in {time.time() - start:.0f}s")


def _progress(label: str, done: int, total: int, start: float, force: bool = False, last=[0.0]):
    now = time.time()
    if not force and now - last[0] < 0.5:
        return
    last[0] = now
    pct = 100 * done / total if total else 0
    elapsed = now - start
    rate = done / elapsed if elapsed > 0 else 0
    print(f"\r{label}: {pct:5.1f}%  ({done}/{total})  {rate:,.0f}/s", end="", flush=True)


def _extract_zip(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        total = len(infos)
        start = time.time()
        for i, info in enumerate(infos, 1):
            zf.extract(info, RAW_DIR)
            _progress("extracting", i, total, start)
        _progress("extracting", total, total, start, force=True)
    print()


def _extract_targz(archive: Path) -> None:
    # tar.gz has no central directory, so we can't know the total member
    # count without a full pre-scan. Stream and report a running count
    # instead of a percentage.
    start = time.time()
    with tarfile.open(archive, "r:gz") as tf:
        n = 0
        for member in tf:
            tf.extract(member, RAW_DIR)
            n += 1
            if n % 200 == 0:
                elapsed = time.time() - start
                rate = n / elapsed if elapsed > 0 else 0
                print(f"\rextracting: {n} entries  {rate:,.0f}/s", end="", flush=True)
    print(f"\rextracting: {n} entries done" + " " * 20)


def count_boxes(counts) -> dict:
    """Sum bounding boxes (one per line) across all label .txt files."""
    totals = {}
    for split in SPLITS:
        lbl_dir = EXTRACT_ROOT / "labels" / split
        total = 0
        if lbl_dir.is_dir():
            for f in lbl_dir.glob("*.txt"):
                with open(f, "r", encoding="utf-8") as fh:
                    total += sum(1 for line in fh if line.strip())
        totals[split] = total
    return totals


def verify() -> None:
    counts = scan_extraction()
    if counts is None:
        sys.exit(f"Nothing extracted at {EXTRACT_ROOT} — extraction did not produce the expected layout.")

    print("\n--- extraction report ---")
    total_images = 0
    ok = True
    for split in SPLITS:
        n_img = counts[split]["images"]
        n_lbl = counts[split]["labels"]
        total_images += n_img
        expected = EXPECTED_COUNTS[split]
        status = "OK" if n_img == n_lbl == expected else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(
            f"  {split:5s}: {n_img:5d} images, {n_lbl:5d} labels "
            f"(expected {expected})  [{status}]"
        )

    print(f"\n  total images found: {total_images}")
    print(f"  expected (this archive, see DECISIONS.md): {EXPECTED_TOTAL_IMAGES}")
    print(f"  canonical published SKU-110K total (reference only): {CANONICAL_TOTAL_IMAGES}")

    print("\n  counting bounding boxes (reading all label files)...")
    box_counts = count_boxes(counts)
    total_boxes = sum(box_counts.values())
    for split in SPLITS:
        print(f"    {split:5s}: {box_counts[split]:,} boxes")
    print(f"    total: {total_boxes:,} boxes")
    print(f"    canonical published reference (~1.73M, for comparison only): {CANONICAL_TOTAL_BOXES:,}")

    if not ok:
        sys.exit(
            "\nExtraction verification FAILED — counts don't match this "
            "archive's known-good baseline (data/raw/ may be incomplete or "
            "corrupted). Delete data/raw/ and re-run this script."
        )

    print("\nExtraction verified OK.")


def main() -> None:
    if already_extracted():
        print(f"data/raw/ already contains a complete SKU-110K extraction at {EXTRACT_ROOT}. Skipping extraction.")
        verify()
        return

    archive = find_archive()
    print(f"Found archive: {archive}")
    extract(archive)
    verify()


if __name__ == "__main__":
    main()
