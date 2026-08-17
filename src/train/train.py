"""
Fine-tune RFDETRNano on SKU-110K.

Dataset format decision (read this before changing anything dataset-related):
rfdetr 1.9.3 has NATIVE YOLO-label support (rfdetr.datasets.yolo) — the
per-image "class cx cy w h" .txt files our Kaggle repackaging ships needed
NO conversion to COCO annotation JSON. What it DID need was a directory
*layout* fix (rfdetr expects <root>/train/images + <root>/train/labels;
our archive ships images/train + labels/train) built from
data/splits/{train,val,test}.txt, never from a raw directory listing. See
src/data/build_yolo_dataset.py for the full explanation and run it first:

    python src/data/build_yolo_dataset.py

This script then reads ONLY data/interim/yolo/ (the materialized view),
never data/raw/ — that boundary is what keeps calibration data out of eval
later (CLAUDE.md hard rule: splits are the source of truth).

Usage:
    python src/train/train.py --smoke-test              # 3 steps, prints peak VRAM, exits
    python src/train/train.py --smoke-test --smoke-steps 5
    python src/train/train.py                            # a real run (not invoked by this task)

Run in .venv-train (workstation only — needs CUDA + rfdetr[train,loggers]).
"""

import argparse
import os
import sys
import time
from pathlib import Path

# rfdetr's COCOEvalCallback prints validation metrics through Rich, which on
# Windows defaults to the console's legacy codepage (cp1252) — an em-dash in
# a table header ("Val (Epoch N) — Overall Metrics") is outside that
# codepage and raises an unhandled UnicodeEncodeError, killing the run
# before any checkpoint is written. This must happen before rfdetr/pytorch
# training imports (they construct Rich consoles bound to sys.stdout at
# import/call time) and before the first print in this file.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_DATASET_DIR = REPO_ROOT / "data" / "interim" / "yolo"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"  # gitignored — see .gitignore "MODEL ARTIFACTS"
WORKSTATION_CONFIG = REPO_ROOT / "configs" / "machine.workstation.yaml"

NUM_CLASSES = 1  # SKU-110K: single "object" class, dense packing (configs/experiment.yaml)


def load_workstation_config() -> dict:
    if not WORKSTATION_CONFIG.is_file():
        sys.exit(f"{WORKSTATION_CONFIG} not found.")
    return yaml.safe_load(WORKSTATION_CONFIG.read_text(encoding="utf-8"))


def parse_args(machine_cfg: dict) -> argparse.Namespace:
    training_defaults = machine_cfg["training"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly --smoke-steps optimizer steps, print peak VRAM after each, then exit. No checkpoint saved.",
    )
    p.add_argument("--smoke-steps", type=int, default=3, help="Optimizer steps for --smoke-test (default: 3)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Cap a REAL run (full checkpoint/EMA/logger stack, unlike --smoke-test) at N optimizer "
            "steps. rfdetr's BestModelCallback.on_fit_end unconditionally writes last_ema.pth when "
            "the run ends, even mid-epoch, so this always produces a loadable checkpoint. Mutually "
            "exclusive with --smoke-test."
        ),
    )
    p.add_argument("--resolution", type=int, default=training_defaults["resolution"])
    p.add_argument("--batch-size", type=int, default=training_defaults["batch_size"])
    p.add_argument("--grad-accum", type=int, default=training_defaults["grad_accum"])
    p.add_argument("--epochs", type=int, default=100, help="Ignored under --smoke-test.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint/log dir. Defaults to checkpoints/smoke_test or checkpoints/rfdetr_nano_sku110k.",
    )
    return p.parse_args()


def count_materialized(split: str) -> int:
    return len(list((YOLO_DATASET_DIR / split / "images").glob("*.jpg")))


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")  # picks up WANDB_API_KEY / WANDB_PROJECT if present; no-op if absent

    machine_cfg = load_workstation_config()
    args = parse_args(machine_cfg)

    if args.smoke_test and args.max_steps is not None:
        sys.exit("--smoke-test and --max-steps are mutually exclusive — use one or the other.")

    if not YOLO_DATASET_DIR.is_dir():
        sys.exit(
            f"{YOLO_DATASET_DIR} not found.\n"
            "Run src/data/build_yolo_dataset.py first — it materializes "
            "data/interim/yolo/ from data/splits/{train,val,test}.txt in the "
            "directory layout rfdetr's YOLO loader expects."
        )

    if not torch.cuda.is_available():
        sys.exit("CUDA not available. This script targets the RTX 5070 workstation — run inside .venv-train.")

    import pytorch_lightning as pl
    from rfdetr import RFDETRNano
    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    class VRAMProbeCallback(pl.Callback):
        """Prints peak CUDA VRAM after every optimizer step. --smoke-test's
        whole purpose is answering "does one step fit in 12 GB", so this
        callback IS the smoke test's output."""

        def __init__(self, total_steps: int, vram_budget_gb: float):
            super().__init__()
            self.total_steps = total_steps
            self.vram_budget_gb = vram_budget_gb
            self._last_step = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step <= self._last_step:
                return  # mid-accumulation micro-batch, not an optimizer-step boundary
            self._last_step = step
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            pct = 100 * peak_gb / self.vram_budget_gb
            print(
                f"[smoke-test] step {step}/{self.total_steps}: "
                f"peak VRAM = {peak_gb:.2f} GB ({pct:.1f}% of {self.vram_budget_gb:.0f} GB budget)"
            )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else CHECKPOINT_ROOT / ("smoke_test" if args.smoke_test else "rfdetr_nano_sku110k")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    use_wandb = bool(wandb_key)
    wandb_project = os.environ.get("WANDB_PROJECT", "shelfsense-q")
    run_name = f"{'smoke-' if args.smoke_test else ''}rfdetr-nano-{int(time.time())}"
    if use_wandb:
        print(f"WANDB_API_KEY found — logging to project '{wandb_project}', run '{run_name}'.")
    else:
        print("No WANDB_API_KEY in environment/.env — training without W&B logging.")

    n_train = count_materialized("train")
    n_val = count_materialized("val")
    print(
        f"Dataset: {YOLO_DATASET_DIR.relative_to(REPO_ROOT)}  "
        f"(train={n_train}, val={n_val} images)"
    )
    print(
        f"Model: RFDETRNano  resolution={args.resolution}  num_classes={NUM_CLASSES}  "
        f"batch_size={args.batch_size}  grad_accum={args.grad_accum}  "
        f"(effective batch {args.batch_size * args.grad_accum})"
    )

    model = RFDETRNano(num_classes=NUM_CLASSES, resolution=args.resolution)

    train_config = TrainConfig(
        dataset_file="yolo",
        dataset_dir=str(YOLO_DATASET_DIR),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        epochs=1 if args.smoke_test else args.epochs,
        output_dir=str(output_dir),
        num_workers=args.num_workers,
        wandb=use_wandb,
        project=wandb_project if use_wandb else None,
        run=run_name if use_wandb else None,
    )

    if args.smoke_test:
        torch.cuda.reset_peak_memory_stats()
        vram_budget_gb = float(machine_cfg["compute"]["vram_gb"])
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}  ({vram_budget_gb:.0f} GB budget per configs/machine.workstation.yaml)")
        print(f"\n=== SMOKE TEST: {args.smoke_steps} optimizer step(s), then exit ===")

        module = RFDETRModelModule(model.model_config, train_config)
        datamodule = RFDETRDataModule(model.model_config, train_config)

        trainer = build_trainer(
            train_config,
            model.model_config,
            accelerator="gpu",
            include_training_callbacks=False,  # no checkpoints/EMA/COCO-eval/loggers for a 3-step probe
            callbacks=[VRAMProbeCallback(total_steps=args.smoke_steps, vram_budget_gb=vram_budget_gb)],
            max_steps=args.smoke_steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        start = time.time()
        trainer.fit(module, datamodule)
        elapsed = time.time() - start

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        pct = 100 * peak_gb / vram_budget_gb
        print(f"\n=== SMOKE TEST DONE ===")
        print(f"{args.smoke_steps} step(s) completed in {elapsed:.1f}s.")
        print(f"Final peak VRAM: {peak_gb:.2f} GB ({pct:.1f}% of {vram_budget_gb:.0f} GB budget).")
        if peak_gb >= vram_budget_gb:
            print("WARNING: peak VRAM reached/exceeded the configured budget.")
        return

    # Real training run — full rfdetr callback stack (checkpoints, EMA, COCO eval,
    # W&B/TensorBoard if configured).
    module = RFDETRModelModule(model.model_config, train_config)
    datamodule = RFDETRDataModule(model.model_config, train_config)
    trainer_kwargs = {}
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps
        print(
            f"Capped at max_steps={args.max_steps}. A full epoch of {n_train} train images at "
            f"batch_size={args.batch_size}/grad_accum={args.grad_accum} is "
            f"~{n_train // (args.batch_size * args.grad_accum)} steps, so this run will stop "
            "mid-epoch — expected. BestModelCallback.on_fit_end still guarantees a checkpoint."
        )
    trainer = build_trainer(train_config, model.model_config, accelerator="gpu", **trainer_kwargs)
    start = time.time()
    trainer.fit(module, datamodule)
    elapsed = time.time() - start
    print(f"\nTraining stopped after {elapsed:.1f}s. Checkpoints (if any) are under {output_dir}.")


if __name__ == "__main__":
    main()
