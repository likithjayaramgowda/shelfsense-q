"""
Export a fine-tuned RF-DETR checkpoint to ONNX, verify PyTorch vs ONNX
Runtime numerical parity on a fixed input, and compare the resulting
graph's input/output shapes + opset against a stock (un-finetuned)
RFDETRNano export.

Verification is not optional here — CLAUDE.md hard rule 5 is about FP32 vs
INT8 preprocessing drift, but the same failure mode (an export that looks
fine but silently computes something different from the PyTorch model) is
just as real between eager PyTorch and ONNX. This script proves parity
before calling an export usable, rather than trusting that export() worked
because it didn't raise.

Usage:
    python src/export/to_onnx.py --checkpoint checkpoints/plumbing_test/last_ema.pth

Run in .venv-train (needs rfdetr[onnx], already in requirements-train.txt).
"""

import argparse
import sys
from pathlib import Path

# See src/train/train.py for why: Windows' legacy console codepage (cp1252)
# can't encode an em-dash, and this script prints a few. Confirmed to crash
# under plain ">" redirection (train.py did); reconfigure before anything
# else prints.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = REPO_ROOT / "exports"  # gitignored — see .gitignore "MODEL ARTIFACTS"

DEFAULT_OPSET = 17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to a fine-tuned RF-DETR .pth checkpoint")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--seed", type=int, default=0, help="Seed for the fixed PyTorch-vs-ONNX comparison input")
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Max-abs-diff threshold for the parity PASS/FAIL verdict (default: 1e-3)",
    )
    p.add_argument(
        "--skip-stock-compare",
        action="store_true",
        help="Skip exporting+diffing a stock (COCO, 90-class) RFDETRNano for comparison",
    )
    return p.parse_args()


def export_model(model, output_dir: Path, opset: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model.export(output_dir=str(output_dir), opset_version=opset, verbose=False)
    return Path(onnx_path)


def graph_io_summary(onnx_path: Path):
    """Return (input_shapes, output_shapes, opset_by_domain) read directly
    from the ONNX graph — not inferred from Python-side config, so this
    reflects what actually got written to disk."""
    import onnx

    m = onnx.load(str(onnx_path))

    def shape_of(value_info):
        dims = []
        for d in value_info.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        return tuple(dims)

    inputs = {vi.name: shape_of(vi) for vi in m.graph.input}
    outputs = {vi.name: shape_of(vi) for vi in m.graph.output}
    opsets = {(op.domain or "ai.onnx"): op.version for op in m.opset_import}
    return inputs, outputs, opsets


def verify_parity(rf_model, onnx_path: Path, seed: int) -> dict:
    """Run the same fixed input through eager PyTorch and the exported ONNX
    graph (CPU, in both cases, for a clean apples-to-apples comparison) and
    return max-abs-diff per output tensor."""
    import onnxruntime as ort

    from rfdetr.export.main import make_infer_image

    resolution = rf_model.model_config.resolution

    # make_infer_image reproduces RF-DETR's real predict()-time preprocessing
    # (resize + ImageNet normalize) rather than raw [0,1] noise — same fixed
    # input generator export() uses internally for its own sanity check.
    np.random.seed(seed)
    torch.manual_seed(seed)
    x = make_infer_image(infer_dir=None, shape=(resolution, resolution), batch_size=1, device="cpu")

    net = rf_model.model.model.to("cpu").eval()
    with torch.no_grad():
        out = net(x)
    pt_dets = out["pred_boxes"].cpu().numpy()
    pt_labels = out["pred_logits"].cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_dets, onnx_labels = sess.run(["dets", "labels"], {"input": x.numpy()})

    return {
        "dets_max_abs_diff": float(np.abs(pt_dets - onnx_dets).max()),
        "labels_max_abs_diff": float(np.abs(pt_labels - onnx_labels).max()),
    }


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    from rfdetr import RFDETR, RFDETRNano

    print(f"Loading fine-tuned checkpoint: {ckpt_path}")
    ft_model = RFDETR.from_checkpoint(str(ckpt_path))
    print(
        f"  resolved class: {type(ft_model).__name__}  "
        f"num_classes={ft_model.model_config.num_classes}  "
        f"resolution={ft_model.model_config.resolution}"
    )

    ft_onnx = export_model(ft_model, EXPORT_ROOT / "finetuned", args.opset)
    print(f"Exported fine-tuned model -> {ft_onnx.relative_to(REPO_ROOT)}")

    ft_inputs, ft_outputs, ft_opsets = graph_io_summary(ft_onnx)
    print(f"  input shapes:  {ft_inputs}")
    print(f"  output shapes: {ft_outputs}")
    print(f"  opset:         {ft_opsets}")

    print("\nVerifying PyTorch vs ONNX Runtime parity on a fixed input...")
    diffs = verify_parity(ft_model, ft_onnx, args.seed)
    print(f"  dets   max abs diff: {diffs['dets_max_abs_diff']:.3e}")
    print(f"  labels max abs diff: {diffs['labels_max_abs_diff']:.3e}")
    overall = max(diffs.values())
    verdict = "PASS" if overall < args.tolerance else "FAIL"
    print(f"  verdict: {verdict} (max diff {overall:.3e} vs tolerance {args.tolerance:.0e})")
    if verdict == "FAIL":
        print(
            "  FAIL means the ONNX graph does not numerically match the PyTorch model closely "
            "enough to trust — do not proceed to quantization/deployment on this export."
        )

    if args.skip_stock_compare:
        return

    print("\nExporting stock (un-finetuned, COCO 90-class) RFDETRNano for comparison...")
    stock_model = RFDETRNano()
    stock_onnx = export_model(stock_model, EXPORT_ROOT / "stock_rfdetr_nano", args.opset)
    print(f"Exported stock model -> {stock_onnx.relative_to(REPO_ROOT)}")

    stock_inputs, stock_outputs, stock_opsets = graph_io_summary(stock_onnx)
    print(f"  input shapes:  {stock_inputs}")
    print(f"  output shapes: {stock_outputs}")
    print(f"  opset:         {stock_opsets}")

    print("\n=== STRUCTURAL COMPARISON: fine-tuned vs stock ===")
    checks = [
        ("input shapes", ft_inputs, stock_inputs),
        ("output shapes", ft_outputs, stock_outputs),
        ("opset", ft_opsets, stock_opsets),
    ]
    any_diff = False
    for label, ft_val, stock_val in checks:
        if ft_val != stock_val:
            any_diff = True
            print(f"  DIFFER — {label}:")
            print(f"    fine-tuned: {ft_val}")
            print(f"    stock:      {stock_val}")
        else:
            print(f"  match — {label}: {ft_val}")
    if not any_diff:
        print("  No structural differences found — the two exports have identical I/O shapes and opset.")


if __name__ == "__main__":
    main()
