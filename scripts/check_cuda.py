"""Blackwell / sm_120 smoke test. Run on the workstation, in .venv-train."""

import sys

try:
    import torch
except ImportError:
    sys.exit("torch not installed. Are you in .venv-train?")

print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit(
        "\nFAIL: no CUDA.\n"
        "Reinstall from the CUDA 12.8+ index:\n"
        "  pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu128"
    )

cap = torch.cuda.get_device_capability()
print(f"device         : {torch.cuda.get_device_name(0)}")
print(f"capability     : {cap}")
print(f"vram           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if cap[0] < 12:
    print(
        "\nWARNING: expected (12, 0) for RTX 5070 Blackwell.\n"
        "A lower capability means torch is not seeing the card correctly."
    )
else:
    # Actually exercise the GPU — availability alone is not proof.
    x = torch.randn(1000, 1000, device="cuda")
    _ = (x @ x).sum().item()
    torch.cuda.synchronize()
    print("\nOK: matmul on sm_120 succeeded.")
