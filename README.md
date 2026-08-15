# ShelfSense-Q

Real-time retail shelf analytics deployed to Qualcomm NPUs.
RF-DETR fine-tuned on densely packed shelves, quantized to INT8, and profiled
on real Qualcomm silicon across five hardware targets.

> **Status:** in development. All numbers marked `TODO(measure)` are
> unmeasured placeholders and must not be cited.

## Why

Qualcomm publishes RF-DETR inference latency across ~20 chipsets — in float
precision only. The INT8 accuracy/latency trade-off is unpublished. This
project measures it, on a dense-detection workload (~147 objects per image),
and reports the failure cases as well as the wins.

## Results

TODO(measure)

## Licensing

- **Models:** RF-DETR is Apache-2.0. Commercially usable.
- **Dataset:** SKU-110K is provided for academic / non-commercial use only.

Because the dataset is research-only, this repository is a **research
demonstration**, not a commercially deployable system. The model weights and
toolchain would permit commercial use; the training data does not.

## Reproducing

See [docs/SETUP.md](docs/SETUP.md).

## Engineering decisions

See [DECISIONS.md](DECISIONS.md) and [FAILURES.md](FAILURES.md).
