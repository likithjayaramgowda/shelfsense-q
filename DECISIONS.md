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
