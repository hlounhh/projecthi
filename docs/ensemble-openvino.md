# Posture ensemble: export and runtime

The opt-in 3-member posture ensemble (`baseline`, `fall-finetuned-v1`,
`fall-finetuned-v2`) can run either through Torch/Ultralytics or through a
verified OpenVINO bundle that shares one letterbox per frame.

## Export

INT8 (CPU default; needs calibration images):

```bash
.venv/Scripts/python.exe tools/export_ensemble_openvino.py \
  --output models/accel/ensemble-416-int8 \
  --size 416 --precision int8 \
  --calibration-images data/datasets/robustness_v1_trainval/images/train \
  --calibration-limit 256
```

FP16 or FP32 (no calibration needed):

```bash
.venv/Scripts/python.exe tools/export_ensemble_openvino.py \
  --output models/accel/ensemble-416-fp16 --size 416 --precision fp16
```

Every member checkpoint is verified against the SHA256 pin in
`ai/posture_ensemble.py` before any export starts, and the 20 MiB cap is
applied to each checkpoint. The parent `ensemble.json` records each member's
source SHA256, export precision and member-manifest SHA256; the runtime
re-verifies all of it on every load.

## Run

```bash
# Ensemble, OpenVINO INT8 (explicit verified bundle)
.venv/Scripts/python.exe main.py --posture-backend ensemble \
  --posture-model-dir models/accel/ensemble-416-int8

# Ensemble, Torch (no bundle: each member runs through Ultralytics)
.venv/Scripts/python.exe main.py --posture-backend ensemble

# Ensemble, OpenVINO FP16 (higher fidelity, ~1.3x slower than INT8)
.venv/Scripts/python.exe main.py --posture-backend ensemble \
  --posture-model-dir models/accel/ensemble-416-fp16
```

`ensemble` never falls back silently: without an explicit bundle directory it
uses Torch, and a missing or tampered bundle raises. Default posture backend
remains `openvino` (single member); the ensemble is opt-in.

Torch members run with `rect=False`, so both backends share the same static
square 416x416 letterbox canvas and therefore the same geometry.

## Measured on this machine

`video3.mp4`, 40 frames, 416 px, 2 CPU threads:

| backend | mean | p50 | p95 |
| --- | --- | --- | --- |
| Torch | 195.8 ms | 189.4 ms | 245.5 ms |
| OpenVINO FP32 | 141.5 ms | 137.6 ms | 166.6 ms |
| OpenVINO FP16 | 107.7 ms | 106.7 ms | 116.0 ms |
| OpenVINO INT8 | 78.6 ms | 75.1 ms | 103.4 ms |

Parity over 160 sampled frames across the local videos (greedy IoU >= 0.5
matching):

| pair | matched | torch-only | ov-only | label mismatch | mean IoU | max conf delta |
| --- | --- | --- | --- | --- | --- | --- |
| Torch vs FP16 | 212 | 0 | 2 | 0 | 0.9993 | 0.0058 |
| Torch vs INT8 | 192 | 20 | 28 | 1 | 0.9164 | 0.3499 |

Fallen-flag agreement (max `fallen` confidence >= 0.4): FP16 disagrees with
Torch on 0/160 frames (max confidence delta 0.0028); INT8 disagrees on 4/160,
all INT8-only flags in the 0.40-0.63 range where Torch had no fallen box.

Annotated fallen-class AP50 on the 72-image degraded split
(`data/datasets/robustness_v1_eval_degraded`): Torch 0.2884, FP16 0.3081,
INT8 0.3123. This split is small and is not a held-out benchmark.

## Precision guidance

- **INT8** is the fastest option and stays the default path for the bundle
  name used by `main.py`. It changes a few borderline fall flags.
- **FP16** is the recommended fidelity fallback: near-identical to Torch
  geometry and confidence, with a ~1.8x speedup over Torch.
- **FP32** is available for parity work only; it is the slowest OpenVINO
  option and is not needed at inference time.
