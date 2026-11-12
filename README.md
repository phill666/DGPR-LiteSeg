# DGPR-LiteSeg

## Introduction

DGPR-LiteSeg is a lightweight industrial surface defect segmentation network
incorporating Dual-Gated Feature Fusion (DGFF) and Groupwise Partial-Channel
Refinement (GPCR) into the PP-LiteSeg-STDC1 decoder.

This repository's current version contains **model source code only**. No
training pipeline, dataset, checkpoint, evaluation code or experimental result
is included. No weights or dataset files are required to construct the model.

## Model architecture

The frozen STDC1 encoder provides features at strides 8, 16 and 32. Its decoder
executes SPPM, stage6, DGFF1, GPCR1, stage7, DGFF2, GPCR2, stage8 and the
segmentation head, followed by bilinear output upsampling. DGFF1 also receives
the stride-16 encoder feature; DGFF2 receives the stride-8 feature.

DGFF aligns its two inputs, applies channel gating to their concatenation,
reduces the channels, and applies a separately generated spatial gate.
GPCR retains the original groupwise partial convolutions, depthwise/pointwise
mixing, SE gating, BatchNorm and learnable residual scaling.

The historical class names DFF and GSHalfConv map to DGFF and GPCR. Parameter
and buffer names and the computed model are preserved. In the official
32/64-channel GPCR instances, the original divisibility guard makes channel
shuffle an identity operation; no new shuffle is introduced.

## Requirements

- Python 3.10 (the tested interpreter version).
- PyTorch 2.0.0+cu118; the research CUDA runtime is 11.8.
- No additional third-party package is required by the model source.
- CUDA is optional for the CPU example below.

## Minimal usage

Run from the repository root:

```python
import torch
from models.dgpr_liteseg import DGPRLiteSeg

model = DGPRLiteSeg(num_classes=2)
model.eval()
x = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    y = model(x)

print(y.shape)  # torch.Size([1, 2, 512, 512])
```

The model is constructed from scratch and does not download or require weights.

## Citation

Working manuscript: *DGPR-LiteSeg: Dual-Gated Feature Fusion and Groupwise
Partial-Channel Refinement for Lightweight Industrial Surface Defect Segmentation*.
Author-approved publication metadata are pending. No DOI or publication
details are asserted.
