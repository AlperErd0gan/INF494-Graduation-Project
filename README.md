# INF494 — Plant Disease Severity Estimation
### Galatasaray University — Computer Engineering Graduation Project

A hybrid CNN + LSTM pipeline for automated, continuous disease severity scoring in plant pathology experiments. The system was developed on *Xanthomonas euvesicatoria* and evaluated under cross-pathogen transfer to *Pseudomonas syringae*.

---

## Overview

Traditional plant disease assessment relies on manual, discrete scoring which is subjective and labour-intensive. This project replaces that process with a two-stage deep learning pipeline:

1. **CNN (PatchSeverityNet)** — A fine-tuned ResNet-18 that scores 64×64 leaf patches extracted via HSV segmentation, producing a noisy but data-driven daily severity signal.
2. **LSTM (DenoiserLSTM)** — A sliding-window denoiser trained on logistic growth anchors that smooths the CNN signal into a clean, biologically consistent disease progression curve.

The logistic ground truth is derived from pathogen-specific anchor points fitted with `scipy.optimize.curve_fit`, providing a continuous and differentiable severity target without manual labelling.

---

## Repository Structure

```
├── xanthomonas_10patch_pipeline.ipynb   # Xa pipeline — 10 images/day
├── xanthomonas_20patch_pipeline.ipynb   # Xa pipeline — 20 images/day
└── transfer_pipeline.ipynb              # Transfer learning → Pseudomonas
```

---

## Pipeline

```
Raw Images
    │
    ▼
HSV Segmentation (segment_leaf_only)
    │
    ▼
Patch Extraction (64×64, stride=32, green_threshold)
    │
    ▼
CNN Inference (PatchSeverityNet / ResNet-18)
    │  sigmoid → per-patch probability
    ▼
Daily Mean Score  ──►  Noisy CNN Signal
    │
    ▼
LSTM Denoiser (sliding window = 3–5 days)
    │
    ▼
Smoothed Severity Score  ──►  Disease Progression Curve
```

---

## Key Design Choices

| Decision | Rationale |
|---|---|
| Patch-based CNN | Handles high-resolution images without resizing; preserves local texture |
| HSV segmentation | Removes soil/pot background before patch extraction |
| Pairwise ranking loss | Enforces monotonic ordering across disease stages |
| Logistic prior as ground truth | Continuous target without manual per-image labelling |
| LSTM window input only (no time index) | Prevents temporal memorisation (verified via ablation) |
| Partial freezing (layer4 + fc) | Sufficient capacity for small transfer datasets (7–10 images/day) |

---

## Transfer Learning

The Xanthomonas base models are transferred to Pseudomonas syringae data with minimal adaptation:

- **CNN**: Partial fine-tuning (layer4 + fc unfrozen, weight decay = 1e-4)
- **LSTM**: Trained from scratch on the new pathogen's CNN score series
- **Zero-shot baseline**: Xanthomonas weights applied directly with no fine-tuning

---

## Requirements

```
torch torchvision
opencv-python
scikit-learn
scipy
pandas numpy matplotlib seaborn
tqdm
```

---

## Coding Standards

This project follows the Galatasaray University Computer Engineering graduation project coding standards:

- Variables & functions — `snake_case`
- Classes — `PascalCase`
- Constants — `UPPER_SNAKE_CASE`
- All functions documented with docstrings (parameters + return values)
- Line length — 80–100 characters
- Comments in English

---

*Galatasaray University — Computer Engineering — 2025–2026*