# Cloud Removal & Explainable AI Project

This workspace contains a VS Code-first implementation scaffold for RICE-based cloud detection, cloud removal, benchmarking, explainability, and a Streamlit UI.

## What’s included

- `datasets/` — RICE dataset loader
- `models/` — baseline segmentation and removal models
- `training/` — training scripts
- `metrics/` — PSNR, SSIM, IoU, Dice helpers
- `explainability/` — Captum utilities
- `ui/` — Streamlit app skeleton
- `configs/` — project settings
- `scripts/` — utility scripts
- `outputs/` — generated results

## Dataset layout

The workspace already includes:

- `RICE_DATASET/RICE1/cloud`
- `RICE_DATASET/RICE1/label`
- `RICE_DATASET/RICE2/cloud`
- `RICE_DATASET/RICE2/label`
- `RICE_DATASET/RICE2/mask`

## Next steps

1. Open this folder in VS Code.
2. Install the Python packages from `requirements.txt`.
3. Run the dataset inspection script.
4. Train the segmentation baseline on `RICE2`.
5. Use the Streamlit UI to compare results.
