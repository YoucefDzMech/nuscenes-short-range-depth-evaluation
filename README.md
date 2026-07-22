# Intrinsics-Aware Comparison of Foundation Monocular Depth Estimators for Short-Range Obstacle Detection in Automotive Applications

Evaluation repository for the paper (AEIT 2026).

## Scope
This repository is **evaluation-only**. It does not include model architectures, training code, or model weights.

Included:
- nuScenes evaluation scripts for `UniDepthV2`, `DA3`, and `Metric3D-v2`
- aggregated summary metrics used in the paper
- per-model requirements files for reproducibility
- plotting script for paper figures

Not included:
- model weights/checkpoints
- nuScenes dataset
- virtual environments
- large per-frame outputs (`per_frame_metrics.csv`, `obstacle_box_metrics.csv`, `frames/`)

## Evaluation setup
- Dataset split: `nuScenes trainval02`
- Camera: `CAM_FRONT` (1600×900, \(f_x \approx 1266\) px) — the rear camera (`CAM_BACK`) is not used because the roof LiDAR returns off the ego bumper at ~0.2 m contaminate the ground truth at short range.
- Frames: **16,182 forward-camera sweep frames**, evaluated identically across all three models (16,131 contain valid in-range ground truth).
- Ground-truth source: `LIDAR_TOP` projected to image plane.
- Evaluation range: \(0 < d \le 5\,m\).
- LiDAR is **evaluation-only** and never used as model input.
- Valid-pixel mask: `gt in (0, 5] m` and prediction finite and positive. **No upper bound is placed on the prediction** — capping predictions to the range would discard exactly the pixels where a model most over-estimates, removing its largest errors from its own statistics (see Corrections below).

### Evaluation pipeline
![Evaluation pipeline](figures/fig1_pipeline.png)

The scripts run each model on RGB images, compute sparse LiDAR-based depth metrics, and compute object-level nearest-depth error inside YOLO detection boxes.

## Repository layout
- [eval/nuscenes_unidepthv2_eval.py](eval/nuscenes_unidepthv2_eval.py)
- [eval/nuscenes_da3_eval.py](eval/nuscenes_da3_eval.py)
- [eval/nuscenes_metric3d_eval.py](eval/nuscenes_metric3d_eval.py)
- [eval/plot_results.py](eval/plot_results.py)
- [results/unidepthv2](results/unidepthv2)
- [results/da3](results/da3)
- [results/metric3d](results/metric3d)
- [requirements](requirements)
- [figures](figures)

## Example qualitative frames (trainval02 CAM_FRONT)
Five generated composite examples per model are included:
- [figures/examples/unidepthv2](figures/examples/unidepthv2)
- [figures/examples/da3](figures/examples/da3)
- [figures/examples/metric3d](figures/examples/metric3d)

Included frame IDs: `7008`, `12809`, `13838`, `15810`, `15885`.

## Key reported results
Pixel-level (summary metrics, CAM_FRONT, 16,182 frames, 0–5 m):

| Model | MAE (m) | RMSE (m) | AbsRel | MedianAE (m) |
|---|---|---|---|---|
| UniDepthV2 (ViT-L) | **0.222** | 2.640 | **5.35%** | **0.095** |
| Metric3D-v2 (ViT-Giant2) | 0.246 | **1.917** | 6.04% | 0.146 |
| DA3 (Metric-Large) | 0.636 | 1.648 | 14.37% | 0.503 |

Object-level nearest-depth (YOLO boxes):

| Model | Boxes | Box MAE (m) | Box median (m) |
|---|---|---|---|
| UniDepthV2 | 2,371 | **0.845** | **0.126** |
| Metric3D-v2 | 2,371 | 0.897 | 0.230 |
| DA3 | 2,274 | 0.927 | 0.585 |

**Finding:** the two strategies that integrate intrinsics into the network's computation — per-layer K conditioning (UniDepthV2) and canonical-space normalization (Metric3D-v2) — reach near-equivalent short-range accuracy, while post-hoc focal-length scaling (DA3) is substantially weaker. UniDepthV2 leads at both the pixel and object level. Across all models RMSE far exceeds MAE, indicating occasional large-magnitude errors that a safety-oriented deployment must account for.

## Generate comparison figure
From repository root:

```bash
python eval/plot_results.py
```

This writes [figures/fig3_metric_comparison.png](figures/fig3_metric_comparison.png).

## Corrections (July 2026)
The results and scripts in this repository were revised after two issues were identified and fixed:

1. **Valid-pixel mask (UniDepthV2 script).** The mask previously required `pred <= max_depth` in addition to the ground-truth range condition. This discarded pixels where the model predicted beyond the 5 m range even though the ground truth was in range — i.e. the model's own largest over-estimates were removed from its statistics, and models were not comparable. The condition was removed so that all three scripts apply the same mask (`gt in (0,5] m` and prediction finite and positive). See `eval/nuscenes_unidepthv2_eval.py`, `compute_metrics`.
2. **Camera and frame set.** Evaluation moved from `CAM_BACK` to `CAM_FRONT` (rear-bumper occlusion) and to a single shared set of 16,182 sweep frames for every model, replacing the earlier per-model frame subsets.

Both changes make the evaluation consistent and comparable across models; the numbers in this README reflect the corrected protocol.

## Reproducibility notes
Each model has its own requirements file:
- [requirements/requirements_unidepthv2.txt](requirements/requirements_unidepthv2.txt)
- [requirements/requirements_da3.txt](requirements/requirements_da3.txt)
- [requirements/requirements_metric3d.txt](requirements/requirements_metric3d.txt)

Create one environment per model and run the corresponding script in [eval](eval).
