# Intrinsics-Aware Comparison of Foundation Monocular Depth Estimators for Short-Range Obstacle Detection in Automotive Applications

IEEE MetroAutomotive 2026 — evaluation repository.

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
- Camera: `CAM_BACK` (1600×900, \(f_x \approx 1266\) px)
- Ground-truth source: `LIDAR_TOP` projected to image plane
- Evaluation range: \(0 < d \le 5\,m\)
- LiDAR is **evaluation-only** and never used as model input.

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

## Example qualitative frames (trainval02 CAM_BACK)
Five generated composite examples per model are included:
- [figures/examples/unidepthv2](figures/examples/unidepthv2)
- [figures/examples/da3](figures/examples/da3)
- [figures/examples/metric3d](figures/examples/metric3d)

Included frame IDs: `3382`, `3385`, `3401`, `3402`, `3403`.

## Key reported results
Pixel-level (summary metrics, 0–5 m):
- UniDepthV2: MAE 0.138 m, RMSE 0.277 m, AbsRel 11.5%
- Metric3D-v2: MAE 0.190 m, RMSE 1.323 m, AbsRel 24.4%
- DA3: MAE 0.376 m, RMSE 1.145 m, AbsRel 25.4%

Object-level (nearest-depth in YOLO boxes):
- DA3: box MAE 0.726 m
- UniDepthV2: box MAE 0.749 m
- Metric3D-v2: box MAE 0.915 m

## Generate comparison figure
From repository root:

```bash
python eval/plot_results.py
```

This writes [figures/fig3_metric_comparison.png](figures/fig3_metric_comparison.png).

## Reproducibility notes
Each model has its own requirements file:
- [requirements/requirements_unidepthv2.txt](requirements/requirements_unidepthv2.txt)
- [requirements/requirements_da3.txt](requirements/requirements_da3.txt)
- [requirements/requirements_metric3d.txt](requirements/requirements_metric3d.txt)

Create one environment per model and run the corresponding script in [eval](eval).
