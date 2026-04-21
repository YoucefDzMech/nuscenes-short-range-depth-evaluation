import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(results_root: Path):
    pixel = {}
    box = {}
    for model in ["unidepthv2", "da3", "metric3d"]:
        s = pd.read_csv(results_root / model / "summary_metrics.csv").iloc[0]
        pixel[model] = {
            "MAE": float(s["mae_m"]),
            "RMSE": float(s["rmse_m"]),
            "AbsRel": float(s["abs_rel"]),
        }
        with open(results_root / model / "obstacle_box_summary.json", "r", encoding="utf-8") as f:
            b = json.load(f)
        box[model] = float(b["mae_min_depth_m"])
    return pixel, box


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"
    fig_out = repo_root / "figures" / "fig3_metric_comparison.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    pixel, box = load_metrics(results_root)
    model_order = ["unidepthv2", "da3", "metric3d"]
    labels = ["UniDepthV2", "DA3", "Metric3D-v2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    x = range(len(model_order))
    axes[0].bar([i - 0.25 for i in x], [pixel[m]["MAE"] for m in model_order], width=0.25, label="MAE")
    axes[0].bar(x, [pixel[m]["RMSE"] for m in model_order], width=0.25, label="RMSE")
    axes[0].bar([i + 0.25 for i in x], [pixel[m]["AbsRel"] for m in model_order], width=0.25, label="AbsRel")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_title("Pixel-level metrics (0 < d ≤ 5 m)")
    axes[0].set_ylabel("Error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(labels, [box[m] for m in model_order])
    axes[1].set_title("Object-level nearest-depth MAE")
    axes[1].set_ylabel("MAE (m)")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Intrinsics-aware comparison on nuScenes trainval02 (CAM_BACK)")
    fig.tight_layout()
    fig.savefig(fig_out, bbox_inches="tight")
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
