import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from safetensors.torch import load_file as load_safetensors_file
from tabulate import tabulate
from ultralytics import YOLO

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


@dataclass
class FrameMetrics:
    sample_token: str
    camera_token: str
    lidar_token: str
    valid_pixels: int
    mae_m: float
    rmse_m: float
    abs_rel: float


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sample_tokens(nusc: NuScenes, camera: str, lidar: str, max_samples: int) -> List[str]:
    tokens: List[str] = []
    for scene in nusc.scene:
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            if camera in sample["data"] and lidar in sample["data"]:
                tokens.append(token)
            token = sample["next"]
            if max_samples > 0 and len(tokens) >= max_samples:
                return tokens
    return tokens


def load_unidepthv2_model(checkpoint_path: str, config_path: str, device: torch.device) -> UniDepthV2:
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    model = UniDepthV2(full_cfg)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = load_safetensors_file(checkpoint_path)
    load_info = model.load_state_dict(state_dict, strict=False)
    if len(load_info.missing_keys) > 0:
        print(f"Warning: missing keys: {len(load_info.missing_keys)}")
    if len(load_info.unexpected_keys) > 0:
        print(f"Warning: unexpected keys: {len(load_info.unexpected_keys)}")

    model = model.to(device).eval()
    return model


def lidar_to_camera_points(
    nusc: NuScenes,
    lidar_token: str,
    camera_token: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sd_lidar = nusc.get("sample_data", lidar_token)
    sd_cam = nusc.get("sample_data", camera_token)

    cs_lidar = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", sd_lidar["ego_pose_token"])

    cs_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", sd_cam["ego_pose_token"])

    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_lidar["filename"]))

    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))

    pc.rotate(Quaternion(pose_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(pose_lidar["translation"]))

    pc.translate(-np.array(pose_cam["translation"]))
    pc.rotate(Quaternion(pose_cam["rotation"]).rotation_matrix.T)

    pc.translate(-np.array(cs_cam["translation"]))
    pc.rotate(Quaternion(cs_cam["rotation"]).rotation_matrix.T)

    points_cam = pc.points[:3, :]
    depths = points_cam[2, :]

    intrinsic = np.array(cs_cam["camera_intrinsic"], dtype=np.float32)
    points_img = view_points(points_cam, intrinsic, normalize=True)
    u = points_img[0, :]
    v = points_img[1, :]

    return points_cam, depths, u, v, intrinsic


def build_sparse_depth_map(
    image_h: int,
    image_w: int,
    depths: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    min_depth_m: float = 0.1,
) -> np.ndarray:
    sparse = np.zeros((image_h, image_w), dtype=np.float32)
    valid = (
        (depths > min_depth_m)
        & (u >= 0)
        & (u < image_w)
        & (v >= 0)
        & (v < image_h)
    )

    uu = np.floor(u[valid]).astype(np.int32)
    vv = np.floor(v[valid]).astype(np.int32)
    uu = np.clip(uu, 0, image_w - 1)
    vv = np.clip(vv, 0, image_h - 1)
    dd = depths[valid].astype(np.float32)

    for x, y, d in zip(uu, vv, dd):
        existing = sparse[y, x]
        if existing == 0.0 or d < existing:
            sparse[y, x] = d

    return sparse


def infer_metric_depth(
    model: UniDepthV2,
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
    intrinsics_mode: str = "tensor",
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous()
    K = torch.from_numpy(intrinsic.astype(np.float32))

    with torch.inference_mode():
        if intrinsics_mode == "none":
            pred = model.infer(rgb)
        elif intrinsics_mode == "tensor":
            pred = model.infer(rgb, K)
        elif intrinsics_mode == "pinhole":
            camera = Pinhole(K=K)
            pred = model.infer(rgb, camera)
        else:
            raise ValueError(f"Unsupported intrinsics mode: {intrinsics_mode}")

    depth_tensor = pred.get("depth", None)
    if depth_tensor is None:
        depth_tensor = pred["radius"]

    depth = depth_tensor.detach().float().cpu().numpy()
    while depth.ndim > 2:
        depth = np.squeeze(depth, axis=0)

    if not np.any(np.isfinite(depth) & (depth > 0)) and "radius" in pred:
        radius = pred["radius"].detach().float().cpu().numpy()
        while radius.ndim > 2:
            radius = np.squeeze(radius, axis=0)
        depth = radius

    return depth.astype(np.float32)


def infer_with_fallback(
    model: UniDepthV2,
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
    device: torch.device,
    cpu_fallback_enabled: bool,
    intrinsics_mode: str,
) -> Tuple[np.ndarray, UniDepthV2, torch.device, bool]:
    pred_metric = infer_metric_depth(model, image_bgr, intrinsic, intrinsics_mode=intrinsics_mode)

    if not np.any(np.isfinite(pred_metric) & (pred_metric > 0)) and device.type == "cuda":
        if not cpu_fallback_enabled:
            print(
                f"Warning: CUDA inference produced invalid depth for mode={intrinsics_mode}. "
                "Falling back to CPU inference."
            )
            model = model.to("cpu").eval()
            device = torch.device("cpu")
            cpu_fallback_enabled = True
        pred_metric = infer_metric_depth(model, image_bgr, intrinsic, intrinsics_mode=intrinsics_mode)

    return pred_metric, model, device, cpu_fallback_enabled


def resolve_image_path(nusc_dataroot: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(nusc_dataroot, filename),
        os.path.join(nusc_dataroot, filename.replace("/samples/", "/sweeps/")),
        os.path.join(nusc_dataroot, filename.replace("\\samples\\", "\\sweeps\\")),
    ]

    base = os.path.basename(filename)
    for folder in ("samples", "sweeps"):
        candidates.append(os.path.join(nusc_dataroot, folder, "CAM_BACK", base))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def compute_metrics(
    pred_metric: np.ndarray,
    gt_sparse_m: np.ndarray,
    max_depth_m: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    valid = (
        (gt_sparse_m > 0.0)
        & (gt_sparse_m <= max_depth_m)
        & np.isfinite(pred_metric)
        & (pred_metric > 0.0)
        & (pred_metric <= max_depth_m)
    )
    pred = pred_metric[valid]
    gt = gt_sparse_m[valid]

    if pred.size == 0:
        return {"valid_pixels": 0, "mae_m": np.nan, "rmse_m": np.nan, "abs_rel": np.nan}, valid

    abs_err = np.abs(pred - gt)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    abs_rel = float(np.mean(abs_err / np.clip(gt, 1e-6, None)))
    return {"valid_pixels": int(pred.size), "mae_m": mae, "rmse_m": rmse, "abs_rel": abs_rel}, valid


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    d = depth_map.copy()
    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    dmin = np.percentile(d[valid], 2)
    dmax = np.percentile(d[valid], 98)
    if dmax <= dmin:
        dmax = dmin + 1e-6

    d_norm = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap("Spectral")
    colored = (cmap((d_norm * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
    return colored[:, :, ::-1]


def overlay_lidar_points(image_bgr: np.ndarray, sparse_depth_m: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    ys, xs = np.where(sparse_depth_m > 0)
    if ys.size == 0:
        return out

    depths = sparse_depth_m[ys, xs]
    dmin = np.percentile(depths, 5)
    dmax = np.percentile(depths, 95)
    if dmax <= dmin:
        dmax = dmin + 1e-6

    for x, y, d in zip(xs, ys, depths):
        t = float(np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0))
        color = (int(255 * (1.0 - t)), int(255 * t), 255)
        cv2.circle(out, (int(x), int(y)), 1, color, -1)

    return out


def evaluate_boxes(
    pred_metric: np.ndarray,
    gt_sparse_m: np.ndarray,
    image_bgr: np.ndarray,
    draw_image_bgr: np.ndarray,
    yolo_model,
    conf_threshold: float,
    max_object_depth_m: Optional[float],
    frame_idx: int,
    sample_token: str,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    results = yolo_model(image_bgr, verbose=False)
    if len(results) == 0:
        return records

    result = results[0]
    if result.boxes is None:
        return records

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()
    names = result.names if hasattr(result, "names") else {}

    h, w = pred_metric.shape

    for box, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        pred_roi = pred_metric[y1:y2, x1:x2]
        gt_roi = gt_sparse_m[y1:y2, x1:x2]

        pred_valid = pred_roi[np.isfinite(pred_roi) & (pred_roi > 0)]
        gt_valid = gt_roi[gt_roi > 0]
        if pred_valid.size == 0 or gt_valid.size == 0:
            continue

        pred_min = float(np.min(pred_valid))
        gt_min = float(np.min(gt_valid))

        if max_object_depth_m is not None and min(pred_min, gt_min) > float(max_object_depth_m):
            continue

        abs_err = abs(pred_min - gt_min)
        abs_rel = abs_err / max(gt_min, 1e-6)

        class_name = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))

        cv2.rectangle(draw_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} p:{pred_min:.1f}m gt:{gt_min:.1f}m"
        y_label = max(20, y1 - 8)
        cv2.putText(draw_image_bgr, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        records.append(
            {
                "frame_idx": int(frame_idx),
                "sample_token": sample_token,
                "class_id": int(cls_id),
                "class_name": class_name,
                "conf": float(conf),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "pred_min_m": pred_min,
                "gt_min_m": gt_min,
                "abs_err_m": abs_err,
                "abs_rel": float(abs_rel),
            }
        )

    return records


def save_composite_output(
    outdir: str,
    idx: int,
    lidar_overlay: np.ndarray,
    pred_metric: np.ndarray,
    metrics: Dict[str, float],
    mode_label: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    depth_vis = colorize_depth(pred_metric)
    composite = cv2.hconcat([lidar_overlay, depth_vis])

    table_lines = [
        "Frame Metrics (UniDepthV2)",
        f"Mode: {mode_label}",
        "Scale: 1.0000 (metric)",
        f"Valid px: {int(metrics['valid_pixels'])}",
        f"MAE (m): {metrics['mae_m']:.4f}",
        f"RMSE (m): {metrics['rmse_m']:.4f}",
        f"AbsRel: {metrics['abs_rel']:.4f}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_h = 24
    padding = 12

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in table_lines]
    box_w = max(w for w, _ in text_sizes) + padding * 2
    box_h = line_h * len(table_lines) + padding

    x2 = composite.shape[1] - 20
    y1 = 20
    x1 = max(0, x2 - box_w)
    y2 = min(composite.shape[0], y1 + box_h)

    overlay = composite.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, composite, 0.25, 0, composite)
    cv2.rectangle(composite, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for i, line in enumerate(table_lines):
        y_text = y1 + padding + (i + 1) * line_h - 8
        cv2.putText(composite, line, (x1 + padding, y_text), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    output_path = os.path.join(outdir, f"{idx:04d}_composite.png")
    cv2.imwrite(output_path, composite)


def main() -> None:
    parser = argparse.ArgumentParser("nuScenes trainval UniDepthV2 + YOLO + LiDAR demo")
    parser.add_argument("--dataroot", type=str, default=r"C:\DataSet\v1.0-trainval02_blobs")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--camera", type=str, default="CAM_BACK")
    parser.add_argument("--lidar", type=str, default="LIDAR_TOP")
    parser.add_argument("--checkpoint", type=str, default=r"C:\RVC\Weights\UniDepth\337d0ee1ac66673e7449612f8ddcf05636c9ad58270e00158705ffcab43822a1")
    parser.add_argument("--config", type=str, default=r"C:\RVC\Weights\UniDepth\unidepth-v2-vitl14\config.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=r"C:\RVC\OtherThanModel\Models'Output\UniDepthV2\trainval02_CAM_BACK_intrinsics_5m")
    parser.add_argument(
        "--intrinsics-mode",
        type=str,
        choices=["none", "tensor", "pinhole"],
        default="pinhole",
        help="Primary inference mode for camera intrinsics.",
    )
    parser.add_argument(
        "--compare-intrinsics-impact",
        action="store_true",
        help="Also evaluate a baseline mode and save impact comparison files.",
    )
    parser.add_argument(
        "--baseline-intrinsics-mode",
        type=str,
        choices=["none", "tensor", "pinhole"],
        default="tensor",
        help="Baseline mode used when --compare-intrinsics-impact is enabled.",
    )
    parser.add_argument("--use-yolo", action="store_true")
    parser.add_argument("--yolo-weights", type=str, default=r"C:\RVC\Weights\Yolo_weights\yolo26s.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument(
        "--max-object-depth-m",
        type=float,
        default=5.0,
        help="Keep YOLO box detections only when nearest depth is <= this threshold. Use negative value to disable.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = load_unidepthv2_model(args.checkpoint, args.config, device)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    sample_tokens = load_sample_tokens(nusc, args.camera, args.lidar, args.max_samples)
    if len(sample_tokens) == 0:
        raise RuntimeError(f"No samples found containing {args.camera} and {args.lidar}.")

    yolo_model = None
    if args.use_yolo:
        if not os.path.isfile(args.yolo_weights):
            raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
        yolo_model = YOLO(args.yolo_weights)

    max_object_depth_m: Optional[float] = args.max_object_depth_m if args.max_object_depth_m >= 0 else None

    cpu_fallback_enabled = False

    frame_metrics: List[FrameMetrics] = []
    baseline_frame_metrics: List[FrameMetrics] = []
    impact_records: List[Dict[str, float]] = []
    box_records: List[Dict[str, float]] = []

    all_pred: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []
    baseline_all_pred: List[np.ndarray] = []
    baseline_all_gt: List[np.ndarray] = []

    frame_outdir = os.path.join(args.outdir, "frames")
    os.makedirs(frame_outdir, exist_ok=True)

    for idx, sample_token in enumerate(sample_tokens):
        sample = nusc.get("sample", sample_token)
        cam_token = sample["data"][args.camera]
        lidar_token = sample["data"][args.lidar]

        sd_cam = nusc.get("sample_data", cam_token)
        image_path = resolve_image_path(nusc.dataroot, sd_cam["filename"])
        if image_path is None:
            continue
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        _, depths, u, v, intrinsic = lidar_to_camera_points(nusc, lidar_token, cam_token)

        h, w = image_bgr.shape[:2]
        sparse_depth = build_sparse_depth_map(h, w, depths, u, v)
        pred_metric, model, device, cpu_fallback_enabled = infer_with_fallback(
            model,
            image_bgr,
            intrinsic,
            device,
            cpu_fallback_enabled,
            intrinsics_mode=args.intrinsics_mode,
        )

        metrics, valid_mask = compute_metrics(pred_metric, sparse_depth, max_depth_m=float(args.max_depth_m))
        frame_metrics.append(
            FrameMetrics(
                sample_token=sample_token,
                camera_token=cam_token,
                lidar_token=lidar_token,
                valid_pixels=metrics["valid_pixels"],
                mae_m=metrics["mae_m"],
                rmse_m=metrics["rmse_m"],
                abs_rel=metrics["abs_rel"],
            )
        )

        if metrics["valid_pixels"] > 0:
            all_pred.append(pred_metric[valid_mask])
            all_gt.append(sparse_depth[valid_mask])

        baseline_metrics: Optional[Dict[str, float]] = None
        if args.compare_intrinsics_impact:
            baseline_pred_metric, model, device, cpu_fallback_enabled = infer_with_fallback(
                model,
                image_bgr,
                intrinsic,
                device,
                cpu_fallback_enabled,
                intrinsics_mode=args.baseline_intrinsics_mode,
            )
            baseline_metrics, baseline_valid_mask = compute_metrics(
                baseline_pred_metric,
                sparse_depth,
                max_depth_m=float(args.max_depth_m),
            )
            baseline_frame_metrics.append(
                FrameMetrics(
                    sample_token=sample_token,
                    camera_token=cam_token,
                    lidar_token=lidar_token,
                    valid_pixels=baseline_metrics["valid_pixels"],
                    mae_m=baseline_metrics["mae_m"],
                    rmse_m=baseline_metrics["rmse_m"],
                    abs_rel=baseline_metrics["abs_rel"],
                )
            )
            if baseline_metrics["valid_pixels"] > 0:
                baseline_all_pred.append(baseline_pred_metric[baseline_valid_mask])
                baseline_all_gt.append(sparse_depth[baseline_valid_mask])

            impact_records.append(
                {
                    "frame_idx": int(idx),
                    "sample_token": sample_token,
                    "primary_mode": args.intrinsics_mode,
                    "baseline_mode": args.baseline_intrinsics_mode,
                    "primary_valid_pixels": int(metrics["valid_pixels"]),
                    "baseline_valid_pixels": int(baseline_metrics["valid_pixels"]),
                    "primary_mae_m": float(metrics["mae_m"]),
                    "baseline_mae_m": float(baseline_metrics["mae_m"]),
                    "delta_mae_m": float(metrics["mae_m"] - baseline_metrics["mae_m"]),
                    "primary_rmse_m": float(metrics["rmse_m"]),
                    "baseline_rmse_m": float(baseline_metrics["rmse_m"]),
                    "delta_rmse_m": float(metrics["rmse_m"] - baseline_metrics["rmse_m"]),
                    "primary_abs_rel": float(metrics["abs_rel"]),
                    "baseline_abs_rel": float(baseline_metrics["abs_rel"]),
                    "delta_abs_rel": float(metrics["abs_rel"] - baseline_metrics["abs_rel"]),
                }
            )

        lidar_overlay_vis = overlay_lidar_points(image_bgr, sparse_depth)
        frame_box_records: List[Dict[str, float]] = []
        if yolo_model is not None:
            frame_box_records = evaluate_boxes(
                pred_metric,
                sparse_depth,
                image_bgr,
                lidar_overlay_vis,
                yolo_model,
                args.yolo_conf,
                max_object_depth_m,
                idx,
                sample_token,
            )
            box_records.extend(frame_box_records)
            if len(frame_box_records) == 0:
                continue

        save_composite_output(
            frame_outdir,
            idx,
            lidar_overlay_vis,
            pred_metric,
            metrics,
            mode_label=args.intrinsics_mode,
        )

    os.makedirs(args.outdir, exist_ok=True)
    per_frame_df = pd.DataFrame([fm.__dict__ for fm in frame_metrics])
    per_frame_df.to_csv(os.path.join(args.outdir, "per_frame_metrics.csv"), index=False)

    if len(all_pred) > 0:
        pred_all = np.concatenate(all_pred)
        gt_all = np.concatenate(all_gt)
        valid = np.isfinite(pred_all) & np.isfinite(gt_all) & (gt_all > 0)
        pred_all = pred_all[valid]
        gt_all = gt_all[valid]

        abs_err = np.abs(pred_all - gt_all)
        summary = {
            "num_frames": int(len(frame_metrics)),
            "valid_pixels": int(pred_all.size),
            "mae_m": float(np.mean(abs_err)),
            "rmse_m": float(np.sqrt(np.mean((pred_all - gt_all) ** 2))),
            "abs_rel": float(np.mean(abs_err / np.clip(gt_all, 1e-6, None))),
        }
    else:
        summary = {
            "num_frames": int(len(frame_metrics)),
            "valid_pixels": 0,
            "mae_m": np.nan,
            "rmse_m": np.nan,
            "abs_rel": np.nan,
        }

    with open(os.path.join(args.outdir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    summary_table = tabulate(
        [[summary["num_frames"], summary["valid_pixels"], summary["mae_m"], summary["rmse_m"], summary["abs_rel"]]],
        headers=["num_frames", "valid_pixels", "mae_m", "rmse_m", "abs_rel"],
        tablefmt="github",
        floatfmt=".4f",
    )
    with open(os.path.join(args.outdir, "summary_table.txt"), "w", encoding="utf-8") as f:
        f.write(summary_table + "\n")

    print("\n=== Aggregate Metrics ===")
    print(summary_table)

    if args.compare_intrinsics_impact:
        baseline_per_frame_df = pd.DataFrame([fm.__dict__ for fm in baseline_frame_metrics])
        baseline_per_frame_df.to_csv(
            os.path.join(args.outdir, f"per_frame_metrics_{args.baseline_intrinsics_mode}.csv"),
            index=False,
        )

        if len(baseline_all_pred) > 0:
            base_pred_all = np.concatenate(baseline_all_pred)
            base_gt_all = np.concatenate(baseline_all_gt)
            base_valid = np.isfinite(base_pred_all) & np.isfinite(base_gt_all) & (base_gt_all > 0)
            base_pred_all = base_pred_all[base_valid]
            base_gt_all = base_gt_all[base_valid]

            base_abs_err = np.abs(base_pred_all - base_gt_all)
            baseline_summary = {
                "num_frames": int(len(baseline_frame_metrics)),
                "valid_pixels": int(base_pred_all.size),
                "mae_m": float(np.mean(base_abs_err)),
                "rmse_m": float(np.sqrt(np.mean((base_pred_all - base_gt_all) ** 2))),
                "abs_rel": float(np.mean(base_abs_err / np.clip(base_gt_all, 1e-6, None))),
            }
        else:
            baseline_summary = {
                "num_frames": int(len(baseline_frame_metrics)),
                "valid_pixels": 0,
                "mae_m": np.nan,
                "rmse_m": np.nan,
                "abs_rel": np.nan,
            }

        impact_summary = {
            "primary_mode": args.intrinsics_mode,
            "baseline_mode": args.baseline_intrinsics_mode,
            "primary": summary,
            "baseline": baseline_summary,
            "delta_primary_minus_baseline": {
                "mae_m": float(summary["mae_m"] - baseline_summary["mae_m"]),
                "rmse_m": float(summary["rmse_m"] - baseline_summary["rmse_m"]),
                "abs_rel": float(summary["abs_rel"] - baseline_summary["abs_rel"]),
            },
        }

        with open(os.path.join(args.outdir, "intrinsics_impact_summary.json"), "w", encoding="utf-8") as f:
            json.dump(impact_summary, f, indent=2)

        pd.DataFrame(impact_records).to_csv(
            os.path.join(args.outdir, "intrinsics_impact_per_frame.csv"),
            index=False,
        )

        impact_table = tabulate(
            [
                [
                    args.intrinsics_mode,
                    summary["valid_pixels"],
                    summary["mae_m"],
                    summary["rmse_m"],
                    summary["abs_rel"],
                ],
                [
                    args.baseline_intrinsics_mode,
                    baseline_summary["valid_pixels"],
                    baseline_summary["mae_m"],
                    baseline_summary["rmse_m"],
                    baseline_summary["abs_rel"],
                ],
            ],
            headers=["mode", "valid_pixels", "mae_m", "rmse_m", "abs_rel"],
            tablefmt="github",
            floatfmt=".4f",
        )
        with open(os.path.join(args.outdir, "intrinsics_impact_table.txt"), "w", encoding="utf-8") as f:
            f.write(impact_table + "\n")

        print("\n=== Intrinsics Impact Comparison ===")
        print(impact_table)

    if len(box_records) > 0:
        box_df = pd.DataFrame(box_records)
        box_df.to_csv(os.path.join(args.outdir, "obstacle_box_metrics.csv"), index=False)

        abs_err = box_df["abs_err_m"].to_numpy()
        gt_min = box_df["gt_min_m"].to_numpy()
        box_summary = {
            "num_boxes": int(len(box_df)),
            "mae_min_depth_m": float(np.mean(abs_err)),
            "rmse_min_depth_m": float(np.sqrt(np.mean(abs_err**2))),
            "abs_rel_min_depth": float(np.mean(abs_err / np.clip(gt_min, 1e-6, None))),
        }
        with open(os.path.join(args.outdir, "obstacle_box_summary.json"), "w", encoding="utf-8") as f:
            json.dump(box_summary, f, indent=2)


if __name__ == "__main__":
    main()
