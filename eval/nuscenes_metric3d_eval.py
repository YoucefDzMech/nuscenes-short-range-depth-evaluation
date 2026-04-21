"""
nuScenes — Metric3D v2 + Camera Intrinsics Demo
===============================================
Metric3D v2 is a foundation model for zero-shot metric depth and surface normal
estimation. It accepts the full K matrix (3×3 camera intrinsic) to produce
metric depth directly.

Architecture: ViT-Giant2 encoder + RAFTDepthNormalDPTDecoder5
Metric output: natively metric (fine-tuned on KITTI, NYU, Matterport3D, etc.)

OD role in this script
----------------------
YOLO runs AFTER the depth map is generated. It has NO influence on the
depth values — used purely to visualise and evaluate per-object predictions
(bounding boxes + `p:X.Xm gt:Y.Ym` labels).

Evaluation scope: only pixels/objects whose LiDAR GT depth ≤ max_depth_m
(parking-scenario focus, default 5 m).

Outputs (written to --outdir):
  frames/
    XXXX_composite.png       — [LiDAR+boxes overlay | predicted depth colormap]
  per_frame_metrics.csv
  obstacle_box_metrics.csv  — per-bounding-box prediction vs GT
  obstacle_box_summary.json
  summary_metrics.json
  summary_metrics.csv
  summary_table.txt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import runpy
import sys
import torch.nn as nn
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from tabulate import tabulate
from ultralytics import YOLO

# Add Metric3D to path
sys.path.insert(0, r"C:\RVC\Metric3D")

from mono.model.monodepth_model import DepthModel
import mono.model.backbones.ViT_DINO_reg as vit_dino_reg

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_DEPTH_M: float = 10.0   # overridden by --max-depth-m
METRIC3D_INPUT_SIZE: int = 960  # Metric3D typically uses 960x960 or similar


# ─── Data classes ────────────────────────────────────────────────────────────
@dataclass
class FrameMetrics:
    sample_token: str
    camera_token: str
    lidar_token: str
    valid_pixels: int
    mae_m: float
    rmse_m: float
    abs_rel: float


# ─── Utilities ───────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def resolve_dataset_path(dataroot: str, filename: str, channel: Optional[str] = None) -> Optional[str]:
    candidates = [
        os.path.join(dataroot, filename),
        os.path.join(dataroot, filename.replace("/samples/", "/sweeps/")),
        os.path.join(dataroot, filename.replace("\\samples\\", "\\sweeps\\")),
    ]

    base = os.path.basename(filename)
    if channel:
        for folder in ("samples", "sweeps"):
            candidates.append(os.path.join(dataroot, folder, channel, base))

    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _load_python_config(config_path: str) -> Dict:
    namespace = runpy.run_path(config_path)
    base_files = namespace.get("_base_", [])
    if isinstance(base_files, str):
        base_files = [base_files]

    merged: Dict = {}
    for base in base_files:
        base_path = os.path.normpath(os.path.join(os.path.dirname(config_path), base))
        merged = _deep_merge_dicts(merged, _load_python_config(base_path))

    local_cfg = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__") and key != "_base_"
    }
    return _deep_merge_dicts(merged, local_cfg)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_namespace(v) for v in obj)
    return obj


def load_metric3d_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple:
    """
    Load Metric3D model from config and checkpoint.
    
    Config: typically C:/RVC/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py
    or similar ViT-Giant2 based config.
    
    Checkpoint: Metric3D v2 weights, typically downloaded from HuggingFace.
    """
    print(f"[Metric3D] Loading config: {config_path}")
    print(f"[Metric3D] Loading checkpoint: {checkpoint_path}")
    print(f"[Metric3D] Device: {device}")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    cfg_dict = _load_python_config(config_path)
    cfg = _to_namespace(cfg_dict)

    # Speed optimization: skip expensive random init for giant ViT backbone.
    # We load full checkpoint weights immediately after construction.
    if hasattr(vit_dino_reg, "DinoVisionTransformer") and hasattr(vit_dino_reg.DinoVisionTransformer, "init_weights"):
        vit_dino_reg.DinoVisionTransformer.init_weights = lambda self: None

    model = DepthModel(cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Metric3D] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Metric3D] unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    
    print(f"[Metric3D] Model ready on {device}")
    return model, cfg


def preprocess_metric3d(
    image_bgr: np.ndarray,
    crop_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, float, Tuple[int, int, int, int]]:
    """Resize, pad, and normalize image for Metric3D inference."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    crop_h, crop_w = crop_size

    scale = min(crop_h / h, crop_w / w)
    resize_h = max(1, int(round(h * scale)))
    resize_w = max(1, int(round(w * scale)))

    resized = cv2.resize(image_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    pad_h = max(crop_h - resize_h, 0)
    pad_w = max(crop_w - resize_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = [123.675, 116.28, 103.53]
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=padding,
    )

    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    tensor = torch.div((tensor - mean), std).unsqueeze(0).to(device)

    return tensor, scale, (pad_top, pad_bottom, pad_left, pad_right)


# ─── LiDAR helpers ───────────────────────────────────────────────────────────

def lidar_to_camera_points(
    nusc: NuScenes,
    lidar_token: str,
    camera_token: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points into camera frame; return 3-D coords, depths, u, v, K."""
    sd_lidar = nusc.get("sample_data", lidar_token)
    sd_cam   = nusc.get("sample_data", camera_token)

    cs_lidar   = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose",           sd_lidar["ego_pose_token"])
    cs_cam     = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    pose_cam   = nusc.get("ego_pose",           sd_cam["ego_pose_token"])

    lidar_path = resolve_dataset_path(nusc.dataroot, sd_lidar["filename"], channel="LIDAR_TOP")
    if lidar_path is None:
        raise FileNotFoundError(f"Could not resolve LiDAR file: {sd_lidar['filename']}")
    pc = LidarPointCloud.from_file(lidar_path)

    # LiDAR → ego → world → ego(cam) → cam
    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))
    pc.rotate(Quaternion(pose_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(pose_lidar["translation"]))
    pc.translate(-np.array(pose_cam["translation"]))
    pc.rotate(Quaternion(pose_cam["rotation"]).rotation_matrix.T)
    pc.translate(-np.array(cs_cam["translation"]))
    pc.rotate(Quaternion(cs_cam["rotation"]).rotation_matrix.T)

    points_cam = pc.points[:3, :]
    depths     = points_cam[2, :]

    intrinsic  = np.array(cs_cam["camera_intrinsic"], dtype=np.float32)
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
    max_depth_m: float = MAX_DEPTH_M,
) -> np.ndarray:
    """Build sparse GT depth map, filtered to [min, max_depth_m]."""
    sparse = np.zeros((image_h, image_w), dtype=np.float32)
    valid = (
        (depths > min_depth_m)
        & (depths <= max_depth_m)
        & (u >= 0) & (u < image_w)
        & (v >= 0) & (v < image_h)
    )
    uu = np.clip(np.floor(u[valid]).astype(np.int32), 0, image_w - 1)
    vv = np.clip(np.floor(v[valid]).astype(np.int32), 0, image_h - 1)
    dd = depths[valid].astype(np.float32)
    for x, y, d in zip(uu, vv, dd):
        existing = sparse[y, x]
        if existing == 0.0 or d < existing:
            sparse[y, x] = d
    return sparse


# ─── Inference ───────────────────────────────────────────────────────────────

def infer_metric_depth(
    model,
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
    crop_size: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """
    Run Metric3D v2 inference with the official Metric3D preprocessing.

    Metric3D v2 is run in its canonical camera space and then de-canonicalized
    using the focal-length scale factor described in the official README.

    Args:
        model: Metric3D v2 model (loaded via get_configured_monodepth_model)
        image_bgr: BGR image (H, W, 3), uint8
        intrinsic: 3×3 camera intrinsic matrix from nuScenes
        device: torch device

    Returns:
        depth map at ORIGINAL image resolution (H, W), float32, metres.
    """
    h_orig, w_orig = image_bgr.shape[:2]

    input_tensor, resize_scale, pad = preprocess_metric3d(image_bgr, crop_size, device)

    data = {"input": input_tensor}
    module = model.module if hasattr(model, "module") else model
    with torch.inference_mode():
        pred_depth, confidence, output_dict = module.inference(data)

    depth = pred_depth
    if isinstance(depth, (list, tuple)):
        depth = depth[0]
    if depth.dim() == 4:
        depth = depth[0, 0]
    elif depth.dim() == 3:
        depth = depth[0]

    depth = depth.detach().float().cpu().numpy()
    depth = depth[pad[0] : depth.shape[0] - pad[1], pad[2] : depth.shape[1] - pad[3]]
    depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    fx = float(intrinsic[0, 0])
    depth = depth * (fx * resize_scale / 1000.0)
    depth = np.clip(depth, 0.0, 300.0)

    return depth.astype(np.float32)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_metric: np.ndarray,
    gt_sparse_m: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    valid = (gt_sparse_m > 0.0) & np.isfinite(pred_metric) & (pred_metric > 0.0)
    pred  = pred_metric[valid]
    gt    = gt_sparse_m[valid]

    if pred.size == 0:
        return {"valid_pixels": 0, "mae_m": np.nan, "rmse_m": np.nan, "abs_rel": np.nan}, valid

    abs_err = np.abs(pred - gt)
    mae     = float(np.mean(abs_err))
    rmse    = float(np.sqrt(np.mean((pred - gt) ** 2)))
    abs_rel = float(np.mean(abs_err / np.clip(gt, 1e-6, None)))
    return {"valid_pixels": int(pred.size), "mae_m": mae, "rmse_m": rmse, "abs_rel": abs_rel}, valid


# ─── Visualisation ───────────────────────────────────────────────────────────

def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    d     = depth_map.copy()
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    dmin = np.percentile(d[valid], 2)
    dmax = np.percentile(d[valid], 98)
    if dmax <= dmin:
        dmax = dmin + 1e-6
    d_norm  = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0)
    cmap_id = getattr(cv2, "COLORMAP_SPECTRAL", cv2.COLORMAP_TURBO)
    colored = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cmap_id)
    return colored


def overlay_lidar_points(image_bgr: np.ndarray, sparse_depth_m: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    ys, xs = np.where(sparse_depth_m > 0)
    if ys.size == 0:
        return out
    depths = sparse_depth_m[ys, xs]
    dmin   = np.percentile(depths, 5)
    dmax   = np.percentile(depths, 95)
    if dmax <= dmin:
        dmax = dmin + 1e-6
    for x, y, d in zip(xs, ys, depths):
        t     = float(np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0))
        color = (int(255 * (1.0 - t)), int(255 * t), 255)
        cv2.circle(out, (int(x), int(y)), 1, color, -1)
    return out


def evaluate_and_draw_boxes(
    pred_metric: np.ndarray,
    gt_sparse_m: np.ndarray,
    image_bgr: np.ndarray,
    draw_bgr: np.ndarray,
    yolo_model,
    conf_threshold: float,
    max_depth_m: float,
    frame_idx: int,
    sample_token: str,
    yolo_result=None,
) -> List[Dict]:
    """
    Run YOLO on image_bgr, draw boxes on draw_bgr, return per-box depth metrics.
    pred_metric is read-only — OD does NOT modify depth values.
    Labels: <class>  /  p:X.Xm gt:Y.Ym
    Only boxes where gt_min ≤ max_depth_m are kept.
    """
    records: List[Dict] = []
    h, w    = pred_metric.shape
    results = [yolo_result] if yolo_result is not None else yolo_model(image_bgr, verbose=False)
    if not results or results[0].boxes is None:
        return records

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs      = results[0].boxes.conf.cpu().numpy()
    clss       = results[0].boxes.cls.cpu().numpy()
    names      = results[0].names if hasattr(results[0], "names") else {}

    for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
        if float(conf) < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        pred_roi   = pred_metric[y1:y2, x1:x2]
        gt_roi     = gt_sparse_m[y1:y2, x1:x2]
        pred_valid = pred_roi[np.isfinite(pred_roi) & (pred_roi > 0)]
        gt_valid   = gt_roi[(gt_roi > 0) & (gt_roi <= max_depth_m)]

        if pred_valid.size == 0 or gt_valid.size == 0:
            continue

        pred_min = float(np.min(pred_valid))
        gt_min   = float(np.min(gt_valid))

        if gt_min > max_depth_m:
            continue

        abs_err    = abs(pred_min - gt_min)
        abs_rel    = abs_err / max(gt_min, 1e-6)
        class_name = names.get(int(cls_id), str(int(cls_id))) \
                     if isinstance(names, dict) else str(int(cls_id))

        cv2.rectangle(draw_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        y_top = max(18, y1 - 18)
        y_bot = max(36, y1 - 2)
        cv2.putText(draw_bgr, class_name,
                    (x1, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(draw_bgr, f"p:{pred_min:.1f}m gt:{gt_min:.1f}m",
                    (x1, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (50, 255, 255), 1, cv2.LINE_AA)

        records.append({
            "frame_idx":  int(frame_idx),
            "sample_token": sample_token,
            "class_name": class_name,
            "conf":       float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pred_min_m": pred_min,
            "gt_min_m":   gt_min,
            "abs_err_m":  abs_err,
            "abs_rel":    abs_rel,
        })

    return records


def has_yolo_detections(image_bgr: np.ndarray, yolo_model, conf_threshold: float) -> Tuple[bool, Optional[object]]:
    """Fast prefilter: return True if YOLO has any box above confidence threshold."""
    results = yolo_model(image_bgr, verbose=False)
    if not results or results[0].boxes is None:
        return False, None

    confs = results[0].boxes.conf.cpu().numpy()
    if confs.size == 0:
        return False, None

    has_any = bool(np.any(confs >= float(conf_threshold)))
    return has_any, results[0]


def save_composite_output(
    outdir: str,
    idx: int,
    draw_bgr: np.ndarray,
    pred_metric: np.ndarray,
    metrics: Dict[str, float],
    max_depth_m: float,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    depth_vis = colorize_depth(pred_metric)

    h1, w1 = draw_bgr.shape[:2]
    h2, w2 = depth_vis.shape[:2]
    if h1 != h2:
        depth_vis = cv2.resize(depth_vis, (w2 * h1 // h2, h1))

    composite = cv2.hconcat([draw_bgr, depth_vis])

    lines = [
        "Metric3D v2",
        f"Scale: metric (ViT-Giant2)",
        f"Max GT: {max_depth_m:.0f} m",
        f"Valid px: {int(metrics['valid_pixels'])}",
        f"MAE    : {metrics['mae_m']:.4f} m",
        f"RMSE   : {metrics['rmse_m']:.4f} m",
        f"AbsRel : {metrics['abs_rel']:.4f}",
    ]
    font, fs, th, lh, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 22, 10
    sizes = [cv2.getTextSize(l, font, fs, th)[0] for l in lines]
    bw    = max(w for w, _ in sizes) + pad * 2
    bh    = lh * len(lines) + pad
    x2_b  = composite.shape[1] - 15
    y1_b  = 15
    x1_b  = max(0, x2_b - bw)
    y2_b  = min(composite.shape[0], y1_b + bh)
    ov    = composite.copy()
    cv2.rectangle(ov, (x1_b, y1_b), (x2_b, y2_b), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.75, composite, 0.25, 0, composite)
    cv2.rectangle(composite, (x1_b, y1_b), (x2_b, y2_b), (255, 255, 255), 1)
    for i, line in enumerate(lines):
        yt = y1_b + pad + (i + 1) * lh - 6
        cv2.putText(composite, line, (x1_b + pad, yt),
                    font, fs, (255, 255, 255), th, cv2.LINE_AA)

    cv2.imwrite(os.path.join(outdir, f"{idx:04d}_composite.png"), composite)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    global MAX_DEPTH_M
    parser = argparse.ArgumentParser(
        description="nuScenes — Metric3D v2 + Camera Intrinsics (metric depth demo)"
    )
    parser.add_argument("--dataroot", default=r"C:\RVC\NUsCENES_DATASET\v1.0-mini",
                        help="Path to nuScenes dataset root.")
    parser.add_argument("--version",  default="v1.0-mini")
    parser.add_argument("--camera",   default="CAM_BACK")
    parser.add_argument("--lidar",    default="LIDAR_TOP")
    parser.add_argument(
        "--config",
        default=r"C:\RVC\Metric3D\mono\configs\HourglassDecoder\vit.raft5.giant2.py",
        help="Path to Metric3D config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Metric3D checkpoint (.pth). If None, will try to auto-download.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. If omitted, a timestamped folder is created.",
    )
    parser.add_argument(
        "--base-outdir",
        default=r"C:\RVC\OtherThanModel\Models'Output\Metric3D",
        help="Base output directory (used when --outdir is not specified).",
    )
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum frames to process (0 = all).")
    parser.add_argument("--max-depth-m", type=float, default=10.0,
                        help="Depth ceiling for GT filtering and evaluation (metres).")
    parser.add_argument("--yolo-weights",
                        default=r"C:\RVC\Weights\Yolo_weights\yolo26s.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    args = parser.parse_args()

    MAX_DEPTH_M = args.max_depth_m

    if args.outdir is None:
        run_name = f"Metric3D_intrinsics_{args.camera}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.outdir = os.path.join(args.base_outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[outdir] {args.outdir}")

    device = get_device()
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[GPU   ] {torch.cuda.get_device_name(device)}")
        print(f"[VRAM  ] {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_metric3d_model(args.config, args.checkpoint, device)

    yolo_model = YOLO(args.yolo_weights)
    print(f"[YOLO ] model loaded: {args.yolo_weights}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    sample_tokens = load_sample_tokens(nusc, args.camera, args.lidar, args.max_samples)
    if not sample_tokens:
        raise RuntimeError(f"No samples found with {args.camera} and {args.lidar}.")
    print(f"[dataset] {len(sample_tokens)} samples to process.")

    frame_metrics: List[FrameMetrics] = []
    box_records:   List[Dict]         = []
    all_pred:      List[np.ndarray]   = []
    all_gt:        List[np.ndarray]   = []

    frame_outdir = os.path.join(args.outdir, "frames")
    os.makedirs(frame_outdir, exist_ok=True)

    for idx, sample_token in enumerate(sample_tokens):
        sample     = nusc.get("sample", sample_token)
        cam_token  = sample["data"][args.camera]
        lidar_token = sample["data"][args.lidar]

        sd_cam     = nusc.get("sample_data", cam_token)
        image_path = resolve_dataset_path(nusc.dataroot, sd_cam["filename"], channel=args.camera)
        if image_path is None:
            print(f"  [warn] could not resolve image path: {sd_cam['filename']}")
            continue
        image_bgr  = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  [warn] could not read image: {image_path}")
            continue

        # Early skip before expensive LiDAR projection + Metric3D inference.
        has_det, yolo_result = has_yolo_detections(image_bgr, yolo_model, args.yolo_conf)
        if not has_det:
            print(f"  [{idx + 1:03d}/{len(sample_tokens)}] skipped (no YOLO detections)")
            continue

        _, depths, u, v, intrinsic = lidar_to_camera_points(nusc, lidar_token, cam_token)
        h, w = image_bgr.shape[:2]

        sparse_depth = build_sparse_depth_map(h, w, depths, u, v,
                                              max_depth_m=args.max_depth_m)

        try:
            pred_metric = infer_metric_depth(model, image_bgr, intrinsic, cfg.data_basic.crop_size, device)
        except Exception as exc:
            print(f"  [warn] inference failed on frame {idx}: {exc}")
            continue

        # ── OD: draw boxes AFTER depth map — does NOT change depth values ────
        draw_bgr = overlay_lidar_points(image_bgr, sparse_depth)
        frame_box_records = evaluate_and_draw_boxes(
            pred_metric, sparse_depth, image_bgr, draw_bgr,
            yolo_model, args.yolo_conf, args.max_depth_m,
            idx, sample_token, yolo_result=yolo_result,
        )

        if not frame_box_records:
            print(f"  [{idx + 1:03d}/{len(sample_tokens)}] skipped "
                  f"(no detections \u2264 {args.max_depth_m} m)")
            continue

        box_records.extend(frame_box_records)

        metrics, valid_mask = compute_metrics(pred_metric, sparse_depth)
        if metrics["valid_pixels"] == 0:
            print(f"  [{idx + 1:03d}/{len(sample_tokens)}] skipped "
                  f"(no GT pixels \u2264 {args.max_depth_m} m)")
            continue

        frame_metrics.append(
            FrameMetrics(
                sample_token=sample_token,
                camera_token=cam_token,
                lidar_token=lidar_token,
                valid_pixels=int(metrics["valid_pixels"]),
                mae_m=metrics["mae_m"],
                rmse_m=metrics["rmse_m"],
                abs_rel=metrics["abs_rel"],
            )
        )
        all_pred.append(pred_metric[valid_mask])
        all_gt.append(sparse_depth[valid_mask])

        save_composite_output(frame_outdir, idx, draw_bgr, pred_metric,
                      metrics, args.max_depth_m)

        print(
            f"  [{idx + 1:03d}/{len(sample_tokens)}]  "
            f"valid_px={metrics['valid_pixels']}  "
            f"MAE={metrics['mae_m']:.3f} m  "
            f"RMSE={metrics['rmse_m']:.3f} m  "
            f"AbsRel={metrics['abs_rel']:.4f}  "
            f"[boxes={len(frame_box_records)}]"
        )

    # ── Per-frame CSV ─────────────────────────────────────────────────────────
    per_frame_df = pd.DataFrame([fm.__dict__ for fm in frame_metrics])
    per_frame_df.to_csv(os.path.join(args.outdir, "per_frame_metrics.csv"), index=False)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    if all_pred:
        pred_all = np.concatenate(all_pred)
        gt_all   = np.concatenate(all_gt)
        mask     = np.isfinite(pred_all) & np.isfinite(gt_all) & (gt_all > 0)
        pred_all, gt_all = pred_all[mask], gt_all[mask]
        abs_err  = np.abs(pred_all - gt_all)
        summary  = {
            "model": "Metric3D-v2",
            "metric_formula": "native metric (ViT-Giant2 encoder + RAFTDepthNormalDPTDecoder5)",
            "K_source": "nuScenes intrinsic matrix (full 3x3)",
            "max_depth_m": args.max_depth_m,
            "num_frames": len(frame_metrics),
            "valid_pixels": int(pred_all.size),
            "mae_m":    float(np.mean(abs_err)),
            "rmse_m":   float(np.sqrt(np.mean((pred_all - gt_all) ** 2))),
            "abs_rel":  float(np.mean(abs_err / np.clip(gt_all, 1e-6, None))),
        }
    else:
        summary = {
            "model": "Metric3D-v2",
            "metric_formula": "native metric (ViT-Giant2 encoder + RAFTDepthNormalDPTDecoder5)",
            "K_source": "nuScenes intrinsic matrix (full 3x3)",
            "max_depth_m": args.max_depth_m,
            "num_frames": len(frame_metrics),
            "valid_pixels": 0,
            "mae_m": float("nan"), "rmse_m": float("nan"), "abs_rel": float("nan"),
        }

    with open(os.path.join(args.outdir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    table = tabulate(
        [[summary["num_frames"], summary["valid_pixels"],
          summary["mae_m"], summary["rmse_m"], summary["abs_rel"]]],
        headers=["num_frames", "valid_pixels", "mae_m", "rmse_m", "abs_rel"],
        tablefmt="github", floatfmt=".4f",
    )
    with open(os.path.join(args.outdir, "summary_table.txt"), "w") as f:
        f.write(table + "\n")

    print("\n=== Aggregate Metrics ===")
    print(table)

    # ── Bounding-box metrics ──────────────────────────────────────────────────
    if box_records:
        box_df   = pd.DataFrame(box_records)
        box_df.to_csv(os.path.join(args.outdir, "obstacle_box_metrics.csv"), index=False)
        abs_errs = box_df["abs_err_m"].to_numpy()
        gts      = box_df["gt_min_m"].to_numpy()
        box_summary = {
            "num_boxes":         int(len(box_df)),
            "mae_min_depth_m":   float(np.mean(abs_errs)),
            "rmse_min_depth_m":  float(np.sqrt(np.mean(abs_errs ** 2))),
            "abs_rel_min_depth": float(np.mean(abs_errs / np.clip(gts, 1e-6, None))),
        }
        with open(os.path.join(args.outdir, "obstacle_box_summary.json"), "w") as f:
            json.dump(box_summary, f, indent=2)
        print("\n=== Obstacle Box Metrics ===")
        print(tabulate(
            [[box_summary["num_boxes"], box_summary["mae_min_depth_m"],
              box_summary["rmse_min_depth_m"], box_summary["abs_rel_min_depth"]]],
            headers=["num_boxes", "mae_m", "rmse_m", "abs_rel"],
            tablefmt="github", floatfmt=".4f",
        ))
    else:
        print(f"[info] No bounding-box records within {args.max_depth_m} m.")

    print(f"\nOutputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
