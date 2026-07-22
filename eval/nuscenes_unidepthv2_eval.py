"""
UniDepthV2 — Consistent re-evaluation script
=============================================
Camera  : CAM_FRONT  (no bumper obstruction)
Frames  : ALL keyframes on disk (~3367)  — no YOLO gate on metrics
Metrics : pixel-level MAE/RMSE/AbsRel at LiDAR GT pixels 0.1–5m
YOLO    : runs AFTER inference, visualization only (boxes on composite PNG)
Output  : D:\\RVC_Model'sOutput\\UnidepthV2\\CAM_FRONT_all_frames_5m\\runN\\

Run:
  & "C:\\RVC\\UnidepthV2\\Uni\\Scripts\\Activate.ps1"
  python nuscenes_unidepthv2_camfront_eval.py
  python nuscenes_unidepthv2_camfront_eval.py --max-samples 100   # quick test
"""

import argparse, bisect, json, os, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2, matplotlib, numpy as np, pandas as pd, torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from safetensors.torch import load_file as load_safetensors_file
from tabulate import tabulate

sys.path.insert(0, os.environ.get("UNIDEPTH_REPO", "path/to/UniDepth"))  # official model repo (not included)
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CKPT   = "path/to/unidepth-v2-vitl14/model.safetensors"
DEFAULT_CFG    = "path/to/unidepth-v2-vitl14/config.json"
DEFAULT_YOLO   = os.environ.get("YOLO_WEIGHTS", "yolo26s.pt")
DEFAULT_OUTDIR = "outputs/unidepthv2_CAM_FRONT_5m"
DATAROOT       = os.environ.get("NUSCENES_DATAROOT", "path/to/nuscenes/trainval02")

@dataclass
class FrameMetrics:
    sample_token: str
    valid_pixels: int
    mae_m: float
    rmse_m: float
    abs_rel: float

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(ckpt, cfg_path, device):
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = UniDepthV2(cfg)
    sd = load_safetensors_file(ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model.resolution_level = 9
    variant = "ViT-S" if "vits" in ckpt.lower() else "ViT-L"
    print(f"[model] UniDepthV2 {variant} ready on {device}")
    model._variant_label = f"UniDepthV2 {variant}"
    return model

def infer(model, image_bgr, intrinsic, device):
    rgb = torch.from_numpy(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).permute(2,0,1).contiguous()
    K   = torch.from_numpy(intrinsic.astype(np.float32))
    with torch.inference_mode():
        pred = model.infer(rgb, Pinhole(K=K))
    d = pred.get("depth", pred.get("radius")).detach().float().cpu().numpy()
    while d.ndim > 2: d = d.squeeze(0)
    return d.astype(np.float32)

def infer_guarded(model, image_bgr, intrinsic, device, stats):
    """infer() with recovery from UniDepthV2's transient all-invalid CUDA output.

    UniDepthV2's CUDA path intermittently returns a depth map that is entirely
    non-finite or non-positive. Without a guard the frame scores valid_pixels=0
    and silently vanishes from frame_metrics.csv — this cost the 2026-06 run 840
    of 16,182 frames on ViT-S and 101 on ViT-L, and the lost frames were harder
    than average, so the reported MAE was optimistic.

    The same guard exists in nuscenes_unidepthv2_intrinsics_demo.py and in the
    Paper-1 eval script; it was dropped when this script was written.

    Recovery order differs from those scripts deliberately. They switch to CPU
    permanently on first failure, which is fine for a short demo but would add
    days to a 16k-frame benchmark. The failure is transient — re-running the lost
    frames reproduces none of them — so retry on the same device first and use
    CPU only for the offending frame.
    """
    def usable(d):
        return bool(np.any(np.isfinite(d) & (d > 0)))

    d = infer(model, image_bgr, intrinsic, device)
    if usable(d):
        return d

    stats["retry"] += 1
    d = infer(model, image_bgr, intrinsic, device)
    if usable(d):
        return d

    # Still invalid: CPU is deterministic here. Move back afterwards regardless.
    stats["cpu_fallback"] += 1
    try:
        model.to("cpu")
        d = infer(model, image_bgr, intrinsic, torch.device("cpu"))
    finally:
        model.to(device)
    if not usable(d):
        stats["unrecovered"] += 1
    return d

# ── nuScenes helpers ───────────────────────────────────────────────────────────
def load_all_frames(nusc, camera, lidar, max_samples):
    """Return (cam_sd_token, lid_sd_token) for ALL frames on disk (keyframes + sweeps)."""
    pairs = []
    for scene in nusc.scene:
        first = nusc.get("sample", scene["first_sample_token"])
        if camera not in first["data"] or lidar not in first["data"]: continue
        lid_chain = []
        lid_cur = nusc.get("sample_data", first["data"][lidar])
        while True:
            if os.path.isfile(os.path.join(nusc.dataroot, lid_cur["filename"])):
                lid_chain.append((lid_cur["timestamp"], lid_cur["token"]))
            if not lid_cur["next"]: break
            lid_cur = nusc.get("sample_data", lid_cur["next"])
        if not lid_chain: continue
        lid_chain.sort(); lid_ts = [x[0] for x in lid_chain]; lid_toks = [x[1] for x in lid_chain]
        cam_cur = nusc.get("sample_data", first["data"][camera])
        while True:
            if not cam_cur["is_key_frame"] and os.path.isfile(os.path.join(nusc.dataroot, cam_cur["filename"])):
                idx = bisect.bisect_left(lid_ts, cam_cur["timestamp"])
                best = min([i for i in (idx-1, idx, idx+1) if 0 <= i < len(lid_ts)],
                           key=lambda i: abs(lid_ts[i] - cam_cur["timestamp"]))
                pairs.append((cam_cur["token"], lid_toks[best]))
            if max_samples > 0 and len(pairs) >= max_samples: return pairs
            if not cam_cur["next"]: break
            cam_cur = nusc.get("sample_data", cam_cur["next"])
    return pairs

def lidar_to_camera(nusc, lidar_token, cam_token):
    sd_l = nusc.get("sample_data", lidar_token)
    sd_c = nusc.get("sample_data", cam_token)
    cs_l = nusc.get("calibrated_sensor", sd_l["calibrated_sensor_token"])
    ep_l = nusc.get("ego_pose", sd_l["ego_pose_token"])
    cs_c = nusc.get("calibrated_sensor", sd_c["calibrated_sensor_token"])
    ep_c = nusc.get("ego_pose", sd_c["ego_pose_token"])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_l["filename"]))
    pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_l["translation"]))
    pc.rotate(Quaternion(ep_l["rotation"]).rotation_matrix)
    pc.translate(np.array(ep_l["translation"]))
    pc.translate(-np.array(ep_c["translation"]))
    pc.rotate(Quaternion(ep_c["rotation"]).rotation_matrix.T)
    pc.translate(-np.array(cs_c["translation"]))
    pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)
    pts = pc.points[:3]
    K   = np.array(cs_c["camera_intrinsic"], dtype=np.float32)
    pi  = view_points(pts, K, normalize=True)
    return pts[2], pi[0], pi[1], K

def sparse_gt(h, w, depths, u, v, min_d=0.1):
    gt = np.zeros((h, w), dtype=np.float32)
    ok = (depths > min_d) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uu = np.clip(np.floor(u[ok]).astype(int), 0, w-1)
    vv = np.clip(np.floor(v[ok]).astype(int), 0, h-1)
    for x, y, d in zip(uu, vv, depths[ok]):
        if gt[y, x] == 0 or d < gt[y, x]:
            gt[y, x] = d
    return gt

def compute_metrics(pred, gt, max_d):
    ok = (gt > 0) & (gt <= max_d) & np.isfinite(pred) & (pred > 0)
    p, g = pred[ok], gt[ok]
    if p.size == 0:
        return {"valid_pixels": 0, "mae_m": np.nan, "rmse_m": np.nan, "abs_rel": np.nan}, ok
    ae = np.abs(p - g)
    return {"valid_pixels": int(p.size),
            "mae_m": float(np.mean(ae)),
            "rmse_m": float(np.sqrt(np.mean((p-g)**2))),
            "abs_rel": float(np.mean(ae / np.clip(g, 1e-6, None)))}, ok

# ── Visualisation ─────────────────────────────────────────────────────────────
def colorize(depth):
    v = np.isfinite(depth);
    if not v.any(): return np.zeros((*depth.shape,3), dtype=np.uint8)
    lo, hi = np.percentile(depth[v], 2), np.percentile(depth[v], 98)
    hi = max(hi, lo+1e-6)
    n = np.clip((depth-lo)/(hi-lo), 0, 1)
    c = matplotlib.colormaps.get_cmap("Spectral")
    return (c((n*255).astype(np.uint8))[:,:,:3]*255).astype(np.uint8)[:,:,::-1]

def overlay_lidar(img, gt):
    out = img.copy()
    ys, xs = np.where(gt > 0)
    if ys.size == 0: return out
    d = gt[ys, xs]
    lo, hi = np.percentile(d,5), np.percentile(d,95); hi=max(hi,lo+1e-6)
    for x, y, dd in zip(xs, ys, d):
        t = float(np.clip((dd-lo)/(hi-lo),0,1))
        cv2.circle(out,(int(x),int(y)),1,(int(255*(1-t)),int(255*t),255),-1)
    return out

def evaluate_and_draw_boxes(img, pred, gt, yolo_model, conf_thr,
                            max_obj_depth, frame_idx, cam_token):
    """Run YOLO: draw boxes on img AND return box-level depth metrics."""
    records = []
    res = yolo_model(img, verbose=False)
    if not res or res[0].boxes is None: return img, records
    out = img.copy(); h, w = pred.shape
    for box, conf, cls in zip(res[0].boxes.xyxy.cpu().numpy(),
                               res[0].boxes.conf.cpu().numpy(),
                               res[0].boxes.cls.cpu().numpy()):
        if float(conf) < conf_thr: continue
        x1,y1,x2,y2 = max(0,int(box[0])),max(0,int(box[1])),min(w,int(box[2])),min(h,int(box[3]))
        if x2<=x1 or y2<=y1: continue
        pv = pred[y1:y2,x1:x2]; gv = gt[y1:y2,x1:x2]
        pv = pv[np.isfinite(pv)&(pv>0)]; gv = gv[gv>0]
        if pv.size==0 or gv.size==0: continue
        pm, gm = float(np.min(pv)), float(np.min(gv))
        if max_obj_depth is not None and min(pm, gm) > max_obj_depth: continue
        ae = abs(pm - gm)
        name = res[0].names.get(int(cls), str(int(cls)))
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(out,f"{name} p:{pm:.1f}m gt:{gm:.1f}m",(x1,max(20,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1,cv2.LINE_AA)
        records.append({"frame_idx": frame_idx, "cam_token": cam_token,
                        "class_name": name, "conf": float(conf),
                        "pred_min_m": pm, "gt_min_m": gm,
                        "abs_err_m": ae, "abs_rel": ae / max(gm, 1e-6)})
    return out, records

def save_composite(outdir, idx, left, pred, metrics, model_label="UniDepthV2"):
    os.makedirs(outdir, exist_ok=True)
    depth_vis = colorize(pred)
    comp = cv2.hconcat([left, depth_vis])
    lines = [model_label,
             f"Valid px: {metrics['valid_pixels']}",
             f"MAE : {metrics['mae_m']:.4f} m",
             f"RMSE: {metrics['rmse_m']:.4f} m",
             f"AbsRel: {metrics['abs_rel']:.4f}"]
    font,fs,th,lh,pad = cv2.FONT_HERSHEY_SIMPLEX,0.55,1,22,10
    bw = max(cv2.getTextSize(l,font,fs,th)[0][0] for l in lines)+pad*2
    bh = lh*len(lines)+pad; x2=comp.shape[1]-20; y1=20
    x1=max(0,x2-bw); y2=min(comp.shape[0],y1+bh)
    ov=comp.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),(20,20,20),-1)
    cv2.addWeighted(ov,0.75,comp,0.25,0,comp)
    cv2.rectangle(comp,(x1,y1),(x2,y2),(255,255,255),1)
    for i,l in enumerate(lines):
        cv2.putText(comp,l,(x1+pad,y1+pad+(i+1)*lh-6),font,fs,(255,255,255),th,cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir,f"{idx:04d}_composite.png"),comp)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser("UniDepthV2 CAM_FRONT consistent eval")
    pa.add_argument("--dataroot",    default=DATAROOT)
    pa.add_argument("--version",     default="v1.0-trainval")
    pa.add_argument("--camera",      default="CAM_FRONT")
    pa.add_argument("--lidar",       default="LIDAR_TOP")
    pa.add_argument("--checkpoint",  default=DEFAULT_CKPT)
    pa.add_argument("--config",      default=DEFAULT_CFG)
    pa.add_argument("--max-depth-m", type=float, default=5.0)
    pa.add_argument("--max-samples", type=int,   default=0)
    pa.add_argument("--outdir",      default=DEFAULT_OUTDIR)
    pa.add_argument("--use-yolo",          action="store_true")
    pa.add_argument("--yolo-weights",      default=DEFAULT_YOLO)
    pa.add_argument("--yolo-conf",         type=float, default=0.25)
    pa.add_argument("--max-object-depth-m",type=float, default=5.0)
    args = pa.parse_args()

    device = get_device(); print(f"[device] {device}")
    model  = load_model(args.checkpoint, args.config, device)

    nusc        = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    frame_pairs = load_all_frames(nusc, args.camera, args.lidar, args.max_samples)
    print(f"[nusc ] {len(frame_pairs)} sweep frames (keyframes excluded)")

    yolo_model = None
    if args.use_yolo:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
        print(f"[yolo ] {args.yolo_weights} (visualization only)")

    # Auto-increment run folder
    base = os.path.normpath(args.outdir); n=1
    while os.path.exists(os.path.join(base, f"run{n}")): n+=1
    run_dir = os.path.join(base, f"run{n}"); os.makedirs(run_dir)
    frame_dir = os.path.join(run_dir, "frames"); os.makedirs(frame_dir)
    print(f"[out  ] {run_dir}")

    frame_metrics=[]; box_records=[]; all_pred=[]; all_gt=[]
    max_obj_depth = args.max_object_depth_m if args.max_object_depth_m >= 0 else None
    # Counts how often the invalid-CUDA-output guard fired (see infer_guarded).
    infer_stats = {"retry": 0, "cpu_fallback": 0, "unrecovered": 0}

    for idx, (ct, lt) in enumerate(frame_pairs):
        sd  = nusc.get("sample_data", ct)
        img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
        if img is None: continue

        depths, u, v, K = lidar_to_camera(nusc, lt, ct)
        h, w = img.shape[:2]
        gt = sparse_gt(h, w, depths, u, v)
        pred = infer_guarded(model, img, K, device, infer_stats)
        if pred.shape != (h,w):
            pred = cv2.resize(pred, (w,h), interpolation=cv2.INTER_LINEAR)

        metrics, ok = compute_metrics(pred, gt, args.max_depth_m)
        frame_metrics.append(FrameMetrics(ct, metrics["valid_pixels"],
                                          metrics["mae_m"], metrics["rmse_m"], metrics["abs_rel"]))
        if metrics["valid_pixels"] > 0:
            all_pred.append(pred[ok]); all_gt.append(gt[ok])

        # YOLO: visualization only — does NOT affect which frames are evaluated
        vis = overlay_lidar(img, gt)
        if yolo_model is not None:
            vis, recs = evaluate_and_draw_boxes(vis, pred, gt, yolo_model,
                                                args.yolo_conf, max_obj_depth,
                                                idx, ct)
            box_records.extend(recs)

        save_composite(frame_dir, idx, vis, pred, metrics, model._variant_label)

        if idx % 10 == 0:
            print(f"  [{idx:04d}] valid={metrics['valid_pixels']:5d} "
                  f"MAE={metrics['mae_m']:.4f} RMSE={metrics['rmse_m']:.4f} AbsRel={metrics['abs_rel']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame([{"sample_token":m.sample_token,"valid_pixels":m.valid_pixels,
                        "mae_m":m.mae_m,"rmse_m":m.rmse_m,"abs_rel":m.abs_rel}
                       for m in frame_metrics if m.valid_pixels>0])
    df.to_csv(os.path.join(run_dir,"frame_metrics.csv"),index=False)

    if all_pred:
        pc=np.concatenate(all_pred); gc=np.concatenate(all_gt); ae=np.abs(pc-gc)
        summary={"model":model._variant_label,"camera":args.camera,
                 "max_depth_m":args.max_depth_m,"n_frames":len(frame_metrics),
                 "n_valid_frames":int(df.shape[0]),"total_pixels":int(pc.size),
                 "mae_m":float(np.mean(ae)),
                 "rmse_m":float(np.sqrt(np.mean((pc-gc)**2))),
                 "abs_rel":float(np.mean(ae/np.clip(gc,1e-6,None))),
                 "median_ae_m":float(np.median(ae)),
                 # Provenance of the invalid-CUDA-output guard, so the run is
                 # self-documenting and comparable against the 2026-06 numbers.
                 "invalid_infer_retries":infer_stats["retry"],
                 "invalid_infer_cpu_fallbacks":infer_stats["cpu_fallback"],
                 "invalid_infer_unrecovered":infer_stats["unrecovered"]}
        print("\n"+"="*55+f"\nSUMMARY — {model._variant_label}  {args.camera}\n"+"="*55)
        print(tabulate([[k,f"{v:.4f}"if isinstance(v,float)else v]
                        for k,v in summary.items()],
                       headers=["Metric","Value"],tablefmt="grid"))
        pd.DataFrame([summary]).to_csv(os.path.join(run_dir,"summary_metrics.csv"),index=False)

    if box_records:
        bdf = pd.DataFrame(box_records)
        bdf.to_csv(os.path.join(run_dir,"box_metrics.csv"),index=False)
        box_summary = {"n_detections": len(bdf),
                       "mae_nearest_m": float(bdf["abs_err_m"].mean()),
                       "median_nearest_m": float(bdf["abs_err_m"].median())}
        pd.DataFrame([box_summary]).to_csv(os.path.join(run_dir,"box_summary.csv"),index=False)
        print(f"\nObject-level: {len(bdf)} detections  MAE={bdf['abs_err_m'].mean():.4f}m")

    # ── Invalid-inference guard report ────────────────────────────────────────
    n_zero = sum(1 for m in frame_metrics if m.valid_pixels == 0)
    n_done = len(frame_metrics)
    # The other five configs lose 51 of 16,182 frames to genuinely empty GT.
    # Scale that rate to however many frames this run actually processed, so the
    # figure is meaningful on --max-samples runs too.
    baseline_expected = 51.0 / 16182.0 * n_done
    print("\n" + "=" * 55 + "\nINVALID-INFERENCE GUARD\n" + "=" * 55)
    print(f"  frames processed          : {n_done}")
    print(f"  frames with 0 valid pixels: {n_zero}   "
          f"(expected ~{baseline_expected:.0f} at this frame count = no LiDAR "
          f"return in 0-{args.max_depth_m:.0f} m)")
    print(f"  transient CUDA failures   : {infer_stats['retry']}")
    print(f"  needed CPU fallback       : {infer_stats['cpu_fallback']}")
    print(f"  unrecovered               : {infer_stats['unrecovered']}")
    if infer_stats["retry"] or infer_stats["cpu_fallback"]:
        print(f"  -> the guard rescued {infer_stats['retry'] + infer_stats['cpu_fallback']} "
              f"frame(s) that the 2026-06 run would have dropped silently.")
    elif n_done < 2000:
        print("  -> guard did not fire; at this sample size that is expected "
              "(~0.6% of frames failed in the 2026-06 run).")
    if n_zero > 2 * baseline_expected + 10:
        print("  !! well above the expected baseline - investigate before using these numbers.")

    print(f"\n[done] {run_dir}")

if __name__ == "__main__":
    main()
