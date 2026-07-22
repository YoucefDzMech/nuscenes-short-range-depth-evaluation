"""
Metric3D v2 — Consistent re-evaluation script
==============================================
Camera  : CAM_FRONT  (no bumper obstruction)
Frames  : ALL keyframes on disk (~3367)  — no YOLO gate on metrics
Metrics : pixel-level MAE/RMSE/AbsRel at LiDAR GT pixels 0.1–5m
YOLO    : runs AFTER inference, visualization only (boxes on composite PNG)
Output  : D:\\RVC_Model'sOutput\\Metric3D\\CAM_FRONT_all_frames_5m\\runN\\

Run:
  & "C:\\RVC\\Metric3D\\M3D\\Scripts\\Activate.ps1"
  python nuscenes_metric3d_camfront_eval.py
  python nuscenes_metric3d_camfront_eval.py --max-samples 100   # quick test
"""

import argparse, bisect, os, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2, matplotlib, numpy as np, pandas as pd, torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from tabulate import tabulate

# Metric3D internal imports
sys.path.insert(0, os.environ.get("METRIC3D_REPO", "path/to/Metric3D"))  # official model repo (not included)
from mono.model.monodepth_model import get_configured_monodepth_model as DepthModel
import mono.model.backbones.ViT_DINO_reg as vit_dino_reg

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CFG    = "path/to/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py"
DEFAULT_CKPT   = "path/to/metric_depth_vit_giant2_800k.pth"
DEFAULT_YOLO   = os.environ.get("YOLO_WEIGHTS", "yolo26s.pt")
DEFAULT_OUTDIR = "outputs/metric3d_CAM_FRONT_5m"
DATAROOT       = os.environ.get("NUSCENES_DATAROOT", "path/to/nuscenes/trainval02")

@dataclass
class FrameMetrics:
    sample_token: str
    valid_pixels: int
    mae_m: float
    rmse_m: float
    abs_rel: float

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
class AttrDict(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and k in base and isinstance(base[k], dict) else v
    return out

def _load_python_config(path):
    import importlib.util
    cfg_dir = os.path.dirname(os.path.abspath(path))
    spec = importlib.util.spec_from_file_location("_cfg_" + os.path.splitext(os.path.basename(path))[0], path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for base_rel in bases:
        base_path = os.path.normpath(os.path.join(cfg_dir, base_rel))
        merged = _deep_merge(merged, _load_python_config(base_path))
    return _deep_merge(merged, raw)

def _to_ns(obj):
    if isinstance(obj, dict): return AttrDict({k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list): return [_to_ns(v) for v in obj]
    return obj

def load_model(cfg_path, ckpt_path, device):
    print(f"[M3D  ] Loading config: {cfg_path}")
    cfg = _to_ns(_load_python_config(cfg_path))
    if hasattr(vit_dino_reg, "DinoVisionTransformer") and \
       hasattr(vit_dino_reg.DinoVisionTransformer, "init_weights"):
        vit_dino_reg.DinoVisionTransformer.init_weights = lambda self: None
    model = DepthModel(cfg)
    ckpt  = torch.load(ckpt_path, map_location="cpu", mmap=True)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    print(f"[M3D  ] Metric3D ({model_name}) ready on {device}")
    return model, cfg

def preprocess(image_bgr, crop_size, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ch, cw = crop_size
    scale = min(ch/h, cw/w)
    rh, rw = max(1,int(round(h*scale))), max(1,int(round(w*scale)))
    resized = cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_LINEAR)
    ph, pw = ch-rh, cw-rw; pt,pb,pl,pr = ph//2, ph-ph//2, pw//2, pw-pw//2
    padded = cv2.copyMakeBorder(resized,pt,pb,pl,pr,cv2.BORDER_CONSTANT,value=[123.675,116.28,103.53])
    mean = torch.tensor([123.675,116.28,103.53],dtype=torch.float32)[:,None,None]
    std  = torch.tensor([58.395,57.12,57.375],  dtype=torch.float32)[:,None,None]
    t = torch.from_numpy(padded.transpose(2,0,1)).float()
    t = torch.div(t-mean,std).unsqueeze(0).to(device)
    return t, scale, (pt,pb,pl,pr)

def infer(model, image_bgr, intrinsic, crop_size, device):
    h0, w0 = image_bgr.shape[:2]
    tensor, scale, (pt,pb,pl,pr) = preprocess(image_bgr, crop_size, device)
    with torch.inference_mode():
        m = model.module if hasattr(model,"module") else model
        pred_depth, _, _ = m.inference({"input": tensor})
    depth = pred_depth.squeeze().float().cpu().numpy()
    ch, cw = crop_size
    depth = depth[pt:ch-pb, pl:cw-pr]
    depth = cv2.resize(depth, (w0,h0), interpolation=cv2.INTER_LINEAR)
    fx = float(intrinsic[0,0])
    depth = depth * (fx * scale / 1000.0)
    return np.clip(depth, 0.0, 300.0).astype(np.float32)

# ── nuScenes helpers ──────────────────────────────────────────────────────────
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
        lid_chain.sort(); lid_ts=[x[0] for x in lid_chain]; lid_toks=[x[1] for x in lid_chain]
        cam_cur = nusc.get("sample_data", first["data"][camera])
        while True:
            if not cam_cur["is_key_frame"] and os.path.isfile(os.path.join(nusc.dataroot, cam_cur["filename"])):
                idx = bisect.bisect_left(lid_ts, cam_cur["timestamp"])
                best = min([i for i in (idx-1,idx,idx+1) if 0<=i<len(lid_ts)],
                           key=lambda i: abs(lid_ts[i]-cam_cur["timestamp"]))
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
    gt = np.zeros((h,w), dtype=np.float32)
    ok = (depths>min_d)&(u>=0)&(u<w)&(v>=0)&(v<h)
    uu = np.clip(np.floor(u[ok]).astype(int),0,w-1)
    vv = np.clip(np.floor(v[ok]).astype(int),0,h-1)
    for x,y,d in zip(uu,vv,depths[ok]):
        if gt[y,x]==0 or d<gt[y,x]: gt[y,x]=d
    return gt

def compute_metrics(pred, gt, max_d):
    ok=(gt>0)&(gt<=max_d)&np.isfinite(pred)&(pred>0)
    p,g=pred[ok],gt[ok]
    if p.size==0: return {"valid_pixels":0,"mae_m":np.nan,"rmse_m":np.nan,"abs_rel":np.nan},ok
    ae=np.abs(p-g)
    return {"valid_pixels":int(p.size),"mae_m":float(np.mean(ae)),
            "rmse_m":float(np.sqrt(np.mean((p-g)**2))),
            "abs_rel":float(np.mean(ae/np.clip(g,1e-6,None)))},ok

# ── Visualisation ─────────────────────────────────────────────────────────────
def colorize(depth):
    v=np.isfinite(depth)
    if not v.any(): return np.zeros((*depth.shape,3),dtype=np.uint8)
    lo,hi=np.percentile(depth[v],2),np.percentile(depth[v],98); hi=max(hi,lo+1e-6)
    n=np.clip((depth-lo)/(hi-lo),0,1)
    c=matplotlib.colormaps.get_cmap("Spectral")
    return (c((n*255).astype(np.uint8))[:,:,:3]*255).astype(np.uint8)[:,:,::-1]

def overlay_lidar(img, gt):
    out=img.copy(); ys,xs=np.where(gt>0)
    if ys.size==0: return out
    d=gt[ys,xs]; lo,hi=np.percentile(d,5),np.percentile(d,95); hi=max(hi,lo+1e-6)
    for x,y,dd in zip(xs,ys,d):
        t=float(np.clip((dd-lo)/(hi-lo),0,1))
        cv2.circle(out,(int(x),int(y)),1,(int(255*(1-t)),int(255*t),255),-1)
    return out

def evaluate_and_draw_boxes(img, pred, gt, yolo_model, conf_thr,
                            max_obj_depth, frame_idx, cam_token):
    records=[]
    res=yolo_model(img,verbose=False)
    if not res or res[0].boxes is None: return img, records
    out=img.copy(); h,w=pred.shape
    for box,conf,cls in zip(res[0].boxes.xyxy.cpu().numpy(),
                             res[0].boxes.conf.cpu().numpy(),
                             res[0].boxes.cls.cpu().numpy()):
        if float(conf)<conf_thr: continue
        x1,y1,x2,y2=max(0,int(box[0])),max(0,int(box[1])),min(w,int(box[2])),min(h,int(box[3]))
        if x2<=x1 or y2<=y1: continue
        pv=pred[y1:y2,x1:x2]; gv=gt[y1:y2,x1:x2]
        pv=pv[np.isfinite(pv)&(pv>0)]; gv=gv[gv>0]
        if pv.size==0 or gv.size==0: continue
        pm,gm=float(np.min(pv)),float(np.min(gv))
        if max_obj_depth is not None and min(pm,gm)>max_obj_depth: continue
        ae=abs(pm-gm)
        name=res[0].names.get(int(cls),str(int(cls)))
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(out,f"{name} p:{pm:.1f}m gt:{gm:.1f}m",(x1,max(20,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1,cv2.LINE_AA)
        records.append({"frame_idx":frame_idx,"cam_token":cam_token,
                        "class_name":name,"conf":float(conf),
                        "pred_min_m":pm,"gt_min_m":gm,
                        "abs_err_m":ae,"abs_rel":ae/max(gm,1e-6)})
    return out, records

def save_composite(outdir, idx, left, pred, metrics):
    os.makedirs(outdir,exist_ok=True)
    comp=cv2.hconcat([left,colorize(pred)])
    lines=["Metric3D ViT-Giant2",f"Valid px: {metrics['valid_pixels']}",
           f"MAE : {metrics['mae_m']:.4f} m",f"RMSE: {metrics['rmse_m']:.4f} m",
           f"AbsRel: {metrics['abs_rel']:.4f}"]
    font,fs,th,lh,pad=cv2.FONT_HERSHEY_SIMPLEX,0.55,1,22,10
    bw=max(cv2.getTextSize(l,font,fs,th)[0][0] for l in lines)+pad*2
    bh=lh*len(lines)+pad; x2=comp.shape[1]-20; y1=20
    x1=max(0,x2-bw); y2=min(comp.shape[0],y1+bh)
    ov=comp.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),(20,20,20),-1)
    cv2.addWeighted(ov,0.75,comp,0.25,0,comp); cv2.rectangle(comp,(x1,y1),(x2,y2),(255,255,255),1)
    for i,l in enumerate(lines):
        cv2.putText(comp,l,(x1+pad,y1+pad+(i+1)*lh-6),font,fs,(255,255,255),th,cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir,f"{idx:04d}_composite.png"),comp)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pa=argparse.ArgumentParser("Metric3D CAM_FRONT consistent eval")
    pa.add_argument("--dataroot",    default=DATAROOT)
    pa.add_argument("--version",     default="v1.0-trainval")
    pa.add_argument("--camera",      default="CAM_FRONT")
    pa.add_argument("--lidar",       default="LIDAR_TOP")
    pa.add_argument("--config",      default=DEFAULT_CFG)
    pa.add_argument("--checkpoint",  default=DEFAULT_CKPT)
    pa.add_argument("--max-depth-m", type=float, default=5.0)
    pa.add_argument("--max-samples", type=int,   default=0)
    pa.add_argument("--outdir",      default=DEFAULT_OUTDIR)
    pa.add_argument("--use-yolo",          action="store_true")
    pa.add_argument("--yolo-weights",      default=DEFAULT_YOLO)
    pa.add_argument("--yolo-conf",         type=float, default=0.25)
    pa.add_argument("--max-object-depth-m",type=float, default=5.0)
    args=pa.parse_args()

    device=get_device(); print(f"[device] {device}")
    model,cfg=load_model(args.config,args.checkpoint,device)
    crop_size=cfg.data_basic.crop_size

    nusc=NuScenes(version=args.version,dataroot=args.dataroot,verbose=True)
    frame_pairs=load_all_frames(nusc,args.camera,args.lidar,args.max_samples)
    print(f"[nusc ] {len(frame_pairs)} sweep frames (keyframes excluded)")

    yolo_model=None
    if args.use_yolo:
        from ultralytics import YOLO
        yolo_model=YOLO(args.yolo_weights); print(f"[yolo ] visualization only")

    base=os.path.normpath(args.outdir); n=1
    while os.path.exists(os.path.join(base,f"run{n}")): n+=1
    run_dir=os.path.join(base,f"run{n}"); os.makedirs(run_dir)
    frame_dir=os.path.join(run_dir,"frames"); os.makedirs(frame_dir)
    print(f"[out  ] {run_dir}")

    frame_metrics=[]; box_records=[]; all_pred=[]; all_gt=[]
    max_obj_depth=args.max_object_depth_m if args.max_object_depth_m>=0 else None

    for idx,(ct,lt) in enumerate(frame_pairs):
        sd=nusc.get("sample_data",ct)
        img=cv2.imread(os.path.join(nusc.dataroot,sd["filename"]))
        if img is None: continue

        depths,u,v,K=lidar_to_camera(nusc,lt,ct)
        h,w=img.shape[:2]
        gt=sparse_gt(h,w,depths,u,v)
        try:
            pred=infer(model,img,K,crop_size,device)
        except Exception as e:
            print(f"  [{idx:04d}] inference error: {e}"); continue
        if pred.shape!=(h,w):
            pred=cv2.resize(pred,(w,h),interpolation=cv2.INTER_LINEAR)

        metrics,ok=compute_metrics(pred,gt,args.max_depth_m)
        frame_metrics.append(FrameMetrics(ct,metrics["valid_pixels"],
                             metrics["mae_m"],metrics["rmse_m"],metrics["abs_rel"]))
        if metrics["valid_pixels"]>0:
            all_pred.append(pred[ok]); all_gt.append(gt[ok])

        vis=overlay_lidar(img,gt)
        if yolo_model is not None:
            vis,recs=evaluate_and_draw_boxes(vis,pred,gt,yolo_model,
                                             args.yolo_conf,max_obj_depth,idx,ct)
            box_records.extend(recs)
        save_composite(frame_dir,idx,vis,pred,metrics)

        if idx%10==0:
            print(f"  [{idx:04d}] valid={metrics['valid_pixels']:5d} "
                  f"MAE={metrics['mae_m']:.4f} RMSE={metrics['rmse_m']:.4f} AbsRel={metrics['abs_rel']:.4f}")

    df=pd.DataFrame([{"sample_token":m.sample_token,"valid_pixels":m.valid_pixels,
                      "mae_m":m.mae_m,"rmse_m":m.rmse_m,"abs_rel":m.abs_rel}
                     for m in frame_metrics if m.valid_pixels>0])
    df.to_csv(os.path.join(run_dir,"frame_metrics.csv"),index=False)

    if all_pred:
        pc=np.concatenate(all_pred); gc=np.concatenate(all_gt); ae=np.abs(pc-gc)
        summary={"model":"Metric3D-ViT-Giant2","camera":args.camera,
                 "max_depth_m":args.max_depth_m,"n_frames":len(frame_metrics),
                 "n_valid_frames":int(df.shape[0]),"total_pixels":int(pc.size),
                 "mae_m":float(np.mean(ae)),"rmse_m":float(np.sqrt(np.mean((pc-gc)**2))),
                 "abs_rel":float(np.mean(ae/np.clip(gc,1e-6,None))),"median_ae_m":float(np.median(ae))}
        print("\n"+"="*55+"\nSUMMARY — Metric3D ViT-Giant2  "+args.camera+"\n"+"="*55)
        print(tabulate([[k,f"{v:.4f}"if isinstance(v,float)else v] for k,v in summary.items()],
                       headers=["Metric","Value"],tablefmt="grid"))
        pd.DataFrame([summary]).to_csv(os.path.join(run_dir,"summary_metrics.csv"),index=False)

    if box_records:
        bdf=pd.DataFrame(box_records)
        bdf.to_csv(os.path.join(run_dir,"box_metrics.csv"),index=False)
        box_summary={"n_detections":len(bdf),
                     "mae_nearest_m":float(bdf["abs_err_m"].mean()),
                     "median_nearest_m":float(bdf["abs_err_m"].median())}
        pd.DataFrame([box_summary]).to_csv(os.path.join(run_dir,"box_summary.csv"),index=False)
        print(f"\nObject-level: {len(bdf)} detections  MAE={bdf['abs_err_m'].mean():.4f}m")

    print(f"\n[done] {run_dir}")

if __name__=="__main__":
    main()
