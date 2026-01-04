#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase4 headless inference (Codex / cloud-friendly)

- Supports image OR video input
- No GUI / no cv2.imshow
- Saves:
  - overlay image/video (optional)
  - per-frame CSV (video) / JSON (image+video)

Usage examples:

Image:
  python phase4_headless.py --input data/xrays/test_patient001.png --model_dir data/models \
    --yolo runs/detect/train/weights/best.pt --class_yaml data/yolo_dummy/data.yaml --out_dir outputs

Video:
  python phase4_headless.py --input path/to/video.mp4 --mode video --model_dir data/models \
    --yolo runs/detect/train/weights/best.pt --class_yaml data/yolo_dummy/data.yaml --out_dir outputs \
    --save_video --save_csv --ema_kp_alpha 0.6 --ema_angle_alpha 0.3
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import lightgbm as lgb
import numpy as np

from scapula_analysis.features import extract_features
from scapula_analysis.yolo_utils import (
    load_class_map_from_data_yaml,
    pick_max_conf_per_class,
)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DEFAULT_CLASS_MAP = {
    "superior": 0,
    "inferior": 1,
    "lateral": 2,
    "clavicle_med": 3,
    "clavicle_lat": 4,
}
REQ_KEYS = ("superior", "inferior", "lateral")
CLAV_KEYS = ("clavicle_med", "clavicle_lat")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_train_metrics(model_dir: str) -> dict:
    p = os.path.join(model_dir, "train_metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def dummy_points(w: int, h: int) -> Dict[str, Tuple[float, float]]:
    # Deterministic fallback points (only for smoke tests)
    return {
        "superior": (0.35 * w, 0.35 * h),
        "inferior": (0.38 * w, 0.72 * h),
        "lateral": (0.62 * w, 0.52 * h),
        "clavicle_med": (0.42 * w, 0.25 * h),
        "clavicle_lat": (0.62 * w, 0.25 * h),
    }


@dataclass
class EMAState:
    kp: Optional[Dict[str, np.ndarray]] = None
    angles: Optional[Dict[str, float]] = None


def ema_update_points(prev: Optional[Dict[str, np.ndarray]], cur: Dict[str, Tuple[float, float]], alpha: float) -> Dict[str, np.ndarray]:
    if prev is None:
        return {k: np.array(v, dtype=float) for k, v in cur.items()}
    out: Dict[str, np.ndarray] = dict(prev)
    for k, v in cur.items():
        v_arr = np.array(v, dtype=float)
        if k in out:
            out[k] = alpha * v_arr + (1.0 - alpha) * out[k]
        else:
            out[k] = v_arr
    return out


def ema_update_angles(prev: Optional[Dict[str, float]], cur: Dict[str, float], alpha: float) -> Dict[str, float]:
    if prev is None:
        return dict(cur)
    out = dict(prev)
    for k, v in cur.items():
        out[k] = alpha * float(v) + (1.0 - alpha) * float(out.get(k, v))
    return out


def draw_overlay(img_bgr: np.ndarray, pts: Dict[str, Tuple[float, float]], angles: Dict[str, float]) -> np.ndarray:
    out = img_bgr.copy()
    for name, (x, y) in pts.items():
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(out, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    txt = f"R:{angles['roll']:.1f} Y:{angles['yaw']:.1f} P:{angles['pitch']:.1f}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def predict_angles(models: Dict[str, lgb.Booster], pts: Dict[str, Tuple[float, float]], use_clav: bool, include_clav_scale: bool) -> Dict[str, float]:
    scap = np.array([pts["superior"], pts["inferior"], pts["lateral"]], dtype=float)
    clav = None
    if use_clav and all(k in pts for k in CLAV_KEYS):
        clav = np.array([pts["clavicle_med"], pts["clavicle_lat"]], dtype=float)

    feats = extract_features(
        scap,
        clavicle_points_2d=clav,
        include_clavicle_scale_hint=include_clav_scale,
    ).reshape(1, -1)

    return {t: float(models[t].predict(feats)[0]) for t in ["pitch", "yaw", "roll"]}


def load_models(model_dir: str) -> Dict[str, lgb.Booster]:
    models: Dict[str, lgb.Booster] = {}
    for t in ["pitch", "yaw", "roll"]:
        p = os.path.join(model_dir, f"lgbm_{t}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found: {p}. Run 03_train_lgbm.py first.")
        models[t] = lgb.Booster(model_file=p)
    return models


def run_image(args, models, yolo, class_map, use_clav, include_clav_scale, out_dir):
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {args.input}")
    h, w = img.shape[:2]

    pts = None
    if yolo is not None:
        res = yolo.predict(img, verbose=False)[0]
        pts = pick_max_conf_per_class(res, class_map, conf_thres=args.conf)
    if pts is None or not all(k in pts for k in REQ_KEYS):
        pts = dummy_points(w, h)

    angles = predict_angles(models, pts, use_clav=use_clav, include_clav_scale=include_clav_scale)

    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(args.input))[0]
    json_path = os.path.join(out_dir, f"{base}_pred.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"input": args.input, "points": pts, "angles": angles}, f, ensure_ascii=False, indent=2)

    if args.save_overlay:
        overlay = draw_overlay(img, pts, angles)
        out_path = os.path.join(out_dir, f"{base}_overlay.png")
        cv2.imwrite(out_path, overlay)

    print("-" * 32)
    print(f"Image: {args.input}")
    print(f"Pitch (前傾): {angles['pitch']:.1f} deg")
    print(f"Yaw   (内旋): {angles['yaw']:.1f} deg")
    print(f"Roll  (上方回旋): {angles['roll']:.1f} deg")
    print(f"Saved: {json_path}")
    if args.save_overlay:
        print(f"Saved: {os.path.join(out_dir, f'{base}_overlay.png')}")
    print("-" * 32)


def run_video(args, models, yolo, class_map, use_clav, include_clav_scale, out_dir):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {args.input}")

    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(args.input))[0]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_path = os.path.join(out_dir, f"{base}_overlay.mp4")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    csv_rows = []
    state = EMAState()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pts = None
        if yolo is not None:
            res = yolo.predict(frame, verbose=False)[0]
            pts = pick_max_conf_per_class(res, class_map, conf_thres=args.conf)

        if pts is None or not all(k in pts for k in REQ_KEYS):
            pts = dummy_points(w, h)

        # Keypoint EMA (optional)
        if args.ema_kp_alpha is not None:
            state.kp = ema_update_points(state.kp, pts, alpha=args.ema_kp_alpha)
            pts_use = {k: (float(v[0]), float(v[1])) for k, v in state.kp.items()}
        else:
            pts_use = pts

        angles = predict_angles(models, pts_use, use_clav=use_clav, include_clav_scale=include_clav_scale)

        # Angle EMA (optional)
        if args.ema_angle_alpha is not None:
            state.angles = ema_update_angles(state.angles, angles, alpha=args.ema_angle_alpha)
            angles_use = state.angles
        else:
            angles_use = angles

        t_sec = frame_idx / float(fps)
        csv_rows.append(
            {
                "frame": frame_idx,
                "time_sec": t_sec,
                "pitch": angles_use["pitch"],
                "yaw": angles_use["yaw"],
                "roll": angles_use["roll"],
            }
        )

        if writer is not None:
            overlay = draw_overlay(frame, pts_use, angles_use)
            writer.write(overlay)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Save outputs
    meta = {
        "input": args.input,
        "fps": fps,
        "num_frames": frame_idx,
        "use_clavicle_features": use_clav,
        "include_clavicle_scale_hint": include_clav_scale,
        "ema_kp_alpha": args.ema_kp_alpha,
        "ema_angle_alpha": args.ema_angle_alpha,
    }
    json_path = os.path.join(out_dir, f"{base}_video_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_csv:
        import csv

        csv_path = os.path.join(out_dir, f"{base}_pred.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=["frame", "time_sec", "pitch", "yaw", "roll"])
            wri.writeheader()
            wri.writerows(csv_rows)
    else:
        csv_path = None

    print("-" * 32)
    print(f"Video: {args.input}")
    print(f"Frames: {frame_idx}  FPS: {fps}")
    print(f"Saved: {json_path}")
    if args.save_csv:
        print(f"Saved: {csv_path}")
    if args.save_video:
        print(f"Saved: {os.path.join(out_dir, f'{base}_overlay.mp4')}")
    print("-" * 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Image or video path")
    ap.add_argument("--mode", choices=["auto", "image", "video"], default="auto")
    ap.add_argument("--model_dir", required=True)

    ap.add_argument("--yolo", default=None, help="YOLO .pt path (optional)")
    ap.add_argument("--class_yaml", default=None, help="YOLO data.yaml to load class-id mapping")
    ap.add_argument("--conf", type=float, default=0.25)

    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    ap.add_argument("--save_overlay", action="store_true", help="Save overlay image for image mode")
    ap.add_argument("--save_video", action="store_true", help="Save overlay video for video mode")
    ap.add_argument("--save_csv", action="store_true", help="Save per-frame csv for video mode")

    # Optional: include clavicle features (should match training)
    ap.add_argument("--use_clavicle_features", action="store_true", help="Use clavicle features if points exist")
    ap.add_argument("--include_clavicle_scale_hint", action="store_true", help="Append raw clavicle length (scale hint)")

    # EMA smoothing (video)
    ap.add_argument("--ema_kp_alpha", type=float, default=None, help="Keypoint EMA alpha (e.g., 0.6)")
    ap.add_argument("--ema_angle_alpha", type=float, default=None, help="Angle EMA alpha (e.g., 0.3)")

    args = ap.parse_args()

    metrics = load_train_metrics(args.model_dir)
    trained_with_clav = bool(metrics.get("include_clavicle_features", False))
    trained_with_clav_scale = bool(metrics.get("include_clavicle_scale_hint", False))

    use_clav = bool(args.use_clavicle_features) or trained_with_clav
    include_clav_scale = bool(args.include_clavicle_scale_hint) or trained_with_clav_scale

    models = load_models(args.model_dir)

    class_map = dict(DEFAULT_CLASS_MAP)
    if args.class_yaml:
        wanted = list(DEFAULT_CLASS_MAP.keys())
        class_map = load_class_map_from_data_yaml(args.class_yaml, wanted)

    yolo = None
    if args.yolo and os.path.exists(args.yolo):
        if YOLO is None:
            raise RuntimeError("ultralytics が import できません。requirements_codex.txt で環境を作ってください。")
        yolo = YOLO(args.yolo)

    # Determine mode
    mode = args.mode
    if mode == "auto":
        ext = os.path.splitext(args.input)[1].lower()
        mode = "video" if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"] else "image"

    out_dir = args.out_dir

    if mode == "image":
        if not args.save_overlay:
            # In headless env, saving overlay is usually desired
            args.save_overlay = True
        run_image(args, models, yolo, class_map, use_clav, include_clav_scale, out_dir)
    else:
        # video
        if not args.save_video:
            args.save_video = True
        if not args.save_csv:
            args.save_csv = True
        run_video(args, models, yolo, class_map, use_clav, include_clav_scale, out_dir)


if __name__ == "__main__":
    main()
