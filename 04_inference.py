from __future__ import annotations

import argparse
import json
import os

import cv2
import lightgbm as lgb
import numpy as np

from scapula_analysis.features import extract_features
from scapula_analysis.yolo_utils import load_class_map_from_data_yaml, pick_max_conf_per_class

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Default class IDs (must match your YOLO training YAML)
DEFAULT_CLASS_MAP = {
    "superior": 0,
    "inferior": 1,
    "lateral": 2,
    "clavicle_med": 3,
    "clavicle_lat": 4,
}

REQ_POINTS = ["superior", "inferior", "lateral"]
CLAV_KEYS = ["clavicle_med", "clavicle_lat"]


def detect_dummy(img):
    h, w = img.shape[:2]
    return {
        "superior": (0.55 * w, 0.35 * h),
        "inferior": (0.50 * w, 0.60 * h),
        "lateral": (0.65 * w, 0.55 * h),
        "clavicle_med": (0.40 * w, 0.25 * h),
        "clavicle_lat": (0.60 * w, 0.25 * h),
    }


def load_train_metrics(model_dir: str) -> dict:
    p = os.path.join(model_dir, "train_metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--yolo", default=None, help="YOLO .pt path (optional)")
    ap.add_argument("--class_yaml", default=None, help="YOLO data.yaml to load class-id mapping")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true")

    # Optional: include clavicle features (should match training)
    ap.add_argument("--use_clavicle_features", action="store_true", help="Use clavicle features if points exist")
    ap.add_argument("--include_clavicle_scale_hint", action="store_true", help="Append raw clavicle length (scale hint)")

    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {args.img}")

    models = {}
    for t in ["pitch", "yaw", "roll"]:
        p = os.path.join(args.model_dir, f"lgbm_{t}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found: {p}. Run 03_train_lgbm.py first.")
        models[t] = lgb.Booster(model_file=p)

    metrics = load_train_metrics(args.model_dir)
    trained_with_clav = bool(metrics.get("include_clavicle_features", False))
    trained_with_clav_scale = bool(metrics.get("include_clavicle_scale_hint", False))

    # If user did not pass flags, follow training config
    use_clav = bool(args.use_clavicle_features) or trained_with_clav
    include_clav_scale = bool(args.include_clavicle_scale_hint) or trained_with_clav_scale

    class_map = dict(DEFAULT_CLASS_MAP)
    if args.class_yaml:
        wanted = list(DEFAULT_CLASS_MAP.keys())
        class_map = load_class_map_from_data_yaml(args.class_yaml, wanted)

    pts = None
    if args.yolo and os.path.exists(args.yolo):
        if YOLO is None:
            raise RuntimeError("ultralytics が import できません。requirements.txt を入れて環境を作り直してください。")
        yolo = YOLO(args.yolo)
        res = yolo.predict(args.img, verbose=False)[0]
        pts = pick_max_conf_per_class(res, class_map, conf_thres=args.conf)
    else:
        print("YOLOモデル未指定 or 見つからないため、ダミー座標で推論します。")
        pts = detect_dummy(img)

    for k in REQ_POINTS:
        if k not in pts:
            raise RuntimeError(f"Missing required keypoint: {k}. detected={list(pts.keys())}")

    scap = np.array([pts["superior"], pts["inferior"], pts["lateral"]], dtype=float)

    clav = None
    if use_clav:
        if all(k in pts for k in CLAV_KEYS):
            clav = np.array([pts["clavicle_med"], pts["clavicle_lat"]], dtype=float)
        else:
            # training had clav but inference doesn't -> be loud
            if trained_with_clav:
                raise RuntimeError(
                    "Model was trained with clavicle features, but clavicle keypoints are missing at inference. "
                    "Fix YOLO class IDs or disable clavicle features / retrain."
                )

    feats = extract_features(
        scap,
        clavicle_points_2d=clav,
        include_clavicle_scale_hint=include_clav_scale,
    ).reshape(1, -1)

    out = {t: float(models[t].predict(feats)[0]) for t in ["pitch", "yaw", "roll"]}
    print("-" * 32)
    print(f"Pitch (前傾): {out['pitch']:.1f} deg")
    print(f"Yaw   (内旋): {out['yaw']:.1f} deg")
    print(f"Roll  (上方回旋): {out['roll']:.1f} deg")
    print("-" * 32)

    if args.draw:
        for name, (x, y) in pts.items():
            cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(img, name, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(
            img,
            f"R:{out['roll']:.1f} Y:{out['yaw']:.1f} P:{out['pitch']:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.imshow("ScapulaAnalysis", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
