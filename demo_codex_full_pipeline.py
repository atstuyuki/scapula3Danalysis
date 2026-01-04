#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex-friendly full pipeline demo (NO GUI):

- dummy CT + dummy X-ray
- auto annotations CSV (so Phase1 click is not needed)
- SSM build
- LightGBM training
- YOLO dummy training
- Phase4 headless inference (saves overlay+json)

Run:
  python demo_codex_full_pipeline.py --device cpu --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def find_best_pt() -> str:
    cands = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
    if not cands:
        raise FileNotFoundError("best.pt not found under runs/detect/**/weights/. Did YOLO training finish?")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_yolo", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10, help="YOLO epochs")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--n_samples", type=int, default=20000, help="LGBM synthetic samples")
    ap.add_argument("--noise_sigma", type=float, default=2.0)
    ap.add_argument("--z0_min", type=float, default=800.0)
    ap.add_argument("--z0_max", type=float, default=1200.0)

    ap.add_argument("--model_dir", default="data/models")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    # 0) dummy + yolo dataset + annotations
    run([sys.executable, "generate_dummy_data.py", "--make_annotations_csv", "--make_yolo_dataset", "--n_yolo", str(args.n_yolo)])

    # 1) SSM
    run([sys.executable, "02_build_ssm.py", "--in_csv", "data/annotations_3d.csv", "--model_dir", args.model_dir])

    # 2) LGBM train
    run([
        sys.executable, "03_train_lgbm.py",
        "--model_dir", args.model_dir, "--out_dir", args.model_dir,
        "--n_samples", str(args.n_samples),
        "--noise_sigma", str(args.noise_sigma),
        "--z0_min", str(args.z0_min),
        "--z0_max", str(args.z0_max),
        "--include_clavicle_features",
    ])

    # 3) YOLO train
    run([
        sys.executable, "train_dummy_yolo.py",
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
    ])

    best_pt = find_best_pt()
    class_yaml = "data/yolo_dummy/data.yaml"

    # 4) Phase4 headless inference (two images)
    for pid in ["patient001", "patient002"]:
        img = f"data/xrays/test_{pid}.png"
        run([
            sys.executable, "phase4_headless.py",
            "--input", img,
            "--model_dir", args.model_dir,
            "--yolo", best_pt,
            "--class_yaml", class_yaml,
            "--out_dir", args.out_dir,
        ])

    print("\nDone.")
    print(f"YOLO weights: {best_pt}")
    print(f"Outputs dir: {args.out_dir}/")
    print("Artifacts:")
    print(f"  {args.out_dir}/test_patient001_overlay.png")
    print(f"  {args.out_dir}/test_patient002_overlay.png")
    print(f"  {args.out_dir}/test_patient001_pred.json")
    print(f"  {args.out_dir}/test_patient002_pred.json")


if __name__ == "__main__":
    main()
