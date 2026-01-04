#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex-friendly end-to-end demo (headless):

1) generate dummy CT + X-ray + YOLO dataset
2) train dummy YOLO (quick)
3) run Phase4 headless inference on the dummy X-rays and save outputs

Run:
  python demo_codex_headless_phase4.py --epochs 10 --imgsz 320 --device cpu

Outputs:
  - runs/detect/.../weights/best.pt
  - outputs/test_patient001_overlay.png (+ json)
  - outputs/test_patient002_overlay.png (+ json)
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def find_best_pt() -> str:
    cands = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
    if not cands:
        raise FileNotFoundError("best.pt not found under runs/detect/**/weights/. Did training finish?")
    # pick most recently modified
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_yolo", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model_dir", default="data/models")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    run([sys.executable, "generate_dummy_data.py", "--make_yolo_dataset", "--n_yolo", str(args.n_yolo)])
    run(
        [
            sys.executable,
            "train_dummy_yolo.py",
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            str(args.device),
        ]
    )

    best_pt = find_best_pt()
    class_yaml = "data/yolo_dummy/data.yaml"

    for pid in ["patient001", "patient002"]:
        img = f"data/xrays/test_{pid}.png"
        run(
            [
                sys.executable,
                "phase4_headless.py",
                "--input",
                img,
                "--model_dir",
                args.model_dir,
                "--yolo",
                best_pt,
                "--class_yaml",
                class_yaml,
                "--out_dir",
                args.out_dir,
            ]
        )

    print("\nDone.")
    print(f"YOLO weights: {best_pt}")
    print(f"Outputs dir: {args.out_dir}/")


if __name__ == "__main__":
    main()
