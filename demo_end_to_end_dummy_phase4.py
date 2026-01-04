"""demo_end_to_end_dummy_phase4.py

ダミーX線 + YOLO で Phase4 推論までを一気に動かすデモスクリプト。

- ダミーデータ生成（CT + X-ray + YOLO dataset）
- YOLO学習（軽量設定）
- Phase4 推論（2枚のテストX線）

注意: CPUだとYOLO学習に少し時間がかかります。必要なら epochs を減らしてください。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # 1) dummy + dataset
    _run([sys.executable, "generate_dummy_data.py", "--make_yolo_dataset", "--n_yolo", "300"])

    # 2) train
    _run([sys.executable, "train_dummy_yolo.py", "--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--device", args.device])

    # 3) find best weights
    best = sorted(Path("runs/detect").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not best:
        raise FileNotFoundError("best.pt not found under runs/detect/**/weights/")
    w = best[0].as_posix()

    # 4) inference
    for img in ["data/xrays/test_patient001.png", "data/xrays/test_patient002.png"]:
        _run([sys.executable, "04_inference.py", "--img", img, "--yolo", w, "--class_yaml", "data/yolo_dummy/data.yaml", "--draw"])


if __name__ == "__main__":
    main()
