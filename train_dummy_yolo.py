"""train_dummy_yolo.py

ダミーX線 + ダミーラベルで YOLO (Ultralytics) を学習し、
Phase4推論（04_inference.py / GUI）に使える .pt を作ります。

手順
----
1) ダミーデータ + YOLOデータセット生成:
   python generate_dummy_data.py --make_yolo_dataset --n_yolo 300

2) 学習:
   python train_dummy_yolo.py

出力
----
Ultralytics標準の runs/detect/train/weights/best.pt ができます。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/yolo_dummy/data.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="初期重み（軽量なら yolov8n.pt 推奨）")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or 0 etc.")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            f"先に `python generate_dummy_data.py --make_yolo_dataset` を実行してください。"
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        verbose=True,
        plots=False,
    )

    # Print best path
    runs = Path("runs/detect")
    best = sorted(runs.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        print("\n✅ best.pt:")
        print(best[0].as_posix())
        print("\nPhase4 example:")
        print(
            "python 04_inference.py --img data/xrays/test_patient001.png "
            f"--yolo {best[0].as_posix()} --class_yaml data/yolo_dummy/data.yaml --draw"
        )
    else:
        print("Training finished, but best.pt not found under runs/detect.")


if __name__ == "__main__":
    main()
