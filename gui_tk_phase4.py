#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScapulaAnalysis Tkinter GUI (Phase4)

- Phase4 推論をGUI化（静止画 / 動画を選択）
- 03_train_lgbm.py で作成した lgbm_pitch/yaw/roll.txt を利用
- YOLO(.pt) は任意（未指定の場合はダミー点で推論）

今回追加したこと（要望対応）
- YOLOのクラスID整合性: `data.yaml` を指定して class_map を自動生成
- 鎖骨特徴量: 学習設定(train_metrics.json)に合わせてON/OFF
- 動画の平滑化: キーポイント EMA / 角度 EMA をON/OFF
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
import lightgbm as lgb

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # ultralytics未導入でもGUIは起動できるようにする

from scapula_analysis.features import extract_features
from scapula_analysis.yolo_utils import pick_max_conf_per_class, load_class_map_from_data_yaml

DEFAULT_CLASS_MAP = {
    "superior": 0,
    "inferior": 1,
    "lateral": 2,
    "clavicle_med": 3,
    "clavicle_lat": 4,
}

REQ_KEYS = ("superior", "inferior", "lateral")
CLAV_KEYS = ("clavicle_med", "clavicle_lat")


@dataclass
class InferenceConfig:
    model_dir: str
    yolo_path: str
    class_yaml: str
    conf_thres: float = 0.25
    preview: bool = True

    # video outputs
    save_video: bool = False
    out_video_path: str = ""
    out_csv_path: str = ""

    # feature options
    use_clavicle_features: bool = False
    include_clavicle_scale_hint: bool = False

    # smoothing (video)
    smooth_keypoints: bool = False
    alpha_kpt: float = 0.3
    smooth_angles: bool = False
    alpha_ang: float = 0.2


def detect_dummy(img: np.ndarray) -> Dict[str, Tuple[float, float]]:
    h, w = img.shape[:2]
    return {
        "superior": (0.55 * w, 0.35 * h),
        "inferior": (0.50 * w, 0.60 * h),
        "lateral": (0.65 * w, 0.55 * h),
        "clavicle_med": (0.40 * w, 0.25 * h),
        "clavicle_lat": (0.60 * w, 0.25 * h),
    }


def load_models(model_dir: str) -> Dict[str, lgb.Booster]:
    models = {}
    for t in ("pitch", "yaw", "roll"):
        p = os.path.join(model_dir, f"lgbm_{t}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model not found: {p}\nまず 03_train_lgbm.py を実行してモデルを作成してください。"
            )
        models[t] = lgb.Booster(model_file=p)
    return models


def load_train_metrics(model_dir: str) -> dict:
    p = os.path.join(model_dir, "train_metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_yolo(yolo_path: str):
    if not yolo_path:
        return None
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    if YOLO is None:
        raise RuntimeError("ultralytics が import できません。requirements.txt を入れて環境を作り直してください。")
    return YOLO(yolo_path)


def infer_from_points(models: Dict[str, lgb.Booster], pts: Dict[str, Tuple[float, float]], cfg: InferenceConfig) -> Dict[str, float]:
    scap = np.array([pts["superior"], pts["inferior"], pts["lateral"]], dtype=float)
    clav = None
    if cfg.use_clavicle_features:
        if all(k in pts for k in CLAV_KEYS):
            clav = np.array([pts["clavicle_med"], pts["clavicle_lat"]], dtype=float)
        else:
            raise RuntimeError("鎖骨特徴量ONですが clavicle_med/lat が検出できません（YOLO class_id を確認）")

    feats = extract_features(
        scap,
        clavicle_points_2d=clav,
        include_clavicle_scale_hint=cfg.include_clavicle_scale_hint,
    ).reshape(1, -1)
    return {t: float(models[t].predict(feats)[0]) for t in ("pitch", "yaw", "roll")}


def draw_overlay(frame: np.ndarray, pts: Dict[str, Tuple[float, float]], out: Optional[Dict[str, float]] = None) -> np.ndarray:
    img = frame.copy()
    for name, (x, y) in pts.items():
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(img, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if out is not None:
        cv2.putText(
            img,
            f"Roll:{out['roll']:.1f}  Yaw:{out['yaw']:.1f}  Pitch:{out['pitch']:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
    return img


def ema_point(prev: Tuple[float, float], curr: Tuple[float, float], alpha: float) -> Tuple[float, float]:
    return (alpha * curr[0] + (1 - alpha) * prev[0], alpha * curr[1] + (1 - alpha) * prev[1])


def ema_angles(prev: Dict[str, float], curr: Dict[str, float], alpha: float) -> Dict[str, float]:
    return {k: float(alpha * curr[k] + (1 - alpha) * prev[k]) for k in curr.keys()}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ScapulaAnalysis GUI (Phase4)")
        self.geometry("980x740")

        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        self.model_dir = tk.StringVar(value="")
        self.yolo_path = tk.StringVar(value="")
        self.class_yaml = tk.StringVar(value="")

        self.input_mode = tk.StringVar(value="image")  # image / video
        self.input_path = tk.StringVar(value="")
        self.conf_thres = tk.StringVar(value="0.25")

        self.preview = tk.BooleanVar(value=True)
        self.save_video = tk.BooleanVar(value=False)
        self.out_video_path = tk.StringVar(value="")
        self.out_csv_path = tk.StringVar(value="")

        # feature options
        self.use_clav = tk.BooleanVar(value=False)
        self.use_clav_scale = tk.BooleanVar(value=False)

        # smoothing
        self.smooth_kpt = tk.BooleanVar(value=True)
        self.alpha_kpt = tk.StringVar(value="0.30")
        self.smooth_ang = tk.BooleanVar(value=True)
        self.alpha_ang = tk.StringVar(value="0.20")

        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        # Model dir
        row0 = ttk.Frame(frm)
        row0.pack(fill="x", **pad)
        ttk.Label(row0, text="Model dir (lgbm_*.txt):", width=32).pack(side="left")
        ttk.Entry(row0, textvariable=self.model_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row0, text="Browse", command=self._browse_model_dir).pack(side="left")
        ttk.Button(row0, text="Load training config", command=self._load_train_config).pack(side="left", padx=(6, 0))

        # YOLO
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="YOLO .pt (optional):", width=32).pack(side="left")
        ttk.Entry(row1, textvariable=self.yolo_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="Browse", command=self._browse_yolo).pack(side="left")

        # Class YAML
        row1b = ttk.Frame(frm)
        row1b.pack(fill="x", **pad)
        ttk.Label(row1b, text="YOLO data.yaml (optional):", width=32).pack(side="left")
        ttk.Entry(row1b, textvariable=self.class_yaml).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1b, text="Browse", command=self._browse_class_yaml).pack(side="left")

        # input mode + conf
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Input:", width=32).pack(side="left")
        ttk.Radiobutton(row2, text="Static image", value="image", variable=self.input_mode).pack(side="left")
        ttk.Radiobutton(row2, text="Video", value="video", variable=self.input_mode).pack(side="left")
        ttk.Label(row2, text="   conf:").pack(side="left", padx=(18, 6))
        ttk.Entry(row2, textvariable=self.conf_thres, width=8).pack(side="left")
        ttk.Checkbutton(row2, text="Preview (OpenCV window)", variable=self.preview).pack(side="left", padx=(18, 0))

        # input path
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Image/Video file:", width=32).pack(side="left")
        ttk.Entry(row3, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="Browse", command=self._browse_input).pack(side="left")

        # feature options
        row_feat = ttk.LabelFrame(frm, text="Features")
        row_feat.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(row_feat, text="Use clavicle features (needs clavicle_med/lat)", variable=self.use_clav).pack(
            side="left", padx=8, pady=6
        )
        ttk.Checkbutton(row_feat, text="Include clavicle scale hint (raw length)", variable=self.use_clav_scale).pack(
            side="left", padx=8, pady=6
        )

        # outputs
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", **pad)
        ttk.Checkbutton(row4, text="Save annotated video (video mode)", variable=self.save_video, command=self._on_save_video_toggle).pack(side="left")
        ttk.Entry(row4, textvariable=self.out_video_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row4, text="Browse", command=self._browse_out_video).pack(side="left")

        row5 = ttk.Frame(frm)
        row5.pack(fill="x", **pad)
        ttk.Label(row5, text="Save per-frame CSV (video mode, optional):", width=32).pack(side="left")
        ttk.Entry(row5, textvariable=self.out_csv_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row5, text="Browse", command=self._browse_out_csv).pack(side="left")

        # smoothing
        row_sm = ttk.LabelFrame(frm, text="Smoothing (video)")
        row_sm.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(row_sm, text="EMA keypoints", variable=self.smooth_kpt).pack(side="left", padx=8, pady=6)
        ttk.Label(row_sm, text="alpha:").pack(side="left")
        ttk.Entry(row_sm, textvariable=self.alpha_kpt, width=6).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row_sm, text="EMA angles", variable=self.smooth_ang).pack(side="left", padx=8)
        ttk.Label(row_sm, text="alpha:").pack(side="left")
        ttk.Entry(row_sm, textvariable=self.alpha_ang, width=6).pack(side="left", padx=(4, 12))

        # actions
        row6 = ttk.Frame(frm)
        row6.pack(fill="x", **pad)
        ttk.Button(row6, text="Run", command=self.run).pack(side="left")
        ttk.Button(row6, text="Stop", command=self.stop, state="disabled").pack(side="left", padx=8)
        ttk.Button(row6, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=8)

        # log
        self.log = ScrolledText(frm, height=22)
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

        self._on_save_video_toggle()

    def _browse_model_dir(self):
        d = filedialog.askdirectory(title="Select model directory")
        if d:
            self.model_dir.set(d)

    def _browse_yolo(self):
        p = filedialog.askopenfilename(title="Select YOLO .pt", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if p:
            self.yolo_path.set(p)

    def _browse_class_yaml(self):
        p = filedialog.askopenfilename(title="Select YOLO data.yaml", filetypes=[("YAML", "*.yaml;*.yml"), ("All files", "*.*")])
        if p:
            self.class_yaml.set(p)

    def _browse_input(self):
        if self.input_mode.get() == "video":
            p = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv"), ("All files", "*.*")])
        else:
            p = filedialog.askopenfilename(title="Select image", filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")])
        if p:
            self.input_path.set(p)

    def _browse_out_video(self):
        p = filedialog.asksaveasfilename(
            title="Save annotated video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")],
        )
        if p:
            self.out_video_path.set(p)

    def _browse_out_csv(self):
        p = filedialog.asksaveasfilename(
            title="Save per-frame angles CSV as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if p:
            self.out_csv_path.set(p)

    def _on_save_video_toggle(self):
        if not self.save_video.get():
            self.out_video_path.set("")

    def _load_train_config(self):
        md = self.model_dir.get().strip()
        if not md:
            messagebox.showinfo("Info", "先に Model dir を指定してください")
            return
        m = load_train_metrics(md)
        if not m:
            messagebox.showwarning("Warning", "train_metrics.json が見つからない/読めませんでした")
            return
        self.use_clav.set(bool(m.get("include_clavicle_features", False)))
        self.use_clav_scale.set(bool(m.get("include_clavicle_scale_hint", False)))
        self.println(f"[TRAIN CONFIG] include_clavicle_features={self.use_clav.get()}  include_clavicle_scale_hint={self.use_clav_scale.get()}")

    def println(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _validate(self) -> Optional[InferenceConfig]:
        md = self.model_dir.get().strip()
        if not md:
            messagebox.showerror("Error", "Model dir を指定してください。")
            return None
        ip = self.input_path.get().strip()
        if not ip:
            messagebox.showerror("Error", "入力ファイル（画像/動画）を指定してください。")
            return None
        if not os.path.exists(ip):
            messagebox.showerror("Error", f"入力ファイルが見つかりません: {ip}")
            return None

        try:
            conf = float(self.conf_thres.get().strip())
        except ValueError:
            messagebox.showerror("Error", "conf は数値で指定してください。例: 0.25")
            return None

        try:
            ak = float(self.alpha_kpt.get().strip())
            aa = float(self.alpha_ang.get().strip())
        except ValueError:
            messagebox.showerror("Error", "alpha は数値で指定してください。例: 0.30")
            return None

        if not (0.0 < ak <= 1.0) or not (0.0 < aa <= 1.0):
            messagebox.showerror("Error", "alpha は 0 < alpha <= 1 で指定してください")
            return None

        cfg = InferenceConfig(
            model_dir=md,
            yolo_path=self.yolo_path.get().strip(),
            class_yaml=self.class_yaml.get().strip(),
            conf_thres=conf,
            preview=bool(self.preview.get()),
            save_video=bool(self.save_video.get()),
            out_video_path=self.out_video_path.get().strip(),
            out_csv_path=self.out_csv_path.get().strip(),
            use_clavicle_features=bool(self.use_clav.get()),
            include_clavicle_scale_hint=bool(self.use_clav_scale.get()),
            smooth_keypoints=bool(self.smooth_kpt.get()),
            alpha_kpt=ak,
            smooth_angles=bool(self.smooth_ang.get()),
            alpha_ang=aa,
        )
        return cfg

    def run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Running", "すでに実行中です。Stop を押してください。")
            return

        cfg = self._validate()
        if cfg is None:
            return

        self.stop_event.clear()
        self.println("=" * 60)
        self.println(f"Mode: {self.input_mode.get()}   Input: {self.input_path.get()}")
        self.println(f"Model dir: {cfg.model_dir}")
        self.println(f"YOLO: {cfg.yolo_path or '(none)'}   conf={cfg.conf_thres}")
        self.println(f"data.yaml: {cfg.class_yaml or '(none)'}")
        self.println(
            f"features: clav={cfg.use_clavicle_features}  clav_scale_hint={cfg.include_clavicle_scale_hint}"
        )
        if self.input_mode.get() == "video":
            self.println(f"smooth: kpt={cfg.smooth_keypoints}(a={cfg.alpha_kpt})  ang={cfg.smooth_angles}(a={cfg.alpha_ang})")
            self.println(f"Save video: {cfg.save_video}  out={cfg.out_video_path or '(none)'}")
            self.println(f"Save CSV: {cfg.out_csv_path or '(none)'}")

        self._set_buttons(running=True)
        self.worker = threading.Thread(target=self._run_worker, args=(cfg,), daemon=True)
        self.worker.start()

    def _set_buttons(self, running: bool):
        def walk(widget):
            for w in widget.winfo_children():
                if isinstance(w, ttk.Button):
                    t = w.cget("text")
                    if t == "Run":
                        w.config(state="disabled" if running else "normal")
                    elif t == "Stop":
                        w.config(state="normal" if running else "disabled")
                walk(w)

        walk(self)

    def stop(self):
        self.stop_event.set()
        self.println("[STOP] stopping... (videoの場合は数フレーム後に停止します)")

    def _run_worker(self, cfg: InferenceConfig):
        try:
            models = load_models(cfg.model_dir)
            yolo = load_yolo(cfg.yolo_path) if cfg.yolo_path else None

            class_map = dict(DEFAULT_CLASS_MAP)
            if cfg.class_yaml:
                class_map = load_class_map_from_data_yaml(cfg.class_yaml, list(DEFAULT_CLASS_MAP.keys()))

            # training consistency hint
            tm = load_train_metrics(cfg.model_dir)
            if tm:
                if bool(tm.get("include_clavicle_features", False)) != cfg.use_clavicle_features:
                    self.println(
                        "[WARN] train_metrics.json とGUIの鎖骨特徴量設定が不一致です（推論がエラー or 精度低下の可能性）"
                    )

            if self.input_mode.get() == "video":
                self._run_video(cfg, models, yolo, class_map)
            else:
                self._run_image(cfg, models, yolo, class_map)

        except Exception as e:
            self.println(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self._set_buttons(running=False)

    def _run_image(self, cfg: InferenceConfig, models, yolo, class_map):
        img_path = self.input_path.get().strip()
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        if yolo is not None:
            res = yolo.predict(source=img, verbose=False)[0]
            pts = pick_max_conf_per_class(res, class_map, conf_thres=cfg.conf_thres)
        else:
            self.println("YOLO未指定のため、ダミー座標で推論します（精度検証用途）。")
            pts = detect_dummy(img)

        missing = [k for k in REQ_KEYS if k not in pts]
        if missing:
            raise RuntimeError(f"必要な点が検出できません: {missing}\nYOLOのクラスID設定/学習を確認してください。")

        out = infer_from_points(models, pts, cfg)
        self.println("-" * 40)
        self.println(f"Pitch (前傾): {out['pitch']:.2f} deg")
        self.println(f"Yaw   (内旋): {out['yaw']:.2f} deg")
        self.println(f"Roll  (上方回旋): {out['roll']:.2f} deg")
        self.println("-" * 40)

        if cfg.preview:
            vis = draw_overlay(img, pts, out)
            cv2.imshow("ScapulaAnalysis - Image", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _run_video(self, cfg: InferenceConfig, models, yolo, class_map):
        vid_path = self.input_path.get().strip()
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {vid_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1

        writer = None
        if cfg.save_video:
            if not cfg.out_video_path:
                raise ValueError("Save annotated video をONにした場合、出力パスを指定してください。")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(cfg.out_video_path, fourcc, fps, (w, h))

        rows: List[Tuple[int, float, float, float]] = []
        last_pts: Optional[Dict[str, Tuple[float, float]]] = None
        ema_pts: Optional[Dict[str, Tuple[float, float]]] = None
        ema_out: Optional[Dict[str, float]] = None

        frame_idx = 0
        t0 = time.time()

        self.println(f"[VIDEO] {w}x{h}  fps={fps:.2f}  frames={total if total > 0 else 'unknown'}")

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            pts = None
            if yolo is not None:
                res = yolo.predict(source=frame, verbose=False)[0]
                pts = pick_max_conf_per_class(res, class_map, conf_thres=cfg.conf_thres)

            if pts is None or any(k not in pts for k in REQ_KEYS):
                if last_pts is not None:
                    pts = last_pts
                else:
                    pts = detect_dummy(frame)

            last_pts = pts

            # smoothing keypoints
            if cfg.smooth_keypoints:
                if ema_pts is None:
                    ema_pts = dict(pts)
                else:
                    for k, v in pts.items():
                        if k in ema_pts:
                            ema_pts[k] = ema_point(ema_pts[k], v, cfg.alpha_kpt)
                        else:
                            ema_pts[k] = v
                pts_use = ema_pts
            else:
                pts_use = pts

            out = infer_from_points(models, pts_use, cfg)

            if cfg.smooth_angles:
                if ema_out is None:
                    ema_out = dict(out)
                else:
                    ema_out = ema_angles(ema_out, out, cfg.alpha_ang)
                out_use = ema_out
            else:
                out_use = out

            rows.append((frame_idx, out_use["pitch"], out_use["yaw"], out_use["roll"]))

            vis = draw_overlay(frame, pts_use, out_use)
            if cfg.preview:
                cv2.imshow("ScapulaAnalysis - Video (press q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break

            if writer is not None:
                writer.write(vis)

            frame_idx += 1
            if frame_idx % 30 == 0:
                dt = time.time() - t0
                self.println(f"[{frame_idx} frames] elapsed={dt:.1f}s  ~{frame_idx / max(dt, 1e-6):.1f} fps(proc)")

        cap.release()
        if writer is not None:
            writer.release()
        if cfg.preview:
            cv2.destroyAllWindows()

        if cfg.out_csv_path:
            import csv

            os.makedirs(os.path.dirname(cfg.out_csv_path) or ".", exist_ok=True)
            with open(cfg.out_csv_path, "w", newline="", encoding="utf-8") as f:
                wtr = csv.writer(f)
                wtr.writerow(["frame", "pitch", "yaw", "roll"])
                wtr.writerows(rows)
            self.println(f"[CSV] saved: {cfg.out_csv_path}")

        arr = np.array(rows, dtype=float)
        if arr.size > 0:
            pitch = arr[:, 1]
            yaw = arr[:, 2]
            roll = arr[:, 3]
            self.println("-" * 40)
            self.println(f"Frames processed: {len(rows)}")
            self.println(f"Pitch mean±std: {pitch.mean():.2f} ± {pitch.std():.2f}")
            self.println(f"Yaw   mean±std: {yaw.mean():.2f} ± {yaw.std():.2f}")
            self.println(f"Roll  mean±std: {roll.mean():.2f} ± {roll.std():.2f}")
            self.println("-" * 40)
        else:
            self.println("[WARN] no frames processed.")


if __name__ == "__main__":
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
