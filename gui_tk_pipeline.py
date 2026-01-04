#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScapulaAnalysis Tkinter GUI Launcher (Phase1〜5)

目的
- Phase1〜Phase5 を1つのGUIから実行できるようにする
- ログ表示、簡易進捗（indeterminate）、Stop（プロセス停止）

使い方
- まず Model Dir / Output Dir などを入力
- 各PhaseのRunを押す

備考
- Phase1 は PyVista のクリックUIが別ウィンドウで開きます
- 長い処理はサブプロセスで実行するためGUIが固まりません
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


def py_exe() -> str:
    # venv環境ならその python を使う
    return sys.executable or "python"


@dataclass
class Cmd:
    title: str
    argv: list[str]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ScapulaAnalysis GUI Launcher (Phase1-5)")
        self.geometry("1040x780")

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()

        # common
        self.project_dir = tk.StringVar(value=os.path.abspath(os.path.dirname(__file__)))

        # phase1
        self.p1_dicom_dir = tk.StringVar(value="")
        self.p1_out_csv = tk.StringVar(value="data/annotations_3d.csv")
        self.p1_patient_id = tk.StringVar(value="Patient001")
        self.p1_iso = tk.StringVar(value="200")

        # phase2
        self.p2_in_csv = tk.StringVar(value="data/annotations_3d.csv")
        self.p2_model_dir = tk.StringVar(value="data/models")
        self.p2_allow_scaling = tk.BooleanVar(value=False)

        # phase3
        self.p3_model_dir = tk.StringVar(value="data/models")
        self.p3_out_dir = tk.StringVar(value="data/models")
        self.p3_n_samples = tk.StringVar(value="50000")
        self.p3_noise = tk.StringVar(value="2.0")
        self.p3_seed = tk.StringVar(value="42")
        self.p3_f = tk.StringVar(value="1000")
        self.p3_z0 = tk.StringVar(value="1000")
        self.p3_z0_min = tk.StringVar(value="")
        self.p3_z0_max = tk.StringVar(value="")
        self.p3_include_clav = tk.BooleanVar(value=False)
        self.p3_clav_scale = tk.BooleanVar(value=False)

        # phase4 -> delegate to gui_tk_phase4.py (open separate window) OR run CLI
        self.p4_open_phase4_gui = tk.BooleanVar(value=True)

        # phase5
        self.p5_model_dir = tk.StringVar(value="data/models")
        self.p5_out_dir = tk.StringVar(value="outputs")
        self.p5_n_samples = tk.StringVar(value="20000")

        self._build()
        self.after(100, self._pump_logs)

    # ---------------- UI ----------------
    def _build(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Project dir:").pack(side="left")
        ttk.Entry(top, textvariable=self.project_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Browse", command=self._browse_project_dir).pack(side="left")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=8)

        nb.add(self._tab_phase1(nb), text="Phase1 CT→3D")
        nb.add(self._tab_phase2(nb), text="Phase2 SSM")
        nb.add(self._tab_phase3(nb), text="Phase3 Train")
        nb.add(self._tab_phase4(nb), text="Phase4 Inference")
        nb.add(self._tab_phase5(nb), text="Phase5 Validate")

        # actions + progress
        act = ttk.Frame(root)
        act.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(act, text="Stop running process", command=self.stop).pack(side="left")
        self.pbar = ttk.Progressbar(act, mode="indeterminate")
        self.pbar.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Button(act, text="Clear log", command=self._clear_log).pack(side="left")

        # log
        self.log = ScrolledText(root, height=18)
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._println("Ready.")

    def _tab_phase1(self, nb):
        f = ttk.Frame(nb)
        pad = {"padx": 8, "pady": 6}

        r1 = ttk.Frame(f); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="DICOM dir:", width=18).pack(side="left")
        ttk.Entry(r1, textvariable=self.p1_dicom_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=lambda: self._browse_dir(self.p1_dicom_dir)).pack(side="left")

        r2 = ttk.Frame(f); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Output CSV:", width=18).pack(side="left")
        ttk.Entry(r2, textvariable=self.p1_out_csv).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=lambda: self._browse_save_csv(self.p1_out_csv)).pack(side="left")

        r3 = ttk.Frame(f); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Patient ID:", width=18).pack(side="left")
        ttk.Entry(r3, textvariable=self.p1_patient_id, width=24).pack(side="left")
        ttk.Label(r3, text="iso(HU):").pack(side="left", padx=(18, 6))
        ttk.Entry(r3, textvariable=self.p1_iso, width=8).pack(side="left")

        r4 = ttk.Frame(f); r4.pack(fill="x", **pad)
        ttk.Button(r4, text="Run Phase1", command=self.run_phase1).pack(side="left")

        ttk.Button(r4, text="Generate Dummy (CT+X-ray)", command=lambda: self.run_dummy(make_yolo=False)).pack(side="left", padx=8)
        ttk.Button(r4, text="Generate Dummy (+YOLO dataset)", command=lambda: self.run_dummy(make_yolo=True)).pack(side="left")
        ttk.Button(r4, text="Train Dummy YOLO", command=self.run_train_dummy_yolo).pack(side="left", padx=8)

        ttk.Label(f, text="※PyVistaの別ウィンドウで3点をクリックして終了するとCSVに保存されます").pack(anchor="w", padx=10, pady=(10, 0))
        return f

    def _tab_phase2(self, nb):
        f = ttk.Frame(nb)
        pad = {"padx": 8, "pady": 6}

        r1 = ttk.Frame(f); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Input CSV:", width=18).pack(side="left")
        ttk.Entry(r1, textvariable=self.p2_in_csv).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=lambda: self._browse_open_csv(self.p2_in_csv)).pack(side="left")

        r2 = ttk.Frame(f); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Model dir:", width=18).pack(side="left")
        ttk.Entry(r2, textvariable=self.p2_model_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=lambda: self._browse_dir(self.p2_model_dir)).pack(side="left")

        r3 = ttk.Frame(f); r3.pack(fill="x", **pad)
        ttk.Checkbutton(r3, text="allow_scaling (GPA)", variable=self.p2_allow_scaling).pack(side="left")

        r4 = ttk.Frame(f); r4.pack(fill="x", **pad)
        ttk.Button(r4, text="Run Phase2", command=self.run_phase2).pack(side="left")
        return f

    def _tab_phase3(self, nb):
        f = ttk.Frame(nb)
        pad = {"padx": 8, "pady": 6}

        def row(label, var, browse=None):
            r = ttk.Frame(f); r.pack(fill="x", **pad)
            ttk.Label(r, text=label, width=18).pack(side="left")
            ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
            if browse:
                ttk.Button(r, text="Browse", command=browse).pack(side="left")

        row("SSM model dir:", self.p3_model_dir, lambda: self._browse_dir(self.p3_model_dir))
        row("Output dir:", self.p3_out_dir, lambda: self._browse_dir(self.p3_out_dir))

        r2 = ttk.Frame(f); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="n_samples:", width=18).pack(side="left")
        ttk.Entry(r2, textvariable=self.p3_n_samples, width=12).pack(side="left")
        ttk.Label(r2, text="noise_sigma:").pack(side="left", padx=(18, 6))
        ttk.Entry(r2, textvariable=self.p3_noise, width=8).pack(side="left")
        ttk.Label(r2, text="seed:").pack(side="left", padx=(18, 6))
        ttk.Entry(r2, textvariable=self.p3_seed, width=8).pack(side="left")

        r3 = ttk.Frame(f); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="camera f:", width=18).pack(side="left")
        ttk.Entry(r3, textvariable=self.p3_f, width=10).pack(side="left")
        ttk.Label(r3, text="z0:").pack(side="left", padx=(18, 6))
        ttk.Entry(r3, textvariable=self.p3_z0, width=10).pack(side="left")
        ttk.Label(r3, text="or z0_min/max:").pack(side="left", padx=(18, 6))
        ttk.Entry(r3, textvariable=self.p3_z0_min, width=8).pack(side="left")
        ttk.Label(r3, text="/").pack(side="left")
        ttk.Entry(r3, textvariable=self.p3_z0_max, width=8).pack(side="left")

        r4 = ttk.Frame(f); r4.pack(fill="x", **pad)
        ttk.Checkbutton(r4, text="include clavicle features", variable=self.p3_include_clav).pack(side="left")
        ttk.Checkbutton(r4, text="clavicle scale hint", variable=self.p3_clav_scale).pack(side="left", padx=(18, 0))

        r5 = ttk.Frame(f); r5.pack(fill="x", **pad)
        ttk.Button(r5, text="Run Phase3", command=self.run_phase3).pack(side="left")
        return f

    def _tab_phase4(self, nb):
        f = ttk.Frame(nb)
        pad = {"padx": 8, "pady": 6}

        ttk.Label(f, text="Phase4は専用GUI（gui_tk_phase4.py）を開くのが基本です。").pack(anchor="w", padx=10, pady=10)
        ttk.Checkbutton(f, text="Open Phase4 GUI", variable=self.p4_open_phase4_gui).pack(anchor="w", padx=10)
        ttk.Button(f, text="Launch Phase4 GUI", command=self.launch_phase4_gui).pack(anchor="w", padx=10, pady=10)
        ttk.Label(f, text="※Phase4 GUI では静止画/動画、平滑化、data.yaml、鎖骨特徴量を設定できます。").pack(anchor="w", padx=10)
        return f

    def _tab_phase5(self, nb):
        f = ttk.Frame(nb)
        pad = {"padx": 8, "pady": 6}

        r1 = ttk.Frame(f); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Model dir:", width=18).pack(side="left")
        ttk.Entry(r1, textvariable=self.p5_model_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=lambda: self._browse_dir(self.p5_model_dir)).pack(side="left")

        r2 = ttk.Frame(f); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Output dir:", width=18).pack(side="left")
        ttk.Entry(r2, textvariable=self.p5_out_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=lambda: self._browse_dir(self.p5_out_dir)).pack(side="left")

        r3 = ttk.Frame(f); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="n_samples:", width=18).pack(side="left")
        ttk.Entry(r3, textvariable=self.p5_n_samples, width=12).pack(side="left")

        r4 = ttk.Frame(f); r4.pack(fill="x", **pad)
        ttk.Button(r4, text="Run Phase5", command=self.run_phase5).pack(side="left")
        return f

    # ---------------- helpers ----------------
    def _browse_project_dir(self):
        d = filedialog.askdirectory(title="Select project directory")
        if d:
            self.project_dir.set(d)

    def _browse_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory(title="Select directory")
        if d:
            var.set(d)

    def _browse_open_csv(self, var: tk.StringVar):
        p = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            var.set(p)

    def _browse_save_csv(self, var: tk.StringVar):
        p = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            var.set(p)

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _println(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def _pump_logs(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._println(line)
        except queue.Empty:
            pass
        self.after(100, self._pump_logs)

    def _run_cmd_async(self, cmd: Cmd):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Running", "すでに別の処理が実行中です。Stopを押してから実行してください。")
            return

        self.stop_event.clear()
        self._println("=" * 72)
        self._println(f"[RUN] {cmd.title}")
        self._println(" " + " ".join(cmd.argv))

        self.pbar.start(10)

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd.argv,
                    cwd=self.project_dir.get(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.log_queue.put(line.rstrip("\n"))
                    if self.stop_event.is_set():
                        break
                if self.stop_event.is_set() and self.proc.poll() is None:
                    try:
                        self.proc.terminate()
                    except Exception:
                        pass
                rc = self.proc.wait()
                self.log_queue.put(f"[DONE] returncode={rc}")
            except Exception as e:
                self.log_queue.put(f"[ERROR] {e}")
            finally:
                self.pbar.stop()
                self.proc = None

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self._println("[STOP] requested")

    # ---------------- phase runners ----------------
    def run_phase1(self):
        dicom_dir = self.p1_dicom_dir.get().strip()
        if not dicom_dir or not os.path.isdir(dicom_dir):
            messagebox.showerror("Error", "Phase1: DICOM dir を指定してください")
            return
        out_csv = self.p1_out_csv.get().strip()
        pid = self.p1_patient_id.get().strip() or "Patient001"
        iso = self.p1_iso.get().strip() or "200"

        cmd = Cmd(
            title="Phase1 CT annotation",
            argv=[
                py_exe(),
                "01_ct_annotation.py",
                "--dicom_dir",
                dicom_dir,
                "--out_csv",
                out_csv,
                "--patient_id",
                pid,
                "--iso",
                iso,
            ],
        )
        self._run_cmd_async(cmd)

    
    def run_dummy(self, make_yolo: bool) -> None:
        args = [sys.executable, "generate_dummy_data.py"]
        if make_yolo:
            args += ["--make_yolo_dataset", "--n_yolo", "300"]
        self._run_subprocess(args, cwd=self.project_dir.get())

    def run_train_dummy_yolo(self) -> None:
        args = [sys.executable, "train_dummy_yolo.py", "--device", "cpu"]
        self._run_subprocess(args, cwd=self.project_dir.get())

def run_phase2(self):
        in_csv = self.p2_in_csv.get().strip()
        model_dir = self.p2_model_dir.get().strip()
        if not in_csv:
            messagebox.showerror("Error", "Phase2: Input CSV を指定してください")
            return
        if not model_dir:
            messagebox.showerror("Error", "Phase2: Model dir を指定してください")
            return

        argv = [py_exe(), "02_build_ssm.py", "--in_csv", in_csv, "--model_dir", model_dir]
        if self.p2_allow_scaling.get():
            argv.append("--allow_scaling")

        self._run_cmd_async(Cmd("Phase2 build SSM", argv))

    def run_phase3(self):
        model_dir = self.p3_model_dir.get().strip()
        out_dir = self.p3_out_dir.get().strip()
        if not model_dir or not out_dir:
            messagebox.showerror("Error", "Phase3: model_dir と out_dir を指定してください")
            return

        argv = [
            py_exe(),
            "03_train_lgbm.py",
            "--model_dir",
            model_dir,
            "--out_dir",
            out_dir,
            "--n_samples",
            self.p3_n_samples.get().strip() or "50000",
            "--noise_sigma",
            self.p3_noise.get().strip() or "2.0",
            "--seed",
            self.p3_seed.get().strip() or "42",
            "--f",
            self.p3_f.get().strip() or "1000",
            "--z0",
            self.p3_z0.get().strip() or "1000",
        ]

        z0min = self.p3_z0_min.get().strip()
        z0max = self.p3_z0_max.get().strip()
        if z0min and z0max:
            argv += ["--z0_min", z0min, "--z0_max", z0max]

        if self.p3_include_clav.get():
            argv.append("--include_clavicle_features")
        if self.p3_clav_scale.get():
            argv.append("--include_clavicle_scale_hint")

        self._run_cmd_async(Cmd("Phase3 train LightGBM", argv))

    def launch_phase4_gui(self):
        if not self.p4_open_phase4_gui.get():
            messagebox.showinfo("Info", "Open Phase4 GUI がOFFです")
            return
        argv = [py_exe(), "gui_tk_phase4.py"]
        # run detached-ish
        self._run_cmd_async(Cmd("Launch Phase4 GUI", argv))

    def run_phase5(self):
        model_dir = self.p5_model_dir.get().strip()
        out_dir = self.p5_out_dir.get().strip()
        if not model_dir or not out_dir:
            messagebox.showerror("Error", "Phase5: model_dir と out_dir を指定してください")
            return

        argv = [
            py_exe(),
            "05_validate.py",
            "--model_dir",
            model_dir,
            "--out_dir",
            out_dir,
            "--n_samples",
            self.p5_n_samples.get().strip() or "20000",
        ]
        self._run_cmd_async(Cmd("Phase5 validation", argv))


if __name__ == "__main__":
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
