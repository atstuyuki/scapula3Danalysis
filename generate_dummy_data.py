"""generate_dummy_data.py

ScapulaAnalysis 用のダミーデータ生成スクリプト。

このスクリプトを実行すると以下を生成します。

生成物
------
1) 患者2名分の胸部CT（DICOMシリーズ）
   - data/ct_dicoms/Patient001/*.dcm
   - data/ct_dicoms/Patient002/*.dcm

2) 検証用の疑似レントゲン（CTからの簡易DRR）
   - data/xrays/test_patient001.png
   - data/xrays/test_patient002.png

3) （オプション）YOLO 用の学習データセット（images/labels + data.yaml）
   - data/yolo_dummy/

目的
----
- パイプライン（Phase1〜Phase5）疎通のための「完全に人工」のダミーデータです。
- 医学的な正確さはありません（幾何学形状で肩甲骨と鎖骨を近似）。

使い方
------
(1) CT + X-ray だけ作る:
    python generate_dummy_data.py

(2) ついでに YOLO 学習データも作る（おすすめ）:
    python generate_dummy_data.py --make_yolo_dataset --n_yolo 300

その後:
    python train_dummy_yolo.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, generate_uid, ImplicitVRLittleEndian


# -------------------------
# Config
# -------------------------

DATA_DIR = "data"
CT_DIR = os.path.join(DATA_DIR, "ct_dicoms")
XRAY_DIR = os.path.join(DATA_DIR, "xrays")
YOLO_DIR = os.path.join(DATA_DIR, "yolo_dummy")


@dataclass
class DummyCTSpec:
    shape: Tuple[int, int, int] = (128, 160, 160)  # (Z,Y,X)
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (row_mm, col_mm, slice_mm)


# -------------------------
# Volume synthesis (CT)
# -------------------------

def _add_sphere(vol: np.ndarray, center_xyz: np.ndarray, radius: float, value_hu: int) -> None:
    """vol is (Z,Y,X). center is voxel index space (x,y,z)."""
    zdim, ydim, xdim = vol.shape
    zz, yy, xx = np.ogrid[:zdim, :ydim, :xdim]
    cx, cy, cz = center_xyz.astype(float)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
    vol[mask] = value_hu


def _add_thick_segment(vol: np.ndarray, p0_xyz: np.ndarray, p1_xyz: np.ndarray, radius: float, value_hu: int) -> None:
    length = float(np.linalg.norm(p1_xyz - p0_xyz))
    n = max(8, int(length * 2))
    for t in np.linspace(0.0, 1.0, n):
        p = p0_xyz + t * (p1_xyz - p0_xyz)
        _add_sphere(vol, p, radius, value_hu)


def create_dummy_scapula(spec: DummyCTSpec, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (HU volume (Z,Y,X), keypoints in voxel index (x,y,z))."""
    rng = np.random.default_rng(seed)
    zdim, ydim, xdim = spec.shape

    vol = np.full(spec.shape, -1000, dtype=np.int16)  # air

    # center
    cz, cy, cx = zdim // 2, ydim // 2, xdim // 2

    # small jitter per patient / sample
    jitter = rng.normal(0.0, 2.0, size=(5, 3))

    # Keypoints (voxel x,y,z) - scapula triangle + clavicle segment
    p_sup = np.array([cx - 26, cy + 0, cz + 34], dtype=float) + jitter[0]
    p_inf = np.array([cx - 14, cy + 0, cz - 44], dtype=float) + jitter[1]
    p_lat = np.array([cx + 34, cy + 10, cz + 10], dtype=float) + jitter[2]

    c_med = np.array([cx + 46, cy - 22, cz + 34], dtype=float) + jitter[3]
    c_lat = np.array([cx -  6, cy - 22, cz + 40], dtype=float) + jitter[4]

    bone_hu = 400

    # vertices
    _add_sphere(vol, p_sup, 8, bone_hu)
    _add_sphere(vol, p_inf, 8, bone_hu)
    _add_sphere(vol, p_lat, 10, bone_hu)

    # edges
    _add_thick_segment(vol, p_sup, p_inf, 5, bone_hu)
    _add_thick_segment(vol, p_inf, p_lat, 5, bone_hu)
    _add_thick_segment(vol, p_lat, p_sup, 5, bone_hu)

    # clavicle
    _add_thick_segment(vol, c_med, c_lat, 4, bone_hu)

    kps = {
        "superior": p_sup,
        "inferior": p_inf,
        "lateral": p_lat,
        "clavicle_med": c_med,
        "clavicle_lat": c_lat,
    }
    return vol, kps


# Backward-compatible wrapper (used nowhere else, but kept for safety)
def create_dummy_scapula_volume(spec: DummyCTSpec, seed: int) -> np.ndarray:
    vol, _ = create_dummy_scapula(spec, seed)
    return vol


# -------------------------
# DICOM series writer
# -------------------------

def save_dicom_series(volume_hu: np.ndarray, patient_id: str, out_dir: str, spec: DummyCTSpec) -> None:
    os.makedirs(out_dir, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    for_uid = generate_uid()

    # File meta template
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = UID("1.2.3.4.5.6.7.8.9")

    zdim, ydim, xdim = volume_hu.shape
    row_mm, col_mm, slice_mm = spec.spacing_mm

    # Axial orientation: rows->X, cols->Y (simple)
    iop = [1.0, 0.0, 0.0,   0.0, 1.0, 0.0]

    # origin in patient mm (dummy)
    origin = np.array([0.0, 0.0, 0.0], dtype=float)

    for iz in range(zdim):
        sop_uid = generate_uid()

        ds = FileDataset(
            filename_or_obj=os.path.join(out_dir, f"slice_{iz:03d}.dcm"),
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Keep UID consistency within the series
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid

        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = iz + 1

        ds.PatientName = patient_id
        ds.PatientID = patient_id

        ds.Rows = ydim
        ds.Columns = xdim
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed

        ds.PixelSpacing = [row_mm, col_mm]
        ds.SliceThickness = slice_mm

        # Position: move along slice direction (Z)
        ipp = origin + np.array([0.0, 0.0, float(iz) * slice_mm])
        ds.ImagePositionPatient = [float(ipp[0]), float(ipp[1]), float(ipp[2])]
        ds.ImageOrientationPatient = iop

        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0

        # pixel array should be (Rows, Cols) == (Y,X)
        ds.PixelData = volume_hu[iz, :, :].astype(np.int16).tobytes()

        # Make file_meta consistent with SOPInstanceUID
        ds.file_meta.MediaStorageSOPInstanceUID = sop_uid

        ds.is_little_endian = True
        ds.is_implicit_VR = True

        ds.save_as(ds.filename, write_like_original=False)

    print(f"Generated DICOM series for {patient_id}: {out_dir}")


# -------------------------
# X-ray (DRR-like) + keypoint projection
# -------------------------

def project_keypoints_to_xray(kps_xyz: Dict[str, np.ndarray], spec: DummyCTSpec) -> Dict[str, Tuple[float, float]]:
    """Project voxel (x,y,z) -> image (x, y) where image is (Z,X) from Y-projection."""
    out = {}
    for name, p in kps_xyz.items():
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        # projection along Y => keep X and Z
        out[name] = (x, z)
    return out


def create_synthetic_xray(volume_hu: np.ndarray, out_png: str) -> None:
    """Simple DRR-like projection. Output is GRAYSCALE png (uint8)."""
    dens = np.clip((volume_hu.astype(np.float32) + 1000.0) / 1400.0, 0.0, 1.0)
    path = np.sum(dens, axis=1)  # (Z,X)
    k = 0.06
    raw = np.exp(-k * path)
    img = 1.0 - raw
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img8 = (img * 255.0).astype(np.uint8)
    img8 = cv2.GaussianBlur(img8, (3, 3), 0)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, img8)
    print(f"Generated synthetic X-ray: {out_png}")


# -------------------------
# YOLO dataset generation
# -------------------------

YOLO_NAMES = ["superior", "inferior", "lateral", "clavicle_med", "clavicle_lat"]


def _write_yolo_label_txt(label_path: str, kps_2d: Dict[str, Tuple[float, float]], img_w: int, img_h: int, box_px: int = 14) -> None:
    """Write YOLO bbox labels around each keypoint."""
    half = box_px / 2.0
    lines = []
    for cls_id, name in enumerate(YOLO_NAMES):
        if name not in kps_2d:
            continue
        x, y = kps_2d[name]
        x0, y0 = x - half, y - half
        x1, y1 = x + half, y + half

        # clip
        x0 = max(0.0, min(float(img_w - 1), x0))
        x1 = max(0.0, min(float(img_w - 1), x1))
        y0 = max(0.0, min(float(img_h - 1), y0))
        y1 = max(0.0, min(float(img_h - 1), y1))

        xc = (x0 + x1) / 2.0 / img_w
        yc = (y0 + y1) / 2.0 / img_h
        bw = (x1 - x0) / img_w
        bh = (y1 - y0) / img_h

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _write_data_yaml(yolo_dir: str) -> None:
    import yaml
    d = {
        "path": os.path.abspath(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(YOLO_NAMES)},
    }
    os.makedirs(yolo_dir, exist_ok=True)
    with open(os.path.join(yolo_dir, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)


def make_yolo_dataset(spec: DummyCTSpec, n_yolo: int, seed: int = 123) -> None:
    """Generate a synthetic keypoint detection dataset for YOLO."""
    os.makedirs(YOLO_DIR, exist_ok=True)
    _write_data_yaml(YOLO_DIR)

    rng = np.random.default_rng(seed)

    # Train samples: purely synthetic variations
    n_train = max(10, int(n_yolo * 0.8))
    n_val = max(2, n_yolo - n_train)

    for split, n in [("train", n_train), ("val", n_val)]:
        for i in range(n):
            s = int(rng.integers(0, 10_000_000))
            vol, kps3 = create_dummy_scapula(spec, s)
            img = _drr_from_volume(vol)  # uint8 (H=Z, W=X)
            kps2 = project_keypoints_to_xray(kps3, spec)

            img_name = f"syn_{split}_{i:04d}.png"
            img_path = os.path.join(YOLO_DIR, "images", split, img_name)
            lab_path = os.path.join(YOLO_DIR, "labels", split, img_name.replace(".png", ".txt"))
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            cv2.imwrite(img_path, img)
            _write_yolo_label_txt(lab_path, kps2, img.shape[1], img.shape[0], box_px=14)

    # Also register the 2 test patient images as extra val examples (domain match)
    for pid in ["Patient001", "Patient002"]:
        test_png = os.path.join(XRAY_DIR, f"test_{pid.lower()}.png")
        if os.path.exists(test_png):
            img = cv2.imread(test_png, cv2.IMREAD_GRAYSCALE)
            # For these, regenerate kps from the same patient seed (stable)
            patient_seed = 1001 if pid == "Patient001" else 2002
            _, kps3 = create_dummy_scapula(spec, patient_seed)
            kps2 = project_keypoints_to_xray(kps3, spec)

            dst_img = os.path.join(YOLO_DIR, "images", "val", f"test_{pid.lower()}.png")
            dst_lab = os.path.join(YOLO_DIR, "labels", "val", f"test_{pid.lower()}.txt")
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            cv2.imwrite(dst_img, img)
            _write_yolo_label_txt(dst_lab, kps2, img.shape[1], img.shape[0], box_px=16)

    print(f"Generated YOLO dataset: {YOLO_DIR}")
    print(f"  data.yaml: {os.path.join(YOLO_DIR, 'data.yaml')}")


def _drr_from_volume(volume_hu: np.ndarray) -> np.ndarray:
    """Return DRR uint8 image (H=Z, W=X)."""
    dens = np.clip((volume_hu.astype(np.float32) + 1000.0) / 1400.0, 0.0, 1.0)
    path = np.sum(dens, axis=1)
    k = 0.06
    raw = np.exp(-k * path)
    img = 1.0 - raw
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img8 = (img * 255.0).astype(np.uint8)
    img8 = cv2.GaussianBlur(img8, (3, 3), 0)
    return img8


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_yolo_dataset", action="store_true", help="YOLO 학習用データセットも作る")
    ap.add_argument("--n_yolo", type=int, default=300, help="YOLO用合成画像数（目安）")
    ap.add_argument("--seed", type=int, default=123, help="乱数seed")
    ap.add_argument("--make_annotations_csv", action="store_true", help="Phase2用に data/annotations_3d.csv を自動生成")
    args = ap.parse_args()

    os.makedirs(CT_DIR, exist_ok=True)
    os.makedirs(XRAY_DIR, exist_ok=True)

    spec = DummyCTSpec()

    # Two patients with fixed seeds -> reproducible
    patient_seeds = {"Patient001": 1001, "Patient002": 2002}

    rows_for_csv = []

    for pid, s in patient_seeds.items():
        vol, kps = create_dummy_scapula(spec, s)
        save_dicom_series(vol, pid, os.path.join(CT_DIR, pid), spec)

        out_png = os.path.join(XRAY_DIR, f"test_{pid.lower()}.png")
        create_synthetic_xray(vol, out_png)
        if args.make_annotations_csv:
            # Convert voxel keypoints (x,y,z) -> patient mm
            row_mm, col_mm, slice_mm = spec.spacing_mm
            def v2p(v):
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                # dicom_utils.voxel_to_patient with our dummy orientation (row_dir=[1,0,0], col_dir=[0,1,0], normal=[0,0,1])
                return (y * row_mm, x * col_mm, z * slice_mm)
            p1 = v2p(kps["superior"])
            p2 = v2p(kps["inferior"])
            p3 = v2p(kps["lateral"])
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_for_csv.append({
                "patient_id": pid,
                "timestamp": ts,
                "p1_x": p1[0], "p1_y": p1[1], "p1_z": p1[2],
                "p2_x": p2[0], "p2_y": p2[1], "p2_z": p2[2],
                "p3_x": p3[0], "p3_y": p3[1], "p3_z": p3[2],
            })

    if args.make_annotations_csv:
        import csv
        out_csv = os.path.join(DATA_DIR, "annotations_3d.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "patient_id","timestamp",
                "p1_x","p1_y","p1_z",
                "p2_x","p2_y","p2_z",
                "p3_x","p3_y","p3_z",
            ])
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f"  {out_csv}  (auto annotations for Phase2)")

    if args.make_yolo_dataset:
        make_yolo_dataset(spec, n_yolo=args.n_yolo, seed=args.seed)

    print("\nDone! Data structure:")
    print(f"  {CT_DIR}/Patient001/*.dcm")
    print(f"  {CT_DIR}/Patient002/*.dcm")
    print(f"  {XRAY_DIR}/test_patient001.png, test_patient002.png")
    if args.make_yolo_dataset:
        print(f"  {YOLO_DIR}/ (YOLO dataset)")
    print("\nNext:")
    print("  1) python 01_ct_annotation.py --dicom_dir data/ct_dicoms/Patient001 --out_csv ...")
    print("  2) python train_dummy_yolo.py")
    print("  3) python 04_inference.py --img data/xrays/test_patient001.png --yolo runs/detect/train/weights/best.pt --class_yaml data/yolo_dummy/data.yaml --draw")


if __name__ == "__main__":
    main()
