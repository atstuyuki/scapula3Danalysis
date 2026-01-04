
from __future__ import annotations
import argparse
import csv
import os
from datetime import datetime
import numpy as np
import pyvista as pv

from scapula_analysis.dicom_utils import load_dicom_series, voxel_to_patient

def build_surface(series, iso_value: float = 200.0):
    # PyVistaは軸が直交前提なので、表示は voxel( index )空間の等方格子として扱う
    vol = series.volume_hu  # (Z,Y,X)
    sx, sy, sz = series.spacing  # x,y,z

    # PyVista UniformGrid expects (X,Y,Z)
    grid = pv.ImageData(
        dimensions=(vol.shape[2], vol.shape[1], vol.shape[0]),
        spacing=(sx, sy, sz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["values"] = vol.transpose(2,1,0).ravel(order="F")  # (X,Y,Z) in Fortran order
    surf = grid.contour([iso_value], scalars="values")
    return surf, (sx, sy, sz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", required=True, help="DICOM series folder (one patient)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (append if exists)")
    ap.add_argument("--patient_id", required=True, help="Patient ID label to store in CSV")
    ap.add_argument("--iso", type=float, default=200.0, help="Iso threshold (HU)")
    args = ap.parse_args()

    series = load_dicom_series(args.dicom_dir)
    surf, spacing = build_surface(series, iso_value=args.iso)

    picked = []

    def _callback(point):
        # point is in displayed world coords where origin=(0,0,0) and axes aligned to voxel indices * spacing
        picked.append(point)
        print(f"Picked {len(picked)}/3: {point}")
        if len(picked) >= 3:
            plotter.close()

    plotter = pv.Plotter()
    plotter.add_text("Click 3 points on scapula surface", font_size=14)
    plotter.add_mesh(surf, opacity=0.7)
    plotter.enable_point_picking(callback=_callback, use_mesh=True, show_message=True, show_point=True)
    plotter.show()

    if len(picked) != 3:
        raise RuntimeError("You must pick exactly 3 points.")

    sx, sy, sz = spacing
    # convert displayed coords -> voxel indices
    vox = []
    for (xw, yw, zw) in picked:
        x = xw / sx
        y = yw / sy
        z = zw / sz
        vox.append((x, y, z))

    # voxel -> patient coord (mm) using DICOM orientation
    pts_mm = [voxel_to_patient(x, y, z, series) for (x,y,z) in vox]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["patient_id", "timestamp",
                        "p1_x","p1_y","p1_z",
                        "p2_x","p2_y","p2_z",
                        "p3_x","p3_y","p3_z"])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [args.patient_id, ts] + [v for p in pts_mm for v in p.tolist()]
        w.writerow(row)

    print(f"Saved 3D points to: {args.out_csv}")

if __name__ == "__main__":
    main()
