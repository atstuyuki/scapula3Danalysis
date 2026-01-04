
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd

from scapula_analysis.gpa import generalized_procrustes
from scapula_analysis.ssm import fit_ssm, save_ssm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--n_components", type=int, default=3)
    ap.add_argument("--allow_scaling", action="store_true", help="If set, GPA also normalizes scale (not recommended for this pipeline)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    req = ["p1_x","p1_y","p1_z","p2_x","p2_y","p2_z","p3_x","p3_y","p3_z"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    shapes = []
    for _, r in df.iterrows():
        pts = np.array([
            [r.p1_x, r.p1_y, r.p1_z],
            [r.p2_x, r.p2_y, r.p2_z],
            [r.p3_x, r.p3_y, r.p3_z],
        ], dtype=float)
        shapes.append(pts)

    mean, aligned, transforms = generalized_procrustes(shapes, allow_scaling=args.allow_scaling)
    ssm = fit_ssm(aligned, n_components=args.n_components)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "scapula_ssm_pca.joblib")
    save_ssm(ssm, model_path)

    meta = {
        "n_shapes": len(shapes),
        "n_points": int(shapes[0].shape[0]),
        "n_components": int(ssm.pca.n_components_),
        "explained_variance_ratio": ssm.pca.explained_variance_ratio_.tolist(),
        "allow_scaling_in_gpa": bool(args.allow_scaling),
    }
    with open(os.path.join(args.model_dir, "ssm_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved SSM PCA to: {model_path}")
    print("Meta:", meta)

if __name__ == "__main__":
    main()
