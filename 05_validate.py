
from __future__ import annotations
import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from scapula_analysis.features import extract_features
from scapula_analysis.ssm import load_ssm, sample_shape
from scapula_analysis.camera import euler_xyz_to_R, project_points

def simulate(ssm_path: str, n_samples: int, noise_sigma: float, seed: int, f: float, z0: float, cx: float, cy: float):
    rng = np.random.default_rng(seed)
    ssm = load_ssm(ssm_path)
    pca = ssm.pca
    std_devs = np.sqrt(pca.explained_variance_)
    X, Y = [], []

    t = np.zeros(3)
    for _ in range(n_samples):
        coeffs = rng.normal(0, 1.0, size=pca.n_components_) * std_devs
        shape = sample_shape(ssm, coeffs)

        pitch = rng.uniform(-20, 20)
        yaw = rng.uniform(-30, 30)
        roll = rng.uniform(0, 60)
        R = euler_xyz_to_R(pitch, yaw, roll, degrees=True)

        pts2 = project_points(shape, R=R, t=t, f=f, z0=z0, cx=cx, cy=cy)
        pts2 = pts2 + rng.normal(0, noise_sigma, size=pts2.shape)

        X.append(extract_features(pts2))
        Y.append([pitch, yaw, roll])

    return np.asarray(X, float), np.asarray(Y, float)

def mae(a, b):
    return float(np.mean(np.abs(a-b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f", type=float, default=1000.0)
    ap.add_argument("--z0", type=float, default=1000.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)
    args = ap.parse_args()

    ssm_path = os.path.join(args.model_dir, "scapula_ssm_pca.joblib")
    if not os.path.exists(ssm_path):
        raise FileNotFoundError("SSM not found. Run 02_build_ssm.py first.")
    models = {}
    for t in ["pitch","yaw","roll"]:
        p = os.path.join(args.model_dir, f"lgbm_{t}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError("Models not found. Run 03_train_lgbm.py first.")
        models[t] = lgb.Booster(model_file=p)

    os.makedirs(args.out_dir, exist_ok=True)

    sigmas = [0.0, 1.0, 2.0, 3.0, 5.0]
    rows = []
    for s in sigmas:
        X, Y = simulate(ssm_path, args.n_samples, noise_sigma=s, seed=args.seed, f=args.f, z0=args.z0, cx=args.cx, cy=args.cy)
        pred = np.column_stack([models[t].predict(X) for t in ["pitch","yaw","roll"]])
        maes = [mae(pred[:,i], Y[:,i]) for i in range(3)]
        rows.append([s] + maes + [float(np.mean(maes))])
        print(f"sigma={s:.1f}  MAE pitch/yaw/roll = {maes[0]:.2f}, {maes[1]:.2f}, {maes[2]:.2f}  mean={np.mean(maes):.2f}")

    csv_path = os.path.join(args.out_dir, "noise_sweep_mae.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_sigma", "mae_pitch", "mae_yaw", "mae_roll", "mae_mean"])
        w.writerows(rows)

    # plot
    sig = [r[0] for r in rows]
    mean_mae = [r[-1] for r in rows]
    plt.figure()
    plt.plot(sig, mean_mae, marker="o")
    plt.xlabel("Noise sigma (px)")
    plt.ylabel("Mean MAE (deg)")
    plt.title("Noise sweep (mean MAE)")
    png_path = os.path.join(args.out_dir, "noise_sweep_mean_mae.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    meta = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "camera": {"f": args.f, "z0": args.z0, "cx": args.cx, "cy": args.cy},
        "sigmas": sig,
        "mean_mae": mean_mae,
        "outputs": {"csv": os.path.basename(csv_path), "png": os.path.basename(png_path)}
    }
    with open(os.path.join(args.out_dir, "validate_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")

if __name__ == "__main__":
    main()
