from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from scapula_analysis.camera import euler_xyz_to_R, project_points
from scapula_analysis.features import extract_features
from scapula_analysis.ssm import load_ssm, sample_shape


def make_dataset(
    ssm_path: str,
    n_samples: int,
    noise_sigma: float,
    seed: int,
    *,
    f: float,
    z0: float,
    cx: float,
    cy: float,
    z0_range: tuple[float, float] | None = None,
    include_clavicle_features: bool = False,
    include_clavicle_scale_hint: bool = False,
    t_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch_range: tuple[float, float] = (-20, 20),
    yaw_range: tuple[float, float] = (-30, 30),
    roll_range: tuple[float, float] = (0, 60),
    coeff_sigma: float = 1.0,
):
    """Generate synthetic (features -> angles) dataset from SSM.

    If `z0_range` is given, we randomize camera distance per-sample. This helps
    robustness to perspective effects.

    If `include_clavicle_features` is True, we synthesize a plausible clavicle line
    from scapula centroid + scale, then project it and include it in features.

    Returns
    -------
    X : (n_samples, n_features)
    Y : (n_samples, 3)  [pitch, yaw, roll]
    meta : dict (camera z0 stats etc.)
    """

    rng = np.random.default_rng(seed)
    ssm = load_ssm(ssm_path)
    pca = ssm.pca

    X: list[np.ndarray] = []
    Y: list[list[float]] = []

    std_devs = np.sqrt(pca.explained_variance_)
    t = np.array(t_xyz, dtype=float)

    z0_used: list[float] = []

    for _ in range(n_samples):
        coeffs = rng.normal(0, coeff_sigma, size=pca.n_components_) * std_devs
        shape = sample_shape(ssm, coeffs)  # (3,3)

        pitch = float(rng.uniform(*pitch_range))
        yaw = float(rng.uniform(*yaw_range))
        roll = float(rng.uniform(*roll_range))
        R = euler_xyz_to_R(pitch, yaw, roll, degrees=True)

        z0_i = float(rng.uniform(*z0_range)) if z0_range is not None else float(z0)
        z0_used.append(z0_i)

        if include_clavicle_features:
            centroid3 = np.mean(shape, axis=0)
            # 3D representative scale (mean edge length)
            s3d = (
                np.linalg.norm(shape[0] - shape[1])
                + np.linalg.norm(shape[1] - shape[2])
                + np.linalg.norm(shape[2] - shape[0])
            ) / 3.0
            s3d = float(max(s3d, 1e-6))
            # A simple, plausible clavicle segment above the scapula centroid
            c_med = centroid3 + np.array([-0.8 * s3d, -0.6 * s3d, 0.0])
            c_lat = centroid3 + np.array([+0.8 * s3d, -0.6 * s3d, 0.0])
            pts3 = np.vstack([shape, c_med[None, :], c_lat[None, :]])  # (5,3)
            pts2_all = project_points(pts3, R=R, t=t, f=f, z0=z0_i, cx=cx, cy=cy)
            pts2_all = pts2_all + rng.normal(0, noise_sigma, size=pts2_all.shape)
            pts2_scap = pts2_all[:3]
            pts2_clav = pts2_all[3:]
            feats = extract_features(
                pts2_scap,
                clavicle_points_2d=pts2_clav,
                include_clavicle_scale_hint=include_clavicle_scale_hint,
            )
        else:
            pts2 = project_points(shape, R=R, t=t, f=f, z0=z0_i, cx=cx, cy=cy)
            pts2 = pts2 + rng.normal(0, noise_sigma, size=pts2.shape)
            feats = extract_features(pts2)

        X.append(feats)
        Y.append([pitch, yaw, roll])

    meta = {
        "z0_used_min": float(np.min(z0_used)) if z0_used else float(z0),
        "z0_used_max": float(np.max(z0_used)) if z0_used else float(z0),
        "z0_used_mean": float(np.mean(z0_used)) if z0_used else float(z0),
    }

    return np.asarray(X, float), np.asarray(Y, float), meta


def train_one_target(X, y, params, *, seed: int, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for _, (tr, va) in enumerate(kf.split(X), start=1):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va])
        m = lgb.train(
            params,
            dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        score = m.best_score["val"][params["metric"]]
        fold_scores.append(float(score))
        best_iters.append(int(m.best_iteration))

    final_iter = int(np.median(best_iters))
    dall = lgb.Dataset(X, label=y)
    final = lgb.train(params, dall, num_boost_round=final_iter)
    return final, {"cv_scores": fold_scores, "best_iters": best_iters, "final_iter": final_iter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=50000)
    ap.add_argument("--noise_sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--f", type=float, default=1000.0)
    ap.add_argument("--z0", type=float, default=1000.0)
    ap.add_argument("--z0_min", type=float, default=None)
    ap.add_argument("--z0_max", type=float, default=None)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)

    ap.add_argument("--include_clavicle_features", action="store_true")
    ap.add_argument("--include_clavicle_scale_hint", action="store_true")

    args = ap.parse_args()

    ssm_path = os.path.join(args.model_dir, "scapula_ssm_pca.joblib")
    if not os.path.exists(ssm_path):
        raise FileNotFoundError(f"SSM not found: {ssm_path}. Run 02_build_ssm.py first.")

    z0_range = None
    if args.z0_min is not None or args.z0_max is not None:
        if args.z0_min is None or args.z0_max is None:
            raise ValueError("--z0_min and --z0_max must be set together")
        if args.z0_max <= args.z0_min:
            raise ValueError("--z0_max must be greater than --z0_min")
        z0_range = (float(args.z0_min), float(args.z0_max))

    X, Y, meta = make_dataset(
        ssm_path=ssm_path,
        n_samples=args.n_samples,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        f=args.f,
        z0=args.z0,
        cx=args.cx,
        cy=args.cy,
        z0_range=z0_range,
        include_clavicle_features=bool(args.include_clavicle_features),
        include_clavicle_scale_hint=bool(args.include_clavicle_scale_hint),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    params = {
        "objective": "regression",
        "metric": "l1",  # MAE
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": args.seed,
    }

    targets = ["pitch", "yaw", "roll"]
    metrics = {
        "dataset": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
        "camera": {
            "f": float(args.f),
            "z0": float(args.z0),
            "z0_range": list(z0_range) if z0_range else None,
            "z0_used_stats": meta,
            "cx": float(args.cx),
            "cy": float(args.cy),
        },
        "noise_sigma": float(args.noise_sigma),
        "seed": int(args.seed),
        "include_clavicle_features": bool(args.include_clavicle_features),
        "include_clavicle_scale_hint": bool(args.include_clavicle_scale_hint),
        "targets": {},
    }

    for ti, name in enumerate(targets):
        print(f"Training {name} ...")
        model, info = train_one_target(X, Y[:, ti], params, seed=args.seed, n_splits=5)
        out_path = os.path.join(args.out_dir, f"lgbm_{name}.txt")
        model.save_model(out_path)
        metrics["targets"][name] = info
        print(f"  saved: {out_path}  (CV MAE mean={np.mean(info['cv_scores']):.3f})")

    with open(os.path.join(args.out_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Done. Metrics saved to train_metrics.json")


if __name__ == "__main__":
    main()
