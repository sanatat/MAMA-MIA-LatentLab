#!/usr/bin/env python3
"""Leakage-aware LOCO ablation for MAMA-MIA-LatentLab (research pilot only).

Compare the same RF hyperparameters on the same patients with:
  std_only = a single post-contrast tumor intensity-STD descriptor
  raw  = 12 original tabular MRI descriptors
  vae  = 4 unsupervised train-only VAE latent means
  both = raw descriptors concatenated with VAE latent means
  median = training-set median of the imaging-derived target

pCR is deliberately not used as a feature, prediction target, or model-selection criterion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from latentlab.images import CHEAP_FEATURES, TARGET
from latentlab.model import train_vae, project

MODES = ("median", "std_only", "raw", "vae", "both")


def parse_seeds(s):
    try:
        seeds = [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError("Pass comma-separated integer seeds") from e
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be nonempty and unique")
    return seeds


def load(path):
    df = pd.read_csv(path)
    needed = ["patient_id", "cohort", TARGET] + CHEAP_FEATURES
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.patient_id.isna().any() or df.patient_id.duplicated().any():
        raise ValueError("Missing/duplicated patient IDs; cannot trust holdout estimates")
    if df.cohort.isna().any():
        raise ValueError("Missing cohort labels")
    for c in CHEAP_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    if df[TARGET].isna().any():
        raise ValueError("Missing/nonfinite surrogate targets; inspect extraction")
    cohorts = sorted(df.cohort.astype(str).unique())
    if len(cohorts) < 3:
        raise ValueError("Need >=3 cohorts for the requested LOCO audit")
    if (df.groupby("cohort").size() < 3).any():
        raise ValueError("Every held-out cohort must have >=3 patients")
    return df, cohorts


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)) if len(y) > 1 else None}


def fit_predict(x_train, x_test, y_train, seed):
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                               n_jobs=-1, random_state=seed)
    rf.fit(x_train, y_train)
    return rf.predict(x_test)


def run(df, cohorts, seeds, epochs, latent):
    rows, preds, reco = [], [], []
    for seed in seeds:
        for cohort in cohorts:
            heldout = df.cohort.astype(str).to_numpy() == cohort
            tr, te = np.where(~heldout)[0], np.where(heldout)[0]
            if len(tr) < 10 or len(te) < 3:
                raise ValueError("Insufficient training or test patients")
            # The held-out cohort is never used when fitting imputation or scaling.
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            scaler = StandardScaler()
            xtr = scaler.fit_transform(imputer.fit_transform(df.iloc[tr][CHEAP_FEATURES])).astype('float32')
            xte = scaler.transform(imputer.transform(df.iloc[te][CHEAP_FEATURES])).astype('float32')
            ytr = df.iloc[tr][TARGET].to_numpy(float)
            yte = df.iloc[te][TARGET].to_numpy(float)
            # Train-only, unsupervised. No target or pCR is ever supplied to VAE.
            net, trace = train_vae(xtr, latent=latent, epochs=epochs, seed=seed)
            ztr, zte = project(net, xtr), project(net, xte)
            with __import__('torch').no_grad():
                import torch
                rec = net(torch.as_tensor(xte, dtype=torch.float32))[0].detach().numpy()
            reco.append({"seed": seed, "heldout_cohort": cohort,
                         "vae_train_loss": float(trace[-1]),
                         "heldout_reconstruction_mse": float(np.mean((rec - xte)**2))})
            pred = {
                "median": np.full(len(te), float(np.median(ytr))),
                "std_only": fit_predict(xtr[:, [CHEAP_FEATURES.index("post_tumor_std_z")]],
                                         xte[:, [CHEAP_FEATURES.index("post_tumor_std_z")]], ytr, seed),
                "raw": fit_predict(xtr, xte, ytr, seed),
                "vae": fit_predict(ztr, zte, ytr, seed),
                "both": fit_predict(np.column_stack((xtr,ztr)), np.column_stack((xte,zte)), ytr, seed)
            }
            for mode in MODES:
                score = metrics(yte, pred[mode])
                rows.append({"seed": seed, "heldout_cohort": cohort,
                             "n_train": len(tr), "n_test": len(te),
                             "model": mode, **score})
                for j, idx in enumerate(te):
                    preds.append({"seed": seed, "heldout_cohort": cohort,
                                  "patient_id": str(df.iloc[idx].patient_id),
                                  "model": mode, "y_true": float(yte[j]),
                                  "y_pred": float(pred[mode][j]),
                                  "abs_error": float(abs(yte[j] - pred[mode][j]))})
            print(f"seed={seed:<5} heldout={cohort:<6} n={len(te):<3} "
                  f"MAE raw={metrics(yte,pred['raw'])['mae']:.4f} "
                  f"vae={metrics(yte,pred['vae'])['mae']:.4f} "
                  f"both={metrics(yte,pred['both'])['mae']:.4f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds), pd.DataFrame(reco)


def bootstrap_gain(preds, seeds, cohorts, bootstraps):
    """Paired stratified patient bootstrap for MAE(raw)-MAE(both).

    We first average per-patient absolute error across seeds; we never count
    the same patient as an independent observation multiple times.
    Conditional on the four observed datasets, not an external-site CI.
    """
    pivot = preds.pivot_table(index=["heldout_cohort", "patient_id"],
                              columns="model", values="abs_error", aggfunc="mean")
    delta = (pivot["raw"] - pivot["both"]).rename("raw_minus_both").reset_index()
    rng = np.random.default_rng(20261020)
    arrays = [delta.loc[delta.heldout_cohort == c, "raw_minus_both"].to_numpy() for c in cohorts]
    sims = np.empty(bootstraps)
    for i in range(bootstraps):
        sims[i] = np.concatenate([a[rng.integers(0,len(a),len(a))] for a in arrays]).mean()
    return {"mae_gain_raw_minus_both": float(delta.raw_minus_both.mean()),
            "percentile_95_interval": [float(x) for x in np.quantile(sims, [.025,.975])],
            "bootstrap_iterations": bootstraps,
            "interpretation": "Stratified paired patient bootstrap, conditional on these 4 sampled cohorts; repeated seeds averaged by patient."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/pilot80_extract/radiomic_features.csv")
    ap.add_argument("--out", default="results/pilot80_ablation")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--seeds", type=parse_seeds, default=[42,123,2026])
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    if args.epochs < 1 or args.latent < 1 or args.bootstrap < 100:
        ap.error("epochs/latent must be positive and bootstrap >= 100")
    df, cohorts = load(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores, preds, reco = run(df, cohorts, args.seeds, args.epochs, args.latent)
    scores.to_csv(out/"by_fold_seed.csv", index=False)
    preds.to_csv(out/"patient_predictions.csv", index=False)
    reco.to_csv(out/"vae_reconstruction.csv", index=False)
    # Average fold metrics across seeds for a clean, readable per-cohort table.
    summary = scores.groupby(["heldout_cohort", "model"], as_index=False)[["mae", "r2"]].agg(["mean", "std"])
    summary.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary.to_csv(out/"cohort_summary.csv", index=False)
    pooled = []
    for seed in args.seeds:
        for mode in MODES:
            p = preds[(preds.seed == seed) & (preds.model == mode)]
            pooled.append({"seed": seed, "model": mode, "n_test": len(p),
                           **metrics(p.y_true.to_numpy(),p.y_pred.to_numpy())})
    pool = pd.DataFrame(pooled)
    pool.to_csv(out/"pooled_by_seed.csv", index=False)
    agg = pool.groupby("model")[["mae","r2"]].agg(["mean","std"])
    agg.columns = ["_".join(x).strip("_") for x in agg.columns]
    agg.to_csv(out/"pooled_summary.csv")
    gain = bootstrap_gain(preds, args.seeds, cohorts, args.bootstrap)
    (out/"paired_gain.json").write_text(json.dumps(gain, indent=2))
    (out/"protocol.json").write_text(json.dumps({
        "input": args.input, "n_patients": len(df), "cohorts": cohorts,
        "seeds": args.seeds, "epochs": args.epochs, "latent": args.latent,
        "features": CHEAP_FEATURES, "target": TARGET,
        "model": "RF 120 trees, min_samples_leaf=3, unchanged across modalities",
        "split": "leave one dataset cohort out; same held-out patients for all models",
        "preprocessing": "train-only median imputation and StandardScaler",
        "vae": "train-only unsupervised tabular VAE with latent means; not a 3D image VAE",
        "outcome": "pCR NOT used for training/evaluation or model selection",
        "limitations": [
          "The surrogate target is computed from the same MRI as the features; this measures approximate image-derived texture, not clinical treatment response.",
          "Same-source image statistics can make proxy target easier to predict; do not interpret as biological marker validity.",
          "Balanced pCR case-control selection is unrepresentative of natural cohort prevalence.",
          "Four named datasets are not necessarily four fully independent scanner sites.",
          "80 patients, 20 per held-out cohort, cannot establish external clinical validity.",
          "Repeated seeds on the same 80 patients are not independent patients.",
          "Model development decisions based on heldout scores would compromise future validation."
        ]}, indent=2))
    print("\n=== Pooled OUT-OF-COHORT results (mean across seeds) ===")
    print(agg.round(4).to_string())
    print("\nPaired MAE(raw) - MAE(both):", round(gain["mae_gain_raw_minus_both"],5),
          "95% bootstrap percentile interval:", gain["percentile_95_interval"])
    print("\nResults saved:", out)

if __name__ == "__main__":
    main()
