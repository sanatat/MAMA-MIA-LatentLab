#!/usr/bin/env python3
"""Exploratory 3-phase DCE-MRI ROI temporal feature extraction for MAMA-MIA.

Uses the already downloaded _0000, _0001, _0002 and expert-mask NIfTI files.
Reads acquisition times from a separately verified phase2_status.csv. Exactly
matched 3D shapes and affines are required; no hidden registration/resampling.

Run from ~/MAMA-MIA-LatentLab after installing nibabel and pandas:
    python mamamia_temporal_extract.py --limit 2
    python mamamia_temporal_extract.py

Output:
    results/pilot80_temporal/temporal_features.csv
    results/pilot80_temporal/temporal_plus_baseline.csv
    results/pilot80_temporal/temporal_qc.csv
    results/pilot80_temporal/summary.json

This is EXPLORATORY research code, not a clinical DCE-MRI pharmacokinetic
model. Time zero is reference acquisition, NOT verified injection time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROI_MIN_VOXELS = 20
FEATURE_NAMES = [
    "temporal_t1_seconds",
    "temporal_t2_seconds",
    "temporal_dt12_seconds",
    "temporal_pre_median_raw",
    "temporal_post1_median_raw",
    "temporal_post2_median_raw",
    "temporal_early_enhancement_median_z",
    "temporal_late_enhancement_median_z",
    "temporal_late_enhancement_iqr_z",
    "temporal_first_to_second_change_median_z",
    "temporal_change_slope_z_per_min",
    "temporal_early_slope_z_per_min",
    "temporal_washout_voxel_fraction",
    "temporal_post2_tumor_std_z",
]
ID_COLS = ["patient_id", "cohort"]


def check_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [x for x in columns if x not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError(f"Missing/duplicate patient IDs in {name}")


def _load_data_and_roi(image, reference_shape, reference_affine, roi):
    if image.shape != reference_shape or not np.allclose(
        image.affine, reference_affine, atol=1e-3, rtol=1e-5
    ):
        raise ValueError("Image/ROI voxel-grid mismatch (shape or affine)")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.shape != reference_shape:
        raise ValueError("Unexpected loaded image shape")
    result = data[roi].copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite MRI intensity in tumor ROI")
    return data, result


def extract_patient(pre_path: Path, post1_path: Path,
                    post2_path: Path, mask_path: Path, t1: float, t2: float):
    import nibabel as nib

    if not (np.isfinite(t1) and np.isfinite(t2) and 0 < t1 < t2):
        raise ValueError(f"Invalid acquisition times: {t1}, {t2}")
    for path in (pre_path, post1_path, post2_path, mask_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty NIfTI: {path}")

    pre_img = nib.load(str(pre_path))
    mask_img = nib.load(str(mask_path))
    if len(pre_img.shape) != 3:
        raise ValueError("Expected 3D MRI")
    if mask_img.shape != pre_img.shape or not np.allclose(
        mask_img.affine, pre_img.affine, atol=1e-3, rtol=1e-5
    ):
        raise ValueError("Expert mask and pre-MRI are not on the same grid")
    roi = np.asarray(mask_img.dataobj) > 0
    nvox = int(np.count_nonzero(roi))
    if nvox < ROI_MIN_VOXELS:
        raise ValueError(f"Mask has too few voxels: {nvox}")

    # Same pre-contrast reference for all phases; do not z-score each phase
    # separately, as that would remove the actual within-case enhancement.
    pre_full, pre_roi = _load_data_and_roi(pre_img, pre_img.shape, pre_img.affine, roi)
    if not np.isfinite(pre_full).all():
        raise ValueError("Nonfinite intensity in baseline image")
    low, pre_ref_median, high = np.percentile(pre_full, [1, 50, 99])
    scale = max(float((high - low) / 2.0), 1e-6)
    del pre_full

    post1_img = nib.load(str(post1_path))
    post1_full, post1_roi = _load_data_and_roi(post1_img, pre_img.shape, pre_img.affine, roi)
    del post1_full
    post2_img = nib.load(str(post2_path))
    post2_full, post2_roi = _load_data_and_roi(post2_img, pre_img.shape, pre_img.affine, roi)
    del post2_full

    pre_z = np.clip((pre_roi - pre_ref_median) / scale, -10, 10)
    post1_z = np.clip((post1_roi - pre_ref_median) / scale, -10, 10)
    post2_z = np.clip((post2_roi - pre_ref_median) / scale, -10, 10)
    early_delta = post1_z - pre_z
    late_delta = post2_z - pre_z
    change_12 = post2_z - post1_z
    q25, q75 = np.percentile(late_delta, [25, 75])

    record = {
        "temporal_t1_seconds": float(t1),
        "temporal_t2_seconds": float(t2),
        "temporal_dt12_seconds": float(t2 - t1),
        "temporal_pre_median_raw": float(np.median(pre_roi)),
        "temporal_post1_median_raw": float(np.median(post1_roi)),
        "temporal_post2_median_raw": float(np.median(post2_roi)),
        "temporal_early_enhancement_median_z": float(np.median(early_delta)),
        "temporal_late_enhancement_median_z": float(np.median(late_delta)),
        "temporal_late_enhancement_iqr_z": float(q75 - q25),
        "temporal_first_to_second_change_median_z": float(np.median(change_12)),
        "temporal_change_slope_z_per_min": float(np.median(change_12) * 60.0 / (t2-t1)),
        "temporal_early_slope_z_per_min": float(np.median(early_delta) * 60.0 / t1),
        "temporal_washout_voxel_fraction": float(np.mean(change_12 < 0)),
        "temporal_post2_tumor_std_z": float(np.std(post2_z)),
    }
    if not all(np.isfinite(v) for v in record.values()):
        raise ValueError("Nonfinite extracted temporal features")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("results/pilot80_manifest.csv"))
    p.add_argument("--status", type=Path, default=Path("results/pilot80_phase2_status.csv"))
    p.add_argument("--baseline", type=Path, default=Path("results/pilot80_extract/radiomic_features.csv"))
    p.add_argument("--out", type=Path, default=Path("results/pilot80_temporal"))
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N new cases; use for first-run smoke test")
    args = p.parse_args()

    for path in (args.manifest, args.status, args.baseline):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    manifest, status, baseline = (pd.read_csv(path) for path in
                                  (args.manifest, args.status, args.baseline))
    check_columns(manifest, ID_COLS + ["pre_nifti", "post_nifti", "expert_mask_nifti"], "manifest")
    check_columns(status, ID_COLS + ["times_valid", "first_post_seconds",
                                      "second_post_seconds", "phase2_path"], "status")
    check_columns(baseline, ID_COLS + ["pcr"], "baseline")
    if len(manifest) != 80 or set(manifest.patient_id) != set(status.patient_id):
        raise ValueError("Expected 80 matching patients in pilot80 manifest and status")
    if set(manifest.patient_id) != set(baseline.patient_id):
        raise ValueError("Baseline features do not cover exactly the selected patients")
    if "cohort" in status:
        x = manifest[ID_COLS].merge(status[ID_COLS], on="patient_id", suffixes=("_m", "_s"))
        if (x.cohort_m != x.cohort_s).any():
            raise ValueError("Cohort mismatch between status and manifest")

    full = manifest.merge(
        status.drop(columns=["cohort"]), on="patient_id", how="left", validate="one_to_one"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    feature_file = args.out / "temporal_features.csv"
    existing = pd.read_csv(feature_file) if feature_file.exists() and feature_file.stat().st_size else pd.DataFrame()
    if not existing.empty:
        check_columns(existing, ID_COLS + FEATURE_NAMES, "prior temporal output")
        if not set(existing.patient_id).issubset(set(manifest.patient_id)):
            raise ValueError("Prior output has patients outside current manifest; use a new --out directory")
    done = set(existing.patient_id) if not existing.empty else set()
    print(f"Selected patients: {len(full)}; already processed: {len(done)}", flush=True)

    qc_rows = []
    new_count = 0
    for i, rec in enumerate(full.to_dict("records"), 1):
        pid = str(rec["patient_id"])
        times_ok = str(rec["times_valid"]).strip().lower() in {"true", "1", "1.0"}
        qc = {"patient_id": pid, "cohort": rec["cohort"], "status": "", "note": ""}
        if not times_ok:
            qc.update(status="excluded_invalid_acquisition_times", note="No reliable t1/t2; image retained")
        elif pid in done:
            qc.update(status="already_extracted")
        elif args.limit is not None and new_count >= args.limit:
            qc.update(status="waiting_next_run")
        else:
            try:
                paths = [Path(str(rec[k])).expanduser() for k in
                         ("pre_nifti", "post_nifti", "phase2_path", "expert_mask_nifti")]
                t1, t2 = (float(rec[k]) for k in ("first_post_seconds", "second_post_seconds"))
                features = extract_patient(*paths, t1=t1, t2=t2)
                output = {"patient_id": pid, "cohort": rec["cohort"], **features}
                pd.DataFrame([output], columns=ID_COLS + FEATURE_NAMES).to_csv(
                    feature_file, index=False, mode="a", header=not feature_file.is_file()
                )
                done.add(pid)
                new_count += 1
                qc.update(status="extracted")
                print(f"[{i:02d}/80] extracted {pid} | t={t1:g},{t2:g}s | change={features['temporal_first_to_second_change_median_z']:+.3f}", flush=True)
            except Exception as exc:
                qc.update(status="failed", note=f"{type(exc).__name__}: {exc}")
                print(f"[{i:02d}/80] FAILED {pid}: {exc}", flush=True)
        qc_rows.append(qc)

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(args.out / "temporal_qc.csv", index=False)
    if feature_file.exists() and feature_file.stat().st_size:
        temporal = pd.read_csv(feature_file)
        check_columns(temporal, ID_COLS + FEATURE_NAMES, "temporal features")
        merged = baseline.merge(temporal.drop(columns=["cohort"]),
                                on="patient_id", how="inner", validate="one_to_one")
        if merged.patient_id.duplicated().any():
            raise ValueError("Duplicate merged patients")
        merged.to_csv(args.out / "temporal_plus_baseline.csv", index=False)
        n_success = len(temporal)
    else:
        n_success = 0
    counts = qc_df["status"].value_counts().to_dict()
    result = {
        "research_scope": "exploratory, nonclinical, non-predictive extraction",
        "selected": len(full),
        "extracted_total": n_success,
        "valid_time_metadata": int(full.times_valid.astype(str).str.lower().isin(["true", "1", "1.0"]).sum()),
        "counts_current_run": {k: int(v) for k, v in counts.items()},
        "features_csv": str(feature_file),
        "combined_csv": str(args.out / "temporal_plus_baseline.csv"),
        "quality_control_csv": str(args.out / "temporal_qc.csv"),
        "warnings": [
            "Acquisition timestamps are relative to pre-contrast reference, not confirmed contrast injection time.",
            "Affine/shape agreement does not prove freedom from between-phase patient motion.",
            "Feature extraction does not establish tumor perfusion kinetics, pCR prediction, or clinical utility.",
            "The case-control pilot intentionally balances pCR and is not representative of clinical prevalence.",
            "Temporal time variables can encode dataset/scanner differences; account for them in modeling.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if not (qc_df.status == "failed").any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
