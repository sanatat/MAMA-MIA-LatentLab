#!/usr/bin/env python3
"""Select a reproducible, outcome-balanced MAMA-MIA pilot and download matched MRI phases/masks.

A deliberately balanced pilot is NOT population-representative and must not be used
for estimating pCR prevalence or clinical performance.

Usage:
  python mamamia_batch.py --per-cohort 8                    # Preview only
  python mamamia_batch.py --per-cohort 8 --download         # 32 cases
  python mamamia_batch.py --per-cohort 20 --download        # 80 cases (pilot extension)

Needs: pandas, openpyxl, synapseclient; authentication via SYNAPSE_AUTH_TOKEN or
Synapse client configuration. Never paste your token into this file or chat.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

COHORTS = ("DUKE", "ISPY1", "ISPY2", "NACT")
IMAGES_FOLDER_ID = "syn64871114"
EXPERT_MASK_FOLDER_ID = "syn64871175"
CLINICAL_COLS = (
    "tumor_subtype", "er", "pr", "her2", "nottingham_grade",
    "age", "site", "manufacturer", "field_strength", "num_phases", "acquisition_times",
)


def select_cases(table: pd.DataFrame, per_cohort: int, seed: int) -> pd.DataFrame:
    if per_cohort < 4 or per_cohort % 2:
        raise ValueError("--per-cohort must be an even integer >= 4")
    df = table.copy()
    for name in ("patient_id", "dataset", "pcr"):
        if name not in df.columns:
            raise ValueError(f"Missing clinical column: {name}")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["dataset"] = df["dataset"].astype(str).str.upper().str.strip()
    df["pcr"] = pd.to_numeric(df["pcr"], errors="coerce")
    if df["patient_id"].duplicated().any():
        raise ValueError("Clinical spreadsheet contains duplicate patient IDs")
    rows = []
    half = per_cohort // 2
    for cohort_i, cohort in enumerate(COHORTS):
        for response in (0, 1):
            subset = df.loc[(df["dataset"] == cohort) & (df["pcr"] == response)]
            if len(subset) < half:
                raise ValueError(f"Not enough {cohort} pCR={response}: {len(subset)} available, {half} requested")
            # Randomize the entire subgroup once, then take the first N.
            # A run with 8 per cohort is a subset of a run with 20 per cohort.
            ordered = subset.sort_values("patient_id").sample(frac=1.0, random_state=seed + cohort_i * 2 + response)
            rows.append(ordered.head(half))
    return pd.concat(rows, ignore_index=True).sort_values(["dataset", "patient_id"]).reset_index(drop=True)


def children_map(syn, parent: str) -> dict:
    return {child["name"]: child["id"] for child in syn.getChildren(parent)}


def get_one(syn, identifier: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[existing] {target.name}", flush=True)
        return
    downloaded = syn.get(identifier, downloadLocation=str(target.parent))
    result = Path(downloaded.path)
    if not result.exists() or result.name != target.name:
        raise RuntimeError(f"Unexpected downloaded file for {identifier}: {result}")
    print(f"[downloaded] {target.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path.home() / "MAMA-MIA-data")
    p.add_argument("--per-cohort", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--download", action="store_true", help="Explicitly allow Synapse downloads")
    p.add_argument("--out", type=Path, default=Path("results/pilot_manifest.csv"))
    args = p.parse_args()

    excel = args.data_root.expanduser() / "clinical_and_imaging_info.xlsx"
    if not excel.is_file():
        p.error(f"Clinical Excel not found: {excel}")
    table = pd.read_excel(excel, sheet_name="dataset_info")
    chosen = select_cases(table, args.per_cohort, args.seed)
    print("\nSelection (balanced case-control research pilot, NOT cohort prevalence):")
    print(pd.crosstab(chosen["dataset"], chosen["pcr"]).to_string())
    print("Selected:", len(chosen), "patients")
    print("Cohorts:", ", ".join(COHORTS))
    if not args.download:
        print("\nPreview only; no images downloaded. Add --download to fetch paired volumes and masks.")
        print(chosen[["patient_id", "dataset", "pcr"]].to_string(index=False))
        return 0

    try:
        import synapseclient
        syn = synapseclient.Synapse()
        syn.login(silent=True)
    except Exception as exc:
        print("Synapse authentication failed. Use SYNAPSE_AUTH_TOKEN in this shell or configure Synapse.", file=sys.stderr)
        print(f"Details: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    image_folders = children_map(syn, IMAGES_FOLDER_ID)
    masks = children_map(syn, EXPERT_MASK_FOLDER_ID)
    output_rows, failed = [], []
    root = args.data_root.expanduser().resolve()
    for i, row in enumerate(chosen.to_dict("records"), 1):
        patient = str(row["patient_id"])
        phase_folder = root / "images" / patient
        mask_path = root / "segmentations" / "expert" / f"{patient}.nii.gz"
        print(f"\n[{i}/{len(chosen)}] {patient}", flush=True)
        try:
            if patient not in image_folders:
                raise RuntimeError("Patient folder unavailable on Synapse")
            if mask_path.name not in masks:
                raise RuntimeError("Expert mask unavailable on Synapse")
            files = children_map(syn, image_folders[patient])
            pre_name, post_name = f"{patient}_0000.nii.gz", f"{patient}_0001.nii.gz"
            if pre_name not in files or post_name not in files:
                raise RuntimeError("Missing 0000 and/or 0001 phase on Synapse")
            pre_path, post_path = phase_folder / pre_name, phase_folder / post_name
            get_one(syn, files[pre_name], pre_path)
            get_one(syn, files[post_name], post_path)
            get_one(syn, masks[mask_path.name], mask_path)
            result = {
                "patient_id": patient, "cohort": row["dataset"],
                "pre_nifti": str(pre_path), "post_nifti": str(post_path),
                "expert_mask_nifti": str(mask_path), "pcr": int(row["pcr"]),
                "sampling_policy": "balanced_case_control_pilot",
            }
            for key in CLINICAL_COLS:
                if key in row:
                    value = row[key]
                    result[key] = "" if pd.isna(value) else str(value)
            output_rows.append(result)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failed.append({"patient_id": patient, "error": f"{type(exc).__name__}: {exc}"})
            print("[FAILED]", failed[-1]["error"], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        pd.DataFrame(output_rows).to_csv(args.out, index=False)
    pd.DataFrame(failed, columns=["patient_id", "error"]).to_csv(
        args.out.with_suffix(".download_failures.csv"), index=False
    )
    print(json.dumps({"complete": len(output_rows), "failed": len(failed),
                      "manifest": str(args.out), "status": "pilot only, not clinical validation"}, indent=2))
    return 0 if output_rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
