#!/usr/bin/env python3
"""Fetch the second post-contrast (_0002) MRI for an existing MAMA-MIA pilot.

Dry run by default. Never puts Synapse credentials on the command line or in files.
The acquisition-time field is in seconds relative to the pre-contrast phase
(as reported in the MAMA-MIA paper); it is not necessarily injection time.

Example:
    python mamamia_phase2.py
    python mamamia_phase2.py --download
"""

import argparse
import ast
import json
import sys
from pathlib import Path
import pandas as pd

IMAGE_ROOT_ID = "syn64871114"


def parse_times(value):
    if pd.isna(value):
        return None
    try:
        vals = ast.literal_eval(str(value)) if isinstance(value, str) else value
        if not isinstance(vals, (list, tuple)):
            return None
        return [float(x) for x in vals]
    except (ValueError, SyntaxError, TypeError):
        return None


def children_map(syn, parent):
    return {obj["name"]: obj["id"] for obj in syn.getChildren(parent)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/pilot80_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "MAMA-MIA-data")
    parser.add_argument("--out", type=Path, default=Path("results/pilot80_phase2_status.csv"))
    parser.add_argument("--download", action="store_true", help="Explicitly download missing second post-contrast images")
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.error(f"Pilot manifest not found: {args.manifest}")
    df = pd.read_csv(args.manifest)
    if not {"patient_id", "cohort"}.issubset(df.columns):
        parser.error("Manifest must contain patient_id and cohort columns")
    if df.patient_id.duplicated().any():
        parser.error("Duplicate patient IDs in manifest")

    root = args.data_root.expanduser().resolve()
    excel = root / "clinical_and_imaging_info.xlsx"
    if not excel.exists():
        parser.error(f"Metadata Excel not found: {excel}")
    meta = pd.read_excel(excel, sheet_name="dataset_info")
    meta = meta[["patient_id", "num_phases", "acquisition_times"]]
    if meta.patient_id.duplicated().any():
        parser.error("Duplicate patient IDs in metadata")
    df = df.drop(columns=["num_phases", "acquisition_times"], errors="ignore").merge(
        meta, on="patient_id", how="left", validate="one_to_one")
    if df.num_phases.isna().any():
        parser.error("Some manifest patients are absent from metadata")

    records = []
    for rec in df.to_dict("records"):
        pid = str(rec["patient_id"])
        times = parse_times(rec["acquisition_times"])
        n = int(rec["num_phases"])
        time_ok = times is not None and len(times) == n and len(times) >= 3 and all(
            times[i] > times[i - 1] for i in range(1, len(times)))
        expected_path = root / "images" / pid / f"{pid}_0002.nii.gz"
        present = expected_path.is_file() and expected_path.stat().st_size > 0
        records.append({
            "patient_id": pid,
            "cohort": str(rec["cohort"]),
            "num_phases": n,
            "times_valid": time_ok,
            "first_post_seconds": times[1] if time_ok else None,
            "second_post_seconds": times[2] if time_ok else None,
            "phase2_path": str(expected_path),
            "phase2_status": "existing" if present else "pending",
            "error": "" if n >= 3 else "fewer than 3 phases"
        })
    status = pd.DataFrame(records)
    print("Selected patients:", len(status), flush=True)
    print("Existing _0002 files:", (status.phase2_status == "existing").sum(), flush=True)
    print("To download:", (status.phase2_status == "pending").sum(), flush=True)
    print("Missing/invalid acquisition times:", (~status.times_valid).sum(), flush=True)
    print("Phase distribution:\n", pd.crosstab(df.cohort, df.num_phases).to_string(), flush=True)
    if (~status.times_valid).any():
        print("Missing/invalid acquisition times in:", ", ".join(status.loc[~status.times_valid, "patient_id"]), flush=True)

    if args.download:
        try:
            from synapseclient import Synapse
            syn = Synapse()
            syn.login(silent=True)
            patient_folders = children_map(syn, IMAGE_ROOT_ID)
        except Exception as exc:
            print(f"Synapse connection error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        for i, row in status.iterrows():
            if row.phase2_status == "existing":
                continue
            pid = str(row.patient_id)
            dest = Path(row.phase2_path)
            try:
                if int(row.num_phases) < 3:
                    raise ValueError("Patient metadata reports fewer than 3 phases")
                folder_id = patient_folders.get(pid)
                if not folder_id:
                    raise FileNotFoundError("Synapse patient folder not found")
                files = children_map(syn, folder_id)
                remote_id = files.get(dest.name)
                if not remote_id:
                    raise FileNotFoundError("Second post-contrast file absent on Synapse")
                dest.parent.mkdir(parents=True, exist_ok=True)
                downloaded = Path(syn.get(remote_id, downloadLocation=str(dest.parent)).path)
                if downloaded != dest or not dest.is_file() or dest.stat().st_size == 0:
                    raise RuntimeError(f"Unexpected download result: {downloaded}")
                status.at[i, "phase2_status"] = "downloaded"
                print(f"[{i + 1}/{len(status)}] OK {pid}", flush=True)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                status.at[i, "phase2_status"] = "failed"
                status.at[i, "error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{i + 1}/{len(status)}] FAILED {pid}: {exc}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(args.out, index=False)
    print(json.dumps({
        "patients": len(status),
        "available_phase2": int(status.phase2_status.isin(["existing", "downloaded"]).sum()),
        "pending": int((status.phase2_status == "pending").sum()),
        "failed": int((status.phase2_status == "failed").sum()),
        "valid_timestamps": int(status.times_valid.sum()),
        "report": str(args.out),
    }, indent=2), flush=True)
    return 1 if (status.phase2_status == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
