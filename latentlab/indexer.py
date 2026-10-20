"""Create a local MAMA-MIA manifest from official downloaded directory layouts.

Default 0001 is only a candidate first post-contrast phase. Verify each case
against source metadata before use. No image is downloaded, altered or registered.
"""
from pathlib import Path
import pandas as pd


def _try_mask(root, patient):
    candidates = [root/"segmentations"/"expert"/f"{patient}.nii.gz",
                  root/"segmentations"/"expert"/f"{patient}.nii",
                  root/"segmentations"/f"{patient}.nii.gz",
                  root/"segmentations"/f"{patient}.nii"]
    for c in candidates:
        if c.is_file():return c
    return None


def _phase(folder, patient, no):
    patterns = [f"{patient}_{no:04d}.nii.gz",f"{patient}_{no:04d}.nii",
                f"phase_{no}.nii.gz",f"phase_{no}.nii"]
    for pat in patterns:
        matches=list(folder.rglob(pat))
        if len(matches)==1:return matches[0]
        if len(matches)>1:return None
    return None


def build_manifest(data_root, output_path, post_phase=1):
    root=Path(data_root).expanduser().resolve()
    images=root/"images"
    if not images.is_dir():raise ValueError(f"Images folder missing: {images}")
    if post_phase<1:raise ValueError("Post phase must be >=1")
    rows=[]; missing=[]
    for p in sorted(images.iterdir()):
        if not p.is_dir():continue
        pid=p.name
        pre,post,mask=_phase(p,pid,0),_phase(p,pid,post_phase),_try_mask(root,pid)
        absent=[name for name,path in [("pre",pre),("post",post),("expert mask",mask)] if path is None]
        if absent:
            missing.append({"patient_id":pid,"error":"Missing or ambiguous: "+", ".join(absent)})
            continue
        cohort=pid.split("_")[0].upper()
        rows.append({"patient_id":pid,"cohort":cohort,"pre_nifti":str(pre),
                     "post_nifti":str(post),"expert_mask_nifti":str(mask),"pcr":""})
    if not rows:raise ValueError("No complete patient directories discovered")
    dest=Path(output_path);dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest,index=False)
    pd.DataFrame(missing,columns=["patient_id","error"]).to_csv(dest.with_suffix(".missing.csv"),index=False)
    return {"indexed":len(rows),"incomplete":len(missing),"manifest":str(dest),
            "important":"Inspect modality/phase metadata and masks before extracting. "+
              "Post phase numbering cannot establish the actual acquisition timing."}
