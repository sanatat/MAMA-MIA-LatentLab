"""Data ingestion, leakage-safe splitting and latent surrogate analysis."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from .images import extract_nifti, CHEAP_FEATURES, TARGET
from .model import train_vae, project


def extract_from_manifest(path, out):
    df = pd.read_csv(path)
    need = ["patient_id", "cohort", "pre_nifti", "post_nifti", "expert_mask_nifti"]
    missing = [k for k in need if k not in df]
    if missing: raise ValueError(f"Manifest missing required columns {missing}")
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient identifiers: do not duplicate patients")
    if not len(df): raise ValueError("Empty manifest")
    source = Path(path).resolve().parent
    results=[]; fails=[]
    for row in df.to_dict("records"):
        try:
            paths = [Path(str(row[k])) for k in need[2:]]
            paths = [p if p.is_absolute() else source / p for p in paths]
            result = extract_nifti(*paths)
            results.append({"patient_id":row["patient_id"], "cohort":row["cohort"],
                            **({"pcr":row["pcr"]} if "pcr" in row else {}),**result})
        except Exception as exc:
            fails.append({"patient_id":row["patient_id"],"error":str(exc)})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/"radiomic_features.csv",index=False)
    pd.DataFrame(fails,columns=["patient_id","error"]).to_csv(out/"failed_cases.csv",index=False)
    (out/"extraction_report.json").write_text(json.dumps({"rows_in":len(df),
       "extracted":len(results),"failed":len(fails), "target":TARGET},indent=2))
    return len(results),len(fails)


def _split(df, heldout, seed):
    if heldout:
        if heldout not in set(df.cohort):raise ValueError(f"Holdout cohort {heldout!r} not found")
        tr=np.where(df.cohort!=heldout)[0]; te=np.where(df.cohort==heldout)[0]
        split=f"leave-one-cohort-out ({heldout})"
    else:
        tr,te=train_test_split(np.arange(len(df)),test_size=.25,random_state=seed)
        split="patient-level random holdout (demo; prefer --holdout-cohort)"
    if len(tr)<10 or len(te)<3:raise ValueError("Insufficient training/holdout cases")
    return np.asarray(tr), np.asarray(te),split


def analyze(input_file,out,heldout=None,epochs=160,seed=42,skip_umap=False):
    df=pd.read_csv(input_file)
    needed=["patient_id","cohort",TARGET]+CHEAP_FEATURES
    missing=[k for k in needed if k not in df.columns]
    if missing:raise ValueError("Missing required columns: "+", ".join(missing))
    if df.patient_id.isna().any() or df.patient_id.duplicated().any():
        raise ValueError("Patient IDs missing or duplicated")
    df=df.dropna(subset=[TARGET,"cohort"]).reset_index(drop=True)
    if df.empty:raise ValueError("No rows with the surrogate target")
    df.loc[:,CHEAP_FEATURES] = df[CHEAP_FEATURES].apply(pd.to_numeric,errors="coerce")
    tr,te,split=_split(df,heldout,seed)
    imp=SimpleImputer(strategy="median",keep_empty_features=True)
    scaler=StandardScaler()
    xtr=scaler.fit_transform(imp.fit_transform(df.loc[tr,CHEAP_FEATURES])).astype("float32")
    xte=scaler.transform(imp.transform(df.loc[te,CHEAP_FEATURES])).astype("float32")
    net,trace=train_vae(xtr,seed=seed,epochs=epochs)
    ztr=project(net,xtr); zte=project(net,xte)
    # Expensive spatial endpoint is excluded from all fitted input transforms.
    surrogate=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,
                                    n_jobs=-1,random_state=seed)
    surrogate.fit(np.concatenate([xtr,ztr],axis=1),df.loc[tr,TARGET].to_numpy())
    preds=surrogate.predict(np.concatenate([xte,zte],axis=1))
    truth=df.loc[te,TARGET].to_numpy()
    null=np.full_like(truth,np.median(df.loc[tr,TARGET]),dtype=float)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result={"input":str(input_file),"analysis_scope":"EXPLORATORY/non-clinical",
      "n_total":len(df),"n_train":len(tr),"n_test":len(te),"split":split,
      "feature_names":CHEAP_FEATURES,"surrogate_endpoint":TARGET,
      "surrogate_holdout_mae":float(mean_absolute_error(truth,preds)),
      "median_null_holdout_mae":float(mean_absolute_error(truth,null)),
      "surrogate_holdout_r2":float(r2_score(truth,preds)) if len(truth)>1 else None,
      "vae_final_train_loss":trace[-1],
      "pcr_use":"plot-only if supplied; never used as VAE or surrogate feature",
      "warnings":["Surrogate emulates MRI texture; it does NOT predict treatment response.",
       "Data must be pre-treatment. No clinical utility or biomarker validity established.",
       "Imaging intensities and scanner protocols can confound latent clusters.",
       "Train-only preprocessing; no training access to held-out cohort targets."]}
    (out/"analysis.json").write_text(json.dumps(result,indent=2))
    result_rows=df.copy()
    result_rows["split"]="train"; result_rows.loc[te,"split"]="test"
    result_rows["surrogate_prediction"]=np.nan
    result_rows.loc[te,"surrogate_prediction"]=preds
    for j in range(ztr.shape[1]):
        result_rows[f"z{j+1}"]=np.nan
        result_rows.loc[tr,f"z{j+1}"]=ztr[:,j]
        result_rows.loc[te,f"z{j+1}"]=zte[:,j]
    result_rows[["patient_id","cohort","split",TARGET,"surrogate_prediction"]+
                [f"z{j+1}" for j in range(ztr.shape[1])]].to_csv(out/"embeddings.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.6,4.6))
    ax.scatter(truth,preds,alpha=.65,s=26)
    lo=min(truth.min(),preds.min());hi=max(truth.max(),preds.max())
    ax.plot([lo,hi],[lo,hi],ls="--",c="0.5")
    ax.set(xlabel="Observed 3D MRI heterogeneity (held-out)",ylabel="Surrogate estimate",
           title="Held-out surrogate residual check")
    fig.tight_layout();fig.savefig(out/"surrogate.png",dpi=150);plt.close(fig)
    if "pcr" in df.columns:
        # Only report label completeness here; pCR is never fitted or imputed.
        pcr = df["pcr"].astype(str).str.strip().str.lower()
        valid = pcr.isin(["0", "1", "0.0", "1.0", "yes", "no", "true", "false"])
        result["pcr_available_for_descriptive_overlay"] = int(valid.sum())
        (out/"analysis.json").write_text(json.dumps(result,indent=2))
    if not skip_umap:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP is unavailable. Install requirements.txt or run --skip-umap") from exc
        import warnings
        # UMAP fit strictly on training embeddings to prevent test-data leakage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=UserWarning)
            mapper=umap.UMAP(n_neighbors=min(15,len(tr)-1),min_dist=.18,
                             random_state=seed,n_jobs=1)
            utrain=mapper.fit_transform(ztr); utest=mapper.transform(zte)
        coords=np.zeros((len(df),2),dtype=float)
        coords[tr]=utrain; coords[te]=utest
        fig,ax=plt.subplots(figsize=(7,5.5))
        for cohort in sorted(df.cohort.astype(str).unique()):
            indices=np.where(df.cohort.astype(str).to_numpy()==cohort)[0]
            ax.scatter(coords[indices,0],coords[indices,1],s=24,alpha=.7,label=cohort)
        ax.set(xlabel="UMAP 1",ylabel="UMAP 2",title="VAE latent UMAP (training fit only)")
        ax.legend(loc="best",fontsize=8)
        fig.tight_layout();fig.savefig(out/"umap_cohorts.png",dpi=160);plt.close(fig)
        result_rows["umap1"]=coords[:,0];result_rows["umap2"]=coords[:,1]
        result_rows[["patient_id","cohort","split","umap1","umap2"]].to_csv(
             out/"umap_coordinates.csv",index=False)
        if "pcr" in df.columns:
            # Not a predictor, merely visual post hoc overlay with explicit unknown.
            pcr = df["pcr"].astype(str).str.strip().str.lower()
            pos = pcr.isin(["1","1.0","yes","true"]).to_numpy()
            neg = pcr.isin(["0","0.0","no","false"]).to_numpy()
            fig,ax=plt.subplots(figsize=(7,5.5))
            for label, mask, marker in [("pCR yes",pos,"o"),("pCR no",neg,"s"),
                                         ("pCR unknown",~(pos|neg),"x")]:
                if mask.any():
                    ax.scatter(coords[mask,0],coords[mask,1],alpha=.58,s=22,
                               marker=marker,label=label)
            ax.set(xlabel="UMAP 1",ylabel="UMAP 2",
                   title="Post-hoc pCR overlay (no outcome-informed embedding)")
            ax.legend(fontsize=8);fig.tight_layout()
            fig.savefig(out/"umap_pcr_exploratory.png",dpi=160);plt.close(fig)
    return result


def make_demo(path, n=100, seed=42):
    rng=np.random.default_rng(seed)
    cohorts=["ISPY1","ISPY2","DUKE","NACT"]
    rows=[]
    for i in range(n):
        cohort=cohorts[i%4];group=cohorts.index(cohort)
        latent=rng.normal(0,1,3)
        # Each feature has a synthetic scanner/domain shift; no real patients.
        feats={name:float(rng.normal(0,1)+latent[j%3]*(.3+(j%4)*.1)+group*.15)
               for j,name in enumerate(CHEAP_FEATURES)}
        target=float(max(.01,.9+0.35*latent[0]-.22*latent[2]+.1*group+rng.normal(0,.12)))
        rows.append({"patient_id":f"SIM_{i:04d}","cohort":cohort,
                     "pcr":int(rng.random()<.4),**feats,TARGET:target})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    return str(path)


def main():
    p=argparse.ArgumentParser(description="MAMA-MIA exploratory VAE + UMAP + surrogate")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("index",help="Index locally downloaded MAMA-MIA MRI folders")
    a.add_argument("--root",required=True);a.add_argument("--out",default="my_mama_mia_manifest.csv")
    a.add_argument("--post-phase",type=int,default=1,
                   help="Candidate first post-contrast phase, must verify metadata")
    a=sub.add_parser("extract",help="Use real, aligned MRI/mask NIfTI files")
    a.add_argument("--manifest",required=True);a.add_argument("--out",default="results/extract")
    a=sub.add_parser("analyze",help="Run VAE + UMAP + spatial-texture surrogate")
    a.add_argument("--input",required=True);a.add_argument("--out",default="results/analysis")
    a.add_argument("--holdout-cohort");a.add_argument("--epochs",type=int,default=160)
    a.add_argument("--seed",type=int,default=42);a.add_argument("--skip-umap",action="store_true")
    a=sub.add_parser("demo",help="GENERATE SYNTHETIC SMOKE TEST, no real results")
    a.add_argument("--out",default="results/demo_input.csv");a.add_argument("--n",type=int,default=100)
    args=p.parse_args()
    if args.command=="index":
        from .indexer import build_manifest
        print(json.dumps(build_manifest(args.root,args.out,args.post_phase),indent=2))
    elif args.command=="extract":
        counts=extract_from_manifest(args.manifest,args.out)
        print(json.dumps({"extracted":counts[0],"failed":counts[1],"out":args.out}))
    elif args.command=="demo": print("SYNTHETIC ONLY",make_demo(args.out,args.n))
    elif args.command=="analyze":
        print(json.dumps(analyze(args.input,args.out,args.holdout_cohort,
                                 args.epochs,args.seed,args.skip_umap),indent=2))
if __name__=="__main__":main()
