# MAMA-MIA LatentLab — Audited Pilot-80 Ablation

**Status: exploratory research software, not a medical device or validated biomarker.**

This audit builds on the project's existing `MAMA-MIA-LatentLab` code and the feature CSV extracted from 80 real MAMA-MIA cases, 20 from each of DUKE, ISPY1, ISPY2 and NACT. It does **not** retrain image encoders from raw MRI, and it does **not** predict chemotherapy response.

## What this experiment asks

> Does a small, unsupervised tabular variational autoencoder add predictive information beyond twelve summary features already extracted from MRI, when a surrogate estimates an expensive spatial texture statistic?

We compare **the same Random Forest hyperparameters** across five alternatives:

1. `median`: training-set median texture endpoint.
2. `std_only`: only `post_tumor_std_z`, to check whether the task has a strong single-feature shortcut.
3. `raw`: all twelve original MRI summary features.
4. `vae`: four VAE latent means learned *only on the training cohorts*.
5. `both`: the original twelve features plus four VAE latent means.

All four cohort-level splits keep a whole dataset out for testing. Imputation, scaling, unsupervised VAE training and supervised Random Forest fitting use training patients only. `pcr` is not used by the experiment at all. The same four cohort holdouts are repeated at three seeded configurations, but those are **not independent patients**.

## Run on Linux Mint

Extract/copy `ablation_pilot80.py` into the **root** of your existing project at `~/MAMA-MIA-LatentLab/`. Keep your existing `latentlab/` package and `results/pilot80_extract/radiomic_features.csv` unchanged.

```bash
cd ~/MAMA-MIA-LatentLab
source .venv/bin/activate
python ablation_pilot80.py \
  --input results/pilot80_extract/radiomic_features.csv \
  --out results/pilot80_ablation \
  --epochs 80 \
  --seeds 42,123,2026 \
  --bootstrap 1000
```

You do **not** need to redownload MRI data or repeat NIfTI feature extraction.

## Files produced

- `pooled_summary.csv`: pooled MAE/R² for each mode, averaged across seeds.
- `pooled_by_seed.csv`: one row per seed and mode, containing all 80 out-of-cohort predictions for that seed.
- `cohort_summary.csv`: per held-out dataset mean/std across seeds.
- `by_fold_seed.csv`: full cohort × seed × model metrics.
- `patient_predictions.csv`: per-patient held-out prediction and absolute error; useful for subgroup analysis; **avoid posting patient-linked records in public repos without reviewing data licensing and sensitivity**.
- `vae_reconstruction.csv`: training loss and held-out feature reconstruction MSE (diagnostic, **not** image reconstruction).
- `paired_gain.json`: paired, cohort-stratified patient-bootstrap interval for the MAE difference between `raw` and `both`.
- `protocol.json`: research definitions and limitations.

## Verified results on the 80-case CSV (80 epochs; seeds 42, 123, 2026)

| Mode | Pooled MAE, mean | Pooled R², mean |
|---|---:|---:|
| Median null | 0.2889 | -0.1928 |
| Single feature: post-tumor intensity STD | 0.1359 | 0.7416 |
| Raw twelve features | **0.1219** | **0.7726** |
| VAE latent means alone | 0.2071 | 0.3374 |
| Raw + VAE | 0.1221 | 0.7694 |

For the paired contrast defined as **MAE(raw) − MAE(raw + VAE)**, the observed mean difference was `-0.00019`, with a 95% bootstrap percentile interval around `[-0.00223, +0.00179]`. This is compatible with **no useful improvement** from the current tabular VAE. Do not interpret the bootstrap interval as uncertainty for new clinical populations or hospitals.

Across all 80 cases, post-tumor intensity STD and the texture surrogate had an exploratory Spearman correlation of about **0.900**. This is descriptive only: both measurements come from the same MRI.

## Code audit: limitations to correct before stronger claims

1. **Tabular, not volumetric, VAE:** the encoder maps 12 radiomic summaries to a 4-dimensional latent vector, not a 3-D MRI to a latent image phenotype.
2. **Proxy-outcome overlap:** both the predictors and `spatial_neighbor_difference_q90` are calculated from the same image and mask. Good performance may reflect mathematical dependence between intensity variance/quantiles and local voxel differences, **not treatment-response biology**.
3. **Physical voxel spacing:** the endpoint takes absolute differences between neighboring voxels along each voxel axis without dividing by physical neighbor distances. Voxel size/protocol may change the result. Consider physically normalized gradients or harmonized spatial resolution with proper sensitivity analysis.
4. **Image orientation:** `images.py` assigns bbox labels `z`, `y`, `x` to the array's axes in this order, without deriving anatomical axes from the NIfTI affine. These labels may be anatomically wrong; use coordinate-safe names or reorient canonical images (and masks) with documented transforms.
5. **Normalization:** `_robust_pre_reference` uses clipped global image percentiles, not conventional z-score using tissue intensity mean and standard deviation; `*_z` names may mislead. Background composition, field-of-view and intensity/protocol shifts may confound the results.
6. **Cohort/site distinction:** the four source collections are not necessarily four single scanner sites. Compare imaging site/manufacturer/field strength from the clinical spreadsheet and prevent hidden repeated subjects.
7. **Selection bias:** these 80 patients were deliberately sampled 50:50 by pCR within each cohort; model estimates do not represent natural population prevalence.
8. **Validation limits:** 20 test patients per cohort, a small VAE and an imaging proxy target are inadequate for clinical utility, biomarker validation, pCR response prediction or causal conclusions.
9. **Model development:** do not repeatedly tune choices on these four test-cohort scores and continue to label the results external validation; use a separate final locked test or appropriate nested model selection.

## A stronger next-stage research question

*Can MRI-derived representations robustly characterize pre-treatment lesion enhancement dynamics across acquisition protocols, and do any such representations add information beyond simple radiomic features and established clinical variables for an appropriately validated treatment-response association?*

A sensible next phase is to (i) audit scanner/spacing and phase timing, (ii) evaluate temporally sampled DCE-MRI enhancement inside a verified expert mask, (iii) compare simple temporal-statistics baselines with lightweight learned ROI representations, and (iv) conduct a separately specified, leakage-safe and calibrated pCR prediction/association study **only if justified by adequate independent data**. Avoid calling this a demonstrated clinical predictor before that validation exists.

## Research provenance

- MAMA-MIA: Garrucho et al., *Scientific Data* 12, 453 (2025), DOI: 10.1038/s41597-025-04707-4.
- Data repository: `syn60868042` on Synapse; follow original source licenses and attribution requirements.
- The audit uses only code and feature rows already provided for this project. No patient MRI files, access tokens or credentials are included in the package.
