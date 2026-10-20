# Research protocol (pre-registered-style outline)

**Project**: MAMA-MIA LatentLab. **Question**: Are unsupervised pre-treatment
DCE-MRI morpho-intensity phenotypes coherent across acquisition cohorts, and can
a compressed representation accelerate a defined voxelwise spatial analysis?

## Source

Garrucho et al., Scientific Data 12, 453 (2025), DOI 10.1038/s41597-025-04707-4.
MAMA-MIA Synapse dataset syn60868042 (CC-BY-NC). Source collections ISPY1,
ISPY2, DUKE, NACT. Coauthors include Apostolia Tsirikoglou and Fredrik Strand.
**No patient MRI or masks are bundled**. Obtain them under Synapse terms.

## Design

1. Load each patient's baseline/pre-treatment pre-contrast and first post-contrast
   T1 DCE-MRI; load expert tumor mask. Require exact shape and affine match.
2. Derive inexpensive physical shape and robust relative intensity summaries.
3. Derive one *withheld* costly 3D neighbor heterogeneity endpoint (90th pct.
   of absolute first-neighbor relative intensity changes within tumor).
4. Hold out all patients from one source collection (suggest DUKE); fit missing
   value imputer and scaler on remaining patients **only**.
5. Train small VAE unsupervised on cheap features. UMAP on VAE means: fit
   training representations only; transform held-out group without refitting.
6. Fit random-forest surrogate with [cheap features, VAE latent means] to
   estimate held-out neighbor heterogeneity; report MAE, R2 and median-null MAE.
7. Inspect cluster composition by source collection and imaging protocol;
   investigate whether scanner dominates apparent biological structure.
8. Overlay pCR **post hoc only**, after inspecting missingness, batch effects,
   selection bias and pre-treatment availability. Do not train on post-treatment
   outcomes or claim predictive utility. pCR is not part of the surrogate target.

## Limitations

- Adjacent-voxel difference is a research-defined image-texture proxy, not
  histopathologic grade, response biomarker, or clinically validated parameter.
- Voxels in the MAMA-MIA scans are not equivalent across scanner protocols;
  standardization removes only limited intensity shifts.
- Mask drawn from first post-contrast phase. Using it in feature extraction
  incorporates an expert annotation, so the surrogate cannot replace that step.
- VAE embeddings and UMAP clusters are descriptive and susceptible to domain
  shift and confounding. No biological clusters established.
- pCR is observed after neoadjuvant chemotherapy; it must not enter encoder
  features or any 'pre-treatment' modeling inputs.
- Surrogate approximates a deterministic image-computational endpoint; it is
  **not** a clinical treatment-response predictor.
- The pipeline does not reproduce the unavailable Zenodo record 23279735.

## Preflight QC

- Inspect MRI phase definitions: 0000 pre; 0001 is not guaranteed the right
  post phase across all cases. Check first post-contrast phase using dataset
  metadata and source notebooks.
- Confirm masks correspond to same first post phase and spatial affine.
- Missing or invalid cases go to failed_cases.csv, not silently resampled.
- Verify sample unit (patient ID), not slices or voxels, before splitting.
- Compare holdouts ISPY1, ISPY2, DUKE, NACT separately before generalization.

## Hypothesis

H1: VAE representations vary by tumor properties after accounting for source
cohort. H0: clustering mostly reflects acquisition/cohort. Check whether
cluster-center shifts shrink after scanner-stratified analysis. These are
future study hypotheses, not evaluated experimental conclusions.
