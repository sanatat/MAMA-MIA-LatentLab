# Three-phase DCE-MRI extension — work in progress

An additional MRI phase (`_0002`) was downloaded for all 80 cases. Acquisition-time metadata were valid in 79 cases and incomplete/invalid in `ISPY1_1184`.

The acquisition-time medians (seconds relative to imaging baseline; **not necessarily time since injection**) differed markedly by source:

| Source | First post-contrast | Second post-contrast |
|---|---:|---:|
| DUKE | 202 | 332 |
| ISPY1 | 386 | 648 |
| ISPY2 | 135 | 222.5 |
| NACT | 422 | 773.5 |

The utility of temporal imaging has **not yet been validated**. A proposed pipeline is to verify phase-to-mask registration, compute simple enhancement/change features using acquisition times, compare static-only and static-plus-temporal baselines with strict cohort holdout, and study the effect of scanner/voxel spacing. Shape-and-affine matching alone is not sufficient to rule out tissue motion. Observed washout-like raw signal patterns must not be equated with formal clinical kinetic categories without validated protocol context.

Local scripts: `mamamia_phase2.py` and `mamamia_temporal_extract.py`. Both require the downloaded original images, clinical spreadsheet and local manifests in `results/`, none of which are included here.
