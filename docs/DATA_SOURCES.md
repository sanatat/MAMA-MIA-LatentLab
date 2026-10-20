# Data provenance / access audit — 19 October 2026

## Professor-linked candidate projects

| Study | Source data | Public access for this project? | Decision |
| --- | --- | --- | --- |
| VAI-B (Cossío et al., 2023, doi:10.1117/1.JMI.10.6.061404) | Swedish mammography and cancer registry, hospital-controlled | Public `mammoai/cobra-db` code; clinical images and outcomes NOT released in the paper | Not runnable without approved access |
| MAMA-MIA (Garrucho et al., 2025, doi:10.1038/s41597-025-04707-4) | 1,506 cases selected from TCIA I-SPY1, I-SPY2, DUKE, NACT | Images, expert masks and harmonized tables under Synapse syn60868042 with terms; supplementary published paper tables | Chosen research base |

The author-contribution section in Garrucho et al. identifies **Lidia Garrucho**
as the study lead. **Apostolia Tsirikoglou** helped segmentation experts and built
paper tables; **Fredrik Strand** was among study supervisors. It is inaccurate
to call this either researcher's exclusive dataset.

## Public dataset links

- https://www.nature.com/articles/s41597-025-04707-4
- https://www.synapse.org/Synapse:syn60868042
- https://github.com/LidiaGarrucho/MAMA-MIA
- https://github.com/mammoai/cobra-db

## Clinical metadata caveat

The original 49 harmonized MAMA-MIA variables combine demographics, clinical
characteristics, treatment/outcomes, and imaging acquisition metadata; they are
not 49 valid *pre-treatment model features*. pCR, survival, recurrence and
post-treatment details must not be supplied to the VAE or imaging surrogate.
This prototype uses ONLY extracted features from baseline scans and optional
post-hoc pCR overlay.

## External example requested by user

https://zenodo.org/records/23279735 could not be fetched through available
interfaces on the audit date. Methods here are independently implemented and
are NOT a reproduction or adaptation of any inaccessible Zenodo source code.
