# Publish to GitHub

Create a new **public, empty** GitHub repository named `MAMA-MIA-LatentLab` under `sanatat` at https://github.com/new. Do not initialize it with a README, gitignore or license; they are included.

From a clean directory extracted from `MAMA-MIA-LatentLab-GitHub.zip`:

```bash
git init -b main
git add .
git status --short
# Inspect this list: it must not contain patient data, local credentials, or Synapse tokens.
git commit -m "Publish MAMA-MIA LatentLab research pilot"
git remote add origin https://github.com/sanatat/MAMA-MIA-LatentLab.git
git push -u origin main
```

GitHub HTTPS pushes need GitHub authentication, **not** Synapse credentials. Do not publish the original clinical spreadsheet, local manifests, NIfTI files, masks, or per-patient predictions. All of these are intentionally excluded from this package.
