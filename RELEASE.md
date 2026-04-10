# Release Checklist

This repository is set up for a first public GitHub push and a Zenodo-backed software release.

## Before The First Public Push

1. Replace the placeholder GitHub URL in `CITATION.cff` with the real repository URL.
2. Review whether all generated outputs under `outputs/` should be published as part of version `0.1.0`.

## Suggested First GitHub Push

If this directory is not yet a Git repository:

```bash
git init -b main
git add .
git commit -m "Prepare public release"
```

Then create the GitHub repository and connect it:

```bash
git remote add origin https://github.com/<owner>/csd-sulcus-model.git
git push -u origin main
```

## Zenodo Preparation

The repository already includes:

- `CITATION.cff` for GitHub citation metadata
- `.zenodo.json` for Zenodo-specific release metadata

Zenodo notes:

- If both files are present, Zenodo uses `.zenodo.json` for GitHub release archiving.
- GitHub still uses `CITATION.cff` to display the "Cite this repository" panel.

## First Zenodo-Backed Release

1. Connect the GitHub repository to Zenodo.
2. Enable the repository in the Zenodo GitHub integration.
3. Create a GitHub release, for example `v0.1.0`.
4. Wait for Zenodo to archive the release and mint the DOI.
5. Update:
   - `CITATION.cff` with the minted DOI
   - `.zenodo.json` if you want to add funding or the related manuscript DOI
   - `manuscript/reframed_submission.tex` data-availability text with the real public repository or DOI

The repository is already configured for MIT licensing.

## Recommended Post-DOI Follow-Up

After Zenodo assigns a DOI, create one small metadata-only commit that updates the repository citation and manuscript availability statement so the archived software and the manuscript point to one another cleanly.
