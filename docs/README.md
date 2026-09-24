# Documentation

This directory contains setup, data preparation, implementation mapping, and result-reproduction documentation for the public SHTA source release. The paper was accepted at IEEE BIBM 2026.

## Files

| File | Purpose |
| --- | --- |
| `ANONYMOUS_RELEASE_STRUCTURE.md` | Historical review-stage release notes; superseded by the public repository README. |
| `REPRODUCE_PAPER_RESULTS.md` | Maps paper settings to released scripts and gives runnable training/evaluation commands. |
| `DATA_PREPARATION.md` | Documents expected processed Synapse/AMOS layouts and split conventions. |
| `EQUATION_TO_CODE.md` | Maps paper notation to implementation variables. |
| `PAPER_RESULTS.md` | States which paper-result rows are command-reproducible and which artifacts are not included. |
| `ANONYMOUS_SYNC.md` | Historical instructions for the review-stage 4open package; not used for this public GitHub release. |
| `CODE_MAP.md` | Maps paper components to executable source files without duplicating code. |
| `RELEASE_CONTENT_AUDIT.md` | Explains why each top-level item is kept or removed. |
| `ANONYMITY_CHECKLIST.md` | Historical review-stage anonymity checklist; no longer applies after acceptance. |
| `assets/shta_overview.png` | Method overview figure used by the repository README. |
| `assets/shta_overview.pdf` | PDF copy of the method overview figure. |

## Recommended Reading Order

1. Read the repository-level `README.md` for setup, data format, training, and evaluation.
2. Read `REPRODUCE_PAPER_RESULTS.md` to match paper settings to scripts.
3. Read `DATA_PREPARATION.md` before preparing local data files.
4. Read `PAPER_RESULTS.md` to understand current table-reproduction boundaries.
5. Read `EQUATION_TO_CODE.md` and `CODE_MAP.md` to locate the implementation of each paper component.
6. Treat the anonymity and 4open documents as archived review history; follow the repository-level README for the current public release.
