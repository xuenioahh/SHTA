# Documentation

This directory contains the review-stage documentation for the anonymous SHTA source release.

## Files

| File | Purpose |
| --- | --- |
| `ANONYMOUS_RELEASE_STRUCTURE.md` | Defines the source-only release layout, included files, excluded artifacts, and external execution order. |
| `REPRODUCE_PAPER_RESULTS.md` | Maps paper settings to released scripts and gives runnable training/evaluation commands. |
| `DATA_PREPARATION.md` | Documents expected processed Synapse/AMOS layouts and split conventions. |
| `EQUATION_TO_CODE.md` | Maps paper notation to implementation variables. |
| `PAPER_RESULTS.md` | States which paper-result rows are command-reproducible and what remains unavailable in the anonymous release. |
| `ANONYMOUS_SYNC.md` | Records the 4open anonymous link and update workflow. |
| `CODE_MAP.md` | Maps paper components to executable source files without duplicating code. |
| `RELEASE_CONTENT_AUDIT.md` | Explains why each top-level item is kept or removed. |
| `ANONYMITY_CHECKLIST.md` | Lists the final anonymity checks before sharing the review link. |
| `assets/shta_overview.png` | Method overview figure used by the repository README. |
| `assets/shta_overview.pdf` | PDF copy of the method overview figure. |

## Recommended Reading Order

1. Read the repository-level `README.md` for setup, data format, training, and evaluation.
2. Read `REPRODUCE_PAPER_RESULTS.md` to match paper settings to scripts.
3. Read `DATA_PREPARATION.md` before preparing local data files.
4. Read `PAPER_RESULTS.md` to understand current table-reproduction boundaries.
5. Read `EQUATION_TO_CODE.md` and `CODE_MAP.md` to locate the implementation of each paper component.
6. Read `ANONYMOUS_RELEASE_STRUCTURE.md` to understand what is intentionally included or excluded.
7. Use `ANONYMITY_CHECKLIST.md` before distributing an anonymous review link.
