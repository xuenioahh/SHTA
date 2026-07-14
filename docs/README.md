# Documentation

This directory contains the review-stage documentation for the anonymous SHTA source release.

## Files

| File | Purpose |
| --- | --- |
| `ANONYMOUS_RELEASE_STRUCTURE.md` | Defines the source-only release layout, included files, excluded artifacts, and external execution order. |
| `REPRODUCE_PAPER_RESULTS.md` | Maps paper settings to released scripts and gives runnable training/evaluation commands. |
| `CODE_MAP.md` | Maps paper components to executable source files without duplicating code. |
| `RELEASE_CONTENT_AUDIT.md` | Explains why each top-level item is kept or removed. |
| `ANONYMITY_CHECKLIST.md` | Lists the final anonymity checks before sharing the review link. |
| `assets/shta_overview.png` | Method overview figure used by the repository README. |
| `assets/shta_overview.pdf` | PDF copy of the method overview figure. |

## Recommended Reading Order

1. Read the repository-level `README.md` for setup, data format, training, and evaluation.
2. Read `REPRODUCE_PAPER_RESULTS.md` to match paper settings to scripts.
3. Read `CODE_MAP.md` to locate the implementation of each paper component.
4. Read `ANONYMOUS_RELEASE_STRUCTURE.md` to understand what is intentionally included or excluded.
5. Use `ANONYMITY_CHECKLIST.md` before distributing an anonymous review link.
