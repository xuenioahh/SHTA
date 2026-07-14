# Anonymity Checklist

Use this checklist before distributing the anonymous review link.

## Required Checks

- `README.md` does not include author names, affiliations, personal emails, or personal GitHub URLs.
- `CITATION.cff` uses `Anonymous Authors`.
- Repository links point to the anonymous 4open root URL.
- Example paths use placeholders such as `/path/to/Synapse` and `/path/to/AMOS`.
- Launchers accept user paths through environment variables such as `ROOT_PATH`, `SPLIT_DIR`, `SAVE_PATH`, and `GA_ROOT`.
- The git-tracked file list does not include datasets, checkpoints, logs, predictions, or local outputs.
- Documentation describes reproducible commands rather than local experiment history.

## Suggested Commands

```bash
git ls-files
rg -n "author|affiliation|github.com|/home|Desktop|SCDL|10\\.12" .
git status --short
```

Expected result:

- `git ls-files` lists only source code, launchers, documentation, and small method assets.
- the `rg` command reports only intentional anonymous metadata or checklist text.
- `git status --short` is empty before upload.

## After Acceptance

- author names in `CITATION.cff`
- final paper citation
- public repository URL
- checkpoint download links, if released
