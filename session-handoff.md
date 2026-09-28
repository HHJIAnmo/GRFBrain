# Session Handoff

## Current objective

The standalone conditional GGRF + Joint Node–Edge Transformer release is
complete. No feature is active and no long-running process exists.

## Verification evidence

| Check | Result |
|---|---|
| `./init.sh` | 5 tests passed; `EEG_CGFM_SMOKE_OK` |
| wheel build | package `0.1.0` built successfully |
| harness audit | 100/100 |
| original architecture compatibility | exact state keys/shapes and parameter counts |
| current seed234 checkpoint import | portable checkpoint load passed |
| four-real-dev replay | max logit error `5.31e-6` |

## Decisions and boundaries

- “Current scheme” means conditional GGRF-pretrained Joint Node–Edge EEG
  representation learning. Default downstream inference is explicitly NFE=0.
- Data entry starts after TUSZ-specific channel alignment/segmentation. Raw data
  and provider-specific split files are not redistributed.
- The default YAML preserves current TUSZ source scales and architecture. New
  datasets require train-only scale calibration and fresh validation.

## Next session startup

1. Read `AGENTS.md`, `README.md`, and both files under `docs/`.
2. Run `./init.sh`.
3. Do not add data/checkpoints before checking permissions.

## Recommended next step

Initialize this directory as its own Git repository, add the chosen remote,
replace generic author metadata, optionally add `CITATION.cff`, and create a
tagged `v0.1.0` release only after a colleague reproduces the smoke in a fresh
environment.
