# Session Progress Log

## 2026-09-20 — initial standalone release complete

- Created an isolated MIT-licensed Python package with no raw TUSZ data,
  checkpoints, result caches, private paths or legacy SDE implementation.
- Added aligned-clip log-FFT preprocessing, train-only normalization, absolute
  Spearman directed top-3 graph construction and a stable NPZ dataset contract.
- Added exact-compatible 125,803-parameter graph forecaster, 36,032-parameter
  temporal condition encoder, 393,581-parameter bidirectional Joint Node–Edge
  field and 65-parameter max-node hidden readout.
- Added conditional GGRF node/edge sampling, normalized joint FM loss, train-only
  q calibration, Flow pretraining, no-transport downstream training, frozen-dev
  threshold evaluation and original-checkpoint import.
- Documented architecture, exact hyperparameters, interpretation, limitations,
  data governance and GitHub workflow.
- Verification: `./init.sh` passed 5 tests and `EEG_CGFM_SMOKE_OK`; all four
  CLIs parsed; wheel SHA-256 `5420f00a...548553`; harness audit 100/100.
- Compatibility: original/released field and graph state keys and tensor shapes
  are exact. Existing seed234 checkpoints converted successfully. Four real dev
  examples replayed with max logit error `5.31e-6`; the tiny difference is from
  comparing fresh CPU inference with a cache originally produced under another
  execution backend, not an architecture mismatch.

No feature is active. Before public upload, the owner should choose repository
name/remote, verify checkpoint redistribution rights, and add author/citation
metadata if desired.

## 2026-09-28 — GitHub preparation

- Excluded duplicate build output and generated package metadata from the source release.
- Extended ignore rules for local credentials, OS files, and EEG array files.
- Added a data-free verification section to the README.
- Fresh verification in isolated Python 3.11: 5 tests passed and
  `EEG_CGFM_SMOKE_OK`. PyTorch emitted its nested-tensor optimization warning.
- Preserved model implementation and original license. Historical real-data
  results above were not rerun during this packaging task.
- Destination requested: private `HHJIAnmo/eeg-conditional-ggrf-joint`.
  Initial source commit uploaded successfully to the private repository.

## 2026-09-28 — interactive HTML integration

- Added the supplied Plotly visualization as `docs/visualization/index.html`
  with document metadata, usage guidance, and a README entry.
- Removed original recording filenames and global source indices from all
  20 sample metadata entries; preserved graph arrays and interactive controls.
- Browser verification: both 3D panels render; switching to signed velocity,
  selecting S01, and advancing transport time to 1 update the page.
- Public online hosting awaits the owner's visibility decision. The existing
  code repository remains private.
