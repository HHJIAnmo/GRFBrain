# Conditional GGRF + Joint Node–Edge Transformer for EEG

This repository is a clean, shareable extraction of the strongest deterministic
downstream representation discovered in the EvoBrain ICLR-2027 study. It learns
an EEG-conditioned Joint Node–Edge velocity network with a graph Gaussian random
field (GGRF) source, then performs seizure classification with the
`node_hidden_no_transport` readout.

## Interactive 3D visualization

Explore predicted FC states and signed edge velocities for non-seizure and
seizure examples. Rotate and zoom the linked 3D views, select class medians or
individual samples, and adjust brain opacity and transport time τ.

**[Open the interactive 3D visualization →](https://hhjianmo.github.io/eeg-conditional-ggrf-visualization/)**

Or **[download the interactive HTML](docs/visualization/index.html)** using the
file page's **Download raw file** button, then open it in a browser. An internet
connection is required to load Plotly. GitHub's README and file preview do not
execute interactive HTML; use the live page linked above.

Original recording filenames and source indices are omitted from this page.
The visualization shows transport trajectories; the default downstream
classifier still uses the deterministic `node_hidden_no_transport` readout
(`NFE=0`).

## Important interpretation

The overall representation is trained by conditional Flow Matching. The default
downstream classifier does **not** sample and integrate a Flow trajectory:

1. GGRF-conditioned Flow pretraining learns the Joint Node–Edge field.
2. Downstream inference sets node and edge residuals to zero (`NFE=0`).
3. The Joint field is evaluated once at transport time `tau=1` using past-only
   temporal, graph, Laplacian and covariance conditions.
4. Post-LayerNorm node tokens are classified by
   `ReLU -> shared Linear(64,1) -> max over 19 nodes`.

Therefore this is a **conditional GGRF-pretrained deterministic representation
readout**, not a claim that test-time ODE transport improves classification.
Physical EEG window time `s` and Flow transport time `tau` are always distinct.

## Data contract

- 19 aligned EEG channels sampled at 200 Hz.
- Twelve non-overlapping one-second windows per example.
- Each window is a 100-dimensional log-amplitude FFT per channel.
- The first 11 windows are the causal condition; window 12 is the future target.
- Dynamic graphs are absolute Spearman similarities with directed top-3
  sparsification. They are not coherence, PLV, or STFT graphs.
- Raw TUSZ files are not distributed. A JSONL manifest points to user-owned
  `.npy`/`.npz` clips and supplies labels and split membership.

See [docs/data_format.md](docs/data_format.md) and
[docs/architecture.md](docs/architecture.md).

## Installation

```bash
conda env create -f environment.yml
conda activate eeg-cgfm
pip install -e '.[dev]'
./init.sh
```

## Verification without EEG data

After installation, run `./init.sh` from the repository root. It runs the five
contract tests and a synthetic forward/backward smoke check; successful output
includes `EEG_CGFM_SMOKE_OK`. No EEG data or pretrained weights are required.

## Quick start

```bash
# 1. Convert aligned 19-channel, 12-second clips to the stable NPZ contract.
eeg-cgfm-prepare \
  --manifest manifests/train.jsonl \
  --output data/processed/train

# Repeat for dev/test. Fit normalization on train only and pass the saved stats
# to dev/test; see --help and docs/data_format.md.

# 2. Pretrain the conditional Joint Node–Edge Flow field.
eeg-cgfm-pretrain \
  --config configs/default.yaml \
  --train-manifest data/processed/train/manifest.jsonl \
  --dev-manifest data/processed/dev/manifest.jsonl \
  --output outputs/flow_pretrain

# 3. Train the deterministic node-hidden readout.
eeg-cgfm-train \
  --config configs/default.yaml \
  --flow-checkpoint outputs/flow_pretrain/best.pt \
  --train-manifest data/processed/train/manifest.jsonl \
  --dev-manifest data/processed/dev/manifest.jsonl \
  --output outputs/node_hidden_no_transport

# 4. Apply the frozen dev-selected checkpoint and threshold to test.
eeg-cgfm-evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/node_hidden_no_transport/best.pt \
  --manifest data/processed/test/manifest.jsonl \
  --output outputs/test
```

The training CLIs are intentionally simple reference runners. For a paper run,
freeze split manifests, normalization statistics, seeds, checkpoints and dev
thresholds before opening test.

`eeg-cgfm-pretrain` re-estimates `q_X/q_A` from training data after fitting the
past-only graph forecaster and stores them in `calibration.json`. The supplied
YAML values reproduce the current TUSZ contract but are not universal constants.

## Importing an existing EvoBrain checkpoint

The released submodule names preserve the original field, temporal encoder,
graph forecaster and readout state dictionaries. If you are an authorized owner
of the original checkpoints, combine them into one portable file:

```bash
eeg-cgfm-import-evobrain \
  --config configs/default.yaml \
  --hidden-checkpoint /path/to/node_hidden_no_transport/best.pt \
  --edge-predictor-checkpoint /path/to/graph_forecaster/best.pth.tar \
  --output checkpoints/node_hidden_no_transport.pt
```

The command copies learned states and frozen source scales but does not copy raw
data, caches or private paths into the model itself. Check checkpoint sharing
permissions before publication.

## Repository contents

- `src/eeg_cgfm/data`: log-FFT preprocessing, Spearman top-k graph construction,
  stable NPZ dataset contract.
- `src/eeg_cgfm/models/ggrf.py`: history-conditioned graph Gaussian operators and
  node/edge source sampling.
- `src/eeg_cgfm/models/graph_forecaster.py`: deterministic conditional edge mean.
- `src/eeg_cgfm/models/joint_transformer.py`: bidirectional node-edge velocity
  field with FiLM, edge-biased node attention and node-to-edge messages.
- `src/eeg_cgfm/models/system.py`: temporal condition, normalized FM objective,
  and deterministic no-transport readout.
- `src/eeg_cgfm/cli`: data, training, evaluation and smoke entry points.
- `tests`: tensor-shape, causality, covariance, symmetry and gradient contracts.

## Reproducibility and licensing

No TUSZ data, pretrained weights, result caches, private paths or legacy SDE code
are included. TUSZ access and redistribution remain governed by its provider.
Code in this extraction is released under MIT; verify third-party dataset and
checkpoint permissions separately before public distribution.
