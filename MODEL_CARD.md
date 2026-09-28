# Model card

## Intended use

Research on causal EEG graph representation learning and 12-second binary
seizure detection. The code is not a medical device and must not be used for
clinical diagnosis or treatment decisions.

## Released method

The model uses eleven one-second history windows to condition a GGRF-trained,
bidirectional Joint Node–Edge Transformer. The default classifier reads node
hidden states at zero residual and `tau=1`; it does not run an ODE solver.

## Existing internal TUSZ evidence

Three-seed frozen-threshold test summary from the source project:

| Readout | Accuracy | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|
| node hidden no transport | 0.87697 ± 0.01103 | 0.87787 ± 0.00354 | 0.51589 ± 0.00367 | 0.52268 ± 0.01400 |
| node hidden max + transport | 0.88524 ± 0.00990 | 0.86701 ± 0.00034 | 0.47669 ± 0.00494 | 0.50489 ± 0.00457 |

These numbers are provenance context, not bundled executable evidence: TUSZ
data, split manifests, caches and weights are intentionally absent. Reproduce
them only with authorized data and frozen original memberships.

## Limitations

- Current graph is absolute Spearman top-3 over log-FFT profiles, not coherence,
  PLV, STFT connectivity, or anatomical connectivity.
- GGRF supplies graph-conditioned anisotropic Gaussian noise; it is not proven
  to lie on a physiological brain manifold.
- Flow transport time is not physical EEG time.
- The best downstream variant bypasses test-time transport; results support
  useful Flow-trained representation learning, not a transport-caused gain.
- TUSZ site, montage and class-distribution shifts require external validation.
- Thresholds must be selected on development data and frozen before test.

## Privacy and data governance

Do not commit EEG signals, patient identifiers, local absolute paths, generated
caches or checkpoints without explicit authorization and de-identification.
