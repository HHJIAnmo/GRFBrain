# Data format

## Input manifest for preprocessing

One JSON object per line:

```json
{"id":"clip_000001","path":"/data/aligned/clip_000001.npy","label":1,"split":"train"}
```

`path` must contain a float array shaped `(19, 2400)` (12 seconds at 200 Hz), or
an NPZ with key `signal`. Channel alignment and TUSZ EDF/TSE segmentation remain
dataset-specific and must be completed before this public pipeline. IDs must be
unique and labels binary.

## Processed sample

Each output NPZ contains:

| Key | Shape | Meaning |
|---|---:|---|
| `history_x` | `(11,19,100)` | standardized one-second log-FFT history |
| `history_a` | `(11,19,19)` | directed absolute-Spearman top-3 graphs |
| `target_x` | `(19,100)` | standardized final-window node target |
| `target_a` | `(19,19)` | final directed graph |
| `target_edges` | `(171,)` | upper triangle of symmetrized final graph |
| `label` | scalar | seizure label for the original 12-second clip |
| `id` | string | stable sample identifier |

Normalization statistics are fitted on **training node windows only**, per FFT
frequency coordinate, and saved to `normalization.npz`. Dev and test must reuse
that file. Never refit preprocessing from test.
