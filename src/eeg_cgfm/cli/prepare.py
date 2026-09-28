from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_cgfm.data.preprocessing import build_sample, fit_normalization, load_signal, log_fft_windows, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stable EEG conditional-Flow NPZ samples")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization", help="Train normalization.npz for dev/test")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    if args.normalization:
        with np.load(args.normalization, allow_pickle=False) as stats:
            mean, std = stats["mean"], stats["std"]
    else:
        features = np.stack([log_fft_windows(load_signal(row["path"])) for row in tqdm(rows, desc="fit")])
        mean, std = fit_normalization(features)
        np.savez(output / "normalization.npz", mean=mean, std=std)
    manifest = output / "manifest.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for row in tqdm(rows, desc="prepare"):
            sample = build_sample(load_signal(row["path"]), int(row["label"]), row["id"], mean, std, args.top_k)
            path = output / f'{row["id"]}.npz'
            with path.open("xb") as handle:
                np.savez_compressed(handle, **sample)
            stream.write(json.dumps({"id": row["id"], "path": path.name, "label": int(row["label"])}) + "\n")


if __name__ == "__main__": main()
