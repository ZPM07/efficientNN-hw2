"""Qualitative inspection: highest / lowest mean-EL2N training examples per class."""

import json

import numpy as np

from . import config


def dump_examples(n_per_class: int = 6) -> None:
    from datasets import load_from_disk

    avg = np.load(config.SUBSETS_DIR / "el2n_mean_scores.npy")
    texts = list(load_from_disk(config.DATA_DIR / "ag_news_raw")["train"]["text"])
    labels = np.array(load_from_disk(config.DATA_DIR / "ag_news_raw")["train"]["label"])

    order = avg.argsort()  # ascending
    out = {"highest": {}, "lowest": {}}
    for c in range(config.NUM_CLASSES):
        hi = [i for i in order[::-1] if labels[i] == c][:n_per_class]
        lo = [i for i in order if labels[i] == c][:n_per_class]
        out["highest"][config.CLASS_NAMES[c]] = [
            {"idx": int(i), "score": round(float(avg[i]), 4), "text": texts[i]} for i in hi
        ]
        out["lowest"][config.CLASS_NAMES[c]] = [
            {"idx": int(i), "score": round(float(avg[i]), 4), "text": texts[i]} for i in lo
        ]
    path = config.RESULTS_DIR / "top_examples.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("written", path)
