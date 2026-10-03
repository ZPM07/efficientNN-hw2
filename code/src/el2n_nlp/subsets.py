"""Subset construction: average scores across runs, keep top-k, random draws."""

import json

import numpy as np

from . import config
from .scores import mean_scores


def keep_top_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Global top-k by score, returned as sorted indices (official keep_max_scores)."""
    return np.sort(scores.argsort()[-k:]).astype(np.int64)


def random_subset(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, k, replace=False)).astype(np.int64)


def select_all() -> None:
    """Build the averaged EL2N ranking, all fixed EL2N subsets and the
    per-run random subsets; record class composition of each EL2N subset."""
    from datasets import load_from_disk

    config.SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = config.scoring_seeds()
    runs = [np.load(config.score_path(s)) for s in seeds]
    avg = mean_scores(runs)
    np.save(config.SUBSETS_DIR / "el2n_mean_scores.npy", avg)

    labels = np.array(load_from_disk(config.DATA_DIR / "ag_news_raw")["train"]["label"])

    stats = {"score_runs": seeds, "el2n_subsets": {}, "random_subsets": {}}
    for pct in config.SIZES_PCT:
        k = config.subset_size(pct)
        idx = keep_top_scores(avg, k)
        np.save(config.el2n_subset_path(pct), idx)
        counts = np.bincount(labels[idx], minlength=config.NUM_CLASSES)
        stats["el2n_subsets"][f"{pct}pct"] = {
            "size": int(k),
            "class_counts": counts.tolist(),
            "mean_score": float(avg[idx].mean()),
        }

    for pct in config.SIZES_PCT:
        k = config.subset_size(pct)
        for r in range(config.RETRAIN_RUNS):
            seed = config.random_subset_seed(pct, r)
            idx = random_subset(config.NUM_TRAIN, k, seed)
            np.save(config.random_subset_path(pct, r), idx)
            counts = np.bincount(labels[idx], minlength=config.NUM_CLASSES)
            stats["random_subsets"][f"{pct}pct_run{r}"] = {
                "size": int(k),
                "seed": seed,
                "class_counts": counts.tolist(),
            }

    with open(config.SUBSETS_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("subsets written to", config.SUBSETS_DIR)


def load_subset(mode: str, pct: int, run: int) -> np.ndarray:
    if mode == "el2n":
        return np.load(config.el2n_subset_path(pct))
    if mode == "random":
        return np.load(config.random_subset_path(pct, run))
    raise ValueError(f"unexpected subset mode: {mode}")
