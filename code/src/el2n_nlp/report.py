"""Collect runs into summary.json, a Markdown table and figures."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from . import config

COLORS = {"el2n": "tab:red", "random": "tab:blue", "full": "black"}


def load_runs() -> dict:
    out = {}
    for f in sorted(config.RUNS_DIR.glob("*.json")):
        d = json.load(open(f))
        out[f.stem] = d
    return out


def summarize(runs: dict) -> dict:
    groups = {}
    for mode in ("full", "el2n", "random"):
        for pct in ([100] if mode == "full" else config.SIZES_PCT):
            accs, bests = [], []
            for r in range(config.RETRAIN_RUNS):
                d = runs.get(config.run_tag(mode, pct, r))
                if d:
                    accs.append(d["final_test_acc"])
                    bests.append(d["best_test_acc"])
            if accs:
                groups[f"{mode}_{pct}"] = {
                    "n": len(accs),
                    "final_mean": float(np.mean(accs)),
                    "final_std": float(np.std(accs)),
                    "final_min": float(np.min(accs)),
                    "final_max": float(np.max(accs)),
                    "best_mean": float(np.mean(bests)),
                    "accs": [round(a, 4) for a in accs],
                }
    return groups


def make_table(summary: dict) -> str:
    lines = [
        "| retained | full | EL2N (keep top) | random | EL2N − random |",
        "|---|---|---|---|---|",
    ]
    full = summary.get("full_100")
    full_s = f"{full['final_mean'] * 100:.2f} ± {full['final_std'] * 100:.2f}" if full else "—"
    for pct in config.SIZES_PCT:
        e, r = summary.get(f"el2n_{pct}"), summary.get(f"random_{pct}")
        es = f"{e['final_mean'] * 100:.2f} ± {e['final_std'] * 100:.2f}" if e else "—"
        rs = f"{r['final_mean'] * 100:.2f} ± {r['final_std'] * 100:.2f}" if r else "—"
        diff = (
            f"{(e['final_mean'] - r['final_mean']) * 100:+.2f} pp" if e and r else "—"
        )
        lines.append(f"| {pct}% | {full_s if pct == config.SIZES_PCT[0] else '—'} | {es} | {rs} | {diff} |")
    note = f"\nFull-dataset runs: {full_s} (same value repeated for reference)."
    return "\n".join(lines) + note


def fig_accuracy_vs_fraction(summary: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ("el2n", "random"):
        pts = [(p, summary[f"{mode}_{p}"]) for p in config.SIZES_PCT if f"{mode}_{p}" in summary]
        if not pts:
            continue
        xs = np.array([pct for pct, _ in pts], dtype=float)
        means = np.array([g["final_mean"] for _, g in pts]) * 100
        stds = np.array([g["final_std"] for _, g in pts]) * 100
        ax.plot(xs, means, "o-", color=COLORS[mode],
                label="EL2N top-score" if mode == "el2n" else "random")
        ax.fill_between(xs, means - stds, means + stds, color=COLORS[mode], alpha=0.15)
        for (pct, g) in pts:
            for a in g["accs"]:
                ax.scatter(pct, a * 100, s=14, color=COLORS[mode], alpha=0.55, zorder=3)
    if "full_100" in summary:
        g = summary["full_100"]
        for a in g["accs"]:
            ax.scatter(100, a * 100, s=14, color="black", alpha=0.55, zorder=3)
        ax.axhline(g["final_mean"] * 100, ls="--", c="gray", lw=1, label="full data (100%)")
    ax.set_xlabel("retained fraction of training set (%)")
    ax.set_ylabel("final test accuracy (%)")
    ax.set_title("EL2N data diet on AG News (DistilBERT): accuracy vs dataset fraction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "accuracy_vs_fraction.png", dpi=150)
    plt.close(fig)


def fig_curves(runs: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    panels = [("full", 100), ("el2n", 10), ("el2n", 50), ("random", 10)]
    for ax, (mode, pct) in zip(axes.flat, panels):
        for r in range(config.RETRAIN_RUNS):
            d = runs.get(config.run_tag(mode, pct, r))
            if not d:
                continue
            steps = [h["step"] for h in d["history"]]
            accs = [h["test_acc"] * 100 for h in d["history"]]
            ax.plot(steps, accs, lw=1.2, color=COLORS[mode], alpha=0.8)
        ax.set_title(f"{mode} {pct}%")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("optimizer step")
    axes[1, 1].set_xlabel("optimizer step")
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[1, 0].set_ylabel("test accuracy (%)")
    fig.suptitle("Test accuracy vs fixed step budget (same steps for every subset size)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "test_curves.png", dpi=150)
    plt.close(fig)


def fig_scores():
    mean_path = config.SUBSETS_DIR / "el2n_mean_scores.npy"
    if not mean_path.exists():
        return
    avg = np.load(mean_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(avg, bins=60, color="tab:red", alpha=0.8)
    for pct in config.SIZES_PCT:
        cut = np.sort(avg)[::-1][config.subset_size(pct) - 1]
        axes[0].axvline(cut, ls="--", lw=1, label=f"top {pct}%")
    axes[0].set_xlabel("mean EL2N score (10 runs)")
    axes[0].set_ylabel("# train examples")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distribution of mean EL2N scores")

    # run-to-run rank consistency (Spearman between the 10 scoring runs)
    runs = [np.load(config.score_path(s)) for s in config.scoring_seeds()]
    rhos = [
        spearmanr(runs[i], runs[j]).statistic
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    axes[1].hist(rhos, bins=20, color="tab:purple", alpha=0.8)
    axes[1].set_xlabel("Spearman ρ between pairs of scoring runs")
    axes[1].set_title(f"Rank agreement across initializations (mean ρ={np.mean(rhos):.3f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "scores_analysis.png", dpi=150)
    plt.close(fig)


def fig_class_balance():
    stats_path = config.SUBSETS_DIR / "stats.json"
    if not stats_path.exists():
        return
    stats = json.load(open(stats_path))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    for i, (kind, color) in enumerate([("el2n_subsets", "tab:red"), ("random_subsets", "tab:blue")]):
        key = f"{config.SIZES_PCT[0]}pct" if kind == "el2n_subsets" else f"{config.SIZES_PCT[0]}pct_run0"
        counts = np.array(stats[kind][key]["class_counts"], dtype=float)
        frac = counts / counts.sum() * 100
        ax.bar(np.arange(4) + i * width - width / 2, frac, width, color=color, label=f"{kind.split('_')[0]} @ {config.SIZES_PCT[0]}%")
    ax.axhline(25, ls="--", c="gray", lw=1, label="balanced (25%)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_ylabel("% of subset")
    ax.set_title("Class composition of selected subsets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def collect():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    summary = summarize(runs)
    table = make_table(summary)
    print(table)
    with open(config.RESULTS_DIR / "table.md", "w") as f:
        f.write(table + "\n")
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fig_accuracy_vs_fraction(summary)
    fig_curves(runs)
    fig_scores()
    fig_class_balance()
    print("figures ->", config.FIGURES_DIR)
