"""CLI: tokenize | score | select | train."""

import argparse

from . import config
from .trainer import train_run


def main():
    p = argparse.ArgumentParser(prog="el2n-nlp")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("tokenize")

    sp = sub.add_parser("score", help="one scoring run: stop early, emit EL2N scores")
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--stop-step", type=int, default=None)
    sp.add_argument("--total-steps", type=int, default=None)
    sp.add_argument("--tag", default="", help="suffix for score file (diagnostics)")

    sp = sub.add_parser("select", help="average scores, build all subsets")

    sub.add_parser("collect", help="summary table + figures from finished runs")

    sub.add_parser("examples", help="top/bottom EL2N training texts per class")

    sp = sub.add_parser("train", help="one retraining run from scratch")
    sp.add_argument("--mode", choices=["full", "el2n", "random"], required=True)
    sp.add_argument("--pct", type=int, default=100)
    sp.add_argument("--run", type=int, required=True)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--total-steps", type=int, default=None)
    sp.add_argument("--out-suffix", default="", help="append to run tag (smoke runs)")

    a = p.parse_args()

    if a.cmd == "tokenize":
        from .data import tokenize_all

        tokenize_all()
    elif a.cmd == "score":
        if a.tag:
            config.score_path = lambda s, t=a.tag: config.SCORES_DIR / f"scores_seed{s}{t}.npy"
        train_run(
            "full",
            100,
            run=-1,
            seed=a.seed,
            device=a.device,
            total_steps=a.total_steps,
            score_mode=True,
            score_stop=a.stop_step,
        )
    elif a.cmd == "select":
        from .subsets import select_all

        select_all()
    elif a.cmd == "collect":
        from .report import collect

        collect()
    elif a.cmd == "examples":
        from .examples import dump_examples

        dump_examples()
    elif a.cmd == "train":
        if a.mode == "full":
            a.pct = 100
        if a.out_suffix:
            orig = config.run_path

            config.run_path = lambda m, pc, r: orig(m, pc, r).with_stem(
                orig(m, pc, r).stem + a.out_suffix
            )
        train_run(
            a.mode,
            a.pct,
            a.run,
            seed=config.retrain_model_seed(a.mode, a.pct, a.run),
            device=a.device,
            total_steps=a.total_steps,
        )


if __name__ == "__main__":
    main()
