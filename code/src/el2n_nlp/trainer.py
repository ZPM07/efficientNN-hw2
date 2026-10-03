"""Fixed-step training loop.

Total optimizer steps and the LR schedule are identical for every run
(full data, EL2N subset, random subset): training cycles over smaller
subsets more times instead of stopping earlier (paper, Appendix B.2).
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import config
from .data import collate, load_tokens
from .modeling import build_model, evaluate, forward_batch
from .scores import compute_train_scores
from .subsets import load_subset


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(warmup, 1)
    return base * (total - step) / max(total - warmup, 1)


def shuffled_batches(idxs: np.ndarray, bs: int, total_steps: int, seed: int):
    """Infinite epoch cycling: fresh permutation per epoch, tail dropped,
    exactly `total_steps` full batches."""
    rng = np.random.RandomState(seed)
    n = len(idxs)
    per_epoch = max(n // bs, 1)
    yielded = 0
    while yielded < total_steps:
        order = rng.permutation(n)[: per_epoch * bs]
        for b in range(per_epoch):
            if yielded >= total_steps:
                return
            yield idxs[order[b * bs : (b + 1) * bs]]
            yielded += 1


def make_optimizer(model) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if "bias" in name or "LayerNorm.weight" in name else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.LR,
    )


def _gather(dataset, idxs):
    return collate([dataset[int(i)] for i in idxs])


def resolve_score_stop(explicit: int | None) -> int:
    """Scores are taken one full epoch into the full-length schedule."""
    return explicit or config.SCORING_STOP_STEP


def train_run(
    mode: str,
    pct: int,
    run: int,
    seed: int,
    device: str = "cuda",
    total_steps: int | None = None,
    score_mode: bool = False,
    score_stop: int | None = None,
) -> dict:
    """One reproduction run. score_mode=True: stop at `score_stop`, emit EL2N
    scores of the full training set. Else train to `total_steps` and report
    final/best test accuracy."""
    total = total_steps or config.TOTAL_STEPS
    t0 = time.time()
    torch.manual_seed(seed)

    train_ds = load_tokens("train")
    test_ds = load_tokens("test")
    model = build_model(seed, device)
    optimizer = make_optimizer(model)

    idxs: np.ndarray = (
        np.arange(config.NUM_TRAIN, dtype=np.int64)
        if mode == "full"
        else load_subset(mode, pct, run)
    )
    batches = shuffled_batches(idxs, config.BATCH_SIZE, total, seed=seed + 1)

    history, best_acc = [], 0.0
    stop_at = resolve_score_stop(score_stop) if score_mode else total
    ce = nn.CrossEntropyLoss()
    model.train()
    running_loss, log_every = 0.0, 50

    for step, batch_idx in enumerate(batches, start=1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, total, config.WARMUP_STEPS, config.LR)
        ids, attn, lab = _gather(train_ds, batch_idx)
        batch = (ids, attn, lab)
        logits, labels = forward_batch(model, batch, device)
        loss = ce(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        running_loss += loss.item()

        if step % log_every == 0:
            print(
                f"[{mode} {pct}% run{run}] step {step}/{total} "
                f"loss {running_loss / log_every:.4f} "
                f"lr {lr_at(step, total, config.WARMUP_STEPS, config.LR):.2e}",
                flush=True,
            )
            running_loss = 0.0

        if step % config.EVAL_EVERY == 0 or step == total:
            acc, tl = evaluate(model, test_ds, device)
            best_acc = max(best_acc, acc)
            history.append({"step": step, "test_acc": acc, "test_loss": tl})
            print(f"    eval @ {step}: acc {acc:.4f} loss {tl:.4f}", flush=True)

        if step == stop_at and score_mode:
            break

    result = {
        "mode": mode,
        "pct": pct,
        "run": run,
        "seed": seed,
        "total_steps": stop_at,
        "subset_size": int(len(idxs)),
        "wall_time_s": round(time.time() - t0, 1),
        "history": history,
    }

    if score_mode:
        scores = compute_train_scores(model, train_ds, device)
        config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.score_path(seed), scores)
        result["final_test_acc"] = history[-1]["test_acc"] if history else None
        with open(config.SCORES_DIR / f"scores_seed{seed}_meta.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"scores saved for seed {seed}", flush=True)
    else:
        result["final_test_acc"] = history[-1]["test_acc"]
        result["best_test_acc"] = best_acc
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.run_path(mode, pct, run), "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"DONE {config.run_tag(mode, pct, run)}: "
            f"final {result['final_test_acc']:.4f} best {best_acc:.4f}",
            flush=True,
        )
    return result
