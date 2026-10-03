"""EL2N score: || p_theta(x) - onehot(y) ||_2 (Paul et al. 2021, eq. 3)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .data import collate
from .modeling import forward_batch


def el2n_from_probs(p, labels) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    onehot = np.eye(p.shape[1])[np.asarray(labels)]
    return np.linalg.norm(p - onehot, axis=1)


def mean_scores(runs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(runs), axis=0)


@torch.no_grad()
def compute_train_scores(model, dataset, device: str) -> np.ndarray:
    """EL2N for every training example, in original dataset order."""
    loader = DataLoader(
        dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    model.eval()
    scores = []
    for batch in loader:
        logits, labels = forward_batch(model, batch, device)
        probs = torch.softmax(logits.float(), dim=-1)
        scores.append(el2n_from_probs(probs.cpu(), labels.cpu()))
    model.train()
    return np.concatenate(scores)
