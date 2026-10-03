"""Model construction and forward helpers."""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from . import config


def build_model(seed: int, device: str = "cuda") -> nn.Module:
    """Pretrained DistilBERT + fresh linear head; `seed` controls head init."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        torch_dtype=torch.float32,
    )
    return model.to(device)


def forward_batch(model, batch, device: str):
    input_ids, attn, labels = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attn)
    return out.logits.float(), labels


@torch.no_grad()
def evaluate(model, dataset, device: str, batch_size: int = config.EVAL_BATCH_SIZE):
    """Test-set accuracy and mean cross-entropy loss."""
    from torch.utils.data import DataLoader

    from .data import collate

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    n_correct, n, loss_sum = 0, 0, 0.0
    for batch in loader:
        logits, labels = forward_batch(model, batch, device)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        n_correct += (logits.argmax(-1) == labels).sum().item()
        n += labels.numel()
    model.train()
    return n_correct / n, loss_sum / n
