"""Tokenized dataset handling: one-time tokenization cached to disk.

Sequences are stored unpadded (truncated at MAX_LEN); batches are padded to
their longest member at collation time (dynamic padding).
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from . import config

PAD_TOKEN_ID = 0  # [PAD] in the DistilBERT vocab


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


def collate(batch):
    ids, attn, labs = zip(*batch)
    return (
        pad_sequence(ids, batch_first=True, padding_value=PAD_TOKEN_ID),
        pad_sequence(attn, batch_first=True, padding_value=0),
        torch.stack(labs),
    )


def tokenize_all() -> None:
    from datasets import load_from_disk

    ds = load_from_disk(config.DATA_DIR / "ag_news_raw")
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    for split in ("train", "test"):
        enc = tok(
            list(ds[split]["text"]),
            truncation=True,
            max_length=config.MAX_LEN,
            padding=False,
            return_attention_mask=True,
        )
        ids = [torch.tensor(r, dtype=torch.long) for r in enc["input_ids"]]
        attn = [torch.tensor(r, dtype=torch.long) for r in enc["attention_mask"]]
        labels = torch.tensor(list(ds[split]["label"]), dtype=torch.long)
        torch.save((ids, attn, labels), config.DATA_DIR / f"tokens_{split}.pt")
        print(f"cached {split}: {len(ids)} examples")


def load_tokens(split: str) -> TokenDataset:
    path = config.DATA_DIR / f"tokens_{split}.pt"
    if not path.exists():
        tokenize_all()
    ids, attn, labels = torch.load(path, weights_only=True)
    return TokenDataset(ids, attn, labels)
