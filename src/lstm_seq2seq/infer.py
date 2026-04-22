from __future__ import annotations

from pathlib import Path

import torch

from .data import SentencePieceTokenizer
from .model import Seq2SeqLSTM
from .train import resolve_device


def predict_text(
    text: str,
    checkpoint_path: str | Path,
    device: str = "auto",
    max_source_tokens: int | None = None,
    max_decode_tokens: int | None = None,
) -> str:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config = checkpoint["config"]
    tokenizer = SentencePieceTokenizer(checkpoint["tokenizer_path"])
    runtime_device = resolve_device(device)

    model = Seq2SeqLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(runtime_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source_limit = max_source_tokens or config["max_source_tokens"]
    decode_limit = max_decode_tokens or config["max_target_tokens"]
    src_ids = tokenizer.encode(text, max_tokens=source_limit)
    src = torch.tensor([src_ids], dtype=torch.long, device=runtime_device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=runtime_device)
    prediction = model.greedy_decode(src, src_lengths, max_steps=decode_limit)[0]
    return tokenizer.decode(prediction)
