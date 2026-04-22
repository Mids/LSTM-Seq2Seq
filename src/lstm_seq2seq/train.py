from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    PAD_TOKEN,
    CsvSeq2SeqDataset,
    SentencePieceTokenizer,
    build_tokenizer,
    collate_batch,
)
from .model import Seq2SeqLSTM


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    train_size: int = 50000
    val_size: int = 2000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 7
    csv_path: str = "all_debate_combined.csv"
    artifact_dir: str = "artifacts"
    vocab_size: int = 16000
    tokenizer_samples: int = 200000
    max_source_tokens: int = 128
    max_target_tokens: int = 128


def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    csv_path = Path(config.csv_path)
    artifact_dir = Path(config.artifact_dir)
    cache_dir = artifact_dir / "cache"
    tokenizer_path = build_tokenizer(
        csv_path=csv_path,
        model_prefix=artifact_dir / "debate_unigram",
        vocab_size=config.vocab_size,
        sample_rows=config.tokenizer_samples,
        seed=config.seed,
    )
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    train_dataset = CsvSeq2SeqDataset.from_cache(
        csv_path=csv_path,
        tokenizer=tokenizer,
        size=config.train_size,
        offset=0,
        max_source_tokens=config.max_source_tokens,
        max_target_tokens=config.max_target_tokens,
        cache_dir=cache_dir,
    )
    val_dataset = CsvSeq2SeqDataset.from_cache(
        csv_path=csv_path,
        tokenizer=tokenizer,
        size=config.val_size,
        offset=config.train_size,
        max_source_tokens=config.max_source_tokens,
        max_target_tokens=config.max_target_tokens,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = Seq2SeqLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            tgt_input = batch.tgt_input.to(device)
            tgt_output = batch.tgt_output.to(device)

            optimizer.zero_grad()
            logits = model(src, src_lengths, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            non_pad = (tgt_output != PAD_TOKEN).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        train_loss = total_loss / max(total_tokens, 1)
        val_loss, sample_predictions = evaluate(model, val_loader, criterion, device, tokenizer)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        for sample in sample_predictions:
            print(sample)


def evaluate(
    model: Seq2SeqLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: SentencePieceTokenizer,
) -> tuple[float, list[str]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    sample_predictions: list[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            tgt_input = batch.tgt_input.to(device)
            tgt_output = batch.tgt_output.to(device)

            logits = model(src, src_lengths, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            non_pad = (tgt_output != PAD_TOKEN).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            if batch_idx == 0:
                predictions = model.greedy_decode(src, src_lengths, max_steps=tgt_output.size(1))
                for row in range(min(3, src.size(0))):
                    sample_predictions.append(
                        "src=[{src}] pred=[{pred}] tgt=[{tgt}]".format(
                            src=tokenizer.decode(batch.src[row].tolist()),
                            pred=tokenizer.decode(predictions[row]),
                            tgt=tokenizer.decode(batch.tgt_output[row].tolist()),
                        )
                    )

    return total_loss / max(total_tokens, 1), sample_predictions


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device was found.")
    return torch.device(requested_device)
