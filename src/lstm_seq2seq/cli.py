from __future__ import annotations

import argparse

from .train import TrainConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-seq2seq",
        description="Train an LSTM-based sequence-to-sequence model on debate CSV data.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv-path", default="all_debate_combined.csv")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--tokenizer-samples", type=int, default=200000)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        csv_path=args.csv_path,
        artifact_dir=args.artifact_dir,
        vocab_size=args.vocab_size,
        tokenizer_samples=args.tokenizer_samples,
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    run_training(config)
