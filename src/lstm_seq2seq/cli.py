from __future__ import annotations

import argparse

from .infer import predict_text
from .train import TrainConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-seq2seq",
        description="Train an LSTM-based sequence-to-sequence model on debate CSV data.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--train-size", type=int, default=50000)
    train_parser.add_argument("--val-size", type=int, default=2000)
    train_parser.add_argument("--embedding-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    train_parser.add_argument("--lr-decay-patience", type=int, default=2)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--csv-path", default="data")
    train_parser.add_argument("--artifact-dir", default="artifacts")
    train_parser.add_argument("--vocab-size", type=int, default=16000)
    train_parser.add_argument("--tokenizer-samples", type=int, default=200000)
    train_parser.add_argument("--max-source-tokens", type=int, default=128)
    train_parser.add_argument("--max-target-tokens", type=int, default=128)
    train_parser.add_argument("--run-name")

    predict_parser = subparsers.add_parser("predict", help="Run inference from a checkpoint.")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--checkpoint", default="artifacts/checkpoint_latest.pt")
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--max-source-tokens", type=int)
    predict_parser.add_argument("--max-decode-tokens", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        print(
            predict_text(
                text=args.text,
                checkpoint_path=args.checkpoint,
                device=args.device,
                max_source_tokens=args.max_source_tokens,
                max_decode_tokens=args.max_decode_tokens,
            )
        )
        return

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_patience=args.lr_decay_patience,
        min_learning_rate=args.min_learning_rate,
        device=args.device,
        seed=args.seed,
        csv_path=args.csv_path,
        artifact_dir=args.artifact_dir,
        vocab_size=args.vocab_size,
        tokenizer_samples=args.tokenizer_samples,
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
        run_name=args.run_name,
    )
    run_training(config)
