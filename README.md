# LSTM Seq2Seq with uv

Minimal `uv` project for training an LSTM-based sequence-to-sequence model on
debate text. The current pipeline uses `all_debate_combined.csv` and trains a
SentencePiece unigram tokenizer over the CSV `input` and `output` columns.

## Requirements

- Python 3.11+
- `uv`
- Local copy of `all_debate_combined.csv`

## Dataset format

The training code expects a CSV with at least these columns:

- `input`
- `output`

The current debate dataset also includes `topic` and `source`, but those are
not used by the model.

## Setup

Install dependencies:

```bash
uv sync
```

PyTorch is configured through `uv` indexes:

- macOS and non-x86 machines use CPU wheels
- Windows and Linux on `x86_64` / `AMD64` use CUDA 12.8 wheels

## Quick start

Run a small smoke test first:

```bash
uv run lstm-seq2seq train --epochs 1 --train-size 2048 --val-size 256 --batch-size 16
```

Then move to a larger run:

```bash
uv run lstm-seq2seq train --epochs 3 --train-size 50000 --val-size 2000 --batch-size 32
```

## What the pipeline does

1. Samples text from the CSV and trains a SentencePiece unigram tokenizer.
2. Builds source-target pairs from `input -> output`.
3. Truncates long examples to fixed token limits.
4. Trains an encoder-decoder LSTM with teacher forcing.
5. Prints validation loss and a few greedy-decoded examples after each epoch.

Tokenizer and training artifacts are written to `artifacts/`.

## Important defaults

- Vocabulary size: `16000`
- Max source tokens: `128`
- Max target tokens: `128`
- Default train subset: `50000`
- Default validation subset: `2000`

These defaults are conservative so the model is easier to run as a baseline.

## Useful commands

Train on a specific CSV path:

```bash
uv run lstm-seq2seq train --csv-path all_debate_combined.csv
```

Use a larger tokenizer vocabulary:

```bash
uv run lstm-seq2seq train --vocab-size 32000
```

Allow longer sequences:

```bash
uv run lstm-seq2seq train --max-source-tokens 256 --max-target-tokens 256
```

## Test Interface

Training now writes checkpoints to `artifacts/`, including `artifacts/checkpoint_latest.pt`.

Run single-example inference with:

```bash
uv run lstm-seq2seq predict "The US should support allies more aggressively."
```

## Project layout

```text
src/lstm_seq2seq/
├── cli.py
├── data.py
├── model.py
└── train.py
```
