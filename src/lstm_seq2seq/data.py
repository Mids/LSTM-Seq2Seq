from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


@dataclass(frozen=True)
class Batch:
    src: torch.Tensor
    src_lengths: torch.Tensor
    tgt_input: torch.Tensor
    tgt_output: torch.Tensor


@dataclass(frozen=True)
class TextExample:
    src_ids: list[int]
    tgt_ids: list[int]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str, max_tokens: int | None = None) -> list[int]:
        token_ids = list(self.processor.encode(text, out_type=int))
        if max_tokens is not None:
            return token_ids[:max_tokens]
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        filtered = [
            token_id
            for token_id in token_ids
            if token_id not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        ]
        if not filtered:
            return ""
        return self.processor.decode(filtered)


class CsvSeq2SeqDataset(Dataset[TextExample]):
    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: SentencePieceTokenizer,
        size: int,
        offset: int = 0,
        max_source_tokens: int = 128,
        max_target_tokens: int = 128,
    ) -> None:
        self.examples: list[TextExample] = []
        target_count = max(size, 0)
        skipped = 0

        for row in iter_csv_rows(csv_path):
            source_text = normalize_text(row.get("input", ""))
            target_text = normalize_text(row.get("output", ""))
            if not source_text or not target_text:
                continue
            if skipped < offset:
                skipped += 1
                continue

            src_ids = tokenizer.encode(source_text, max_tokens=max_source_tokens)
            tgt_ids = tokenizer.encode(target_text, max_tokens=max_target_tokens)
            if not src_ids or not tgt_ids:
                continue

            self.examples.append(TextExample(src_ids=src_ids, tgt_ids=tgt_ids))
            if len(self.examples) >= target_count:
                break

        if not self.examples:
            raise ValueError(
                f"No usable training rows found in {csv_path}. "
                "Check the CSV path and token limits."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TextExample:
        return self.examples[index]

    @classmethod
    def from_cache(
        cls,
        csv_path: str | Path,
        tokenizer: SentencePieceTokenizer,
        size: int,
        offset: int = 0,
        max_source_tokens: int = 128,
        max_target_tokens: int = 128,
        cache_dir: str | Path | None = None,
    ) -> "CsvSeq2SeqDataset":
        dataset = cls.__new__(cls)
        csv_path = Path(csv_path)
        cache_path = None

        if cache_dir is not None:
            cache_path = build_dataset_cache_path(
                csv_path=csv_path,
                tokenizer=tokenizer,
                size=size,
                offset=offset,
                max_source_tokens=max_source_tokens,
                max_target_tokens=max_target_tokens,
                cache_dir=Path(cache_dir),
            )
            if cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu")
                dataset.examples = [
                    TextExample(src_ids=example["src_ids"], tgt_ids=example["tgt_ids"])
                    for example in payload["examples"]
                ]
                return dataset

        built = cls(
            csv_path=csv_path,
            tokenizer=tokenizer,
            size=size,
            offset=offset,
            max_source_tokens=max_source_tokens,
            max_target_tokens=max_target_tokens,
        )
        dataset.examples = built.examples

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "examples": [
                        {"src_ids": example.src_ids, "tgt_ids": example.tgt_ids}
                        for example in dataset.examples
                    ]
                },
                cache_path,
            )

        return dataset


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def resolve_csv_files(csv_path: str | Path) -> list[Path]:
    path = Path(csv_path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    else:
        files = [path]
    if not files:
        raise ValueError(f"No CSV files found in {csv_path}.")
    return files


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict[str, str]]:
    for file_path in resolve_csv_files(csv_path):
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row


def build_dataset_cache_path(
    csv_path: Path,
    tokenizer: SentencePieceTokenizer,
    size: int,
    offset: int,
    max_source_tokens: int,
    max_target_tokens: int,
    cache_dir: Path,
) -> Path:
    csv_files = resolve_csv_files(csv_path)
    cache_key = "|".join(
        list(build_csv_signature(csv_files))
        + [
            str(Path(tokenizer.model_path).resolve()),
            str(size),
            str(offset),
            str(max_source_tokens),
            str(max_target_tokens),
        ]
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    filename = (
        f"{csv_path.stem}_offset{offset}_size{size}_"
        f"src{max_source_tokens}_tgt{max_target_tokens}_{digest}.pt"
    )
    return cache_dir / filename


def build_tokenizer(
    csv_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 16000,
    sample_rows: int = 200000,
    seed: int = 7,
) -> Path:
    model_prefix = Path(model_prefix)
    model_path = model_prefix.with_suffix(".model")
    if model_path.exists():
        return model_path

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = model_prefix.parent / f"{model_prefix.name}_corpus.txt"
    write_tokenizer_corpus(
        csv_path=Path(csv_path),
        output_path=corpus_path,
        sample_rows=sample_rows,
        seed=seed,
    )

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        pad_id=PAD_TOKEN,
        bos_id=BOS_TOKEN,
        eos_id=EOS_TOKEN,
        unk_id=UNK_TOKEN,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        shuffle_input_sentence=True,
    )
    corpus_path.unlink(missing_ok=True)
    return model_path


def write_tokenizer_corpus(
    csv_path: Path,
    output_path: Path,
    sample_rows: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0

    for row in iter_csv_rows(csv_path):
        for field in ("input", "output"):
            text = normalize_text(row.get(field, ""))
            if not text:
                continue
            seen += 1
            if len(reservoir) < sample_rows:
                reservoir.append(text)
                continue

            replace_at = rng.randint(0, seen - 1)
            if replace_at < sample_rows:
                reservoir[replace_at] = text

    with output_path.open("w", encoding="utf-8") as handle:
        for text in reservoir:
            handle.write(text)
            handle.write("\n")


def build_csv_signature(csv_files: list[Path]) -> list[str]:
    signature: list[str] = []
    for file_path in csv_files:
        stat = file_path.stat()
        signature.extend(
            [
                str(file_path.resolve()),
                str(stat.st_size),
                str(int(stat.st_mtime)),
            ]
        )
    return signature


def collate_batch(samples: list[TextExample]) -> Batch:
    batch_size = len(samples)
    src_lengths = torch.tensor([len(sample.src_ids) for sample in samples], dtype=torch.long)
    max_src_len = int(src_lengths.max().item())
    max_tgt_len = max(len(sample.tgt_ids) for sample in samples) + 1

    src_tensor = torch.full((batch_size, max_src_len), PAD_TOKEN, dtype=torch.long)
    tgt_input = torch.full((batch_size, max_tgt_len), PAD_TOKEN, dtype=torch.long)
    tgt_output = torch.full((batch_size, max_tgt_len), PAD_TOKEN, dtype=torch.long)

    for row, sample in enumerate(samples):
        src_tensor[row, : len(sample.src_ids)] = torch.tensor(sample.src_ids, dtype=torch.long)

        decoder_input = [BOS_TOKEN, *sample.tgt_ids]
        decoder_output = [*sample.tgt_ids, EOS_TOKEN]
        tgt_input[row, : len(decoder_input)] = torch.tensor(decoder_input, dtype=torch.long)
        tgt_output[row, : len(decoder_output)] = torch.tensor(decoder_output, dtype=torch.long)

    return Batch(
        src=src_tensor,
        src_lengths=src_lengths,
        tgt_input=tgt_input,
        tgt_output=tgt_output,
    )
