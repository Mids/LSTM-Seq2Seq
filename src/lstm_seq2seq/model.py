from __future__ import annotations

import torch
from torch import nn

from .data import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_input: torch.Tensor,
    ) -> torch.Tensor:
        embedded_src = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_src,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)

        embedded_tgt = self.embedding(tgt_input)
        decoded, _ = self.decoder(embedded_tgt, hidden)
        return self.output(decoded)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_steps: int,
    ) -> list[list[int]]:
        self.eval()
        embedded_src = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_src,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)

        batch_size = src.size(0)
        current = torch.full(
            (batch_size, 1),
            BOS_TOKEN,
            dtype=torch.long,
            device=src.device,
        )
        sequences = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_steps):
            embedded = self.embedding(current)
            decoded, hidden = self.decoder(embedded, hidden)
            logits = self.output(decoded[:, -1, :])
            current = logits.argmax(dim=-1, keepdim=True)

            for idx, token in enumerate(current.squeeze(1).tolist()):
                if finished[idx]:
                    continue
                if token == EOS_TOKEN:
                    finished[idx] = True
                    continue
                sequences[idx].append(token)

            if bool(finished.all()):
                break

        return sequences
