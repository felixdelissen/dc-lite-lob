"""DC-Lite: a byte-level autoregressive model with learned chunking.

The model reads raw bytes (vocab = 256) and learns where message boundaries
fall instead of relying on a fixed tokenizer. A router emits a boundary
probability per position; boundaries are binarised in the forward pass and
differentiated through with a straight-through estimator, so the segmentation
is trained jointly with the encoder.

Pipeline: byte embedding -> encoder -> router -> chunk pooling -> chunk
transformer -> fusion with the byte-level states -> next-byte head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: causal attention then a feed-forward MLP."""

    def __init__(
        self, d_model: int, n_heads: int, mlp_mult: float = 2.0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        hidden = int(mlp_mult * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


def straight_through_step(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard threshold in the forward pass, sigmoid gradient in the backward pass."""
    p = torch.sigmoid(logits)
    return (p > threshold).to(p.dtype) - p.detach() + p


def causal_ema(x: torch.Tensor, tau: float, tol: float = 1e-7) -> torch.Tensor:
    """Causal exponential moving average along time, computed as a convolution.

    Implements y[0] = x[0] and y[t] = tau * y[t-1] + (1 - tau) * x[t], which
    expands to y[t] = (1 - tau) * sum_k tau^(t-k) x[k] + tau^(t+1) x[0]. The
    kernel is truncated where tau^L falls below ``tol``, so the cost is a single
    parallel conv1d rather than T sequential steps.

    Args:
        x: tensor of shape [B, T, C].
        tau: smoothing coefficient in [0, 1).
        tol: truncation tolerance on the kernel tail.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must lie in [0, 1), got {tau}")
    B, T, C = x.shape
    if tau == 0.0:
        return x.clone()

    kernel_len = min(T, max(1, int(math.ceil(math.log(tol) / math.log(tau)))))
    decay = tau ** torch.arange(kernel_len, device=x.device, dtype=x.dtype)

    # conv1d correlates, so the kernel is reversed to make the filter causal.
    weight = decay.flip(0).view(1, 1, kernel_len).expand(C, 1, kernel_len)
    padded = F.pad(x.transpose(1, 2), (kernel_len - 1, 0))
    y = F.conv1d(padded, weight, groups=C).transpose(1, 2)

    # Boundary term carrying the y[0] = x[0] initialisation.
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    init = (tau ** (t + 1)).view(1, T, 1) * x[:, :1]
    return (1.0 - tau) * y + init


def chunk_lengths(boundaries: torch.Tensor) -> torch.Tensor:
    """Chunk lengths implied by a boundary mask, flattened across the batch.

    Args:
        boundaries: bool tensor [B, T], True where a chunk ends.

    Returns:
        1-D int64 tensor of chunk lengths (the trailing partial chunk of each
        row is included).
    """
    B, T = boundaries.shape
    closed = boundaries.clone()
    closed[:, -1] = True  # close the trailing chunk of every row
    idx = closed.nonzero(as_tuple=False)
    rows, cols = idx[:, 0], idx[:, 1]
    prev = torch.full_like(cols, -1)
    prev[1:] = cols[:-1]
    row_start = torch.ones_like(rows, dtype=torch.bool)
    row_start[1:] = rows[1:] != rows[:-1]
    prev = torch.where(row_start, torch.full_like(prev, -1), prev)
    return cols - prev


@dataclass
class DCLiteConfig:
    """Hyperparameters of the DC-Lite model.

    Attributes:
        d_model_byte: width of the byte-level encoder and decoder.
        d_model_chunk: width of the chunk-level transformer.
        target_chunk_len: chunk length the boundary-rate penalty pulls towards.
        boundary_rate_weight: weight of that penalty in the total loss.
        smooth_tau: EMA coefficient applied to the router logits.
    """

    d_model_byte: int = 256
    d_model_chunk: int = 384
    n_layers_byte: int = 2
    n_heads_byte: int = 4
    n_layers_chunk: int = 4
    n_heads_chunk: int = 6
    n_layers_decoder: int = 1
    mlp_mult: float = 2.0
    dropout: float = 0.1
    target_chunk_len: int = 64
    boundary_rate_weight: float = 0.05
    smooth_tau: float = 0.6
    boundary_threshold: float = 0.5


class DCLiteLM(nn.Module):
    """Byte-level language model with a learned, differentiable segmentation."""

    def __init__(self, config: DCLiteConfig | None = None, **overrides) -> None:
        super().__init__()
        self.config = config or DCLiteConfig(**overrides)
        cfg = self.config

        self.byte_embed = nn.Embedding(VOCAB_SIZE, cfg.d_model_byte)
        self.pos_byte = PositionalEncoding(cfg.d_model_byte)
        self.encoder = nn.ModuleList(
            TransformerBlock(cfg.d_model_byte, cfg.n_heads_byte, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_byte)
        )
        self.encoder_ln = nn.LayerNorm(cfg.d_model_byte)

        self.router = nn.Sequential(
            nn.Linear(cfg.d_model_byte, 128), nn.GELU(), nn.Linear(128, 1)
        )

        self.chunk_in = nn.Linear(cfg.d_model_byte, cfg.d_model_chunk)
        self.pos_chunk = PositionalEncoding(cfg.d_model_chunk)
        self.chunk_blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model_chunk, cfg.n_heads_chunk, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_chunk)
        )
        self.chunk_ln = nn.LayerNorm(cfg.d_model_chunk)

        self.fuse = nn.Linear(cfg.d_model_byte + cfg.d_model_chunk, cfg.d_model_byte)
        self.decoder = nn.ModuleList(
            TransformerBlock(cfg.d_model_byte, cfg.n_heads_byte, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_decoder)
        )
        self.decoder_ln = nn.LayerNorm(cfg.d_model_byte)
        self.head = nn.Linear(cfg.d_model_byte, VOCAB_SIZE, bias=False)

    def segment(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route byte states to chunk ids.

        Returns:
            boundaries: [B, T] soft-hard boundary indicators in {0, 1}.
            chunk_ids: [B, T] int64 chunk index of each position.
        """
        logits = causal_ema(self.router(h), tau=self.config.smooth_tau)
        boundaries = straight_through_step(logits, self.config.boundary_threshold)
        boundaries = boundaries.squeeze(-1)
        # A chunk always opens at t=0; done out of place to keep autograd happy.
        opening = torch.ones_like(boundaries[:, :1])
        boundaries = torch.cat([opening, boundaries[:, 1:]], dim=1)
        chunk_ids = (boundaries.cumsum(dim=1) - 1.0).long().clamp_min_(0)
        return boundaries, chunk_ids

    @staticmethod
    def pool_chunks(h: torch.Tensor, chunk_ids: torch.Tensor, n_chunks: int) -> torch.Tensor:
        """Mean-pool byte states within each chunk.

        Args:
            h: [B, T, C] byte-level states.
            chunk_ids: [B, T] chunk index per position.
            n_chunks: padded number of chunks.

        Returns:
            [B, n_chunks, C] chunk representations, zero where the row has fewer
            chunks than ``n_chunks``.
        """
        B, T, C = h.shape
        offsets = torch.arange(B, device=h.device).unsqueeze(1) * n_chunks
        flat_ids = (chunk_ids + offsets).reshape(-1)
        sums = h.new_zeros(B * n_chunks, C).index_add_(0, flat_ids, h.reshape(-1, C))
        counts = h.new_zeros(B * n_chunks, 1).index_add_(
            0, flat_ids, h.new_ones(B * T, 1)
        )
        return (sums / counts.clamp_min(1.0)).view(B, n_chunks, C)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_chunk_lengths: bool = False,
    ):
        """Predict the next byte at every position.

        Args:
            x: [B, T] int64 byte ids in [0, 255].
            return_aux: also return the boundary-rate penalty and routing stats.
            return_chunk_lengths: include a tensor of chunk lengths in the stats.

        Returns:
            ``logits`` of shape [B, T, 256], or ``(logits, aux_loss, stats)``
            when ``return_aux`` is set.
        """
        B, T = x.shape
        h = self.pos_byte(self.byte_embed(x))
        for block in self.encoder:
            h = block(h)
        h = self.encoder_ln(h)

        boundaries, chunk_ids = self.segment(h)
        n_chunks = int(chunk_ids.max().item()) + 1  # single host sync per step

        c = self.chunk_in(self.pool_chunks(h, chunk_ids, n_chunks))
        c = self.pos_chunk(c)
        for block in self.chunk_blocks:
            c = block(c)
        c = self.chunk_ln(c)

        # Broadcast each chunk state back to the positions it covers.
        gather_idx = chunk_ids.unsqueeze(-1).expand(-1, -1, c.size(-1))
        chunk_per_byte = c.gather(1, gather_idx)

        y = self.fuse(torch.cat([h, chunk_per_byte], dim=-1))
        for block in self.decoder:
            y = block(y)
        logits = self.head(self.decoder_ln(y))

        if not return_aux:
            return logits

        n_boundaries = boundaries.sum(dim=1)
        rate = n_boundaries / T
        target_rate = 1.0 / self.config.target_chunk_len
        aux_loss = self.config.boundary_rate_weight * ((rate - target_rate) ** 2).mean()

        stats = {
            "avg_chunk_len": T / n_boundaries.mean().clamp_min(1.0).item(),
            "avg_boundaries": n_boundaries.mean().item(),
            "boundary_rate": rate.mean().item(),
        }
        if return_chunk_lengths:
            with torch.no_grad():
                stats["chunk_lengths"] = chunk_lengths(boundaries > 0.5)
        return logits, aux_loss, stats

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
