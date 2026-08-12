"""Pretraining loop for DC-Lite on serialized LOBSTER byte streams.

Example:
    python train_dc_lite.py \
        --train-files data/train.bit_packed.bin \
        --val-files data/val.bit_packed.bin \
        --seq-len 1024 --batch-size 64 --epochs 24 --amp

Cross-entropy is reported in nats per byte; perplexity and bits per byte are
derived from it. The boundary-rate penalty is tracked separately so that the
language-modelling numbers stay comparable across runs with different penalty
weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dc_lite import VOCAB_SIZE, DCLiteConfig, DCLiteLM  # noqa: E402


class ByteSequenceDataset(Dataset):
    """Fixed-length windows over a concatenated stream of raw bytes.

    Args:
        paths: files to concatenate, in order.
        seq_len: window length in bytes.
        stride: step between window starts; defaults to ``seq_len`` (no overlap).
        bytes_cap: keep only the first N bytes of the stream (0 = keep all).
    """

    def __init__(
        self,
        paths: list[str],
        seq_len: int = 2048,
        stride: int | None = None,
        bytes_cap: int = 0,
    ) -> None:
        raw = b"".join(Path(p).read_bytes() for p in paths)
        if bytes_cap > 0:
            raw = raw[:bytes_cap]
        if len(raw) < seq_len + 1:
            raise ValueError(
                f"stream has {len(raw)} bytes, need at least {seq_len + 1} for seq_len={seq_len}"
            )
        self.data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.starts = torch.arange(0, len(self.data) - seq_len - 1, self.stride)

    def __len__(self) -> int:
        return self.starts.numel()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        return (
            self.data[s : s + self.seq_len].clone(),
            self.data[s + 1 : s + self.seq_len + 1].clone(),
        )


def pick_device() -> torch.device:
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype_for(device: torch.device, enabled: bool):
    """Half-precision dtype to use for autocast, or None when disabled."""
    if not enabled:
        return None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype=None,
    collect_chunks: bool = False,
    max_samples: int = 50_000,
) -> dict:
    """Validation pass.

    Returns a dict with cross-entropy (nats/byte), perplexity, bits per byte,
    the boundary-rate penalty, the mean chunk length, and optionally a sample of
    chunk lengths for the histogram.
    """
    model.eval()
    ce_sum, aux_sum, n_bytes, n_batches = 0.0, 0.0, 0, 0
    chunk_len_sum, chunk_lengths = 0.0, []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, aux_loss, stats = model(
                x, return_aux=True, return_chunk_lengths=collect_chunks
            )
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="sum"
            )
        ce_sum += ce.item()
        aux_sum += aux_loss.item()
        chunk_len_sum += stats["avg_chunk_len"]
        n_bytes += y.numel()
        n_batches += 1

        if collect_chunks and len(chunk_lengths) < max_samples:
            chunk_lengths.extend(stats["chunk_lengths"].tolist())

    ce = ce_sum / n_bytes
    return {
        "ce": ce,
        "ppl": math.exp(ce),
        "bpb": ce / math.log(2.0),
        "aux": aux_sum / n_batches,
        "avg_chunk_len": chunk_len_sum / n_batches,
        "chunk_lengths": chunk_lengths[:max_samples],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype=None,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
) -> dict:
    """One pass over the training set. Returns mean CE, aux loss and chunk length."""
    model.train()
    ce_sum, aux_sum, n_bytes, n_batches = 0.0, 0.0, 0, 0
    chunk_len_sum = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, aux_loss, stats = model(x, return_aux=True)
            ce = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss = ce + aux_loss

        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ce_sum += ce.item() * y.numel()
        aux_sum += aux_loss.item()
        chunk_len_sum += stats["avg_chunk_len"]
        n_bytes += y.numel()
        n_batches += 1

    return {
        "ce": ce_sum / n_bytes,
        "aux": aux_sum / n_batches,
        "avg_chunk_len": chunk_len_sum / n_batches,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = p.add_argument_group("data")
    data.add_argument("--train-files", nargs="+", required=True)
    data.add_argument("--val-files", nargs="+", required=True)
    data.add_argument("--seq-len", type=int, default=2048)
    data.add_argument("--train-bytes-cap", type=int, default=0,
                      help="truncate the training stream to N bytes (0 = all)")
    data.add_argument("--val-bytes-cap", type=int, default=0)
    data.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))

    model = p.add_argument_group("model")
    model.add_argument("--target-chunk-len", type=int, default=64)
    model.add_argument("--aux-weight", type=float, default=0.05,
                       help="weight of the boundary-rate penalty")
    model.add_argument("--tau", type=float, default=0.6,
                       help="EMA coefficient on the router logits")
    model.add_argument("--dropout", type=float, default=0.1)

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--batch-size", type=int, default=24)
    optim.add_argument("--epochs", type=int, default=8)
    optim.add_argument("--lr", type=float, default=2e-3)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--amp", action="store_true", help="mixed-precision training")
    optim.add_argument("--patience", type=int, default=0,
                       help="stop after N epochs without validation improvement (0 = off)")
    optim.add_argument("--min-delta", type=float, default=0.0)
    optim.add_argument("--seed", type=int, default=0)

    io = p.add_argument_group("io")
    io.add_argument("--outdir", type=Path, default=Path("runs/dc_lite"))
    io.add_argument("--resume", type=Path, default=None)
    io.add_argument("--save-last", action="store_true")
    io.add_argument("--save-every", type=int, default=0)
    io.add_argument("--collect-chunk-hist", action="store_true")
    io.add_argument("--max-hist-samples", type=int, default=50_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device()
    amp_dtype = autocast_dtype_for(device, args.amp)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"device={device} amp={amp_dtype}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    loaders = {}
    for split, files, cap in (
        ("train", args.train_files, args.train_bytes_cap),
        ("val", args.val_files, args.val_bytes_cap),
    ):
        ds = ByteSequenceDataset(files, seq_len=args.seq_len, bytes_cap=cap)
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"{split}: {len(ds.data):,} bytes, {len(ds):,} windows")

    config = DCLiteConfig(
        dropout=args.dropout,
        target_chunk_len=args.target_chunk_len,
        boundary_rate_weight=args.aux_weight,
        smooth_tau=args.tau,
    )
    model = DCLiteLM(config).to(device)
    print(f"model: {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: dict[str, list] = {
        k: [] for k in
        ("train_ce", "train_aux", "val_ce", "val_ppl", "val_bpb", "val_aux", "avg_chunk_len", "lr")
    }
    best_val = float("inf")
    start_epoch, stale_epochs = 1, 0

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history.update(ckpt.get("history", {}))
        best_val = ckpt.get("best_val", best_val)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch} (best val CE {best_val:.4f})")

    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train = train_one_epoch(
                model, loaders["train"], optimizer, device, amp_dtype,
                args.accum_steps, args.grad_clip,
            )
            val = evaluate(
                model, loaders["val"], device, amp_dtype,
                collect_chunks=args.collect_chunk_hist,
                max_samples=args.max_hist_samples,
            )
            lr = optimizer.param_groups[0]["lr"]

            for key, value in (
                ("train_ce", train["ce"]), ("train_aux", train["aux"]),
                ("val_ce", val["ce"]), ("val_ppl", val["ppl"]),
                ("val_bpb", val["bpb"]), ("val_aux", val["aux"]),
                ("avg_chunk_len", val["avg_chunk_len"]), ("lr", lr),
            ):
                history[key].append(value)
            if args.collect_chunk_hist:
                history["chunk_lengths"] = val["chunk_lengths"]

            print(
                f"epoch {epoch:02d} | lr {lr:.3g} | train CE {train['ce']:.4f} | "
                f"val CE {val['ce']:.4f} | ppl {val['ppl']:.2f} | bpb {val['bpb']:.3f} | "
                f"chunk len {val['avg_chunk_len']:.1f}"
            )

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "history": history,
                "args": {k: str(v) for k, v in vars(args).items()},
                "config": asdict(config),
            }
            if best_val - val["ce"] > args.min_delta:
                best_val = val["ce"]
                checkpoint["best_val"] = best_val
                torch.save(checkpoint, args.outdir / "best.pt")
                stale_epochs = 0
            else:
                stale_epochs += 1
            if args.save_last:
                torch.save(checkpoint, args.outdir / "last.pt")
            if args.save_every and epoch % args.save_every == 0:
                torch.save(checkpoint, args.outdir / f"epoch_{epoch}.pt")

            if args.patience and stale_epochs >= args.patience:
                print(f"early stop at epoch {epoch}: no improvement for {stale_epochs} epochs")
                break
            scheduler.step()
    except KeyboardInterrupt:
        print(f"interrupted at epoch {epoch}")

    with open(args.outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"artifacts written to {args.outdir}")


if __name__ == "__main__":
    main()
