"""Figures from a training run's ``history.json``.

Example:
    python plots.py runs/dc_lite --title "DC-Lite (bit-packed)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=160)
    plt.close(fig)


def plot_history(history: dict, outdir: Path, title: str = "") -> None:
    """Write loss, perplexity, chunk-length and learning-rate figures."""
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["val_ce"]) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, history["train_ce"], label="train")
    ax.plot(epochs, history["val_ce"], label="validation")
    ax.set(xlabel="epoch", ylabel="cross-entropy (nats/byte)", title=f"{title} loss".strip())
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, outdir, "loss.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["val_ppl"])
    ax.set(xlabel="epoch", ylabel="perplexity", title=f"{title} validation perplexity".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "perplexity.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["avg_chunk_len"])
    ax.set(xlabel="epoch", ylabel="mean chunk length (bytes)",
           title=f"{title} learned chunk length".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "chunk_length.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["lr"])
    ax.set(xlabel="epoch", ylabel="learning rate", title=f"{title} schedule".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "learning_rate.png")

    lengths = np.asarray(history.get("chunk_lengths", []), dtype=np.int64)
    if lengths.size:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        left.hist(lengths, bins=min(100, max(32, int(np.median(lengths)) * 2)))
        left.set(xlabel="chunk length (bytes)", ylabel="count", title="histogram")
        left.grid(alpha=0.3)
        ordered = np.sort(lengths)
        right.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size)
        right.set(xlabel="chunk length (bytes)", ylabel="ECDF", title="ECDF")
        right.grid(alpha=0.3)
        fig.suptitle(f"{title} chunk-length distribution".strip())
        _save(fig, outdir, "chunk_length_distribution.png")


def compare_runs(runs: dict[str, Path], outdir: Path) -> None:
    """Overlay validation perplexity across serialization schemes."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for label, path in runs.items():
        history = json.loads(Path(path).read_text())
        ax.plot(range(1, len(history["val_ppl"]) + 1), history["val_ppl"], label=label)
    ax.set(xlabel="epoch", ylabel="validation perplexity",
           title="Validation perplexity by serialization scheme")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, outdir, "scheme_comparison.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rundir", type=Path, help="directory containing history.json")
    p.add_argument("--title", default="")
    args = p.parse_args()
    history = json.loads((args.rundir / "history.json").read_text())
    plot_history(history, args.rundir / "figures", args.title)
    print(f"figures written to {args.rundir / 'figures'}")


if __name__ == "__main__":
    main()
