# DC-Lite: Tokenizer-Free Byte-Level Modelling of Limit Order Books

A byte-level autoregressive model for limit order book message streams that
learns its own segmentation instead of using a fixed tokenizer or hand-designed
snapshot features.

MSc Statistical Science dissertation, University of Oxford.
Supervisors: Mihai Cucuringu, Stefan Zohren, Oxford-Man Institute of Quantitative Finance.

**[Read the thesis](Delissen_2025_DC_Lite_LOB.pdf)**

---

## Motivation

BPE and its variants are a poor fit for order book data. They are fitted to
character frequency, not to record structure, so numeric fields split
inconsistently depending on their digits, and every token receives the same
compute regardless of how predictable it is. DC-Lite drops the vocabulary
entirely: LOBSTER messages are serialized to raw bytes, and a router learns
where chunk boundaries fall so that predictable prefixes (timestamps, event
codes) are compressed into long chunks while volatile fields (price, size)
receive more capacity.

## Method

A simplified H-Net adapted to order book messages:

1. **Serialization** — LOBSTER rows to raw bytes under 2 schemes
   (`serialize_lobster.py`), so the effect of field layout on the learned
   segmentation can be measured.
2. **Byte encoder** — embedding over a 256-symbol alphabet plus a shallow causal
   transformer, producing one contextual vector per byte.
3. **Router and dynamic chunking** — an MLP emits a boundary logit per position,
   smoothed by a causal EMA. The logit is thresholded in the forward pass and
   differentiated with a straight-through estimator, so the segmentation trains
   jointly with the rest of the network. A boundary-rate penalty pulls the mean
   chunk length towards a target.
4. **Chunk transformer** — operates on mean-pooled chunk representatives, so the
   main backbone runs on a sequence shorter by roughly the mean chunk length.
5. **Decoder** — chunk states are broadcast back to byte positions, fused with
   the byte-level states, and passed to a next-byte head.

## Serialization schemes

| Scheme | Record size | Layout |
|---|---|---|
| `utf8_delim` | variable | UTF-8 text, pipe-separated, newline end-of-message |
| `byte_aligned` | 22 bytes | little-endian fixed fields |

## Data

LOBSTER Level-3 message files. **No market data is included in this repository**
— LOBSTER data is licensed and must be obtained separately. The serialization
script expects the standard LOBSTER message CSV layout: time, event type, order
ID, size, price (dollars x 10000), direction.

## Usage

```bash
pip install -r requirements.txt

# 1. Serialize messages to byte streams
python serialize_lobster.py --csv data/messages.csv --outdir data/bytes --no-header

# 2. Pretrain
python train_dc_lite.py \
    --train-files data/bytes/train.bit_packed.bin \
    --val-files   data/bytes/val.bit_packed.bin \
    --seq-len 1024 --batch-size 64 --epochs 24 --amp \
    --target-chunk-len 64 --patience 3 --outdir runs/bit_packed

# 3. Figures
python plots.py runs/bit_packed --title "DC-Lite (bit-packed)"
```

Cross-entropy is reported in nats per byte; perplexity and bits per byte follow
from it. The boundary-rate penalty is logged separately so that language-model
numbers stay comparable across runs with different penalty weights.

## Results

| Serialization | Val. bits/byte | Val. perplexity | Coverage |
|---|---|---|---|
| `utf8_delim` | 1.5155 | 2.859 | 70.0% |
| `byte_aligned` | 1.7180 | 3.290 | 97.8% |

Main finding: UTF-8 achieves the lower overall perplexity (2.859 vs 3.290; 1.5155 vs 1.7180 bits per byte), but only over the 70% of messages it decodes reliably, whereas the fixed-width byte-aligned scheme reaches 97.8% coverage with faster, smoother convergence and lower per-field perplexity on the hardest numerical channels: diff_tick_size, size and inter-arrival time, making it the preferable representation end-to-end.

## Files

```
dc_lite.py             model: byte encoder, router, chunking, chunk transformer, head
serialize_lobster.py   LOBSTER CSV to byte streams (three schemes)
train_dc_lite.py       pretraining loop with early stopping and checkpointing
plots.py               figures from a run's history.json
pipeline.ipynb         end-to-end Colab walkthrough
```

## Limitations and next steps

DC-Lite is pretrained on a single NYSE instrument (Public Storage, PSA) over 252 trading days and evaluated on next-byte prediction. It is a
pretraining backbone, not a foundation model: establishing that would require
multi-instrument, multi-venue pretraining and evaluation on several unseen
downstream tasks with frozen representations. That is the natural extension of
this work.

## References

- Hwang et al. (2025), *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*
- Pagnoni et al. (2024), *Byte Latent Transformer: Patches Scale Better Than Tokens*
- Zhang, Zohren & Roberts (2019), *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*
- Sirignano & Cont (2019), *Universal features of price formation in financial markets*

## License

MIT — see [LICENSE](LICENSE).
