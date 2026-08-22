# DC-Lite: Tokenizer-Free Byte-Level Modelling of Limit Order Books

A byte-level autoregressive model for limit order book message streams that learns its own segmentation instead of using a fixed tokenizer or hand-designed snapshot features.

MSc Statistical Science dissertation, University of Oxford, 2025.
Supervisors: Mihai Cucuringu, Stefan Zohren, Department of Statistics Oxford.

**[Read the thesis](https://github.com/felixdelissen/dc-lite-lob/blob/main/Delissen_2025_DC_Lite_LOB.pdf)**

---

## Motivation

BPE and its variants are a poor fit for order book data. They are fitted to character frequency, not to record structure, so numeric fields split inconsistently depending on their digits, and every token receives the same compute regardless of how predictable it is. DC-Lite drops the vocabulary entirely: LOBSTER messages are serialized to raw bytes, and a router learns where chunk boundaries fall so that predictable prefixes (timestamps, event codes) are compressed into long chunks while volatile fields (price, size) receive more capacity.

## Method

A simplified H-Net adapted to order book messages:

1. **Serialization** — LOBSTER rows to raw bytes under two schemes (`serialize_lobster.py`), so the effect of field layout on the learned segmentation can be measured.
2. **Byte encoder** — embedding over a 256-symbol alphabet plus a shallow causal transformer (2 pre-norm blocks, 4 heads, `d=256`), producing one contextual vector per byte.
3. **Router and dynamic chunking** — an MLP (256→128→1, GELU) emits a boundary logit per position, smoothed by a causal EMA. The logit is thresholded in the forward pass and differentiated with a straight-through estimator, so the segmentation trains jointly with the rest of the network. A boundary-rate penalty pulls the mean chunk length towards a target.
4. **Chunk transformer** — 4 pre-norm blocks, 6 heads, operating on mean-pooled chunk representatives, so the main backbone runs on a sequence shorter by roughly the mean chunk length.
5. **Decoder** — chunk states are broadcast back to byte positions, fused with the byte-level states, and passed to a next-byte head over the 256-byte vocabulary.

Total: 6.7M parameters, trainable on a single A100.

## Serialization schemes

| Scheme         | Record size | Layout                                                          |
| -------------- | ----------- | --------------------------------------------------------------- |
| `utf8_delim`   | variable (~26 B) | UTF-8 text, space-separated, `;` end-of-message              |
| `byte_aligned` | 18 B fixed  | little-endian: `iat:u64`, `type:u8`, `size:u32`, `diff_tick:i32`, `side:i8` |

Fields are ordered from lowest to highest entropy (`side → type → size → diff_tick_size → iat`), so that predictable fields provide low-variance context for the harder ones.

## Data

LOBSTER Level-3 message files. **No market data is included in this repository** — LOBSTER data is licensed and must be obtained separately. The serialization script expects the standard LOBSTER message CSV layout: time, event type, order ID, size, price (dollars × 10,000), direction.

Experiments use a single NYSE instrument (Public Storage, PSA) over 252 trading days in 2016, split chronologically: 214 days train, 25 validation, 13 test.

## Usage

```bash
pip install -r requirements.txt

# 1. Serialize messages to byte streams
python serialize_lobster.py --csv data/messages.csv --outdir data/bytes --no-header

# 2. Train
python train_dc_lite.py \
    --train-files data/bytes/train.byte_aligned.bin \
    --val-files   data/bytes/val.byte_aligned.bin \
    --seq-len 1024 --batch-size 64 --epochs 30 --amp \
    --target-chunk-len 48 --patience 5 --outdir runs/byte_aligned

# 3. Figures
python plots.py runs/byte_aligned --title "DC-Lite (byte-aligned)"
```

Cross-entropy is reported in nats per byte; perplexity and bits per byte follow from it. The boundary-rate penalty is logged separately so that language-model numbers stay comparable across runs with different penalty weights.

## Results

| Serialization  | Val. loss | Val. perplexity | Val. bits/byte | Coverage |
| -------------- | --------- | --------------- | -------------- | -------- |
| `utf8_delim`   | 1.0504    | 2.859           | 1.5155         | 70.8%    |
| `byte_aligned` | 1.1908    | 3.290           | 1.7180         | 97.7%    |

Per-field perplexity at the best checkpoint:

| Field            | `utf8_delim` | `byte_aligned` |
| ---------------- | ------------ | -------------- |
| `side`           | 1.435        | 1.725          |
| `type`           | 1.960        | 1.918          |
| `size`           | 1.594        | **1.399**      |
| `diff_tick_size` | 3.014        | **1.932**      |
| `iat`            | 9.535        | **7.635**      |

**Main finding.** UTF-8 attains the lower overall perplexity, but only over the 70.8% of messages it decodes reliably — variable-length formatting shifts field boundaries, and a single missing separator corrupts an entire record. The fixed-width byte-aligned scheme reaches 97.7% coverage with faster, smoother convergence and lower per-field perplexity on the three hardest numerical channels, making it the preferable representation end-to-end.

Both schemes recover the categorical marginals well (`side` within ~1pp, dominant event types correctly ranked) but under-calibrate the upper tails: large order sizes and long inter-arrival gaps are systematically underweighted.

## Limitations and next steps

DC-Lite is pretrained on a single NYSE instrument and evaluated on next-byte prediction under teacher forcing. It is a pretraining backbone, not a foundation model: establishing that would require multi-instrument, multi-venue pretraining and evaluation on several unseen downstream tasks with frozen representations. Free-running generation and calibration of the inter-arrival tail are the two clearest open problems.

## References

- Hwang, Wang & Gu (2025), *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, arXiv:2507.07955
- Pagnoni et al. (2025), *Byte Latent Transformer: Patches Scale Better Than Tokens*, ACL 2025
- Nagy et al. (2023), *Generative AI for End-to-End Limit Order Book Modelling*, arXiv:2309.00638
- Bouchaud, Farmer & Lillo (2009), *How Markets Slowly Digest Changes in Supply and Demand*

## Citation

```bibtex
@mastersthesis{delissen2025dclite,
  title  = {Dynamic Chunking for Limit Order Book Predictions},
  author = {Delissen, F{\'e}lix},
  school = {University of Oxford},
  year   = {2025},
  type   = {{MSc} dissertation, Department of Statistics}
}
```

## License

MIT — see [LICENSE](https://github.com/felixdelissen/dc-lite-lob/blob/main/LICENSE).
