"""Serialize LOBSTER message files into raw byte streams.

Three encodings are provided, so that the effect of the serialization scheme on
the learned segmentation can be measured:

  utf8_delim    UTF-8 text, pipe-separated fields, newline end-of-message.
                Variable length; field widths depend on the values.
  byte_aligned  fixed 22-byte records, little-endian.
  bit_packed    fixed 21-byte records; event type and direction share one byte.

A LOBSTER message row is:
    Time (seconds after midnight), EventType (1-7), OrderID, Size,
    Price (dollars x 10000), Direction (-1 sell, +1 buy)

Example:
    python serialize_lobster.py --csv messages.csv --outdir bytes/ --schemes all
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path
from typing import Callable, Iterator, Sequence

UINT32_MAX = 2**32 - 1
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
DAY_NS = 86_400 * 1_000_000_000

BYTE_ALIGNED = struct.Struct("<QBIIiB")  # t_ns, event, order_id, size, price, direction
BIT_PACKED = struct.Struct("<QIIiB")  # t_ns, order_id, size, price, event|direction


def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into [lo, hi]."""
    return max(lo, min(x, hi))


def read_messages(csv_path: Path, has_header: bool) -> Iterator[tuple]:
    """Yield parsed LOBSTER rows, skipping malformed ones.

    Yields:
        (time_sec, event_type, order_id, size, price, direction)
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                yield (
                    float(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                )
            except ValueError:
                continue


def to_nanoseconds(time_sec: float) -> int:
    """Seconds after midnight to integer nanoseconds, clamped to one day."""
    return clamp(round(time_sec * 1_000_000_000), 0, DAY_NS)


def encode_direction(direction: int) -> int:
    """LOBSTER direction (-1 sell, +1 buy) to an unsigned byte (2 sell, 1 buy)."""
    return 1 if direction >= 0 else 2


def pack_event_and_direction(event_type: int, direction: int) -> int:
    """Pack event type into bits 0-2 and direction into bit 3 of a single byte."""
    return (clamp(event_type, 1, 7) & 0b111) | ((encode_direction(direction) - 1) << 3)


def write_utf8_delim(
    csv_path: Path, out_path: Path, has_header: bool, delimiter: str = "|"
) -> int:
    """Write messages as delimited UTF-8 text, one message per line."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            record = delimiter.join(str(v) for v in (t, event, order_id, size, price, direction))
            w.write(record.encode("utf-8") + b"\n")
            count += 1
    return count


def write_byte_aligned(csv_path: Path, out_path: Path, has_header: bool) -> int:
    """Write fixed-width 22-byte records."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            w.write(
                BYTE_ALIGNED.pack(
                    to_nanoseconds(t),
                    clamp(event, 1, 7),
                    clamp(order_id, 0, UINT32_MAX),
                    clamp(size, 0, UINT32_MAX),
                    clamp(price, INT32_MIN, INT32_MAX),
                    encode_direction(direction),
                )
            )
            count += 1
    return count


def write_bit_packed(csv_path: Path, out_path: Path, has_header: bool) -> int:
    """Write fixed-width 21-byte records with event type and direction packed."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            w.write(
                BIT_PACKED.pack(
                    to_nanoseconds(t),
                    clamp(order_id, 0, UINT32_MAX),
                    clamp(size, 0, UINT32_MAX),
                    clamp(price, INT32_MIN, INT32_MAX),
                    pack_event_and_direction(event, direction),
                )
            )
            count += 1
    return count


SCHEMES: dict[str, tuple[str, Callable[[Path, Path, bool], int]]] = {
    "utf8": ("utf8_delim", write_utf8_delim),
    "aligned": ("byte_aligned", write_byte_aligned),
    "packed": ("bit_packed", write_bit_packed),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True, help="LOBSTER message CSV")
    p.add_argument("--outdir", type=Path, required=True, help="directory for the .bin files")
    p.add_argument("--no-header", action="store_true", help="the CSV has no header row")
    p.add_argument("--schemes", nargs="+", default=["all"], choices=["all", *SCHEMES])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = list(SCHEMES) if "all" in args.schemes else args.schemes
    stem = args.csv.stem

    for key in selected:
        suffix, writer = SCHEMES[key]
        out_path = args.outdir / f"{stem}.{suffix}.bin"
        count = writer(args.csv, out_path, not args.no_header)
        print(f"{suffix}: {count:,} messages -> {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
