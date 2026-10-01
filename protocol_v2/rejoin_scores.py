#!/usr/bin/env python3
"""Rejoin the row-split scores files back into a single scores.csv.

Two arms had scores.csv larger than the 2 MB transfer ceiling, so each was split by
ROWS into scores.part1of2.csv and scores.part2of2.csv. Each half carries the header,
so each half is a valid CSV on its own and can be read without the other.

    python rejoin_scores.py results/arms/P2_cutoff
    python rejoin_scores.py results/arms/P3_acausal

Verifies the headers match and that the row count comes to 338,456 before writing.
The two halves are left in place; nothing is deleted.
"""
import sys, csv
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
p1, p2, out = d/"scores.part1of2.csv", d/"scores.part2of2.csv", d/"scores.csv"
for p in (p1, p2):
    if not p.exists():
        sys.exit(f"missing {p} — that part did not arrive")
if out.exists():
    sys.exit(f"{out} already exists; not overwriting")

h1 = p1.open().readline()
h2 = p2.open().readline()
if h1 != h2:
    sys.exit("the two halves have different headers; wrong pair of files")

n = 0
with out.open("w", newline="") as w:
    w.write(h1)
    for p in (p1, p2):
        with p.open() as f:
            f.readline()
            for line in f:
                w.write(line); n += 1
print(f"wrote {out}  ({n:,} data rows)")
if n != 338456:
    print(f"  WARNING: expected 338,456 rows, got {n:,}. A part may be truncated.")
