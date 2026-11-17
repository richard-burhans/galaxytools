#!/usr/bin/env python3
"""Pivot per-masker bedtools genomecov output into a MultiQC custom-content table.

Each input is ``bedtools genomecov`` (hist, ``-max 1``) output for one strain. From each,
the script takes the genome-wide depth>=1 row
(``genome <TAB> 1 <TAB> covered_bp <TAB> genome_bp <TAB> fraction``) and writes:

    Sample <TAB> dustmasker <TAB> windowmasker <TAB> tantan <TAB> fastan <TAB> union
    PvP01  <TAB> 13.63 ...

The header has no leading ``#``, so MultiQC reads it as a table header, not as YAML config.
Each masker option takes space-separated ``identifier=path`` tokens, one per strain.
"""

import argparse
from pathlib import Path


def parse_pairs(tokens: list[str]) -> dict[str, str]:
    """Map identifier to path for ``identifier=path`` tokens, skipping empty tokens."""
    pairs: dict[str, str] = {}
    for token in tokens:
        if token == "":
            continue
        identifier, _, path = token.partition("=")
        pairs[identifier] = path
    return pairs


def genome_masked_pct(genomecov_path: Path) -> float:
    """Return the percent masked from the ``genome`` depth-1 row of a genomecov hist."""
    with genomecov_path.open(encoding="utf-8") as fh:
        for line in fh:
            # chrom, depth, covered_bp, chrom_bp, fraction
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 5 and cols[0] == "genome" and cols[1] == "1":
                return round(100.0 * float(cols[4]), 2)
    # No depth-1 genome row: nothing is masked.
    return 0.0


def masked_pct(pairs: dict[str, str], strain: str) -> float:
    """Return the strain's percent masked, or 0.0 if it has no input path."""
    path = pairs.get(strain, "")
    if path == "":
        return 0.0
    return genome_masked_pct(Path(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # `--arrived` is the soft-mask the assembly shipped with. Its default is None, not [],
    # so that omitting it drops the column: 0.00 would wrongly claim nothing arrived masked.
    parser.add_argument("--arrived", nargs="*", default=None)
    parser.add_argument("--dustmasker", nargs="*", default=[])
    parser.add_argument("--windowmasker", nargs="*", default=[])
    parser.add_argument("--tantan", nargs="*", default=[])
    parser.add_argument("--fastan", nargs="*", default=[])
    parser.add_argument("--union", nargs="*", default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    arrived_pairs = parse_pairs(args.arrived) if args.arrived is not None else {}
    # An external checker reads this list by regex to verify the column order, so keep it
    # literal, complete, in order, and free of square brackets. `arrived` comes first: it
    # also feeds the union, so the table shows what arrived, what each masker found, and the
    # union that was published.
    maskers = [
        ("arrived", arrived_pairs),
        ("dustmasker", parse_pairs(args.dustmasker)),
        ("windowmasker", parse_pairs(args.windowmasker)),
        ("tantan", parse_pairs(args.tantan)),
        ("fastan", parse_pairs(args.fastan)),
        ("union", parse_pairs(args.union)),
    ]
    if args.arrived is None:
        maskers = [masker for masker in maskers if masker[0] != "arrived"]

    # Strain order follows the first non-empty masker.
    strains = next((list(pairs) for _, pairs in maskers if pairs), [])

    with args.out.open("w", encoding="utf-8") as out:
        out.write("Sample\t" + "\t".join(name for name, _ in maskers) + "\n")
        for strain in strains:
            values = [masked_pct(pairs, strain) for _, pairs in maskers]
            out.write(strain + "\t" + "\t".join(f"{v:.2f}" for v in values) + "\n")


if __name__ == "__main__":
    main()
