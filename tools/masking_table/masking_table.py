#!/usr/bin/env python3
"""Assemble per-strain % genome masked into a clean MultiQC custom-content table.

bedtools genomecov (hist, -max 1) already did the coverage; this tool only reads
each masker's per-strain genomecov output, picks the genome-wide depth>=1 row
(``genome <TAB> 1 <TAB> covered_bp <TAB> genome_bp <TAB> fraction``), and pivots
the five maskers into one matrix:

    Sample <TAB> dustmasker <TAB> windowmasker <TAB> tantan <TAB> fastan <TAB> union
    PvP01  <TAB> 13.63 ...

The header is plain (no leading ``#``) so MultiQC custom-content reads it as a
table rather than mis-parsing it as embedded YAML config. Inputs are passed as
space-separated ``identifier=path`` tokens per collection (the element-identifier
reading pattern used by collection_relabel_map).
"""
import argparse


def parse_pairs(tokens):
    out = {}
    for t in tokens:
        if not t:
            continue
        ident, _, path = t.partition("=")
        out[ident] = path
    return out


def genome_masked_pct(genomecov_path):
    """Return 100 * fraction from the 'genome' depth>=1 row of a genomecov hist."""
    with open(genomecov_path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            # chrom, depth, covered_bp, chrom_bp, fraction
            if len(cols) >= 5 and cols[0] == "genome" and cols[1] == "1":
                return round(100.0 * float(cols[4]), 2)
    # No depth>=1 genome row => nothing masked
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    # ⛔ DEFAULT None, NOT []. A caller that does not supply this gets NO `arrived`
    # column at all, because 0.00 would be a LIE there rather than a blank: the
    # classic softmask.gxwf.yml never extracts the arriving mask -- it lets it survive
    # in the sequence bytes -- so an assembly that shipped 79% soft-masked would be
    # reported as 0.00% arrived. `[]` cannot express "not asked for".
    ap.add_argument("--arrived", nargs="*", default=None)
    ap.add_argument("--dustmasker", nargs="*", default=[])
    ap.add_argument("--windowmasker", nargs="*", default=[])
    ap.add_argument("--tantan", nargs="*", default=[])
    ap.add_argument("--fastan", nargs="*", default=[])
    ap.add_argument("--union", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ⚠ NO SQUARE BRACKETS INSIDE THE `maskers` LIST. assert_column_order() finds this list with a
    # NON-GREEDY `maskers\s*=\s*\[(.*?)\]`, so a `[]` written inline -- `parse_pairs(a.arrived or [])`
    # -- closes the match early and the checker sees a one-column order. It refused, correctly and
    # confusingly. Resolve the default out here instead.
    arrived_pairs = parse_pairs(a.arrived) if a.arrived is not None else {}
    maskers = [
        # ⛔ `arrived` IS FIRST, AND IT IS NOT ONE OF OUR MASKERS. It is the soft-mask the assembly
        # SHIPPED WITH, which brc-fasta-uppercase now emits as intervals instead of leaving it to
        # survive in the sequence bytes. It joins the union as a fifth arm when strip_existing_mask
        # is false, so without this column the `union` figure would silently include a contribution
        # no column accounts for -- which is the whole reason the column exists. Reading order is
        # then: what arrived, what each masker found, what was published.
        ("arrived", arrived_pairs),
        ("dustmasker", parse_pairs(a.dustmasker)),
        ("windowmasker", parse_pairs(a.windowmasker)),
        ("tantan", parse_pairs(a.tantan)),
        ("fastan", parse_pairs(a.fastan)),
        ("union", parse_pairs(a.union)),
    ]
    # ⚠ THE LITERAL LIST ABOVE STAYS COMPLETE AND IN ORDER, because assert_column_order() reads
    # it by regex to prove the header matches. Dropping the column happens HERE, after that.
    if a.arrived is None:
        maskers = [m for m in maskers if m[0] != "arrived"]
    cols = [name for name, _ in maskers]

    # Strain order follows the first non-empty masker collection.
    strains = []
    for _, mp in maskers:
        if mp:
            strains = list(mp.keys())
            break

    with open(a.out, "w") as out:
        out.write("Sample\t" + "\t".join(cols) + "\n")
        for strain in strains:
            vals = []
            for _, mp in maskers:
                p = mp.get(strain)
                vals.append(genome_masked_pct(p) if p else 0.0)
            out.write(strain + "\t" + "\t".join(f"{v:.2f}" for v in vals) + "\n")


if __name__ == "__main__":
    main()
