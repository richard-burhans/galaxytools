#!/usr/bin/env python3
"""
Build a global UCSC assembly-hub genomes.txt from a per-assembly metadata table.

The genomes.txt file lists every assembly in an assembly hub. Each assembly is
described by a stanza of one ``key value`` line per field, stanzas separated by
a blank line. This script emits the exact stanza layout committed in the
Pv4-pangenome assembly hub:

    genome <accession>
    trackDb <accession>/trackDb.txt
    groups <accession>/groups.txt
    description <description>
    twoBitPath <accession>/<accession>.2bit
    organism <organism>
    defaultPos <chrom>:<start>-<end>
    scientificName <scientific name>
    htmlPath <accession>/description.html

Input is a TSV metadata table with a header row. Required columns:

    accession    assembly accession used as the genome id / directory name
    defaultPos   default browser position, formatted chrom:start-end
    organism     organism token (underscored, e.g. Plasmodium_vivax)
    scientificName   scientific name (e.g. Plasmodium vivax)
    description  human-readable assembly description

Optional columns (sensible defaults derived from the accession when absent):

    twoBitPath   path to the .2bit (default <accession>/<accession>.2bit)
    trackDb      path to trackDb.txt (default <accession>/trackDb.txt)
    groups       path to groups.txt (default <accession>/groups.txt)
    htmlPath     path to description.html (default <accession>/description.html)

Requires: Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

# Accepts chrom:start-end, e.g. LT635625.2:1264700-1277700
DEFAULTPOS_RE = re.compile(r"^[^\s:]+:\d+-\d+$")

# Order of fields in each emitted stanza.
FIELD_ORDER = [
    "genome",
    "trackDb",
    "groups",
    "description",
    "twoBitPath",
    "organism",
    "defaultPos",
    "scientificName",
    "htmlPath",
]

REQUIRED_COLUMNS = [
    "accession",
    "defaultPos",
    "organism",
    "scientificName",
    "description",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a global UCSC assembly-hub genomes.txt "
        "from a per-assembly metadata table."
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Per-assembly metadata table (TSV with header).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the assembled genomes.txt.",
    )
    return p.parse_args(argv)


def build_record(row, line_no):
    """Build the ordered field map for one assembly from a metadata row."""
    acc = (row.get("accession") or "").strip()
    if not acc:
        raise ValueError(f"row {line_no}: empty 'accession'")

    defaultpos = (row.get("defaultPos") or "").strip()
    if not DEFAULTPOS_RE.match(defaultpos):
        raise ValueError(
            f"row {line_no} ({acc}): 'defaultPos' must be chrom:start-end, "
            f"got {defaultpos!r}"
        )

    def col(name, default):
        val = (row.get(name) or "").strip()
        return val if val else default

    return {
        "genome": acc,
        "trackDb": col("trackDb", f"{acc}/trackDb.txt"),
        "groups": col("groups", f"{acc}/groups.txt"),
        "description": col("description", "").strip()
        or _require(row, "description", line_no, acc),
        "twoBitPath": col("twoBitPath", f"{acc}/{acc}.2bit"),
        "organism": _require(row, "organism", line_no, acc),
        "defaultPos": defaultpos,
        "scientificName": _require(row, "scientificName", line_no, acc),
        "htmlPath": col("htmlPath", f"{acc}/description.html"),
    }


def _require(row, name, line_no, acc):
    val = (row.get(name) or "").strip()
    if not val:
        raise ValueError(f"row {line_no} ({acc}): empty required column {name!r}")
    return val


def format_record(record):
    return "\n".join(f"{key} {record[key]}" for key in FIELD_ORDER)


def main(argv=None):
    args = parse_args(argv)

    with open(args.metadata, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("error: metadata table is empty (no header row)")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                "error: metadata table missing required column(s): "
                + ", ".join(missing)
            )

        stanzas = []
        for i, row in enumerate(reader, start=2):  # header is line 1
            # skip wholly blank lines
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                record = build_record(row, i)
            except ValueError as exc:
                raise SystemExit(f"error: {exc}")
            stanzas.append(format_record(record))

    if not stanzas:
        raise SystemExit("error: no assembly rows found in metadata table")

    with open(args.output, "w") as out:
        out.write("\n\n".join(stanzas))
        out.write("\n")

    sys.stderr.write(f"Wrote {len(stanzas)} assembly stanza(s) to {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
