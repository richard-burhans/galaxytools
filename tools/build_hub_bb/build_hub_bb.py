#!/usr/bin/env python3
"""
Build UCSC hub BigBed inputs for a reference assembly from HyPhy BUSTED results
and an orthogroup table.

Consolidates the original two hardcoded scripts
(build_selection_bb.py + build_orthogroup_bb.py) into one parameterized tool.

Emits, as plain BED files (ready for bedToBigBed):

  * selection BED12+5  (strict, and optionally relaxed)  -- one row per gene
    with a BUSTED result that maps to an orthogroup. Extra fields:
      orthogroup_id, n_strains, busted_pvalue, busted_qvalue_fdr, gene_family
  * orthogroup membership BED12 -- one row per ref gene that maps to an OG.

BUSTED p-values are read from busted.json files, either from a directory tree
(``<dir>/<GENE_ID>/busted.json``) or from a ``.tar.gz`` archive with the same
layout. q-values are computed with Benjamini-Hochberg FDR across mapped genes.

Ported from:
  Pv4-pangenome/v3/ucsc_hub/build_selection_bb.py
  Pv4-pangenome/v3/ucsc_hub/build_orthogroup_bb.py
Both originals had no CLI; every hardcoded constant is now an argument.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import tarfile
from collections import defaultdict


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sizes(fai):
    """Read a .fa.fai (or 2-col sizes) file -> dict chrom -> length."""
    sizes = {}
    with open(fai) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            sizes[parts[0]] = int(parts[1])
    return sizes


def _open_maybe_gzip(path):
    """Open a possibly-gzipped text file."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path)


def load_bed12(bedpath):
    """Return dict: gene_id_base -> bed12_fields (list).

    Strips the trailing ``.N`` isoform suffix and keeps the first (primary)
    isoform.  Accepts plain or gzipped BED12.
    """
    bed = {}
    with _open_maybe_gzip(bedpath) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 12:
                continue
            name = f[3]
            base = name.rsplit('.', 1)[0] if '.' in name else name
            if base not in bed:
                bed[base] = f[:12]
    return bed


def load_ortholog_table(tsvpath, ref_column, gene_prefix):
    """Return (og_map, og_info).

    og_map:  gene_id -> (og_id, label, n_strains)
    og_info: og_id   -> (label, n_strains)

    The reference column may list several genes separated by ``|``; only genes
    starting with ``gene_prefix`` are kept.
    """
    og_map = {}
    og_info = {}
    with _open_maybe_gzip(tsvpath) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if ref_column not in (reader.fieldnames or []):
            sys.exit(
                f"ERROR: reference column {ref_column!r} not found in ortholog "
                f"table; available columns: {reader.fieldnames}"
            )
        for row in reader:
            og = row['orthogroup_id']
            label = row.get('label', '')
            n = int(row['n_strains'])
            og_info[og] = (label, n)
            ref_field = row.get(ref_column, '-')
            if ref_field == '-' or not ref_field:
                continue
            for gene in ref_field.split('|'):
                gene = gene.strip()
                if gene.startswith(gene_prefix):
                    og_map[gene] = (og, label, n)
    return og_map, og_info


def _gene_id_from_member(name):
    """Gene ID for an archive member, for either accepted layout, or None.

    Two layouts are accepted deliberately. ``<GENE_ID>/busted.json`` is what the fixture
    and Phase H's directory form use. ``<gene_id>.json`` at the tar ROOT is what
    ``workflows/ucsc_hub/ucsc_hub.gxwf.yml`` documents for ``busted_strict``.

    ⛔ THE TWO DISAGREED AND NOTHING NOTICED. Only the first was recognised, so the
    documented archive yielded an empty result dict, an empty selection BED, and a job
    that exited 0 -- a published track with no selection in it. Which form Phase H
    actually emits is not determinable from this repository, so both are read and
    ``extract_busted_jsons`` refuses an archive that yields nothing.
    """
    base = name.rsplit('/', 1)[-1]
    if base == 'busted.json':
        parts = [q for q in name.split('/') if q]
        return parts[-2] if len(parts) >= 2 else None
    if base.endswith('.json') and '/' not in name.strip('./'):
        return base[:-len('.json')] or None
    return None


def extract_busted_jsons(source):
    """``_extract_busted_jsons`` plus the one assertion that makes a miss visible.

    ⛔ ZERO GENES IS NOT A RESULT, IT IS A FAILURE TO READ THE INPUT. An unrecognised
    archive layout, a wrong tarball, or JSONs with no ``test results.p-value`` all land
    here identically, and the caller printed "Found 0 gene BUSTED results", wrote an empty
    selection BED and exited 0 -- publishing a track with nothing in it. Nothing
    downstream checks the count, so this is the only place it can be caught.
    """
    results = _extract_busted_jsons(source)
    if not results:
        raise SystemExit(
            f"error: no gene BUSTED results read from {source!r}. Accepted layouts are "
            f"'<GENE_ID>/busted.json' and '<gene_id>.json' at the archive root, and each "
            f"JSON must carry 'test results' -> 'p-value'. An empty result would otherwise "
            f"produce an empty selection track from a job that exited 0.")
    return results


def _extract_busted_jsons(source):
    """Return dict: gene_id -> p-value from busted.json.

    ``source`` is either a ``.tar.gz`` archive or a directory. See
    ``_gene_id_from_member`` for the two archive layouts accepted and why.

    ⛔ Refuses an input that yields no gene at all: that is indistinguishable from a
    successful run with nothing significant, and it published an empty track once.
    """
    results = {}
    if os.path.isdir(source):
        for root, _dirs, files in os.walk(source):
            if 'busted.json' not in files:
                continue
            gene_id = os.path.basename(root.rstrip('/'))
            try:
                with open(os.path.join(root, 'busted.json')) as fh:
                    d = json.load(fh)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN: failed to parse {root}/busted.json: {e}")
                continue
            pval = d.get('test results', {}).get('p-value', None)
            if pval is not None:
                results[gene_id] = float(pval)
        return results

    seen_json = 0
    with tarfile.open(source, 'r:gz') as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith('.json'):
                continue
            seen_json += 1
            gene_id = _gene_id_from_member(member.name)
            if gene_id is None:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                d = json.load(fh)
                pval = d.get('test results', {}).get('p-value', None)
                if pval is not None:
                    results[gene_id] = float(pval)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN: failed to parse {member.name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Stats / color helpers  (unchanged from the originals)
# ---------------------------------------------------------------------------

def bh_fdr(pvals_dict):
    """Benjamini-Hochberg FDR. Returns dict: key -> qvalue."""
    items = sorted(pvals_dict.items(), key=lambda x: x[1])
    n = len(items)
    qvals = {}
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        key, p = items[i]
        q = p * n / (i + 1)
        q = min(q, prev_q)
        qvals[key] = min(q, 1.0)
        prev_q = q
    return qvals


def qval_to_rgb(q):
    """Color by q-value bin."""
    if q < 0.01:
        return "255,0,0"      # red
    elif q < 0.05:
        return "255,128,0"    # orange
    elif q < 0.10:
        return "200,200,0"    # yellow
    else:
        return "128,128,128"  # gray


def qval_to_score(q):
    """0-1000 score, higher is more significant."""
    if q <= 0:
        return 1000
    try:
        s = int(-math.log10(q) * 100)
    except Exception:  # noqa: BLE001
        s = 0
    return max(0, min(1000, s))


def rgb_to_int(rgb_str):
    """Convert '255,0,0' to integer."""
    r, g, b = map(int, rgb_str.split(','))
    return (r << 16) | (g << 8) | b


def is_variant_antigen(gene_id, label):
    """Simple heuristic for variant antigen family."""
    va_labels = {'VIR', 'PIR', 'PHIST', 'DBP', 'RBP', 'SURFIN', 'SERA'}
    for va in va_labels:
        if va.lower() in label.lower():
            return label
    return 'other'


# ---------------------------------------------------------------------------
# BED builders
# ---------------------------------------------------------------------------

def build_selection_bed(busted_results, qvals, og_map, bed12, sizes, out_bed):
    """Write BED12+5 for selection track. Returns number of rows written."""
    written = 0
    skipped_no_og = 0
    skipped_no_bed = 0
    skipped_no_chrom = 0

    lines = []
    for gene_id, pval in busted_results.items():
        if gene_id not in og_map:
            skipped_no_og += 1
            continue
        og_id, label, n_strains = og_map[gene_id]
        if gene_id not in bed12:
            skipped_no_bed += 1
            continue
        b = bed12[gene_id]
        chrom = b[0]
        if chrom not in sizes:
            skipped_no_chrom += 1
            continue
        qval = qvals.get(gene_id, 1.0)
        rgb = qval_to_rgb(qval)
        score = qval_to_score(qval)
        rgb_int = rgb_to_int(rgb)
        gene_family = is_variant_antigen(gene_id, label)
        # BED12+5: chrom,start,end,name,score,strand,thickStart,thickEnd,
        #          itemRgb,blockCount,blockSizes,blockStarts,
        #          orthogroup_id,n_strains,busted_pvalue,busted_qvalue_fdr,gene_family
        row = '\t'.join([
            b[0], b[1], b[2], og_id, str(score), b[5],
            b[6], b[7], str(rgb_int),
            b[9], b[10], b[11],
            og_id, str(n_strains),
            f"{pval:.6g}", f"{qval:.6g}", gene_family,
        ])
        lines.append((b[0], int(b[1]), row))

    lines.sort(key=lambda x: (x[0], x[1]))
    with open(out_bed, 'w') as fh:
        for _chrom, _start, row in lines:
            fh.write(row + '\n')
            written += 1

    print(f"  Selection written: {written}, skipped_no_og: {skipped_no_og}, "
          f"skipped_no_bed: {skipped_no_bed}, skipped_no_chrom: {skipped_no_chrom}")
    return written


def build_orthogroup_bed(og_map, bed12, sizes, out_bed):
    """Write orthogroup membership BED12. Returns number of rows written."""
    lines = []
    clipped = 0
    for gene_id, (og_id, _label, n_strains) in og_map.items():
        if gene_id not in bed12:
            continue
        b = bed12[gene_id]
        chrom = b[0]
        if chrom not in sizes:
            continue
        start = int(b[1])
        end = int(b[2])
        chrom_size = sizes[chrom]
        block_sizes = [int(x) for x in b[10].rstrip(',').split(',') if x]
        block_starts_rel = [int(x) for x in b[11].rstrip(',').split(',') if x]
        if block_sizes and block_starts_rel:
            true_end = start + block_starts_rel[-1] + block_sizes[-1]
        else:
            true_end = end
        if true_end > chrom_size or end > chrom_size or start >= end:
            clipped += 1
            continue
        # Color by n_strains: few=red, many=green
        r = max(0, int(255 * (1 - (n_strains - 1) / 7)))
        g = max(0, int(255 * ((n_strains - 1) / 7)))
        rgb_int = (r << 16) | g
        score = int(n_strains * 125)
        row = '\t'.join([
            chrom, str(start), str(end), og_id, str(score), b[5],
            b[6], b[7], str(rgb_int), b[9], b[10], b[11],
        ])
        lines.append((chrom, start, row))

    lines.sort(key=lambda x: (x[0], x[1]))
    with open(out_bed, 'w') as fh:
        for _chrom, _start, row in lines:
            fh.write(row + '\n')
    print(f"  Orthogroup written: {len(lines)} rows, clipped {clipped}")
    return len(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_one_selection(name, source, og_map, bed12, sizes, out_bed):
    print(f"\n=== {name} BUSTED selection ===")
    results = extract_busted_jsons(source)
    print(f"  Found {len(results)} gene BUSTED results")
    for_fdr = {g: p for g, p in results.items() if g in og_map}
    qvals = bh_fdr(for_fdr)
    n01 = sum(1 for q in qvals.values() if q < 0.01)
    n05 = sum(1 for q in qvals.values() if q < 0.05)
    print(f"  Significant at q<0.01: {n01}, q<0.05: {n05}")
    return build_selection_bed(results, qvals, og_map, bed12, sizes, out_bed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build selection BED12+5 and orthogroup BED12 for a UCSC hub."
    )
    p.add_argument('--ortholog-table', required=True,
                   help='Ortholog table TSV (optionally .gz).')
    p.add_argument('--ref-bed12', required=True,
                   help='Reference gene models in BED12 (optionally .gz).')
    p.add_argument('--sizes', required=True,
                   help='Chromosome sizes (.fa.fai or 2-column sizes).')
    p.add_argument('--ref-acc', default='reference',
                   help='Reference accession label (informational).')
    p.add_argument('--ref-column', default='PvP01',
                   help='Column in the ortholog table holding reference genes.')
    p.add_argument('--gene-prefix', default='PVP01_',
                   help='Only genes with this prefix are kept from the column.')
    p.add_argument('--busted-strict', required=True,
                   help='Strict BUSTED source: directory or .tar.gz of '
                        '<GENE>/busted.json.')
    p.add_argument('--busted-relaxed', default=None,
                   help='Optional relaxed BUSTED source (directory or .tar.gz).')
    p.add_argument('--out-selection-strict', required=True,
                   help='Output path for the strict selection BED12+5.')
    p.add_argument('--out-selection-relaxed', default=None,
                   help='Output path for the relaxed selection BED12+5.')
    p.add_argument('--out-orthogroup', required=True,
                   help='Output path for the orthogroup membership BED12.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Reference accession: {args.ref_acc}")
    print("Loading genome sizes...")
    sizes = load_sizes(args.sizes)
    print(f"  {len(sizes)} sequences")

    print("Loading reference BED12...")
    bed12 = load_bed12(args.ref_bed12)
    print(f"  {len(bed12)} genes")

    print("Loading ortholog table...")
    og_map, og_info = load_ortholog_table(
        args.ortholog_table, args.ref_column, args.gene_prefix)
    print(f"  {len(og_map)} gene->OG mappings, {len(og_info)} OGs")

    # --- Strict selection (required) ---
    build_one_selection("Strict", args.busted_strict, og_map, bed12, sizes,
                        args.out_selection_strict)

    # --- Relaxed selection (optional) ---
    if args.busted_relaxed and args.out_selection_relaxed:
        build_one_selection("Relaxed", args.busted_relaxed, og_map, bed12,
                            sizes, args.out_selection_relaxed)

    # --- Orthogroup membership ---
    print("\n=== Orthogroup membership ===")
    build_orthogroup_bed(og_map, bed12, sizes, args.out_orthogroup)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
