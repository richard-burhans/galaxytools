#!/usr/bin/env python3
"""
Process a MAF file for mafToBigMaf:
1. Drop blocks where the reference species (ref_acc) is absent
2. Reorder s-lines within each block so the reference is FIRST
3. Sort blocks by (ref_chrom, ref_start)
4. Write resulting MAF (overlaps preserved — multiz blocks legitimately overlap)

Usage: process_maf.py <ref_acc> <input_maf> <output_maf>
ref_acc: the accession prefix in s-line names (e.g. GCA_900093555.2)
"""
import sys

ref_acc, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]


def parse_blocks(fh):
    header_lines = []
    blocks = []
    current = []
    in_body = False
    for line in fh:
        if not in_body:
            header_lines.append(line)
            if line.startswith('##maf'):
                in_body = True
            continue
        if line.startswith('a '):
            if current:
                blocks.append(current)
            current = [line]
        elif line.strip() == '':
            if current:
                blocks.append(current)
                current = []
        else:
            if current:
                current.append(line)
    if current:
        blocks.append(current)
    return header_lines, blocks


def species_of_sline(line):
    """Extract species name from an s-line. MAF s-line: 's <species>.<chrom> ...'.
    Multiple dots possible (e.g. GCA_900093555.2.LT635612.2). We treat everything
    before the LAST dot-separated chrom-suffix as species. UCSC convention:
    species is everything up to the first dot."""
    name = line.split()[1]
    # If name starts with GCA_NNNNNNNNN.N (accession with one dot), grab first two parts
    parts = name.split('.')
    if len(parts) >= 2 and parts[0].startswith(('GCA_', 'GCF_')):
        return parts[0] + '.' + parts[1]
    return parts[0]


def find_ref_idx(block, ref_acc):
    """Return index (0-based among s-lines) of the first s-line whose species is ref_acc."""
    s_idx = 0
    for line in block:
        if line.startswith('s '):
            if species_of_sline(line) == ref_acc:
                return s_idx
            s_idx += 1
    return None


def reorder_block(block, ref_idx):
    """Move the s-line at index `ref_idx` to the first s-line position."""
    s_lines = [l for l in block if l.startswith('s ')]
    other = [l for l in block if not l.startswith('s ')]
    ref_line = s_lines.pop(ref_idx)
    new_s = [ref_line] + s_lines
    return other + new_s


def ref_coords(block, ref_acc):
    """Return (chrom, start, end) for the reference s-line. Block must have one."""
    for line in block:
        if line.startswith('s '):
            if species_of_sline(line) == ref_acc:
                parts = line.split()
                chrom = parts[1]
                start = int(parts[2])
                span = int(parts[3])
                return chrom, start, start + span
    return None


with open(src) as fh:
    header_lines, blocks = parse_blocks(fh)
print(f"  Parsed {len(blocks)} blocks", flush=True)

# Keep blocks with ref row, reorder if needed, record coords
kept = []
dropped_no_ref = 0
for block in blocks:
    idx = find_ref_idx(block, ref_acc)
    if idx is None:
        dropped_no_ref += 1
        continue
    if idx != 0:
        block = reorder_block(block, idx)
    coords = ref_coords(block, ref_acc)
    kept.append((coords[0], coords[1], coords[2], block))

print(f"  With ref row: {len(kept)} (dropped {dropped_no_ref} blocks lacking ref)", flush=True)

# Sort by ref (chrom, start)
kept.sort(key=lambda t: (t[0], t[1]))

# Write output
with open(dst, 'w') as out:
    for h in header_lines:
        out.write(h)
    for _, _, _, block in kept:
        for l in block:
            out.write(l)
        out.write('\n')

print(f"  Wrote {len(kept)} blocks to {dst}", flush=True)
