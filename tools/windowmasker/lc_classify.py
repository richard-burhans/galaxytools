#!/usr/bin/env python3
"""lc_classify.py <fasta> <bed3|-> -> BED6

For each low-complexity interval, classify what it is made of so the 4th column
carries real content instead of the (redundant) tool name:

  name   = repeat-unit signature: 'polyA'/'polyT'/... (period 1),
           '(AT)n'/'(AAT)n'/... (short tandem, period 2-6), or 'lc' (complex)
  score  = purity * 1000 (fraction of bases matching the best periodic consensus;
           drives UCSC grayscale), clamped 0-1000
  strand = '.'

Smallest period that explains the interval well is preferred.

⚠ PERIODS 1-6 ONLY, AND THAT IS WHY THE fastan TRACK DOES NOT USE THIS. Anything with a longer
unit comes back `lc`. Measured on 103,341 real FASTAN intervals (ASM1303036v1), 75.9% have a unit
longer than 6 -- 65,084 of them 20 bp or more, median interval 62 bp, longest 22,755. FASTAN
measures the unit from its own tandem-array model and reports divergence, so classifying its
output here would fail on three quarters of the intervals and overwrite a measured value with a
guess. Its BED6 therefore carries a DIFFERENT vocabulary in columns 4 and 5 -- unit LENGTH
(`u65`) and 1000 - 10 x divergence, against the signature (`polyA`, `(AT)n`) and purity x 1000
this produces. Both are 0-1000 and higher-is-better, so nothing errors when they are mixed.

⛔ THE FASTA IS STREAMED ONE RECORD AT A TIME, AND THAT IS A MEMORY FIX, NOT A SPEED ONE.
This used to build a dict of every sequence in the assembly, uppercased. MEASURED on vgp:
five brc-lc-classify jobs peaked at 3,790 MB against a 3,788 MB grant -- 100.0%, with no
`ram_min` set on the tool at all. They survived, but a 585-job stage sitting exactly on its
ceiling fails by SIGKILL, silently as far as the tool is concerned.

The cost was never the sequence itself: it was holding every record at once, plus the
per-record `buf` list of line strings (a 1.5 Gb genome is ~31M small str objects before the
join) plus a whole-genome `.upper()` copy. Streaming holds ONE record, and uppercases only
the interval slices actually classified -- 3.7M bases on a real genome rather than 1.56e9.

⚠ SPEED WAS NOT THE REASON, AND DO NOT EXPECT ANY. Profiled on 288 real records with 62,027
real dustmasker intervals: loading the FASTA was 0.09 s of 1.08 s -- 8.5%. `classify()` is
where the time goes, and an obvious rewrite of it (Counter over strided slices) measured only
1.3x, because the intervals average 59 bases and Counter's overhead eats the win on strings
that short. lc_classify is 0.58 wall-hours of a run whose maskers are 6.48 EACH, so there is
no speed problem here worth taking risk for.

⚠ ON A FATAL ERROR THIS NOW WRITES NOTHING, WHERE IT USED TO EMIT A PARTIAL BED6. Output is
buffered because rows are COMPLETED in FASTA-record order and must be PRINTED in BED order, so
there is nothing to stream. Measured difference: on a BED naming a missing sequence the old
script had already put 21 bytes on stdout before exiting 1. Both exit 1 with the identical
message, and Galaxy discards the output of a failed job either way -- but a truncated BED6 that
looks well-formed is exactly the artifact this script's other guards exist to refuse, so
emitting none of it is the better of the two behaviours, not merely a side effect.

⛔ BED ORDER IS NOT ASSUMED, DELIBERATELY. The BED is read first and grouped by sequence, and
output is emitted in the ORIGINAL BED line order regardless of the order the FASTA presents
its records in. Requiring the BED to arrive grouped in FASTA order would have halved the
memory again and is true of every masker that feeds this today -- but it would make a
correct-looking input fail, and it would couple this script to an ordering no caller
declares. Byte-identical output for ANY input order is worth the extra dict.
"""
import re
import sys


def iter_fasta(path):
    """Yield (name, sequence) one record at a time. Never holds two records."""
    name = None
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


#: maximal runs of non-gap bases within an interval
NON_GAP = re.compile(r"[^N]+")


def classify(s):
    L = len(s)
    if L == 0:
        return "lc", 0
    best_frac = 0.0
    best_unit = ""
    best_p = 0
    for p in range(1, 7):
        if p > L:
            break
        cols = [{} for _ in range(p)]
        for i, ch in enumerate(s):
            cols[i % p][ch] = cols[i % p].get(ch, 0) + 1
        match = 0
        unit = ""
        for j in range(p):
            b = max(cols[j], key=cols[j].get)
            unit += b
            match += cols[j][b]
        frac = match / L
        if frac > best_frac + 1e-9:
            best_frac, best_unit, best_p = frac, unit, p
        if frac >= 0.85:   # smallest period that explains it well -> stop
            best_frac, best_unit, best_p = frac, unit, p
            break
    sc = int(round(best_frac * 1000))
    if best_frac < 0.60:
        return "lc", sc
    if best_p == 1:
        return "poly" + best_unit, sc
    return "(" + best_unit + ")n", sc


def main():
    fasta = sys.argv[1]
    bed = sys.argv[2] if len(sys.argv) > 2 else "-"
    who = bed if bed != "-" else "stdin"
    src = sys.stdin if bed == "-" else open(bed)

    # Pass 1: the BED, grouped by sequence, each interval remembering where it must be printed.
    by_seq = {}
    rows = []
    for n, line in enumerate(src, 1):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        c = f[0]
        a = int(f[1])
        b = int(f[2])
        by_seq.setdefault(c, []).append((n, a, b, len(rows)))
        rows.append(None)

    # ⛔ ERRORS ARE COLLECTED AND THE EARLIEST BED LINE WINS, because records no longer arrive
    # in BED order. Exiting at the first fault FOUND would name whichever line happened to sit
    # in the first offending FASTA record -- a different line number than this script has always
    # reported for the same pair of inputs, for no reason a reader could reconstruct.
    faults = []

    # Pass 2: one record at a time.
    for name, s in iter_fasta(fasta):
        want = by_seq.pop(name, None)
        if not want:
            continue
        L = len(s)
        for n, a, b, slot in want:
            if b > L or a < 0 or a >= b:
                faults.append((n, "lc_classify: %s line %d has interval %d-%d, outside %r "
                                  "(length %d)." % (who, n, a, b, name, L)))
                continue
            # ⛔ `.upper()` STAYS EVEN THOUGH THE WORKFLOW NOW GUARANTEES UPPERCASE INPUT.
            # MEASURED on 62,027 real intervals: on correctly-uppercase input it costs **+0.5%**
            # and returns identical results. Remove it and a soft-masked FASTA -- a standalone
            # run, a rewired workflow, a future caller -- reclassifies **49.4% of intervals** and
            # exits 0: `polyT` becomes `polyt`, `(AT)n` becomes `(at)n`. Six columns, legal
            # scores, well-formed BED6, and every repeat signature split into case variants that
            # propagate straight into the UCSC hub. The invariant it would lean on is created by
            # the same change that would remove it, and this is the only place that would notice
            # if that invariant were ever wrong.
            # ⛔ ASSEMBLY GAP IS NOT LOW COMPLEXITY. dustmasker reports every N run as an interval, so
            # an interval is split at N and each non-gap piece classified on its own; a pure-gap
            # interval is dropped. Otherwise gap reaches the union mask, `maskfasta -soft` turns it to
            # `n`, and masking_table counts it as masked: measured on one 888 Mb assembly with 16%
            # gap, dustmasker covered 100% of its 139.6 Mb of N and the union read 65.65% masked
            # against 59.14% of its non-gap bases. The other three maskers covered 0% of that gap.
            seg = s[a:b].upper()
            rows[slot] = "".join(
                "%s\t%d\t%d\t%s\t%d\t.\n" % (name, a + m.start(), a + m.end(), *classify(m.group()))
                for m in NON_GAP.finditer(seg))

    # A BED naming a sequence the FASTA does not hold is a MISMATCHED PAIR of inputs, not an
    # interval to classify. Falling back to "" made classify() return ("lc", 0) and this wrote
    # a complete, well-formed, entirely wrong BED6 line and exited 0 -- and every downstream
    # check passes it: the row count is preserved, there are six columns, 0 is a legal score,
    # and "lc" is a legal signature. Nothing anywhere would notice.
    for c, want in by_seq.items():
        n = want[0][0]
        faults.append((n, "lc_classify: %s line %d names sequence %r, which is not in %s. "
                          "The BED and the FASTA do not describe the same assembly."
                       % (who, n, c, fasta)))
    if faults:
        sys.exit(min(faults)[1])

    sys.stdout.writelines(rows)


if __name__ == "__main__":
    main()
