# FasTAN annotation (.1ano, ONEcode) -> BED6, in SCAFFOLD coordinates.
#
# THE .1ano SCHEMA, TAKEN FROM FasTAN's OWN SOURCE (ANO.c, `anoSchemaText`) rather than inferred:
#
#     O S 1 6 STRING              id for a scaffold
#     D G 1 3 INT                 gap of given length
#     D C 1 3 INT                 contig of given length
#     O M 3 3 INT 3 INT 3 INT     scaffold index, beg,end pair
#     D L 1 6 STRING              optional label for preceeding M
#     D X 1 3 INT                 optional score for the preceeding M
#
# So `M` carries a scaffold INDEX (not a name) and a beg,end pair that the schema itself calls
# scaffold-relative -- which is why this reads the annotation directly and does not accumulate the
# C/G records. The index is the ORDER of the S lines, 0-based, which is what `nscaf` tracks.
#
# ⛔ ONEcode ASCII LENGTH-PREFIXES EVERY STRING FIELD, AND THAT IS WHY THE COLUMNS DIFFER BY TYPE.
# ONElib reads a STRING as `readInt` then `readString`, and writes it as `" %lld %s"` (ONElib.c,
# writeStringList) -- so an `S` line is `S 4 chrX` and the VALUE is $3, while an `X` line is a bare
# INT and its value is $2. Getting that backwards shifts every field silently and yields intervals
# under the wrong scaffold, which is exactly the failure mode this file exists to avoid.
#
# ⛔ READ THE ANNOTATION FILE, NOT ANOtoBED's BED. `ANOtoBED` emits CONTIG-relative coordinates
# under the SCAFFOLD's name, so on any assembly with internal N-gaps every interval after the first
# gap is reported at the wrong place -- and the file it names cannot be used with it. Measured with
# planted ground truth: a scaffold `chrX` holding a 20-mer array at 500-1700 and a 25-mer array at
# 3500-4750, split by an 800 bp N-gap at 2200:
#
#     planted    chrX  500 1700   |  chrX 3500 4750
#     tantan     chrX  519 1702   |  chrX 3523 4752      (independent, correct)
#     ANOtoBED   chrX  500 1682   |  chrX  499 1725      <- second array 3 kb to its left,
#                                                           and the two intervals OVERLAP
#
# Nothing upstream loses anything. The .1ano itself is CORRECT -- `M 0 500 1682` and
# `M 0 3499 4725`, scaffold-relative, per its own declared schema
# ("~ O M 3 3 INT 3 INT 3 INT   scaffold index, beg,end pair"). ANOtoBED subtracts a contig offset
# it should not. So this reads the authoritative file through `ONEview`, its declared reader, and
# is byte-identical to the old pipeline wherever ANOtoBED was already right (a gapless scaffold).
#
# ONEcode records used, in file order:
#     S <len> <name>          a scaffold; they appear in index order 0, 1, 2, ...
#     M <scaf> <beg> <end>    an annotated interval, ALREADY scaffold-relative
#     L <len> <unit>          label for the preceding M -- FasTAN puts the repeat unit length here
#     X <divergence>          score for the preceding M
# Flush on the NEXT M or at END, never on a trailing optional record: `P` (the parse partition) is
# present for FasTAN and need not be for another annotator.
BEGIN { OFS = "\t"; nscaf = -1; have = 0 }

function flush() {
    if (!have) return
    # ⛔ X IS DIVERGENCE, AND THIS FILE USED TO CALL IT IDENTITY. Measured by planting a 30-mer
    # array 40 times over and substituting bases at a known rate: 0% -> X=0, 2% -> 1, 5% -> 7,
    # 10% -> 12, 20% -> 28. So X RISES as the array degrades, on roughly a percent scale -- while
    # the old code passed it straight into BED column 5 as "identity, clamped 0-1000 (UCSC
    # grayscale)". In a hub track that renders the PUREST arrays palest, inverted against every
    # other track here, and compressed into the bottom 3% of the range. Mapped to an
    # identity-like 0-1000 where higher is better, which is what column 5 means.
    d = score + 0; if (d < 0) d = 0; if (d > 100) d = 100
    print name[s], b, e, "u" unit, int(1000 - 10 * d), "."
    have = 0
}

$1 == "S" { nscaf++; name[nscaf] = $3; next }
$1 == "M" { flush(); s = $2 + 0; b = $3; e = $4; unit = "?"; score = 0; have = 1; next }
$1 == "L" && have { unit = $3; next }
$1 == "X" && have { score = $2; next }
END { flush() }
