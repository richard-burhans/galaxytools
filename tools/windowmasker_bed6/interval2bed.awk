# NCBI dustmasker/windowmasker `-outfmt interval` -> BED3 (chrom, start, end).
#
# ⛔ WHY THIS FILE EXISTS AT ALL, WHEN `lc2bed.awk` ALREADY DOES THE SAME JOB. dustmasker and
# windowmasker BOTH support `-outfmt fasta`, which emits the soft-masked sequence tantan already
# emits -- so both maskers could take the lowercase-run route and this converter could be deleted
# outright. That would remove a whole coordinate-convention to misread, and this file is the proof
# that such a convention CAN be misread: see the off-by-one below.
#
# ⚠ IT IS KEPT ANYWAY, AND THE REDUNDANCY IS THE POINT. scripts/check_coordinates.py validates
# THIS file's BED against the lowercase runs of the SAME tool's `-outfmt fasta` on the SAME input --
# two independent routes out of one tool, agreeing base for base. That is the strongest coordinate
# check in this repository, and it is what caught the off-by-one below. Move the maskers onto the
# fasta route and the check collapses into comparing this awk against a Python reimplementation of
# the same idea, which is exactly what the tantan case already is and is measurably weaker: it can
# only catch a bug that one of the two implementations has and the other does not.
#
# So two converters is not duplication to be tidied away. Deleting either one is how you lose the
# ability to tell whether the survivor is right.
#
# ⛔ THE INTERVAL FORMAT IS 0-BASED AND INCLUSIVE AT BOTH ENDS, and this file spent its whole life
# converting it as if it were 1-based. MEASURED against dustmasker's own `-outfmt fasta` on the
# same sequence, which is the only ground truth that settles it:
#
#     -outfmt interval      873 - 880          1500 - 1800
#     lowercase runs        [873, 881)         [1500, 1801)      <- what was actually masked
#     what this emitted     872   880          1499   1800       <- one base LEFT, right width
#
# `s = $1 - 1; print c, s, $2` preserves the WIDTH, which is why nothing caught it: every coverage
# percentage, every `bedtools genomecov` column of the masking table, and
# scripts/verify_softmask_outputs.py (which compares the union against the lowercase runs of a
# FASTA masked from that same union) all agree with themselves. Only the POSITION is wrong -- by
# one base, at every dustmasker and windowmasker interval in every strain. Downstream that
# lower-cases one non-repeat base and leaves one repeat base uppercase at each boundary, and it
# puts `lc_classify` out of phase: a `(AT)n` array was reported as `(TA)n`, a `(CAGGT)n` unit as
# `(GTCAG)n`, and a pure polyA run at purity 997 instead of 1000.
#
# At a sequence start the old clamp turned the shift into a TRUNCATION: dustmasker reporting
# `0 - 199` for a 200 bp leading repeat became `0 199`, losing one masked base outright. With the
# correct conversion the clamp has nothing to do -- `$1` is already >= 0 -- so it is gone.
#
# (content for cols 4-6 is added downstream by lc_classify.py)
BEGIN { OFS = "\t" }
/^>/  { c = substr($1, 2); next }
/[0-9]+ *- *[0-9]+/ { gsub(/-/, " "); print c, $1, $2 + 1 }
