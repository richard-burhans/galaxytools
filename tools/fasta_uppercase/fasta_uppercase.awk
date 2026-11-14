/^>/ { print; next }
{
    total += length($0)
    lower += gsub(/[a-z]/, "&")
    print toupper($0)
}
END {
    if (total == 0) {
        print "fasta_uppercase: no sequence found in input" > "/dev/stderr"
        exit 1
    }
    printf("upper-cased %d residues; discarded soft-masking on %d (%.2f%%)\n",
           total, lower, 100 * lower / total) > "/dev/stderr"
}
