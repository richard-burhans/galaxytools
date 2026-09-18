#!/usr/bin/env python3
"""Generate the Growler LASTZ test data: two .2bit sequences and a two-strand pair file.

    python3 make_test_data.py [output_dir]

The test asserts that BOTH strands appear in the output, so the data has to contain a
plus-strand and a minus-strand alignment. That is the tool's whole contract: a pair file
holds both strands, written plus-before-minus, and LASTZ is invoked with no --strand.

The generated files were run through real LASTZ before being committed. The first attempt
packed the .2bit header as one uint32 plus two uint16 and LASTZ rejected it with
"bad 2bit version (00010000)" -- the header is FOUR uint32 fields.
"""
import gzip
import pathlib
import random
import struct
import sys

COMPLEMENT = str.maketrans("ACGT", "TGCA")

#: .2bit packs four bases per byte, two bits each, in this order.
BASE_BITS = {"T": 0, "C": 1, "A": 2, "G": 3}

SIGNATURE = 0x1A412743
SHARED_LENGTH = 150
FILLER_LENGTH = 50


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def pack_dna(sequence):
    """Two bits per base, four per byte, the last byte zero-padded on the right."""
    packed = bytearray()
    accumulator = 0
    held = 0
    for base in sequence.upper():
        accumulator = (accumulator << 2) | BASE_BITS.get(base, 0)
        held += 1
        if held == 4:
            packed.append(accumulator)
            accumulator = 0
            held = 0
    if held:
        packed.append(accumulator << (2 * (4 - held)))
    return bytes(packed)


def write_2bit(path, sequences):
    names = list(sequences)
    index_length = sum(1 + len(name) + 4 for name in names)
    header_length = 16 + index_length

    blobs = {}
    offsets = {}
    position = header_length
    for name in names:
        sequence = sequences[name]
        # dnaSize, nBlockCount, maskBlockCount, reserved, then the packed bases
        blobs[name] = (
            struct.pack("<IIII", len(sequence), 0, 0, 0) + pack_dna(sequence)
        )
        offsets[name] = position
        position += len(blobs[name])

    with open(path, "wb") as handle:
        # signature, version, sequenceCount, reserved -- four uint32 fields
        handle.write(struct.pack("<IIII", SIGNATURE, 0, len(names), 0))
        for name in names:
            handle.write(struct.pack("B", len(name)))
            handle.write(name.encode())
            handle.write(struct.pack("<I", offsets[name]))
        for name in names:
            handle.write(blobs[name])


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rnd = random.Random(20260917)

    def filler(length):
        return "".join(rnd.choice("ACGT") for _ in range(length))

    shared_plus = filler(SHARED_LENGTH)
    shared_minus = filler(SHARED_LENGTH)

    target = shared_plus + filler(FILLER_LENGTH) + shared_minus + filler(FILLER_LENGTH)
    query = (
        shared_plus
        + filler(FILLER_LENGTH)
        + reverse_complement(shared_minus)
        + filler(FILLER_LENGTH)
    )

    write_2bit(out / "target.2bit", {"target1": target})
    write_2bit(out / "query.2bit", {"query1": query})

    # Segment coordinates are one-based and inclusive. Minus-strand query intervals are
    # counted along the reverse strand, so the forward region 201..350 of a 400 bp query
    # is 400-350+1 .. 400-201+1 = 51..200 there.
    # the reverse-complemented block sits at these FORWARD coordinates in both sequences
    forward_start = SHARED_LENGTH + FILLER_LENGTH + 1
    forward_end = SHARED_LENGTH * 2 + FILLER_LENGTH
    # reverse-strand coordinates are (length - forward + 1), which swaps the endpoints
    minus_start = len(query) - forward_end + 1
    minus_end = len(query) - forward_start + 1
    segments = [
        "target1\t1\t{}\tquery1\t1\t{}\t+\t3000".format(SHARED_LENGTH, SHARED_LENGTH),
        "target1\t{}\t{}\tquery1\t{}\t{}\t-\t3000".format(
            forward_start, forward_end, minus_start, minus_end
        ),
    ]
    with gzip.open(out / "pair_two_strand.segments.gz", "wb", compresslevel=6) as handle:
        handle.write(("\n".join(segments) + "\n").encode())

    print("wrote target.2bit query.2bit pair_two_strand.segments.gz")
    print("target {} bp, query {} bp".format(len(target), len(query)))


if __name__ == "__main__":
    main()
