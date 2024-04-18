#!/usr/bin/env python

"""
Diagonal partitioning for segment files output by SegAlign.

Usage:
diagonal_partition <max-segments> <lastz-command>
"""

import os
import sys


def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


if __name__ == "__main__":

    DELETE_AFTER_CHUNKING = True

    # input_params = "10000 sad sadsa sad --segments=tmp10.block5.r1239937044.plus.segments dsa sa --strand=plus --output=out.maf sadads 2> logging.err"
    # sys.argv = [sys.argv[0]] + input_params.split(' ')
    chunk_size = int(sys.argv[1])  # first parameter contains chunk size
    params = sys.argv[2:]

    # Parsing command output from SegAlign
    segment_key = "--segments="
    segment_index = None
    input_file = None

    for index, value in enumerate(params):
        if value[: len(segment_key)] == segment_key:
            segment_index = index
            input_file = value[len(segment_key):]
            break
    if segment_index is None:
        print(f"Error: could not segment key {segment_key} in parameters {params}")
        exit(0)

    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} does not exist")
        exit(0)

    if (
        chunk_size == 0 or sum(1 for _ in open(input_file)) <= chunk_size
    ):  # no need to sort if number of lines <= chunk_size
        print(" ".join(params), flush=True)
        exit(0)

    # Find rest of relevant parameters
    output_key = "--output="
    output_index = None
    output_alignment_file = None
    output_alignment_file_base = None
    output_format = None

    strand_key = "--strand="
    strand_index = None
    for index, value in enumerate(params):
        if value[: len(output_key)] == output_key:
            output_index = index
            output_alignment_file = value[len(output_key):]
            output_alignment_file_base, output_format = output_alignment_file.rsplit(
                ".", 1
            )
        if value[: len(strand_key)] == strand_key:
            strand_index = index
    if segment_index is None:
        print(f"Error: could not output key {output_key} in parameters {params}")
        exit(0)
    if strand_index is None:
        print(f"Error: could not output key {strand_key} in parameters {params}")
        exit(0)

    err_index = -1  # error file is at very end
    err_name_base = params[-1].split(".err", 1)[0]

    data = {}  # dict of list of tuple (x, y, str)

    direction = None
    if "plus" in params[strand_index]:
        direction = "f"
    elif "minus" in params[strand_index]:
        direction = "r"
    else:
        print(
            f"Error: could not figure out direction from strand value {params[strand_index]}"
        )
        exit(0)

    for line in open(input_file, "r"):
        if line == "":
            continue
        (
            seq1_name,
            seq1_start,
            seq1_end,
            seq2_name,
            seq2_start,
            seq2_end,
            _dir,
            score,
        ) = line.split()
        # data.append((int(seq1_start), int(seq2_start), line))
        half_dist = int((int(seq1_end) - int(seq1_start)) // 2)
        assert int(seq1_end) > int(seq1_start)
        assert int(seq2_end) > int(seq2_start)
        seq1_mid = int(seq1_start) + half_dist
        seq2_mid = int(seq2_start) + half_dist
        data.setdefault((seq1_name, seq2_name), []).append((seq1_mid, seq2_mid, line))

    # If there are chromosome pairs with segment count <= chunk_size
    # then no need to sort and split these pairs into separate files.
    # It is better to keep these pairs in a single segment file.
    skip_pairs = []  # pairs that have count <= chunk_size.
    # these will not be sorted
    if len(data.keys()) > 1:
        for pair in data.keys():
            if len(data[pair]) <= chunk_size:
                skip_pairs.append(pair)

    # sorting for forward segments
    if direction == "r":
        for pair in data.keys():
            if pair not in skip_pairs:
                data[pair] = sorted(
                    data[pair], key=lambda coord: (coord[1] - coord[0], coord[0])
                )
    # sorting for reverse segments
    elif direction == "f":
        for pair in data.keys():
            if pair not in skip_pairs:
                data[pair] = sorted(
                    data[pair], key=lambda coord: (coord[1] + coord[0], coord[0])
                )
    else:
        print(f"INVALID DIRECTION VALUE: {direction}")
        exit(0)

    # Writing file in chunks
    ctr = 0
    for pair in data.keys() - skip_pairs:
        for chunk in chunks(list(zip(*data[pair]))[2], chunk_size):
            ctr += 1
            name_addition = f".split{ctr}"
            fname = input_file.split(".segments", 1)[0] + name_addition + ".segments"

            with open(fname, "w") as f:
                f.writelines(chunk)
            # update segment file in command
            params[segment_index] = segment_key + fname
            # update output file in command
            params[output_index] = (
                output_key
                + output_alignment_file_base
                + name_addition
                + "."
                + output_format
            )
            # update error file in command
            params[-1] = err_name_base + name_addition + ".err"
            print(" ".join(params), flush=True)

    # writing unsorted skipped pairs
    if len(skip_pairs) > 0:
        skip_pairs_with_len = sorted(
            [(len(data[p]), p) for p in skip_pairs]
        )  # list of tuples of (pair length, pair)
        aggregated_skip_pairs = []  # list of list of pair names
        current_count = 0
        aggregated_skip_pairs.append([])
        for count, pair in skip_pairs_with_len:
            if current_count + count <= chunk_size:
                current_count += count
                aggregated_skip_pairs[-1].append(pair)
            else:
                aggregated_skip_pairs.append([])
                current_count = count
                aggregated_skip_pairs[-1].append(pair)

        for aggregate in aggregated_skip_pairs:
            ctr += 1
            name_addition = f".split{ctr}"
            fname = input_file.split(".segments", 1)[0] + name_addition + ".segments"
            with open(fname, "w") as f:
                for pair in sorted(aggregate, key=lambda p: (p[1], p[0])):
                    chunk = list(zip(*data[pair]))[2]
                    f.writelines(chunk)
            # update segment file in command
            params[segment_index] = segment_key + fname
            # update output file in command
            params[output_index] = (
                output_key
                + output_alignment_file_base
                + name_addition
                + "."
                + output_format
            )
            # update error file in command
            params[-1] = err_name_base + name_addition + ".err"
            print(" ".join(params), flush=True)

    if DELETE_AFTER_CHUNKING:
        os.remove(input_file)
