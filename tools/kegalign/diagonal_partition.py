#!/usr/bin/env python


"""
Diagonal partitioning for segment files output by KegAlign.

Usage:
diagonal_partition.py <max-segments> <lastz-command>

set <max-segments> = 0 to skip partitioning, -1 to estimate best parameter
"""

import collections
import os
import statistics
import sys
import typing


def chunks(lst: tuple[str, ...], n: int) -> typing.Iterator[tuple[str, ...]]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    # TODO: make these optional user defined parameters

    # deletes original segment file after splitting
    DELETE_AFTER_CHUNKING = True

    # don't partition segment files with line count below this value
    MIN_CHUNK_SIZE = 5000

    # only used when segment size is being estimated
    MAX_CHUNK_SIZE = 50000

    # include chosen split size in file name
    DEBUG = False

    # first parameter contains chunk size
    chunk_size = int(sys.argv[1])
    params = sys.argv[2:]

    # don't do anything if 0 chunk size
    if chunk_size == 0:
        print(" ".join(params), flush=True)
        sys.exit(0)

    # Parsing command output from KegAlign
    segment_key = "--segments="
    segment_index = None
    input_file = None

    for index, value in enumerate(params):
        if value[:len(segment_key)] == segment_key:
            segment_index = index
            input_file = value[len(segment_key):]
            break

    if segment_index is None:
        sys.exit(f"Error: could not get segment key {segment_key} from parameters {params}")

    if input_file is None:
        sys.exit(f"Error: could not get segment file from parameters {params}")

    if not os.path.isfile(input_file):
        sys.exit(f"Error: File {input_file} does not exist")

    # each char is 1 byte
    line_size = None
    file_size = os.path.getsize(input_file)
    with open(input_file, "r") as f:
        # add 1 for newline
        line_size = len(f.readline())

    estimated_lines = file_size // line_size

    # check if chunk size should be estimated
    if chunk_size < 0:
        # optimization, do not need to get each file size in this case
        if estimated_lines < MIN_CHUNK_SIZE:
            print(" ".join(params), flush=True)
            sys.exit(0)

        # get size of each segment assuming DELETE_AFTER_CHUNKING == True
        # takes into account already split segments
        files = [i for i in os.listdir(".") if i.endswith(".segments")]

        if len(files) < 2:
            # if not enough segment files for estimation, use MAX_CHUNK_SIZE
            chunk_size = MAX_CHUNK_SIZE
        else:
            fdict: typing.DefaultDict[str, int] = collections.defaultdict(int)
            for filename in files:
                size = os.path.getsize(filename)
                f_ = filename.split(".split", 1)[0]
                fdict[f_] += size

            if len(fdict) < 7:
                # outliers can heavily skew prediction if <7 data points
                # to be safe, use 50% quantile
                chunk_size = int(statistics.quantiles(fdict.values())[1] // line_size)
            else:
                # otherwise use 75% quantile
                chunk_size = int(statistics.quantiles(fdict.values())[-1] // line_size)
            # if not enough data points, there is a chance of getting unlucky
            # minimize worst case by using MAX_CHUNK_SIZE

            chunk_size = min(chunk_size, MAX_CHUNK_SIZE)

    # no need to sort if number of lines <= chunk_size
    if (estimated_lines <= chunk_size):
        print(" ".join(params), flush=True)
        sys.exit(0)

    # Find rest of relevant parameters
    output_key = "--output="
    output_index = None
    output_alignment_file = None
    output_alignment_file_base = None
    output_format = None

    strand_key = "--strand="
    strand_index = None
    for index, value in enumerate(params):
        if value[:len(output_key)] == output_key:
            output_index = index
            output_alignment_file = value[len(output_key):]
            output_alignment_file_base, output_format = output_alignment_file.rsplit(".", 1)

        if value[:len(strand_key)] == strand_key:
            strand_index = index

    if output_alignment_file_base is None:
        sys.exit(f"Error: could not get output alignment file base from parameters {params}")

    if output_format is None:
        sys.exit(f"Error: could not get output format from parameters {params}")

    if output_index is None:
        sys.exit(f"Error: could not get output key {output_key} from parameters {params}")

    if strand_index is None:
        sys.exit(f"Error: could not get strand key {strand_key} from parameters {params}")

    # error file is at very end
    err_index = -1
    err_name_base = params[-1].split(".err", 1)[0]

    # dict of list of tuple (x, y, str)
    data: dict[tuple[str, str], list[tuple[int, int, str]]] = {}

    direction = None
    if "plus" in params[strand_index]:
        direction = "f"
    elif "minus" in params[strand_index]:
        direction = "r"
    else:
        sys.exit(f"Error: could not figure out direction from strand value {params[strand_index]}")

    for line in open(input_file, "r"):
        if line == "":
            continue
        seq1_name, seq1_start, seq1_end, seq2_name, seq2_start, seq2_end, _dir, score = line.split()
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

    # pairs that have count <= chunk_size. these will not be sorted
    skip_pairs = []

    # save query key order
    # for lastz segment files: 'Query sequence names must appear in the same
    # order as they do in the query file'

    # NOTE: assuming data.keys() preserves order of keys. Requires Python 3.7+

    query_key_order = list(dict.fromkeys([i[1] for i in data.keys()]))

    if len(data.keys()) > 1:
        for pair in data.keys():
            if len(data[pair]) <= chunk_size:
                skip_pairs.append(pair)

    # sorting for forward segments
    if direction == "r":
        for pair in data.keys():
            if pair not in skip_pairs:
                data[pair] = sorted(data[pair], key=lambda coord: (coord[1] - coord[0], coord[0]))

    # sorting for reverse segments
    elif direction == "f":
        for pair in data.keys():
            if pair not in skip_pairs:
                data[pair] = sorted(data[pair], key=lambda coord: (coord[1] + coord[0], coord[0]))
    else:
        sys.exit(f"INVALID DIRECTION VALUE: {direction}")

    # Writing file in chunks
    ctr = 0
    # [i for i in data_keys if i not in set(skip_pairs)]:
    for pair in (data.keys() - skip_pairs):
        for chunk in chunks(list(zip(*data[pair]))[2], chunk_size):
            ctr += 1
            name_addition = f".split{ctr}"

            if DEBUG:
                name_addition = f".{chunk_size}{name_addition}"

            fname = input_file.split(".segments", 1)[0] + name_addition + ".segments"

            assert len(chunk) != 0
            with open(fname, "w") as f:
                f.writelines(chunk)
            # update segment file in command
            params[segment_index] = segment_key + fname
            # update output file in command
            params[output_index] = output_key + output_alignment_file_base + name_addition + "." + output_format
            # update error file in command
            params[-1] = err_name_base + name_addition + ".err"
            print(" ".join(params), flush=True)

    # writing unsorted skipped pairs
    if len(skip_pairs) > 0:
        # list of tuples of (pair length, pair)
        skip_pairs_with_len = sorted([(len(data[p]), p) for p in skip_pairs])
        # NOTE: This sorting can violate lastz query key order requirement, this is fixed later

        # used for sorting
        query_key_order_table = {item: idx for idx, item in enumerate(query_key_order)}

        # list of list of pair names
        aggregated_skip_pairs: list[list[tuple[str, str]]] = []
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

            if DEBUG:
                name_addition = f".{chunk_size}{name_addition}"

            fname = input_file.split(".segments", 1)[0] + name_addition + ".segments"

            with open(fname, "w") as f:
                # fix possible lastz query key order violations
                # p[1] is query key
                for pair in sorted(aggregate, key=lambda p: query_key_order_table[p[1]]):
                    chunk = list(zip(*data[pair]))[2]
                    f.writelines(chunk)
            # update segment file in command
            params[segment_index] = segment_key + fname
            # update output file in command
            params[output_index] = output_key + output_alignment_file_base + name_addition + "." + output_format
            # update error file in command
            params[-1] = err_name_base + name_addition + ".err"
            print(" ".join(params), flush=True)

    if DELETE_AFTER_CHUNKING:
        os.remove(input_file)
