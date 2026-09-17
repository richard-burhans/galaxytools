"""Minimal .2bit writer + Growler test-data generator.

⛔ The test data must actually contain a MINUS-strand alignment, or the test that asserts on
both strands is decoration. So this generator writes the pair, then the caller runs real lastz
against it and checks. Nothing here is trusted on inspection.
"""
import struct, gzip, random, pathlib, sys

COMP = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(COMP)[::-1]

def pack_dna(seq):
    vals = {"T":0,"C":1,"A":2,"G":3}
    out = bytearray(); acc = 0; n = 0
    for ch in seq.upper():
        acc = (acc << 2) | vals.get(ch, 0); n += 1
        if n == 4: out.append(acc); acc = 0; n = 0
    if n: out.append(acc << (2*(4-n)))
    return bytes(out)

def write_2bit(path, seqs):
    names = list(seqs)
    index = b"".join(struct.pack("B", len(n)) + n.encode() + struct.pack("<I", 0) for n in names)
    header_len = 16 + len(index)
    offsets, cur = {}, header_len
    blobs = {}
    for n in names:
        s = seqs[n]
        blob = struct.pack("<I", len(s)) + struct.pack("<I", 0) + struct.pack("<I", 0) + struct.pack("<I", 0) + pack_dna(s)
        blobs[n] = blob; offsets[n] = cur; cur += len(blob)
    with open(path, "wb") as fh:
        # signature, version, sequenceCount, reserved -- FOUR uint32s. Packing version and
        # count as uint16 gives "bad 2bit version (00010000)": the fields shift by two bytes.
        fh.write(struct.pack("<IIII", 0x1A412743, 0, len(names), 0))
        for n in names:
            fh.write(struct.pack("B", len(n)) + n.encode() + struct.pack("<I", offsets[n]))
        for n in names:
            fh.write(blobs[n])

rnd = random.Random(20260917)
bases = "ACGT"
filler = lambda k: "".join(rnd.choice(bases) for _ in range(k))

shared_plus  = filler(150)
shared_minus = filler(150)
target = shared_plus + filler(50) + shared_minus + filler(50)          # 400 bp
query  = shared_plus + filler(50) + rc(shared_minus) + filler(50)      # 400 bp

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
write_2bit(out/"target.2bit", {"target1": target})
write_2bit(out/"query.2bit",  {"query1": query})

# plus anchor: target 1..150  <-> query 1..150
# minus anchor: target 201..350. On the reverse strand the query's matching region 201..350
# (1-based forward) counts as 400-350+1 .. 400-201+1 = 51..200.
lines = [
    "target1\t1\t150\tquery1\t1\t150\t+\t3000",
    "target1\t201\t350\tquery1\t51\t200\t-\t3000",
]
with gzip.open(out/"keg_two_strand.segments.gz", "wb", compresslevel=6) as fh:
    fh.write(("\n".join(lines) + "\n").encode())
print("wrote target.2bit query.2bit keg_two_strand.segments.gz")
print("target len", len(target), "query len", len(query))
