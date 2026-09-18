# Growler

**One genome pair in, a pair-keyed alignment collection out.**

*On the name:* **Growler** is just the system's name. It carries no relationship to *pair file* beyond
both being words — in particular it implies nothing about relative size, and there is no
container metaphor to keep consistent. A **pair file** is the unit: one chromosome pair's segments.

KegAlign emits a collection of **pair files** — one gzipped segments file per `(target, query)`
chromosome pair, both strands — plus the two 2bits as their own datasets. `growler_lastz` is
mapped over that collection. No tarball, no `commands.json`, no `format.txt`, no collapse.

```
target.fa ─┐
           ├─▶ KegAlign ──┬─▶ collection of pairs ─────┐
query.fa  ─┘  (GPU)       ├─▶ target.2bit          ├─▶ growler_lastz  ──▶ collection of
scores ───────────────────┴─▶ query.2bit  ─────────┘   (mapped, N jobs)    alignments
```

## Why a subworkflow

Mapped over a genome-pair collection this emits `list:list` — outer = genome pair, inner =
chromosome pair — with outer identifiers preserved. Measured on usegalaxy.org 26.1 on
2026-09-17, in this exact shape (two parallel lists dot-producted into two `data` inputs of a
subworkflow that emits a collection):

```
collection_type: list:list | outer: 2
  'p1': inner list x2 -> [('p1','p1'), ('p1q','p1q')]
  'p2': inner list x2 -> [('p2','p2'), ('p2q','p2q')]
```

That matters because **Galaxy does not broadcast a dataset across nesting levels**. brc-tools works
around it by materialising parallel `list:list` collections with matching identifiers. Inside one
Growler invocation there is only *one* level — a single genome pair, two 2bits as plain `data`
inputs, a flat collection of pair files — so there is nothing to broadcast. The subworkflow does not
solve the nesting problem; it removes it.

## Why every mapped job runs the same command line

A real bundle's 5,177 lastz commands carry only **8 distinct argument signatures** (strand × two
target blocks × two query blocks); `--output`/`--segments` are the only per-command values. Two
measurements collapse those 8 to 1:

| | result |
|---|---|
| drop `subset=`, use the whole 2bit | byte-identical output; costs +888 MB peak RSS, +25% time |
| one pair file carrying both strands, no `--strand` | byte-identical to the two single-strand runs concatenated — 970 blocks = 542 + 428, same order |

So nothing per-element has to be routed, and **no element identifier has to reach a text
parameter** — the limitation recorded in brc-tools' `multiz_fold.xml` as *"how every fold in the
2026-06-13 runs died on the name validator."*

## Why the pair files are gzipped

Segments are 97.1% of a bundle, and a bundle compresses ~3.6× (14.58 GB → 4.10 GB, measured).
Shipping pair files as plain datasets would roughly triple the object store — ~300 GB against ~1.06 TB
across a 90-pair panel.

⚠ But **lastz cannot read a gzipped segments file**, and it does not say so loudly:
`--segments=x.gz` reports `FAILURE: bad field (x.gz: line 1, …)` *and still writes an empty output
file*. `growler_lastz` therefore decompresses its own pair file — ~140 MB per job, against the tarball
path inflating the whole 14.58 GB into the job directory before running anything.

## Where it stops

Growler is every step that stays a collection. Downstream, `axtChain` and `chainSort` map per
chromosome pair too, and multiz runs per hinge chromosome — but `chainPreNet` and `chainNet` each
walk one score-ordered chain stream marking **target and query simultaneously**
(`chainUsed(chain, qChrom, tChrom)`; `addChainQ` then `addChainT`), so a chain on `(tchr1, qchr5)`
consumes query space a chain on `(tchr3, qchr5)` would otherwise claim.

**No chromosome partition of either tool is correct.** Both `errAbort` on unsorted input, so a
wrong split fails loudly. Merge with `chainMergeSort` (`-inputList=…`, a k-way merge over the
collection) — and ⛔ **never `-saveId`**: independent `axtChain` runs each number their chains from
1, `netChainSubset` matches nets to chains by id, and the default renumbering is what repairs the
collision.

## Status

⬜ Not run end to end. `growler_lastz` is not installed on any server yet, and the KegAlign
`collection` output needs a release before it can be. The workflow imports cleanly and its
connections bind as intended (validated against usegalaxy.org 26.1); the only import error is
`growler_lastz` reporting "Tool is not installed", which is expected until it ships.

## Where the tools live

All three live in `tools/kegalign/` as one toolshed repository: KegAlign (GPU), Batched LASTZ
(gapped extension over a tarball) and Growler LASTZ (gapped extension over one chromosome pair).

They were separate directories, which cost two things. `alignment_type_option.xml` had to be a
**symlink** from `batched_lastz/` into `kegalign/`, and editing it changed the rendered XML of both
tools while `git diff --name-only` reported one path -- which is what CI derives its
changed-repository list from, so only one tool got linted (2026-09-17). And the `lastz` version pin
existed **twice**, once per directory, with nothing asserting the two agreed.

One directory removes both. The symlink is now a plain file, and `macros.xml` carries one
`@LASTZ_VERSION@` shared by the two LASTZ tools, alongside `@KEGALIGN_VERSION@` for the GPU tool --
separate tokens because `lastz` and `kegalign-full` release on separate cadences, and each tool
carries its own suffix so a change to one bumps only that one.

⚠ **Merging to `main` publishes.** `pr.yaml`'s `deploy` job runs `planemo shed_update` on every
push to `main` in this owner's repository, so a merge is a release. That is why the tool suffix has
to be bumped in the same PR as any change to a tool's rendered XML: the previous suffix is already
the latest installable revision by the time the next PR is linted.

⚠ **This changes Batched LASTZ's full tool id** from
`.../repos/richard-burhans/batched_lastz/batched_lastz/...` to
`.../repos/richard-burhans/kegalign/batched_lastz/...`. TPV entries and any workflow referencing the
old id need updating; brc-tools' WF-C is one.

See `galaxytools#149` for the full design record and the measurements behind each claim.
