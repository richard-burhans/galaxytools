#!/usr/bin/env python3
"""Verify the tools COPIED from nekrut/brc-tools have not drifted from what was copied.

    check_brc_provenance.py              # non-zero if any copied file changed or appeared
    check_brc_provenance.py --write      # re-stamp after a deliberate re-copy
    check_brc_provenance.py --self-test

⛔ THESE TOOLS ARE COPIES, AND A COPY WITHOUT A LINK BECOMES A FORK QUIETLY. nekrut/brc-tools is
the source of truth: a change belongs there first and is re-copied here. Nothing in git enforces
that -- an edit made here looks exactly like an edit made upstream and copied down -- so this
re-hashes every copied file against the manifest written when it was copied.

⚠ THE PRECEDENT IS CONCRETE. KegAlign's helper scripts exist in FOUR drifted copies across
repositories, and the only way to tell which one a job actually runs is to read the wrapper's
`$KEGALIGN_BIN` vs `$__tool_directory__`. That is the end state this check exists to prevent, and
by then the question "which is right" has no answer anyone can reconstruct.

⚠ AN ADDED FILE IS DRIFT TOO, not just a changed one. A new file inside a copied directory is a
local invention with no upstream counterpart, which is how a fork starts; it is reported by name.
A file the manifest lists and the tree lacks is reported the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

MANIFEST = pathlib.Path("tools/.brc-tools-provenance.tsv")
UPSTREAM = "https://github.com/nekrut/brc-tools"

#: The directories copied wholesale. Anything under them is covered by the manifest.
COPIED_DIRS = ["ucsc_kent", "build_genomes_txt", "build_hub_bb", "chain_to_bigChain",
               "maf_to_bigmaf_bed", "process_maf"]

#: ⛔ DELIBERATELY LOCAL, AND NAMED SO ITS ABSENCE IS A DECISION RATHER THAN AN OVERSIGHT.
#: `.shed.yml` carries the PUBLISHING IDENTITY -- `owner` and `remote_repository_url`. Upstream it
#: says `owner: nekrut` and points at brc-tools; here it must say `richard-burhans` and point at
#: galaxytools, or a shed publish from this repo would target the wrong namespace. So these files
#: diverge BY DESIGN on every copy, and hashing them would make the check cry wolf forever until
#: someone stopped reading it.
#: ⚠ `homepage_url` is NOT retargeted: it names where the project lives, which is still brc-tools.
LOCAL_FILES = {".shed.yml"}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def on_disk(root: pathlib.Path) -> dict[str, str]:
    """`{path: sha256}` for every file under the copied directories."""
    out = {}
    for d in COPIED_DIRS:
        for f in sorted((root / "tools" / d).rglob("*")):
            if f.is_file() and f.name not in LOCAL_FILES:
                out[str(f.relative_to(root))] = digest(f)
    return out


def read_manifest(path: pathlib.Path) -> tuple[dict[str, str], str]:
    """`({path: sha256}, upstream_commit)`."""
    rows, commit = {}, ""
    for line in path.read_text().splitlines():
        if line.startswith("# upstream:"):
            commit = line.rsplit("@", 1)[-1].strip()
        if not line or line.startswith("#") or line.startswith("path\t"):
            continue
        p, h = line.split("\t", 1)
        rows[p] = h.strip()
    return rows, commit


def compare(manifest: dict[str, str], disk: dict[str, str]) -> dict[str, list[str]]:
    """`{"changed": [...], "added": [...], "missing": [...]}`. Pure, so it is testable.

    ⚠ Three kinds, reported separately, because the remedy differs: a CHANGED file was edited
    here instead of upstream; an ADDED one is a local invention; a MISSING one means the manifest
    is describing a tree that no longer exists.
    """
    return {
        "changed": sorted(p for p in manifest if p in disk and disk[p] != manifest[p]),
        "added": sorted(p for p in disk if p not in manifest),
        "missing": sorted(p for p in manifest if p not in disk),
    }


def _self_test() -> int:
    bad = 0
    cases = [
        ("identical", {"a": "1", "b": "2"}, {"a": "1", "b": "2"},
         {"changed": [], "added": [], "missing": []}),
        ("one edited here", {"a": "1"}, {"a": "9"},
         {"changed": ["a"], "added": [], "missing": []}),
        ("a local invention", {"a": "1"}, {"a": "1", "new": "7"},
         {"changed": [], "added": ["new"], "missing": []}),
        ("manifest describes a vanished file", {"a": "1", "gone": "3"}, {"a": "1"},
         {"changed": [], "added": [], "missing": ["gone"]}),
        ("empty both ways is clean", {}, {}, {"changed": [], "added": [], "missing": []}),
    ]
    # ⚠ The local-file set must be non-empty and must hold .shed.yml, or the check would flag the
    # publishing identity on every single copy and teach everyone to ignore it.
    if ".shed.yml" not in LOCAL_FILES:
        print("not ok - .shed.yml must be exempt; it diverges by design")
        bad += 1
    else:
        print("ok - .shed.yml is exempt, so the publishing identity may differ")
    for name, man, disk, want in cases:
        got = compare(man, disk)
        ok = got == want
        bad += not ok
        print(f"{'ok' if ok else 'not ok'} - {name}: {got}")
    # ⛔ MUTATION: an "added" file must not be silently tolerated. If the added-detection were
    # dropped, a local invention would read as clean, which is precisely how a fork begins.
    if compare({"a": "1"}, {"a": "1", "new": "7"})["added"] != ["new"]:
        print("not ok - mutation: an added file was not reported")
        bad += 1
    else:
        print("ok - mutation: an added file is reported, not tolerated")
    print(f"# {len(cases) + 2} checks, {bad} failed")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="re-stamp the manifest after a DELIBERATE re-copy from upstream")
    ap.add_argument("--commit", default="",
                    help="with --write: the brc-tools commit copied from")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()

    root = pathlib.Path.cwd()
    disk = on_disk(root)
    if a.write:
        prev_commit = read_manifest(MANIFEST)[1] if MANIFEST.exists() else ""
        commit = a.commit or prev_commit
        if not commit:
            ap.error("--write needs --commit the first time, so the stamp names a real upstream")
        head = [
            "# GENERATED by scripts/check_brc_provenance.py --write. Do not hand-edit.",
            "#",
            "# These tools are COPIES. nekrut/brc-tools is the source of truth; a change belongs",
            "# there first and is re-copied here. This file records the upstream commit and each",
            "# file's sha256, so divergence is a failing check rather than a later discovery.",
            f"# upstream: {UPSTREAM} @ {commit}",
            "path\tsha256",
        ]
        MANIFEST.write_text("\n".join(head + [f"{p}\t{h}" for p, h in sorted(disk.items())]) + "\n")
        print(f"stamped {len(disk)} file(s) against {commit}")
        return 0

    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}; run --write --commit <sha>")
        return 1
    manifest, commit = read_manifest(MANIFEST)
    diff = compare(manifest, disk)
    print(f"{len(manifest)} copied file(s), stamped against {UPSTREAM} @ {commit or '?'}")
    for kind, paths in diff.items():
        for p in paths:
            print(f"  {kind.upper():8s} {p}")
    total = sum(len(v) for v in diff.values())
    if total:
        print(f"  -- {total} divergence(s). A change belongs in brc-tools first; re-copy, then "
              f"`--write --commit <sha>`.")
        return 1
    print("  -- no divergence from what was copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
