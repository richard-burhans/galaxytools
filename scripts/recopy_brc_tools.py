#!/usr/bin/env python3
"""Re-copy the hub tools from a brc-tools checkout, re-applying only the local overrides.

    recopy_brc_tools.py --from /path/to/brc-tools --commit <sha>
    recopy_brc_tools.py --self-test

⛔ WHY THIS IS A SCRIPT AND NOT A PROCEDURE. The first re-copy was done by hand and PRESERVED
`.shed.yml` wholesale, on the grounds that it is "deliberately local". That is true of exactly two
FIELDS in it -- `owner` and `remote_repository_url` -- and false of the rest. brc-tools#113 then
fixed an illegal toolshed repository name and an invalid category, both of which live in
`.shed.yml`, and preserving the file would have kept both bugs here while the upstream fix sat
one directory away. The local part is a pair of fields, so only that pair is re-applied.

⚠ AND IT ASSERTS, rather than hoping: after copying it checks that each override took, and that
every other line of `.shed.yml` matches upstream. A silent miss here means the shed publish goes
to the wrong namespace, or a known-bad field survives a fix.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

DIRS = ["ucsc_kent", "build_genomes_txt", "build_hub_bb", "chain_to_bigChain",
        "maf_to_bigmaf_bed", "process_maf"]

#: The ONLY fields that are local. Everything else in .shed.yml tracks upstream.
#: `owner` is the publishing identity; `remote_repository_url` is this file's own location.
#: ▶ `homepage_url` is deliberately NOT here: it names where the PROJECT lives, still brc-tools.
OVERRIDES = {
    "owner": "richard-burhans",
}
URL_RE = (r"^remote_repository_url: https://github\.com/nekrut/brc-tools/tree/main/tools/(\S+)$",
          r"remote_repository_url: https://github.com/richard-burhans/galaxytools/tree/main/tools/\1")


def apply_overrides(text: str) -> str:
    """Re-apply the local fields to an upstream `.shed.yml`. Pure."""
    for key, value in OVERRIDES.items():
        text, n = re.subn(rf"^{key}: \S+$", f"{key}: {value}", text, flags=re.M)
        if n != 1:
            raise SystemExit(f"expected exactly one `{key}:` line, found {n}")
    text, n = re.subn(URL_RE[0], URL_RE[1], text, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected exactly one upstream remote_repository_url, found {n}")
    return text


def _self_test() -> int:
    bad = 0
    src = ("categories:\n- Sequence Analysis\n"
           "name: chain_to_bigchain\n"
           "owner: nekrut\n"
           "homepage_url: https://github.com/nekrut/brc-tools\n"
           "remote_repository_url: https://github.com/nekrut/brc-tools/tree/main/tools/x\n")
    got = apply_overrides(src)
    checks = [
        ("owner is retargeted", "owner: richard-burhans" in got),
        ("url is retargeted",
         "remote_repository_url: https://github.com/richard-burhans/galaxytools/tree/main/tools/x"
         in got),
        # ⛔ the brc-tools#113 case: a FIX that lives in .shed.yml must survive the copy
        ("an upstream name fix survives", "name: chain_to_bigchain" in got),
        ("homepage_url still credits brc-tools",
         "homepage_url: https://github.com/nekrut/brc-tools" in got),
        ("no nekrut owner remains", "owner: nekrut" not in got),
    ]
    for name, ok in checks:
        bad += not ok
        print(f"{'ok' if ok else 'not ok'} - {name}")
    # a file missing the fields must FAIL loudly, not pass through unchanged
    for broken, why in ((src.replace("owner: nekrut\n", ""), "no owner line"),
                        (src.replace("remote_repository_url", "other_url"), "no upstream url")):
        try:
            apply_overrides(broken)
            print(f"not ok - {why} should have raised")
            bad += 1
        except SystemExit:
            print(f"ok - {why} raises instead of passing through")
    print(f"# {len(checks) + 2} checks, {bad} failed")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", type=pathlib.Path,
                    help="a brc-tools checkout at the commit to copy from")
    ap.add_argument("--commit", help="the brc-tools commit, for the provenance stamp")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    if not a.src or not a.commit:
        ap.error("--from and --commit are both required")

    for d in DIRS:
        dst = pathlib.Path("tools") / d
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(a.src / "tools" / d, dst)
        shed = dst / ".shed.yml"
        shed.write_text(apply_overrides(shed.read_text()))
        # ⚠ ASSERT the rest matches upstream, so a field that is NOT local cannot drift here.
        up = (a.src / "tools" / d / ".shed.yml").read_text().splitlines()
        mine = shed.read_text().splitlines()
        differ = [i for i, (x, y) in enumerate(zip(up, mine), 1) if x != y]
        unexpected = [i for i in differ
                      if not re.match(r"^(owner|remote_repository_url):", mine[i - 1])]
        if len(up) != len(mine) or unexpected:
            raise SystemExit(f"{d}/.shed.yml differs from upstream beyond the local fields "
                             f"at line(s) {unexpected or 'length'}")
        print(f"  {d:20s} copied; .shed.yml differs only at {len(differ)} local line(s)")

    rc = subprocess.run([sys.executable, "scripts/check_brc_provenance.py",
                         "--write", "--commit", a.commit]).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
