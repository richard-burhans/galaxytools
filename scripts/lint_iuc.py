#!/usr/bin/env python3
"""Lint a Galaxy tool XML against the IUC rules planemo does NOT check.

    lint_iuc.py tools/ucsc_fatotwobit/fatotwobit.xml
    lint_iuc.py tools/            # every .xml beneath
    lint_iuc.py --self-test

⛔ THIS IS NOT A PLANEMO REPLACEMENT. planemo lint checks the schema, element order and the
obviously-missing. It passes a tool that every IUC reviewer will send back, because the review
standards are prose in a skill and prose is not run. The `tool-dev` skill carries a table titled
"Will Definitely Be Flagged" -- eighteen rows, derived from 25 inline comments across 3 reviewers
on one submission. Most are decidable from the XML. This decides them.

⚠ A clean run here plus a clean planemo lint is still not a review. Judgement calls -- is the help
useful, are the tests meaningful, is the parameter set the right shape -- are not here and cannot be.

The rules, each one a review comment someone actually received:

  * Cheetah inside an `<xml>` macro DOES NOT WORK. `#if`/`#for` belong in a `<token>`.
  * `detect_errors` must be set, and `aggressive` is the IUC default.
  * A version must come from `@TOOL_VERSION@`, never be hardcoded.
  * `profile` must be recent; an old profile silently opts out of newer Galaxy behaviour.
  * `display="checkboxes"` is always flagged.
  * Units in labels are SI-lowercase: `kb`/`Mb`, never `KB`/`MB`.
  * Every `<test>` needs `expect_num_outputs`.
  * `optional="true"` next to a `value=` is contradictory -- pick one.
  * A param that never appears in `<command>` is orphaned.
  * Tests address sections by NESTING, never `section|param` pipe syntax.
  * A boolean puts its flag in `truevalue`, not in an `#if` in the command.
  * Cheetah booleans are the STRINGS "true"/"false"; `#if $flag` is unreliable.
  * A numeric `0` is falsy, so `#if $n` silently drops a valid zero -- use `str($n)`.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

CHEETAH = re.compile(r"^\s*#(if|for|end|else|elif)\b", re.M)
PIPE_PARAM = re.compile(r'name="[^"]+\|[^"]+"')
MIN_PROFILE = "23.0"


def text_of(el) -> str:
    return "".join(el.itertext()) if el is not None else ""


def lint(path: pathlib.Path) -> list[str]:
    out: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        # ⚠ `--` inside an XML comment is the house em-dash style and is FATAL here.
        return [f"{path.name}: XML does not parse ({e}); a `--` inside a comment is the usual cause"]
    if root.tag != "tool":
        return []
    name = path.name

    prof = root.get("profile", "")
    if not prof:
        out.append(f"{name}: no `profile` on <tool>")
    elif prof < MIN_PROFILE:
        out.append(f"{name}: profile {prof} is older than {MIN_PROFILE}; an old profile opts out "
                   f"of newer Galaxy behaviour silently")
    ver = root.get("version", "")
    if "@TOOL_VERSION@" not in ver:
        out.append(f"{name}: version `{ver}` is hardcoded; use @TOOL_VERSION@+galaxy@VERSION_SUFFIX@")

    cmd = root.find("command")
    if cmd is None:
        out.append(f"{name}: no <command>")
        cmd_text = ""
    else:
        cmd_text = text_of(cmd)
        de = cmd.get("detect_errors")
        if not de:
            out.append(f"{name}: <command> has no `detect_errors`; IUC expects \"aggressive\"")
        elif de != "aggressive":
            out.append(f"{name}: detect_errors=\"{de}\"; IUC expects \"aggressive\"")

    if root.find("xrefs") is None:
        out.append(f"{name}: no <xrefs>; add a bio.tools cross-reference")
    if root.find("help") is None:
        out.append(f"{name}: no <help>")
    if root.find("citations") is None and root.find("./macros") is None:
        out.append(f"{name}: no <citations>")

    # ⛔ Cheetah in an <xml> macro silently does nothing. The single most common review comment.
    for macros in root.findall("macros"):
        for x in macros.findall("xml"):
            if CHEETAH.search(text_of(x)):
                out.append(f"{name}: macro <xml name=\"{x.get('name')}\"> contains Cheetah "
                           f"(#if/#for); it MUST be a <token> or it will not work")

    params = root.findall(".//inputs//param")
    for p in params:
        pname = p.get("name") or (p.get("argument") or "").lstrip("-").replace("-", "_")
        label = p.get("label", "")
        if p.get("display") == "checkboxes":
            out.append(f"{name}: param `{pname}` uses display=\"checkboxes\"; always flagged")
        if re.search(r"\b\d+\s?[KMG]B\b", label):
            out.append(f"{name}: param `{pname}` label uses uppercase units "
                       f"({label.strip()[:40]}); IUC wants SI lowercase kb/Mb")
        if p.get("optional") == "true" and p.get("value") not in (None, ""):
            out.append(f"{name}: param `{pname}` is optional=\"true\" AND has value="
                       f"\"{p.get('value')}\"; pick one")
        if p.get("type") == "boolean" and not p.get("truevalue") and cmd_text:
            if re.search(rf"#if\s+\$?[\w.]*\b{re.escape(pname)}\b", cmd_text):
                out.append(f"{name}: boolean `{pname}` drives an #if; put the flag in "
                           f"truevalue/falsevalue instead")
        if pname and cmd_text and not re.search(rf"\${{?[\w.]*\b{re.escape(pname)}\b", cmd_text):
            out.append(f"{name}: param `{pname}` never appears in <command> (orphaned)")

    # Cheetah truthiness traps, both of which pass tests and fail on real values
    for m in re.finditer(r"#if\s+\$([\w.]+)\s*$", cmd_text, re.M):
        ref = m.group(1).rsplit(".", 1)[-1]
        el = next((p for p in params if (p.get("name") or "") == ref), None)
        if el is None:
            continue
        if el.get("type") == "boolean":
            out.append(f"{name}: `#if $...{ref}` tests a boolean as truthy; Galaxy renders them "
                       f"as the STRINGS \"true\"/\"false\" — use str() == \"true\"")
        if el.get("type") in ("integer", "float"):
            out.append(f"{name}: `#if $...{ref}` drops a valid 0, which is falsy; use str({ref})")

    tests = root.findall(".//tests/test")
    if not tests:
        out.append(f"{name}: no <test>")
    for i, t in enumerate(tests, 1):
        if t.get("expect_num_outputs") is None:
            out.append(f"{name}: test {i} has no `expect_num_outputs`")
    if PIPE_PARAM.search(path.read_text()):
        out.append(f"{name}: a test addresses a param with `section|param` pipe syntax; "
                   f"IUC wants explicit nesting")
    return out


def self_test() -> int:
    """⚠ Every rule is injected into a minimal tool and must be caught."""
    import tempfile
    GOOD = '''<tool id="t" name="T" version="@TOOL_VERSION@+galaxy@VERSION_SUFFIX@" profile="25.0">
    <description>d</description>
    <macros><token name="@TOOL_VERSION@">1</token></macros>
    <xrefs><xref type="bio.tools">x</xref></xrefs>
    <command detect_errors="aggressive"><![CDATA[ t --flag '$inp' --n $num ]]></command>
    <inputs>
        <param name="inp" type="data" format="txt" label="in"/>
        <param name="num" type="integer" value="1" label="n"/>
    </inputs>
    <outputs><data name="o" format="txt"/></outputs>
    <tests><test expect_num_outputs="1"><param name="inp" value="a"/></test></tests>
    <help>h</help>
    <citations><citation type="doi">10.1/x</citation></citations>
</tool>'''
    CASES = [
        ("a clean tool passes", GOOD, None),
        ("an old profile is caught", GOOD.replace('profile="25.0"', 'profile="21.01"'), "older than"),
        ("a hardcoded version is caught",
         GOOD.replace('version="@TOOL_VERSION@+galaxy@VERSION_SUFFIX@"', 'version="1.0"'),
         "hardcoded"),
        ("a missing detect_errors is caught",
         GOOD.replace(' detect_errors="aggressive"', ""), "no `detect_errors`"),
        ("detect_errors=exit_code is caught",
         GOOD.replace('"aggressive"', '"exit_code"'), "IUC expects"),
        ("a missing xref is caught",
         GOOD.replace('<xrefs><xref type="bio.tools">x</xref></xrefs>', ""), "no <xrefs>"),
        ("display=checkboxes is caught",
         GOOD.replace('name="inp" type="data"', 'name="inp" display="checkboxes" type="data"'),
         "checkboxes"),
        ("uppercase units are caught",
         GOOD.replace('label="n"', 'label="Block size 1MB"'), "SI lowercase"),
        ("optional with a value is caught",
         GOOD.replace('name="num" type="integer" value="1"',
                      'name="num" type="integer" optional="true" value="1"'), "pick one"),
        ("an orphaned param is caught",
         GOOD.replace('<param name="num" type="integer" value="1" label="n"/>',
                      '<param name="num" type="integer" value="1" label="n"/>'
                      '<param name="ghost" type="text" value="" label="g"/>'), "orphaned"),
        ("a test with no expect_num_outputs is caught",
         GOOD.replace(' expect_num_outputs="1"', ""), "expect_num_outputs"),
        ("pipe syntax in a test is caught",
         GOOD.replace('<param name="inp" value="a"/>', '<param name="sec|inp" value="a"/>'),
         "pipe syntax"),
        ("Cheetah in an <xml> macro is caught",
         GOOD.replace('<macros><token name="@TOOL_VERSION@">1</token></macros>',
                      '<macros><token name="@TOOL_VERSION@">1</token>'
                      '<xml name="m">#if $x\n--a\n#end if</xml></macros>'),
         "MUST be a <token>"),
        ("a numeric tested for truthiness is caught",
         GOOD.replace("--n $num", "\n#if $num\n--n $num\n#end if\n"), "drops a valid 0"),
    ]
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, xml, expect in CASES:
            p = pathlib.Path(tmp) / "t.xml"
            p.write_text(xml)
            found = lint(p)
            hit = (not found) if expect is None else any(expect in f for f in found)
            print(f"{'ok' if hit else 'not ok'} - {label}")
            if not hit:
                print(f"     findings: {found}")
                fails.append(label)
    print("\nall tests passed" if not fails else f"\n{len(fails)} check(s) FAILED")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="tool XML files or directories")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.paths:
        ap.error("give a path, or --self-test")
    files: list[pathlib.Path] = []
    for raw in a.paths:
        p = pathlib.Path(raw)
        files += sorted(p.rglob("*.xml")) if p.is_dir() else [p]
    bad = 0
    for f in files:
        for msg in lint(f):
            print(f"  ⛔ {msg}")
            bad += 1
    print(f"lint_iuc: {len(files)} file(s), {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
