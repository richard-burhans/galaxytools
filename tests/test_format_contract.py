"""The format.txt contract between kegalign and batched_lastz.

`package_output.py` (kegalign) writes `format.txt` into the tarball; `run_lastz_tarball.py`
(batched_lastz) reads it back and uses it to assign the Galaxy datatype of the final output. That
is a contract between two tools in two repositories, with a text file as its only interface and
nothing anywhere asserting the two ends agree.

They do not. Both ends drifted, in opposite directions, and BOTH mislabel output today -- silently,
because the reader defaults to "tabular" when it does not recognise a value. A wrongly typed dataset
does not fail; it just refuses to feed the next tool, or feeds it garbage.

These tests are written BEFORE any repair, deliberately. Two of them state the contract as it
SHOULD be (and currently fail); the characterisation test records what each copy does today, so the
repair shows up as an intentional change rather than an unexplained diff.

Run:  python -m pytest tests/ -v
      KEGALIGN_SRC=../KegAlign python -m pytest tests/ -v   # also checks the upstream copies
"""
import importlib.util
import os
import pathlib
import types
import xml.etree.ElementTree as ET

import pytest

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"
CORE = pathlib.Path(os.environ["KEGALIGN_SRC"]).resolve() / "scripts" if os.environ.get("KEGALIGN_SRC") else None


def _load(path: pathlib.Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_formats() -> list[tuple[str, str]]:
    """(Galaxy selector, LASTZ format) for everything the tool can emit.

    DERIVED, never listed here: a hand-maintained copy of this is the same failure the contract
    itself has -- two places that must agree with nothing checking that they do. A format added to
    the XML is covered from the moment it is added.

    Both halves are needed because the two copies take different inputs. The wrapper's
    package_output.py records the LASTZ format; the KegAlign copy still derives its value from the
    Galaxy selector. Supplying both lets one test drive either.
    """
    xml = ET.parse(TOOLS / "kegalign" / "output_options.xml")
    options: dict[str, list[str]] = {}
    for param in xml.iter("param"):
        name = param.get("name")
        if name:
            options[name] = [o.get("value") for o in param.iter("option") if o.get("value")]
    pairs: list[tuple[str, str]] = []
    for selector, sub in (("bam", "bam_options"), ("maf", "maf_type"),
                          ("axt", "axt_type"), ("lav", "lav_type")):
        for flavour in options.get(sub, []):
            pairs.append((selector, flavour))
    # The selectors whose format kegalign.xml writes as a literal rather than a sub-select.
    pairs += [("general_def", "general-"), ("general_full", "general-:score,name1"),
              ("blastn", "BLASTN-"), ("differences", "differences")]
    assert pairs, "no formats found in output_options.xml"
    return pairs


# The contract: the Galaxy datatype each LASTZ output format should carry.
#
# ⚠ THE "BAM" OPTION EMITS SAM, AND THE HELP TEXT SAYING OTHERWISE IS INHERITED, NOT TRUE.
# output_options.xml still carries LASTZ's own wording -- "Lastz actually outputs SAM data but
# Galaxy converts it into BAM to save space" -- from the tool this one was adapted from. Here the
# conversion was never wired up: kegalign.xml carries it commented out under `## todo :: rplot,
# bam`. So the honest datatype for every sam flavour is `sam`; asserting `bam` would encode the
# documentation's claim rather than the tool's behaviour.
#
# `sam-` and `softsam-` suppress the SAM header. They are typed `sam` here because that is what they
# contain, but a headerless SAM is not valid input to most SAM-consuming tools -- an open question
# recorded rather than silently decided.
EXPECTED_DATATYPE = {
    "sam": "sam", "softsam": "sam", "sam-": "sam", "softsam-": "sam",
    "maf": "maf", "maf+": "maf", "maf-": "maf",
    "axt": "axt", "axt+": "axt",
    "lav": "lav", "lav+text": "lav",
    "differences": "interval",
    "BLASTN-": "tabular",
    "general-": "tabular",
    "general-:score,name1": "tabular",
}


def write_format(package_output: types.ModuleType, selector: str, lastz_format: str,
                 tmp_path: pathlib.Path) -> str:
    """What this copy of package_output.py puts in format.txt for this format."""
    writer = package_output.bashCommandLineFile.__new__(package_output.bashCommandLineFile)
    # Both attributes: the wrapper copy reads output_format, the KegAlign copy format_selector.
    writer.args = types.SimpleNamespace(output_format=lastz_format, format_selector=selector)
    writer.package_file = types.SimpleNamespace(add_format=lambda _f: None)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)
        writer._write_format()
        return (tmp_path / "format.txt").read_text().strip()
    finally:
        os.chdir(cwd)


def read_format(run_lastz_tarball: types.ModuleType, written: str, tmp_path: pathlib.Path) -> str:
    """What this copy of run_lastz_tarball.py types the output as, given `written`."""
    (tmp_path / "galaxy").mkdir(exist_ok=True)
    (tmp_path / "galaxy" / "format.txt").write_text(written + "\n")
    reader = run_lastz_tarball.BatchTar.__new__(run_lastz_tarball.BatchTar)
    reader.format_name = "tabular"          # the default __init__ sets before _load_format runs
    reader.pathname = str(tmp_path)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)
        reader._load_format()
        return reader.format_name
    finally:
        os.chdir(cwd)


PAIRS = {
    "wrapper": (TOOLS / "kegalign" / "package_output.py",
                TOOLS / "batched_lastz" / "run_lastz_tarball.py"),
}
if CORE:
    PAIRS["core"] = (CORE / "package_output.py", CORE / "run_lastz_tarball.py")
    PAIRS["core-writer/wrapper-reader"] = (CORE / "package_output.py",
                                           TOOLS / "batched_lastz" / "run_lastz_tarball.py")
    PAIRS["wrapper-writer/core-reader"] = (TOOLS / "kegalign" / "package_output.py",
                                           CORE / "run_lastz_tarball.py")


@pytest.mark.parametrize("pair", sorted(PAIRS))
@pytest.mark.parametrize(("selector", "lastz_format"), _selector_formats())
def test_round_trip_yields_the_right_datatype(pair: str, selector: str, lastz_format: str,
                                              tmp_path: pathlib.Path) -> None:
    """Writing a selector and reading it back must produce the datatype the output really is.

    Parameterised over the pairs because the mixed ones are not hypothetical: a tarball built by one
    version can be read by another whenever the two tools are updated separately, which is exactly
    what having two copies in two repositories makes routine.
    """
    writer_path, reader_path = PAIRS[pair]
    writer = _load(writer_path, f"pkg_{pair}")
    reader = _load(reader_path, f"run_{pair}")
    written = write_format(writer, selector, lastz_format, tmp_path)
    got = read_format(reader, written, tmp_path)
    assert got == EXPECTED_DATATYPE[lastz_format], (
        f"[{pair}] {selector}/{lastz_format!r} -> format.txt {written!r} -> datatype {got!r}, "
        f"expected {EXPECTED_DATATYPE[lastz_format]!r}")


@pytest.mark.parametrize(("selector", "lastz_format"), _selector_formats())
def test_writer_and_reader_speak_the_same_vocabulary(selector: str, lastz_format: str,
                                                     tmp_path: pathlib.Path) -> None:
    """Whatever the writer emits must be a value the reader actually recognises.

    Separate from the round trip because it isolates the root cause: `--format_selector` carries a
    GALAXY selector (`bam`, `general_def`), while the reader's table is keyed on LASTZ format names
    (`sam-`, `maf+`). Where those vocabularies fail to overlap the reader silently falls back to
    "tabular" instead of failing, so nothing downstream ever reports the mismatch.
    """
    writer = _load(PAIRS["wrapper"][0], "pkg_vocab")
    reader = _load(PAIRS["wrapper"][1], "run_vocab")
    written = write_format(writer, selector, lastz_format, tmp_path)
    got = read_format(reader, written, tmp_path)
    fell_back = got == "tabular" and EXPECTED_DATATYPE[lastz_format] != "tabular"
    assert not fell_back, (
        f"writer emitted {written!r} for {selector}/{lastz_format!r}; the reader has no entry for it and "
        f"silently defaulted to 'tabular'")


@pytest.mark.parametrize(("selector", "lastz_format"), _selector_formats())
def test_characterise_current_wrapper_behaviour(selector: str, lastz_format: str,
                                                tmp_path: pathlib.Path, record_property) -> None:
    """Record what the shipped wrapper does today. Never fails -- it is the before-picture."""
    writer = _load(PAIRS["wrapper"][0], "pkg_char")
    reader = _load(PAIRS["wrapper"][1], "run_char")
    written = write_format(writer, selector, lastz_format, tmp_path)
    got = read_format(reader, written, tmp_path)
    record_property("selector", selector)
    record_property("lastz_format", lastz_format)
    record_property("format_txt", written)
    record_property("datatype", got)
    print(f"  {selector:13s} {lastz_format:22s} -> format.txt {written!r:24s} -> datatype {got!r}"
          f"{'   <-- WRONG, expected ' + EXPECTED_DATATYPE[lastz_format] if got != EXPECTED_DATATYPE[lastz_format] else ''}")
