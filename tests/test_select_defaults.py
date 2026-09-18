"""A single-select `<param>` must mark at most one `<option selected="true">`.

Galaxy resolves a select's default by scanning its options; with two marked selected the tool still
loads, the form still renders, and the job still runs -- so nothing fails and nothing warns. What the
user sees is a form with two options both drawn as chosen, and what the tool gets is whichever one
the resolution order happens to reach.

⚠ TODAY THAT ORDER FAVOURS THE FIRST, which is why `kegalign`'s `alignment_type` behaved correctly
despite carrying two: measured against a live Galaxy, `/api/tools?io_details=true` reported
`value='vertebrates_default'` while both it and `vertebrates_same_species` were flagged. So this is a
LATENT defect, not a behavioural one -- and that is precisely why a test is worth more than the
one-line fix. Reorder the options, or have Galaxy change how it resolves, and the default moves
silently with no diff to blame.

⛔ `multiple="true"` SELECTS ARE EXEMPT. Several selected options is the whole point of a multi-select,
so the check keys on the `multiple` attribute rather than on option count. A version of this test
that flagged them would have to be suppressed everywhere it fired, which is how a check stops being
read.

This is a STATIC check: it parses the XML, needs no Galaxy, no container and no GPU, and covers every
select in every tool rather than the one that happened to be noticed.
"""
import pathlib
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def _xml_files() -> list[pathlib.Path]:
    """Every tool and macro XML, with symlinks collapsed to their target.

    ⚠ `alignment_type_option.xml` USED TO BE a symlink from `batched_lastz/` into `kegalign/`, so walking the
    tree naively reports the same defect twice and a fix appears to only half-land. Resolving to the
    real path and de-duplicating is what makes the count mean "distinct files".
    """
    seen: dict[pathlib.Path, pathlib.Path] = {}
    for p in sorted(TOOLS.rglob("*.xml")):
        seen.setdefault(p.resolve(), p)
    return sorted(seen.values())


def _selects(path: pathlib.Path):
    """Yield (param_element, source_path) for every single-select param in the file."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        pytest.fail(f"{path.relative_to(ROOT)} is not parseable XML: {exc}")
    for param in root.iter("param"):
        if param.get("type") != "select":
            continue
        if param.get("multiple", "false").lower() in ("true", "1"):
            continue
        yield param


@pytest.mark.parametrize("path", _xml_files(), ids=lambda p: str(p.relative_to(TOOLS)))
def test_single_select_has_at_most_one_default(path: pathlib.Path) -> None:
    offenders = []
    for param in _selects(path):
        chosen = [
            o.get("value")
            for o in param.findall("option")
            if o.get("selected", "false").lower() in ("true", "1")
        ]
        if len(chosen) > 1:
            name = param.get("name") or param.get("argument") or "<unnamed>"
            offenders.append(f"{name}: {len(chosen)} marked selected -> {chosen}")
    assert not offenders, (
        f"{path.relative_to(ROOT)} has single-select param(s) with more than one default:\n  "
        + "\n  ".join(offenders)
        + "\nGalaxy resolves one of them and warns about none; keep exactly one `selected=\"true\"`."
    )
