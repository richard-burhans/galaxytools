"""Every `$section.…` reference in a tool's command must resolve to a declared parameter.

Galaxy parameters nest: a `<param>` inside a `<conditional>` inside a `<section>` is addressed by the
full path, `$section.conditional.param`. Get a level wrong and Cheetah does not fail loudly at lint
time -- the tool ships, and the error surfaces only when a user selects the option that reaches that
branch. `kegalign`'s Ambiguous Nucleotides settings were wired one level too shallow for exactly that
reason: the default short-circuits the block, so nothing ever evaluated it.

This is a STATIC check. It parses the command block for parameter references, parses the macro files
for what is actually declared, and asserts the first resolves against the second. No Galaxy, no
container, no GPU -- so it runs anywhere and covers every parameter in the tool rather than the one
that happened to be reported.
"""
import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"

# Cheetah locals -- assigned with #set inside the command block, not parameters.
LOCALS = re.compile(r"#set\s+\$(\w+)")
# A parameter reference: $section.a.b or ${section.a.b}
REFERENCE = re.compile(r"\$\{?([a-zA-Z_][\w.]*)\}?")


def _declared_paths(tool_xml: pathlib.Path) -> dict[str, str]:
    """Every addressable parameter path declared by a tool's XML and its macros, path -> tag.

    Sections and conditionals contribute a path level; repeats do too but are addressed by
    iteration, so they are recorded and their contents skipped rather than guessed at. The TAG is
    kept because it decides whether a longer path is a nested parameter or an attribute access --
    see `_resolves`.
    """
    paths: dict[str, str] = {}

    def declared_name(node: ET.Element) -> str | None:
        """A param's addressable name.

        ⚠ `argument=` COUNTS. Galaxy derives the parameter name from `argument` when `name` is
        absent -- `argument="--step"` is addressed as `$section.step`. A checker that reads only
        `name` reports every such parameter as undeclared, which is how the first version of this
        test produced 18 false alarms against a tool that works.
        """
        if node.get("name"):
            return node.get("name")
        argument = node.get("argument")
        if argument:
            return argument.lstrip("-").replace("-", "_")
        return None

    def walk(node: ET.Element, prefix: str) -> None:
        for child in node:
            name = declared_name(child)
            if child.tag in ("section", "conditional", "repeat") and name:
                here = f"{prefix}{name}"
                paths[here] = child.tag
                if child.tag != "repeat":
                    walk(child, here + ".")
            elif child.tag == "param" and name:
                paths[f"{prefix}{name}"] = "param"
            elif child.tag in ("when", "xml", "inputs", "macros"):
                walk(child, prefix)
            else:
                walk(child, prefix)

    # ⚠ THE TOOL'S OWN IMPORTS, NOT EVERY XML BESIDE IT. A tool directory can hold more than one
    # tool -- ncbi_egapx/ holds two -- and globbing the directory lets one tool's parameters satisfy
    # another's references, which both hides real defects and invents false ones.
    root = ET.parse(tool_xml).getroot()
    sources = [tool_xml]
    for imported in root.iter("import"):
        if imported.text:
            candidate = tool_xml.parent / imported.text.strip()
            if candidate.exists():
                sources.append(candidate)
    for xml_file in sources:
        walk(ET.parse(xml_file).getroot(), "")
    return paths


def _resolves(ref: str, declared: dict[str, str]) -> bool:
    """Is this reference legitimate?

    Either it names a declared parameter outright, or it is an ATTRIBUTE ACCESS on one --
    `$input_reads.ext`, `$in.element_identifier`, `$out.files_path` are Cheetah reaching into a
    dataset object, not addressing a nested parameter. The two are told apart by what the prefix is:
    attributes hang off a `param`, nested parameters hang off a `section`, `conditional` or `repeat`.
    Without that distinction this check reports every dataset attribute in the repository as a defect.
    """
    if ref in declared:
        return True
    parts = ref.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        if declared.get(prefix) == "param":
            return True
    return False


def _command_text(tool_xml: pathlib.Path) -> str:
    for node in ET.parse(tool_xml).getroot():
        if node.tag in ("command", "configfiles"):
            return node.text or ""
    return ""


def _references(command: str, roots: set[str]) -> set[str]:
    """Referenced paths that are rooted at a declared section, minus Cheetah locals."""
    locals_ = set(LOCALS.findall(command))
    found = set()
    for ref in REFERENCE.findall(command):
        head = ref.split(".")[0]
        if head in roots and head not in locals_ and "." in ref:
            found.add(ref.rstrip("."))
    return found


TOOL_XMLS = sorted(p for p in TOOLS.glob("*/*.xml") if p.parent.name == p.stem)

# Tools carrying the same defect this check was written for, not yet repaired. Recorded rather than
# excluded: `strict` means a fix makes the test fail as UNEXPECTEDLY PASSING, which is the prompt to
# delete the entry. A plain skip would let the defect sit here indefinitely without anyone noticing
# it had been fixed -- or that it had not.
# Keyed by (tool, check) because a tool may carry one defect and not the other -- ncbi_egapx's
# references all resolve, they are simply missing their conditional prefix.
#
# `segalign` is ABANDONED and is deliberately not being repaired. Its entries stay so the check runs
# repository-wide rather than being narrowed to the tools that happen to pass, and so that anyone
# reviving the tool finds the defect already located.
KNOWN_DEFECTS = {
    ("segalign", "resolves"): "abandoned tool — scoring_options ambiguous params wired one level too "
                              "shallow, the same defect as kegalign, in the tool it was derived from",
    ("segalign", "bare"): "abandoned tool — ambiguous_selector referenced without its section prefix",
    ("ncbi_egapx", "bare"): "$genome and $yamlin referenced without their conditional prefix, while "
                            "the #if directly above them uses the full path",
}


def _known(tool_xml: pathlib.Path, check: str, request) -> None:
    reason = KNOWN_DEFECTS.get((tool_xml.parent.name, check))
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("tool_xml", TOOL_XMLS, ids=lambda p: p.parent.name)
def test_every_parameter_reference_resolves(tool_xml: pathlib.Path, request) -> None:
    _known(tool_xml, "resolves", request)
    declared = _declared_paths(tool_xml)
    sections = {p for p, tag in declared.items() if "." not in p and tag != "param"}
    referenced = _references(_command_text(tool_xml), sections)
    if not referenced:
        pytest.skip(f"{tool_xml.parent.name} has no section-rooted parameter references")

    unresolved = sorted(r for r in referenced if not _resolves(r, declared))
    assert not unresolved, (
        f"{tool_xml.parent.name}: {len(unresolved)} reference(s) name no declared parameter — "
        f"Cheetah will fail only when a user reaches that branch:\n  " + "\n  ".join(unresolved))


@pytest.mark.parametrize("tool_xml", TOOL_XMLS, ids=lambda p: p.parent.name)
def test_no_section_parameter_referenced_bare(tool_xml: pathlib.Path, request) -> None:
    """A parameter inside a section must never be addressed without its section prefix.

    `$ambiguous_selector` and `$scoring_options.ambiguous_selector` sat three lines apart in the same
    conditional; the first is not the parameter, it is an undefined name. Catching this needs its own
    assertion because the reference above is not section-rooted and so is invisible to that test.
    """
    _known(tool_xml, "bare", request)
    command = _command_text(tool_xml)
    declared = _declared_paths(tool_xml)
    locals_ = set(LOCALS.findall(command))
    # leaf names that are ONLY reachable through a section
    nested_leaves = {p.split(".")[-1] for p, tag in declared.items() if "." in p and tag == "param"}
    top_level = {p for p in declared if "." not in p}

    bare = set()
    for ref in REFERENCE.findall(command):
        if "." in ref or ref in locals_ or ref in top_level:
            continue
        if ref in nested_leaves:
            bare.add(ref)
    assert not bare, (
        f"{tool_xml.parent.name}: parameter(s) referenced without their section prefix: "
        f"{', '.join(sorted(bare))}")
