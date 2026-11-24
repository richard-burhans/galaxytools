"""Unit and static contract tests for tools/ncbi_egapx."""

import gzip
import importlib.util
import os
import pathlib
import re
import shutil
import types
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
EGAPX_DIR = ROOT / "tools" / "ncbi_egapx"


def _load_module(path: pathlib.Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load_module(EGAPX_DIR / "prepare_egapx_config.py", "prepare_egapx_config")
RESOURCE_CFG = _load_module(
    EGAPX_DIR / "docker" / "assets" / "galaxy-resource-config.py",
    "galaxy_resource_config",
)
PARAM_PATHS = _load_module(ROOT / "tests" / "test_param_paths.py", "test_param_paths_mod")


def _parse_egapx_reads_table(path: pathlib.Path) -> dict[str, list[str]]:
    """Mirror upstream egapx.py _parse_reads_table_file whitespace splitting."""
    prefixes: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        mo = re.match(r"([^ \t]+)[ \t]+(.*)", stripped)
        assert mo is not None, f"unparseable reads table line: {line!r}"
        sample = mo.group(1)
        files = mo.group(2).strip().split()
        prefixes.setdefault(sample, []).extend(files)
    return prefixes


def test_macro_expanded_param_references_resolve() -> None:
    """Verify Cheetah references in all three ncbi_egapx XMLs AFTER expanding @TOKEN@s."""
    macros_root = ET.parse(EGAPX_DIR / "macros.xml").getroot()
    tokens = {
        node.get("name"): (node.text or "")
        for node in macros_root.findall("token")
        if node.get("name")
    }

    for xml_name in ("ncbi_egapx.xml", "ncbi_egapx_prepare_input.xml", "ncbi_egapx_execute.xml"):
        tool_xml = EGAPX_DIR / xml_name
        declared = PARAM_PATHS._declared_paths(tool_xml)
        sections = {p for p, tag in declared.items() if "." not in p and tag != "param"}
        raw_text = PARAM_PATHS._command_text(tool_xml)
        for tok_name, tok_body in tokens.items():
            raw_text = raw_text.replace(tok_name, tok_body)

        referenced = PARAM_PATHS._references(raw_text, sections)
        assert referenced, f"{xml_name}: expected section-rooted parameter references"
        unresolved = sorted(r for r in referenced if not PARAM_PATHS._resolves(r, declared))
        assert not unresolved, f"{xml_name}: unresolved references after token expansion: {unresolved}"

        # Also check no nested leaf is referenced bare inside expanded tokens.
        for_vars = set(re.findall(r"#for\s+\$(\w+)", raw_text))
        locals_ = set(PARAM_PATHS.LOCALS.findall(raw_text)) | for_vars
        nested_leaves = {p.split(".")[-1] for p, tag in declared.items() if "." in p and tag == "param"}
        top_level = {p for p in declared if "." not in p}
        bare = {
            ref
            for ref in PARAM_PATHS.REFERENCE.findall(raw_text)
            if "." not in ref and ref not in locals_ and ref not in top_level and ref in nested_leaves
        }
        assert not bare, f"{xml_name}: bare nested references after token expansion: {sorted(bare)}"


def test_workflow_pins_match_macros_version() -> None:
    """Keep test-workflows/egapx-prepare-execute.gxwf.yml in lockstep with macros.xml."""
    macros_root = ET.parse(EGAPX_DIR / "macros.xml").getroot()
    tokens = {n.get("name"): (n.text or "").strip() for n in macros_root.findall("token")}
    expected = f"{tokens['@TOOL_VERSION@']}+galaxy{tokens['@VERSION_SUFFIX@']}"

    wf = yaml.safe_load((EGAPX_DIR / "test-workflows" / "egapx-prepare-execute.gxwf.yml").read_text())
    for step_name in ("prepare_input", "execute"):
        actual = wf["steps"][step_name]["tool_version"]
        assert actual == expected, f"workflow step {step_name} pins {actual}, expected {expected}"


def test_fillform_online_stages_symlinks_and_valid_yaml(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Online mode stages symlinks, strips spaces from dataset identifiers, and emits int deny-taxids."""
    monkeypatch.chdir(tmp_path)

    # Simulate Galaxy .dat files (gzipped genome & proteins without .gz filename suffix!)
    genome_dat = tmp_path / "dataset_101.dat"
    with gzip.open(genome_dat, "wb") as gz:
        gz.write(b">chr1\nACGTACGT\n")

    add_prot_dat = tmp_path / "dataset_102.dat"
    with gzip.open(add_prot_dat, "wb") as gz:
        gz.write(b">prot1\nMKTLL\n")

    r1_dat = tmp_path / "dataset_103.dat"
    r2_dat = tmp_path / "dataset_104.dat"
    r1_dat.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    r2_dat.write_text("@r2\nTGCA\n+\nIIII\n", encoding="utf-8")

    se_dat = tmp_path / "dataset_105.dat"
    se_dat.write_text(">seq1\nACGTACGT\n", encoding="utf-8")

    extra_file = tmp_path / "extra.yaml"
    extra_file.write_text(
        "tasks:\n  convert_from_bam:\n    sam2asn: -filter 'pct_identity_gap >= 85'\n",
        encoding="utf-8",
    )

    out_yaml = tmp_path / "egapx.yaml"
    rc = PREPARE.main([
        "--output-yaml", str(out_yaml),
        "--stage-dir", "galaxy_staged",
        "--genome-file", str(genome_dat),
        "--genome-ext", "auto",
        "--taxid", "6954",
        "--locus-tag-prefix", "egapxtmp",
        "--short-read-single", "FASTP on data 5 (single).fasta", str(se_dat), "fasta",
        "--short-read-pair", "Sample A (SRR8506572)", str(r1_dat), "fastqsanger", str(r2_dat), "fastqsanger",
        "--protein-aligner", "prosplign",
        "--proteins-best-n-orgs", "5",
        "--additional-proteins", str(add_prot_dat), "fasta.gz",
        "--proteins-deny-taxids", " 7227, 7240 ",
        "--trnascan",
        "--max-intron", "700000",
        "--annotation-provider", "Galaxy Test Lab",
        "--annotation-name-prefix", "D_farinae_v1",
        "--extra-yaml-file", str(extra_file),
    ])
    assert rc == 0

    cfg = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert cfg["genome"] == "galaxy_staged/genome/genome.fasta.gz"
    assert pathlib.Path(cfg["genome"]).is_symlink()
    assert cfg["taxid"] == 6954
    assert cfg["locus_tag_prefix"] == "egapxtmp"
    assert cfg["protein_aligner_name"] == "prosplign"
    assert cfg["proteins_best_n_orgs"] == 5
    assert cfg["additional_proteins"] == "galaxy_staged/proteins/additional_proteins.fasta.gz"
    # Upstream egapx.py runs set(map(int, inputs.get('proteins_deny_taxids', [])))
    assert cfg["proteins_deny_taxids"] == [7227, 7240]
    assert set(map(int, cfg["proteins_deny_taxids"])) == {7227, 7240}
    assert cfg["trnascan"] == {"enabled": True}
    assert cfg["cmsearch"] == {"enabled": False}
    assert cfg["max_intron"] == 700000
    assert cfg["annotation_provider"] == "Galaxy Test Lab"
    assert cfg["annotation_name_prefix"] == "D_farinae_v1"
    assert cfg["tasks"]["convert_from_bam"]["sam2asn"] == "-filter 'pct_identity_gap >= 85'"
    # Empty long reads in history mode must not emit a dangling or empty key
    assert "long_reads" not in cfg

    # Verify short_reads table parses cleanly with upstream whitespace rules
    reads_map = _parse_egapx_reads_table(pathlib.Path(cfg["short_reads"]))
    assert len(reads_map) == 2
    for sample_id, files in reads_map.items():
        assert " " not in sample_id
        assert len(files) in (1, 2)
        for fpath in files:
            assert pathlib.Path(fpath).is_file()
            assert not fpath.endswith(".fastqsanger")


def test_prepare_to_execute_cache_relocation(tmp_path: pathlib.Path, monkeypatch) -> None:
    """History datasets staged into prepare_input's cache remain valid when relocated to execute's job dir."""
    prep_cwd = tmp_path / "job_prepare"
    prep_cwd.mkdir()
    src_dir = tmp_path / "galaxy_objects"
    src_dir.mkdir()
    prep_cache = tmp_path / "prep_cache_files"

    genome_dat = src_dir / "dataset_genome.dat"
    genome_dat.write_text(">ctg1\nACGTACGTACGT\n", encoding="utf-8")
    prot_dat = src_dir / "dataset_custom_prot.dat"
    prot_dat.write_text(">p1\nMKTLLLTLVVVTIVCLDLGYT\n", encoding="utf-8")
    long_dat = src_dir / "dataset_long.dat"
    with gzip.open(long_dat, "wb") as gz:
        gz.write(b"@lr1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")

    exported_yaml = tmp_path / "exported_history_config.yaml"

    monkeypatch.chdir(prep_cwd)
    assert PREPARE.main([
        "--output-yaml", "egapx.yaml",
        "--export-yaml", str(exported_yaml),
        "--stage-dir", "galaxy_staged",
        "--cache-dir", str(prep_cache),
        "--genome-file", str(genome_dat),
        "--genome-ext", "fasta",
        "--taxid", "6954",
        "--short-reads-query", "txid6954[Organism] AND biomol_rna[Properties]",
        "--long-read-single", "ONT IsoSeq Run 1.fastq.gz", str(long_dat), "fastqsanger.gz",
        "--proteins", str(prot_dat), "fasta",
    ]) == 0

    # Staged files inside prep_cache must be real files (Pulsar rule: no external symlinks)
    for item in (prep_cache / "galaxy_staged").rglob("*"):
        if item.is_file():
            assert not item.is_symlink(), f"cache file {item} must be a real file, not a symlink"

    # Remove both prep_cwd and original galaxy_objects, and relocate cache to execute's path
    shutil.rmtree(prep_cwd)
    shutil.rmtree(src_dir)
    exec_cache = tmp_path / "relocated_exec_cache"
    shutil.move(str(prep_cache), str(exec_cache))

    exec_cwd = tmp_path / "job_execute"
    exec_cwd.mkdir()
    monkeypatch.chdir(exec_cwd)

    # Mirror ncbi_egapx_execute.xml <command>
    os.symlink(exec_cache / "galaxy_staged", exec_cwd / "galaxy_staged")
    assert PREPARE.main([
        "--base-yaml", str(exported_yaml),
        "--output-yaml", "egapx.yaml",
    ]) == 0

    cfg = yaml.safe_load((exec_cwd / "egapx.yaml").read_text(encoding="utf-8"))
    assert pathlib.Path(cfg["genome"]).is_file()
    assert pathlib.Path(cfg["proteins"]).is_file()
    assert cfg["short_reads"] == "txid6954[Organism] AND biomol_rna[Properties]"
    long_map = _parse_egapx_reads_table(pathlib.Path(cfg["long_reads"]))
    assert len(long_map) == 1
    (long_files,) = long_map.values()
    assert len(long_files) == 1
    assert long_files[0].endswith(".fastq.gz")
    assert pathlib.Path(long_files[0]).is_file()


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--genome-uri", "file:///etc/passwd", "--taxid", "6954"],
        ["--genome-uri", "https://example.org/g.fa;id", "--taxid", "6954"],
        ["--genome-uri", "https://example.org/g.fa", "--taxid", "0"],
        ["--genome-uri", "https://example.org/g.fa", "--taxid", "6954", "--locus-tag-prefix", "bad;tag"],
        ["--genome-uri", "https://example.org/g.fa", "--taxid", "6954", "--short-reads-list", "not_sra_or_url"],
        ["--genome-uri", "https://example.org/g.fa", "--taxid", "6954", "--proteins-deny-taxids", "7227,abc"],
    ],
)
def test_prepare_rejects_invalid_inputs(tmp_path: pathlib.Path, bad_args: list[str]) -> None:
    out_yaml = tmp_path / "out.yaml"
    with pytest.raises(SystemExit) as exc:
        PREPARE.main(["--output-yaml", str(out_yaml)] + bad_args)
    assert exc.value.code != 0


def test_prepare_rejects_bad_extra_yaml(tmp_path: pathlib.Path) -> None:
    bad_extra = tmp_path / "bad.yaml"
    bad_extra.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        PREPARE.main([
            "--output-yaml", str(tmp_path / "out.yaml"),
            "--genome-uri", "https://example.org/g.fa",
            "--taxid", "6954",
            "--extra-yaml-file", str(bad_extra),
        ])


def test_xml_regex_validators_are_anchored_and_reject_hostile_text() -> None:
    """Every regex validator in macros.xml must be anchored (^...$) and reject shell injection."""
    macros_root = ET.parse(EGAPX_DIR / "macros.xml").getroot()
    validators: dict[str, list[str]] = {}
    for param in macros_root.iter("param"):
        pname = param.get("name") or ""
        for val in param.findall("validator"):
            if val.get("type") == "regex" and val.text:
                pattern = val.text.strip()
                assert pattern.startswith("^") and pattern.endswith("$"), (
                    f"{pname}: regex validator is not anchored at both ends: {pattern!r}"
                )
                validators.setdefault(pname, []).append(pattern)

    hostile_samples = {
        "uri": ["https://example.org/g.fa; touch pwned", "file:///etc/passwd", "https://a/b'c"],
        "locus_tag_prefix": ["egapxtmp; id", "-flag", "has space", "quote'here"],
        "short_reads": ["SRR8506572; id", "not_sra_or_url", "https://a/b`id`"],
        "long_reads": ["ERR12345; rm", "local/path.fq"],
        "short_reads_query": ["txid6954[Organism]'; id", "bad;query"],
        "long_reads_query": ["txid6954[Organism]'; id", "bad`cmd`"],
        "proteins_deny_taxids": ["7227; id", "0", "-5", "7227,abc"],
        "annotation_provider": ["Lab'; touch pwned", "Lab`id`"],
        "annotation_name_prefix": ["Asm; rm -rf /", "Asm$(id)"],
    }
    for pname, bad_values in hostile_samples.items():
        assert pname in validators, f"expected regex validator on {pname}"
        for pattern in validators[pname]:
            compiled = re.compile(pattern)
            for bad in bad_values:
                assert compiled.match(bad) is None, f"{pname} regex {pattern!r} accepted {bad!r}"


def test_resource_config_memavailable_and_small_mem_floor(tmp_path: pathlib.Path) -> None:
    """MemAvailable beats tiny MemFree, and a 6 GB floor clamps small_mem <= executor.memory."""
    meminfo = tmp_path / "meminfo"
    # 2 GB MemFree, 128 GB MemAvailable (typical Linux node with page cache)
    meminfo.write_text(
        "MemTotal:       268435456 kB\n"
        "MemFree:          2097152 kB\n"
        "MemAvailable:   134217728 kB\n",
        encoding="utf-8",
    )
    assert RESOURCE_CFG.get_allocated_memory("65536", meminfo_path=str(meminfo)) == 64
    assert RESOURCE_CFG.get_allocated_memory("262144", meminfo_path=str(meminfo)) == 128

    # Fallback to MemFree when MemAvailable is absent
    meminfo_legacy = tmp_path / "meminfo_legacy"
    meminfo_legacy.write_text("MemFree: 33554432 kB\n", encoding="utf-8")
    assert RESOURCE_CFG.get_allocated_memory("65536", meminfo_path=str(meminfo_legacy)) == 32

    # Minimum 6 GB allocation must clamp small_mem to 6.GB so Nextflow does not abort
    overlay_6gb = RESOURCE_CFG.build_galaxy_overlay(galaxy_slots=4, galaxy_memory_gb=6, gigabase_pairs=1)
    assert "memory = 6.GB" in overlay_6gb
    assert "memory = 8.GB" not in overlay_6gb

    # Normal >= 8 GB allocation preserves 8.GB small_mem tier
    overlay_64gb = RESOURCE_CFG.build_galaxy_overlay(galaxy_slots=16, galaxy_memory_gb=64, gigabase_pairs=2)
    assert "withLabel: 'small_mem' {\n        memory = 8.GB\n    }" in overlay_64gb
    assert "params.gigabase_pairs = 2" in overlay_64gb
