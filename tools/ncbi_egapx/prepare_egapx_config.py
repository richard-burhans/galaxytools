#!/usr/bin/env python3
"""Build an EGAPx YAML configuration and stage local inputs for Galaxy."""

import argparse
import os
import pathlib
import re
import shutil
import sys
from typing import NoReturn

import yaml

__version__ = "1.0.1"

URI_RE = re.compile(r"^(https?|ftp)://[^\s'\"`;$|&<>()\\]+$")
SRA_RUN_RE = re.compile(r"^[DES]RR\d+$")
LOCUS_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
SEQ_EXT_RE = re.compile(
    r"\.(?:fastqsanger|fastq|fq|fasta|fa|fna)(?:\.gz)?$",
    re.IGNORECASE,
)


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def is_gzipped(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def sanitize_token(raw: str, fallback: str = "sample") -> str:
    cleaned = SEQ_EXT_RE.sub("", raw.strip())
    cleaned = re.sub(r"[^\w.-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    if not cleaned:
        return fallback
    return cleaned[:80]


def normalize_fasta_ext(ext: str, src_path: pathlib.Path) -> str:
    ext_clean = (ext or "").strip().lstrip(".").lower()
    gz = ext_clean.endswith(".gz") or src_path.name.lower().endswith(".gz") or is_gzipped(src_path)
    return ".fasta.gz" if gz else ".fasta"


def normalize_read_ext(ext: str, src_path: pathlib.Path) -> str:
    ext_clean = (ext or "").strip().lstrip(".").lower()
    gz = ext_clean.endswith(".gz") or src_path.name.lower().endswith(".gz") or is_gzipped(src_path)
    base_ext = ext_clean[:-3] if ext_clean.endswith(".gz") else ext_clean
    is_fasta = base_ext in {"fasta", "fa", "fna"} or src_path.name.lower().endswith(
        (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
    )
    stem = "fasta" if is_fasta else "fastq"
    return f".{stem}.gz" if gz else f".{stem}"


def stage_file(src: pathlib.Path, dst: pathlib.Path, copy_files: bool) -> None:
    if not src.is_file():
        fail(f"Input file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def merge_yaml_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_yaml_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_extra_yaml(path: pathlib.Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML in Additional YAML field: {exc}")
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        fail("Additional YAML must parse to a YAML mapping (key: value pairs)")
    return parsed


def parse_deny_taxids(raw: str) -> list[int]:
    taxids: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit() or int(token) <= 0:
            fail(f"Invalid denied taxid '{token}'; expected positive integers separated by commas")
        taxids.append(int(token))
    return taxids


def parse_reads_list(raw: str, label: str) -> list[str]:
    tokens = raw.split()
    if not tokens:
        fail(f"{label} list is empty; provide at least one SRA run accession or remote URI")
    for token in tokens:
        if not (SRA_RUN_RE.match(token) or URI_RE.match(token)):
            fail(
                f"Invalid {label} entry '{token}'; expected an SRA run accession "
                "(SRR/ERR/DRR...) or an http(s)/ftp URI"
            )
    return tokens


def stage_reads_for_kind(
    kind: str,
    singles: list[list[str]],
    pairs: list[list[str]],
    stage_root: pathlib.Path,
    stage_rel_prefix: str,
    copy_files: bool,
) -> str | None:
    if not singles and not pairs:
        return None

    reads_dir = stage_root / "reads"
    reads_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for idx, (ident, src_str, ext) in enumerate(singles):
        src = pathlib.Path(src_str)
        safe_id = sanitize_token(ident, fallback=f"{kind}_se_{idx}")
        norm_ext = normalize_read_ext(ext, src)
        filename = f"{kind}_se_{idx}_{safe_id}{norm_ext}"
        stage_file(src, reads_dir / filename, copy_files=copy_files)
        rel_path = f"{stage_rel_prefix}/reads/{filename}"
        sample_id = f"{kind}_se_{idx}_{safe_id}"
        lines.append(f"{sample_id} {rel_path}\n")

    for idx, (ident, fwd_str, fwd_ext, rev_str, rev_ext) in enumerate(pairs):
        fwd_src = pathlib.Path(fwd_str)
        rev_src = pathlib.Path(rev_str)
        safe_id = sanitize_token(ident, fallback=f"{kind}_pe_{idx}")
        fwd_name = f"{kind}_pe_{idx}_{safe_id}_1{normalize_read_ext(fwd_ext, fwd_src)}"
        rev_name = f"{kind}_pe_{idx}_{safe_id}_2{normalize_read_ext(rev_ext, rev_src)}"
        stage_file(fwd_src, reads_dir / fwd_name, copy_files=copy_files)
        stage_file(rev_src, reads_dir / rev_name, copy_files=copy_files)
        fwd_rel = f"{stage_rel_prefix}/reads/{fwd_name}"
        rev_rel = f"{stage_rel_prefix}/reads/{rev_name}"
        sample_id = f"{kind}_pe_{idx}_{safe_id}"
        lines.append(f"{sample_id} {fwd_rel} {rev_rel}\n")

    table_name = f"{kind}_reads.txt"
    (reads_dir / table_name).write_text("".join(lines), encoding="utf-8")
    return f"{stage_rel_prefix}/reads/{table_name}"


def handle_base_yaml(
    base_yaml: pathlib.Path,
    extra_yaml_file: pathlib.Path | None,
    output_yaml: pathlib.Path,
    export_yaml: pathlib.Path | None,
) -> None:
    if not base_yaml.is_file():
        fail(f"Input YAML configuration does not exist: {base_yaml}")
    try:
        parsed = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        fail(f"Invalid input YAML configuration: {exc}")
    if not isinstance(parsed, dict):
        fail("Input YAML configuration must be a YAML mapping")
    input_section = parsed.get("input") if isinstance(parsed.get("input"), dict) else parsed
    if "genome" not in input_section or "taxid" not in input_section:
        fail("Input YAML configuration must define both 'genome' and 'taxid'")

    extra = load_extra_yaml(extra_yaml_file)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    if extra:
        merged = merge_yaml_dicts(parsed, extra)
        output_yaml.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    elif base_yaml.resolve() != output_yaml.resolve():
        shutil.copyfile(base_yaml, output_yaml)

    if export_yaml is not None and export_yaml.resolve() != output_yaml.resolve():
        shutil.copyfile(output_yaml, export_yaml)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-yaml", type=pathlib.Path, help="Existing EGAPx YAML configuration")
    parser.add_argument("--output-yaml", type=pathlib.Path, required=True, help="Output YAML path")
    parser.add_argument("--export-yaml", type=pathlib.Path, help="Optional history output copy of generated YAML")
    parser.add_argument("--stage-dir", default="galaxy_staged", help="Relative directory name for staged files")
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="Optional local cache directory; when set, staged inputs are copied into <cache-dir>/<stage-dir>",
    )

    parser.add_argument("--genome-file", type=pathlib.Path, help="Local or indexed genome FASTA path")
    parser.add_argument("--genome-ext", default="auto", help="Galaxy datatype extension for --genome-file")
    parser.add_argument("--genome-uri", help="Remote http(s)/ftp URI for the genome FASTA")
    parser.add_argument("--taxid", type=int, help="NCBI Taxonomy ID")
    parser.add_argument("--locus-tag-prefix", default="", help="Locus tag prefix")

    parser.add_argument("--short-read-single", nargs=3, action="append", default=[], metavar=("ID", "PATH", "EXT"))
    parser.add_argument(
        "--short-read-pair",
        nargs=5,
        action="append",
        default=[],
        metavar=("ID", "FWD_PATH", "FWD_EXT", "REV_PATH", "REV_EXT"),
    )
    parser.add_argument("--short-reads-list", help="Whitespace-separated SRA accessions and/or URIs")
    parser.add_argument("--short-reads-query", help="SRA Entrez query for short reads")

    parser.add_argument("--long-read-single", nargs=3, action="append", default=[], metavar=("ID", "PATH", "EXT"))
    parser.add_argument(
        "--long-read-pair",
        nargs=5,
        action="append",
        default=[],
        metavar=("ID", "FWD_PATH", "FWD_EXT", "REV_PATH", "REV_EXT"),
    )
    parser.add_argument("--long-reads-list", help="Whitespace-separated SRA accessions and/or URIs")
    parser.add_argument("--long-reads-query", help="SRA Entrez query for long reads")

    parser.add_argument("--protein-aligner", choices=["miniprot", "prosplign"], default="miniprot")
    parser.add_argument("--proteins-best-n-orgs", type=int, default=0)
    parser.add_argument("--additional-proteins", nargs=2, metavar=("PATH", "EXT"))
    parser.add_argument("--proteins-deny-taxids", default="")
    parser.add_argument("--proteins", nargs=2, metavar=("PATH", "EXT"))

    parser.add_argument("--trnascan", action="store_true", default=False)
    parser.add_argument("--cmsearch", action="store_true", default=False)

    parser.add_argument("--max-intron", type=int)
    parser.add_argument("--annotation-provider", default="")
    parser.add_argument("--annotation-name-prefix", default="")
    parser.add_argument("--extra-yaml-file", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.base_yaml is not None:
        handle_base_yaml(args.base_yaml, args.extra_yaml_file, args.output_yaml, args.export_yaml)
        return 0

    stage_rel_prefix = args.stage_dir.strip("/")
    if not stage_rel_prefix or "/" in stage_rel_prefix or stage_rel_prefix in {".", ".."}:
        fail(f"Invalid --stage-dir '{args.stage_dir}'")

    copy_files = args.cache_dir is not None
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        stage_root = args.cache_dir / stage_rel_prefix
    else:
        stage_root = pathlib.Path(stage_rel_prefix)

    config: dict = {}

    if args.genome_file is not None:
        norm_ext = normalize_fasta_ext(args.genome_ext, args.genome_file)
        genome_name = f"genome{norm_ext}"
        stage_file(args.genome_file, stage_root / "genome" / genome_name, copy_files=copy_files)
        config["genome"] = f"{stage_rel_prefix}/genome/{genome_name}"
    elif args.genome_uri is not None:
        uri = args.genome_uri.strip()
        if not URI_RE.match(uri):
            fail(f"Invalid genome URI '{uri}'; expected an http(s) or ftp URI without spaces or quotes")
        config["genome"] = uri
    else:
        fail("A reference genome (history, built-in index, or URI) is required")

    if args.taxid is None or args.taxid < 1:
        fail("NCBI Taxon ID (--taxid) must be a positive integer (>= 1)")
    config["taxid"] = args.taxid

    locus_tag_prefix = args.locus_tag_prefix.strip()
    if locus_tag_prefix:
        if not LOCUS_TAG_RE.match(locus_tag_prefix):
            fail(
                f"Invalid locus_tag_prefix '{locus_tag_prefix}'; use 1-32 alphanumeric, '_' or '-' "
                "characters starting with a letter or digit"
            )
        config["locus_tag_prefix"] = locus_tag_prefix

    if args.short_reads_list is not None:
        config["short_reads"] = parse_reads_list(args.short_reads_list, "short RNA-seq")
    elif args.short_reads_query is not None:
        query = args.short_reads_query.strip()
        if not query:
            fail("Short RNA-seq SRA Entrez query cannot be empty")
        config["short_reads"] = query
    else:
        short_table = stage_reads_for_kind(
            "short",
            args.short_read_single,
            args.short_read_pair,
            stage_root,
            stage_rel_prefix,
            copy_files=copy_files,
        )
        if short_table is not None:
            config["short_reads"] = short_table

    if args.long_reads_list is not None:
        config["long_reads"] = parse_reads_list(args.long_reads_list, "long RNA-seq")
    elif args.long_reads_query is not None:
        query = args.long_reads_query.strip()
        if not query:
            fail("Long RNA-seq SRA Entrez query cannot be empty")
        config["long_reads"] = query
    else:
        long_table = stage_reads_for_kind(
            "long",
            args.long_read_single,
            args.long_read_pair,
            stage_root,
            stage_rel_prefix,
            copy_files=copy_files,
        )
        if long_table is not None:
            config["long_reads"] = long_table

    config["protein_aligner_name"] = args.protein_aligner
    if args.proteins is not None:
        prot_src = pathlib.Path(args.proteins[0])
        prot_name = f"proteins{normalize_fasta_ext(args.proteins[1], prot_src)}"
        stage_file(prot_src, stage_root / "proteins" / prot_name, copy_files=copy_files)
        config["proteins"] = f"{stage_rel_prefix}/proteins/{prot_name}"
    else:
        if args.proteins_best_n_orgs > 0:
            config["proteins_best_n_orgs"] = args.proteins_best_n_orgs
        if args.additional_proteins is not None:
            add_src = pathlib.Path(args.additional_proteins[0])
            add_name = f"additional_proteins{normalize_fasta_ext(args.additional_proteins[1], add_src)}"
            stage_file(add_src, stage_root / "proteins" / add_name, copy_files=copy_files)
            config["additional_proteins"] = f"{stage_rel_prefix}/proteins/{add_name}"
        if args.proteins_deny_taxids.strip():
            deny_list = parse_deny_taxids(args.proteins_deny_taxids)
            if deny_list:
                config["proteins_deny_taxids"] = deny_list

    config["trnascan"] = {"enabled": bool(args.trnascan)}
    config["cmsearch"] = {"enabled": bool(args.cmsearch)}

    if args.max_intron is not None:
        if args.max_intron < 1:
            fail("--max-intron must be a positive integer")
        config["max_intron"] = args.max_intron
    if args.annotation_provider.strip():
        config["annotation_provider"] = args.annotation_provider.strip()
    if args.annotation_name_prefix.strip():
        config["annotation_name_prefix"] = args.annotation_name_prefix.strip()

    extra = load_extra_yaml(args.extra_yaml_file)
    if extra:
        config = merge_yaml_dicts(config, extra)

    # When staging into a local cache directory (prepare_input), link ./<stage-dir>
    # in the working directory so egapx.py --download-needed resolves relative paths
    # immediately, just as ncbi_egapx_execute will at execution time.
    if args.cache_dir is not None and stage_root.is_dir():
        cwd_link = pathlib.Path(stage_rel_prefix)
        if cwd_link.resolve() != stage_root.resolve():
            if cwd_link.exists() or cwd_link.is_symlink():
                cwd_link.unlink()
            os.symlink(stage_root.resolve(), cwd_link)

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    if args.export_yaml is not None and args.export_yaml.resolve() != args.output_yaml.resolve():
        shutil.copyfile(args.output_yaml, args.export_yaml)

    return 0


if __name__ == "__main__":
    sys.exit(main())
