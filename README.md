[![Galaxy Tool Linting and Tests for push and PR](https://github.com/richard-burhans/galaxytools/actions/workflows/pr.yaml/badge.svg?branch=main)](https://github.com/richard-burhans/galaxytools/actions/workflows/pr.yaml)
[![Weekly global Tool Linting and Tests](https://github.com/richard-burhans/galaxytools/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/richard-burhans/galaxytools/actions/workflows/ci.yaml)

# Galaxy tools

Galaxy tool wrappers maintained by [@richard-burhans](https://github.com/richard-burhans), in the
[IUC](https://github.com/galaxyproject/tools-iuc) layout. Tools are published to the
[Main Tool Shed](https://toolshed.g2.bx.psu.edu/) under the owner `richard-burhans`, automatically,
when CI passes on `main`.

## Tools

| tool | what it does | notes |
|---|---|---|
| [`kegalign`](tools/kegalign) | GPU whole-genome pairwise alignment on LASTZ's seed–filter–extend paradigm | needs a GPU; **emits work for `batched_lastz`, not alignments** |
| [`batched_lastz`](tools/batched_lastz) | runs the LASTZ commands KegAlign produced | CPU only; second half of the pair above |
| [`segalign`](tools/segalign) | KegAlign's predecessor | |
| [`ncbi_egapx`](tools/ncbi_egapx) | NCBI Eukaryotic Genome Annotation Pipeline (EGAPx) | |
| [`ncbi_fcs_adaptor`](tools/ncbi_fcs_adaptor) | detects adaptor and vector contamination in genome sequences | |
| [`rdeval`](tools/rdeval) | multithreaded read analysis and manipulation | |

### KegAlign and Batched LASTZ are one pipeline in two tools

KegAlign does seeding and ungapped extension on the GPU and writes a **tarball** — the surviving
HSPs as `.segments` files, the LASTZ command line to run over each, the 2bit sequences, and the
scoring file. Batched LASTZ takes that tarball and performs the gapped extension, on CPU.

The split exists because the two halves want different hardware: the GPU node is released as soon as
seeding is done, rather than sitting idle for the much longer CPU stage.

⚠ **They are versioned independently and must be kept in step.** The interface is the tarball —
`commands.json` and `format.txt` — with no schema anywhere to enforce it, so `tests/test_format_contract.py`
asserts that the writer and the reader still agree. They have drifted before.

⚠ **`tools/batched_lastz/run_lastz_tarball.py` is a deliberate second copy** of the script in the
KegAlign source. It is vendored here because Batched LASTZ must install on CPU-only nodes and the
`kegalign` conda package is a CUDA build. Changing one copy means considering the other.

## Tests

`tests/` holds static checks that need no Galaxy, no container and no GPU, so they run anywhere:

| test | what it pins |
|---|---|
| `test_format_contract.py` | the tarball datatype contract between `kegalign` and `batched_lastz` |
| `test_param_paths.py` | every `$section.…` in a tool's command resolves to a declared parameter |
| `test_select_defaults.py` | a single-select `<param>` marks at most one option `selected="true"` |
| `test_batched_lastz_failures.py` | a failed LASTZ command makes the tool exit nonzero |

```bash
pip install pytest
python -m pytest tests/ -q
```

Each exists because the thing it checks went wrong once. They are cheap; run them before pushing.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

- **Bump the version when you change a tool.** `@TOOL_VERSION@` tracks the wrapped software;
  `@VERSION_SUFFIX@` increments for wrapper-only changes and resets to `0` when `@TOOL_VERSION@`
  moves. Planemo's `ShedVersion` linter fails the build otherwise, because the Tool Shed cannot tell
  two revisions apart at the same version.
- ⚠ **A macro shared between tools may be a symlink** — `tools/batched_lastz/alignment_type_option.xml`
  points at the `kegalign` copy. Editing it changes the rendered XML of *both* tools, but `git diff`
  reports one path, and CI derives its changed-tool list from that. Bump both.
- Python is linted with `flake8` (see `setup.cfg`); tool XML with `planemo shed_lint`.

`/run-all-tool-tests branch=release_25.1 fork=galaxyproject` triggers the weekly CI against a chosen
Galaxy branch. It needs the `PAT` secret.

## Reporting problems

Tool bugs: open an issue here. Problems with the shared CI workflows themselves belong upstream with
the [IUC](https://github.com/galaxyproject/tools-iuc), which this repository's layout and workflows
are derived from.
