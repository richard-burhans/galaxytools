# EGAPx workflow integration test

`egapx-prepare-execute.gxwf.yml` chains the two EGAPx tools so the local cache
produced by `ncbi_egapx_prepare_input` flows directly into `ncbi_egapx_execute`:

```
config_yaml ─► prepare_input (input_style=history) ─► output (directory cache)
      │                                                      │
      └─► execute.input_config         execute.input_cache ◄─┘ ─► complete.genomic.gff
```

A standalone tool test cannot feed one tool's output into another, so this is
the only way to give `execute` a populated `directory` cache (its content lives
in `extra_files_path`, which is only staged when the dataset is produced by an
upstream step). The standalone `ncbi_egapx_execute` tool test is therefore
marked `expect_test_failure="true"`; this workflow provides its real coverage.

## Running the test

Run it against the **workflow file** (not the `-tests.yml`, and not a
directory). From this `tools/ncbi_egapx/` directory:

```bash
planemo test \
    --biocontainers \
    --extra_tools ncbi_egapx_prepare_input.xml \
    --extra_tools ncbi_egapx_execute.xml \
    test-workflows/egapx-prepare-execute.gxwf.yml
```

- `--extra_tools` is **required**: when planemo tests a *workflow* it does not
  auto-install the local tools the way it does for a *tool* test. Without it the
  run errors with `required tools are not installed` before any step is invoked.
- planemo auto-discovers the adjacent `egapx-prepare-execute.gxwf-tests.yml`,
  whose job uses `../test-data/input.yaml`.
- The run needs network access and the `quay.io/galaxy/egapx` container:
  `prepare_input` runs with `--download-needed` (downloads reference data) and
  `execute` runs a full annotation. It is **not** runnable in an offline
  CI/toolshed environment.

## Packaging

`test-workflows/` is excluded from the ToolShed package via `.shed.yml`; these
files are integration-test artifacts, not Galaxy tool files.
