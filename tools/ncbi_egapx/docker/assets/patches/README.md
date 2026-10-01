# EGAPx source patches

`bootstrap.bash` clones EGAPx fresh from GitHub at the requested tag and then
applies every `*.patch` in this directory to the checkout, in filename order.

Each patch must either apply cleanly or already be applied; if a patch fails to
apply (e.g. it has gone stale because the change landed upstream or the
surrounding code changed), the container build fails instead of silently
shipping an unpatched image. When a patch becomes obsolete, delete it.

Patches are standard `git diff` output and are applied with `git apply` (so the
paths use the usual `a/`,`b/` prefixes).

## Patches

- **0001-ftpdownloader-resilient-download.patch** — hardens `FtpDownloader` in
  `ui/egapx.py`: connection timeout, size checks, resume, and retries on
  flaky FTP transfers. Rebased onto EGAPx `v1.0.1` (the surrounding code moved
  and `FTP(...)` is now `ftplib.FTP(...)`); still not upstream.

- **0002-export-argument-order.patch** — fixes the `export(...)` call in
  `nf/ui.nf`. Nextflow binds process inputs by position, and in `v1.0.1` the
  call passes four channels out of order relative to the `export` process
  declaration. As a result the filtered protein alignments (`align.asn`) were
  published under `stats/rnaseq_long/` instead of `filtered_protein_alignments/`,
  and with long reads the minimap2 stats landed in `filtered_protein_alignments/`.
  The patch reorders the arguments to match the declaration. Reported upstream as
  ncbi/egapx#274 (fix: ncbi/egapx#275); delete this patch once a release includes it.

## Retired patches

- **0002-relocatable-sra-read-cache.patch** — *retired for EGAPx `v1.0.1`*.
  This patch made the downloaded SRA read cache relocatable: the cache used to
  record absolute paths to the downloaded read files in
  `<cache>/sra_dir/runs.yaml`, so moving the cache directory (for example, when
  a Galaxy `directory` dataset is handed from one job to another) broke the STAR
  step with "No such file or directory". As of `v1.0.1` the same fix is upstream
  — `download_sra_query` now writes cache-relative paths and `read_cache`
  rebases both relative and legacy-absolute entries onto the current cache
  directory by file name. The patch is therefore no longer needed and has been
  removed.
