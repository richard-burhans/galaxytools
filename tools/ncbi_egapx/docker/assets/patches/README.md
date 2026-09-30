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
