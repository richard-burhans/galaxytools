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
  flaky FTP transfers.

- **0002-relocatable-sra-read-cache.patch** — makes the downloaded SRA read
  cache relocatable. `download_sra_query` records absolute paths to the
  downloaded read files in `<cache>/sra_dir/runs.yaml`. When the cache
  directory is moved (for example, when a Galaxy `directory` dataset is handed
  from one job to another), those absolute paths no longer exist and the STAR
  step fails with "No such file or directory". The patch rebases each cached
  read path onto the current cache location (by file name) when the cache is
  read back. This is the candidate fix for an upstream PR to ncbi/egapx.
