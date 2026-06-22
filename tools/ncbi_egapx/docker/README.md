# Augmented EGAPx container

Builds `quay.io/galaxy/egapx:<version>` — the EGAPx release image augmented with
the bits the Galaxy tools need (Nextflow conda env, a Python venv, the EGAPx
runner, resource-config helper, and binary symlinks).

## Build

`doit.bash` builds and pushes a tag for each supported EGAPx release. The quay
robot token comes from the environment (never commit it):

```bash
QUAY_ROBOT_PASSWORD=… ./doit.bash
```

`bootstrap.bash` (run inside the image build) clones EGAPx from GitHub at the
release tag, then applies every patch in `assets/patches/` to the checkout — see
`assets/patches/README.md`. A stale patch fails the build instead of silently
shipping an unpatched image.

## How the Galaxy tools consume this image — NOT via BioContainers

The `ncbi_egapx*` tools reference this image as an **explicit** container:

```xml
<container type="docker">quay.io/galaxy/egapx:@TOOL_VERSION@</container>
```

Galaxy / planemo resolve that by pulling `quay.io/galaxy/egapx:<version>`
directly from quay.io. This image is therefore **not** a BioContainer and does
not flow through the usual bioconda pipeline:

- it is not built from a Conda recipe, so it never appears under
  `quay.io/biocontainers/`;
- the Galaxy Singularity depot (`depot.galaxyproject.org/singularity/`) is fed
  by `singularity-build-bot`, which mirrors **only** `quay.io/biocontainers` —
  it does not watch `quay.io/galaxy`, so this image never auto-appears there.

There is nothing to "wait for" after pushing: as soon as the tag is on quay.io,
the tools can pull it.

## Re-pushing the same tag: bust caches before testing

Tags here are rebuilt in place (e.g. `0.5.2` gets a new digest on each rebuild).
Any host that already pulled that tag has the **old** image cached by name:tag,
so a re-test will silently run the previous build. Force a refresh first:

```bash
# Docker
docker pull quay.io/galaxy/egapx:<version>          # re-pulls the new digest

# Singularity (planemo --biocontainers runs via singularity; .sif keyed by name:tag)
singularity cache clean -f                          # or delete the cached egapx .sif
```

Confirm the host has the intended build by matching the digest to quay.io:

```bash
docker inspect --format '{{index .RepoDigests 0}}' quay.io/galaxy/egapx:<version>
```
