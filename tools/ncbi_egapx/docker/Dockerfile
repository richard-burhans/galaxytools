ARG EGAPx_TAG=latest
FROM docker.io/ncbi/egapx:$EGAPx_TAG
ARG EGAPx_TAG

# - install OS updates
# - install venv module for python3
RUN --mount=type=bind,source=assets,target=/root/bin <<UPDATE
#!/usr/bin/env bash
# Fail the layer on any error so a flaky Ubuntu mirror (e.g. 503s from the
# EC2 archive) stops the build instead of silently shipping an image that is
# missing python3-venv and later breaks in bootstrap.bash.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# apt-get update can hit transient mirror errors; retry a few times.
for i in 1 2 3 4 5; do
    apt-get update && break
    echo "apt-get update failed (attempt $i), retrying..." >&2
    sleep 10
done
apt-get -y install --no-install-recommends python3-venv
apt-get -y dist-upgrade
apt-get -y autoremove
apt-get -y clean
rm -rf /var/lib/apt/lists/*
UPDATE

# - install latest release of miniforge
# - create conda env for nextflow
# - install EGAPx runner script
RUN --mount=type=bind,source=assets,target=/root/bin <<AUGMENT
#!/usr/bin/env bash
/root/bin/bootstrap.bash $EGAPx_TAG
AUGMENT

# /galaxy/.venv/bin
# /galaxy/miniforge3/envs/nextflow/bin
# /galaxy/miniforge3/condabin
# /img/gp/bin
# /usr/local/sbin
# /usr/local/bin
# /usr/sbin
# /usr/bin
# /sbin
# /bin
# /sratoolkit.3.0.7-ubuntu64/bin

# /img/gp/bin
# /img/gp/third-party/STAR/bin/Linux_x86_64
# $PATH
