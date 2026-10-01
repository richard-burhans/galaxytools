ARG EGAPx_TAG=latest
FROM docker.io/ncbi/egapx:$EGAPx_TAG
ARG EGAPx_TAG

# - install OS updates
# - install venv module for python3
RUN --mount=type=bind,source=assets,target=/root/bin <<UPDATE
#!/usr/bin/env bash
# Fail the layer on any error so a flaky Ubuntu mirror stops the build instead
# of silently shipping an image that is missing python3-venv and later breaks
# in bootstrap.bash.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# The base ncbi/egapx image points apt at the AWS EC2 regional mirror
# (us-east-2.ec2.archive.ubuntu.com). That mirror is a poor fit off-EC2 and has
# been returning persistent 503s; repoint at the canonical archive.ubuntu.com.
# Handles both classic sources.list and Noble's deb822 *.sources.
sed -i -E 's#https?://[a-z0-9.-]*\.ec2\.(archive|ports)\.ubuntu\.com#http://archive.ubuntu.com#g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true
# apt-get update only warns on a flaky mirror, so retry update+install together
# and only accept success when the package is actually fetched and installed.
ok=0
for i in 1 2 3 4 5; do
    if apt-get update && apt-get -y install --no-install-recommends --fix-missing python3-venv; then
        ok=1
        break
    fi
    echo "apt update/install failed (attempt $i), retrying..." >&2
    sleep 10
done
[ "$ok" = 1 ] || { echo "error: could not install python3-venv after retries" >&2; exit 1; }
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
