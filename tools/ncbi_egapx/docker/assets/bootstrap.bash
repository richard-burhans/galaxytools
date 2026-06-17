#sh !/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o posix

egapx_tag=""
if [ $# -eq 1 ]; then
    egapx_tag="$1"
fi

project_root="/galaxy"
version_file="$project_root/version"
miniforge_root="$project_root/miniforge3"
repo="conda-forge/miniforge"

if [ ! -e "$project_root" ]; then
    mkdir -m 0755 -p "$project_root"
fi

echo "$egapx_tag" > "$version_file"

if [ ! -e "$miniforge_root" ]; then
    ## get tag_name of latest release of miniforge
    tag=$(curl --silent "https://api.github.com/repos/conda-forge/miniforge/releases/latest" \
            | python3 -m json.tool \
            | awk '$1 == "\"tag_name\":" {print $2}' \
            | cut -d\" -f2)

    if [ -z "$tag" ]; then
        echo "error: unable to get latest release of miniforge"
        exit 1
    fi
fi
    
## download and verify miniforge
filename="Miniforge3-${tag}-$(uname -s)-$(uname -m).sh"
url="https://github.com/conda-forge/miniforge/releases/download/$tag/$filename"

curl --silent --location --output "$project_root/$filename" "$url"
curl --silent --location --output "$project_root/${filename}.sha256" "${url}.sha256"

expected_checksum=$(awk '{print $1}' "$project_root/${filename}.sha256")
actual_checksum=$(openssl dgst -sha256 "$project_root/$filename" | awk '{print $2}')

if [ "$expected_checksum" != "$actual_checksum" ]; then
    echo "error: checksum mismatch"
    echo "  expected: $expected_checksum"
    echo "  awk '{print \$1}' \"$project_root/${filename}.sha256\""
    echo "    actual: $actual_checksum"
    exit 1
fi

## install miniforge
bash "$project_root/$filename" -b -p "$miniforge_root"

cat > "$project_root/env.bash" << EOF
source "$miniforge_root/etc/profile.d/conda.sh"
export MAMBA_EXE="$miniforge_root/bin/mamba"
export MAMBA_ROOT_PREFIX="$miniforge_root"
__mamba_setup="\$("\$MAMBA_EXE" shell hook --shell bash --root-prefix "\$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__mamba_setup"
else
    alias mamba="\$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
EOF

source "$project_root/env.bash"

nextflow_env_installed=$(mamba env list | egrep '^nextflow\b' || true)
if [ -z "$nextflow_env_installed" ]; then
    mamba create \
        --name nextflow \
        --channel conda-forge \
        --channel bioconda \
        --override-channels \
        --strict-channel-priority \
        --yes \
        git \
        "nextflow==23.10.1" \
        "python>=3.11"

    echo "mamba activate nextflow" >> "$project_root/env.bash"
    echo "export NXF_DISABLE_CHECK_LATEST=true" >> "$project_root/env.bash"
fi

mamba activate nextflow
if [ ! -e "$project_root/egapx" ]; then
    cd "$project_root"
    if [ -n "$egapx_tag" ]; then
        git clone --branch "v$egapx_tag" https://github.com/ncbi/egapx.git
    else
        git clone https://github.com/ncbi/egapx.git
    fi
fi

if [ ! -e "$project_root/.venv" ]; then
    python3 -m venv "$project_root/.venv"
    echo "source \"$project_root/.venv/bin/activate\"" >> "$project_root/env.bash"
fi

source "$project_root/.venv/bin/activate"
pip install --upgrade pip

cd "$project_root/egapx"

if [ -e "requirements.txt" ]; then
    pip install -r "requirements.txt"
fi

python3 ui/egapx.py ./examples/input_D_farinae_small.yaml -o example_out || true

cat > "$project_root/egapx/egapx_config/galaxy.config" <<EOF
docker.enabled = false
env {
    GP_HOME = "/img/gp/"
    PATH = "/img/gp/bin/:/img/gp/bin/third-party/:/img/gp/bin/third-party/sratoolkit.3.0.7-ubuntu64/bin/:/netmnt/vast01/egapx/bin/gp/:\$PATH"
}
EOF

ln "$project_root/egapx/egapx_config/galaxy.config" "$project_root/egapx//ui/assets/config/user/galaxy.config"
rm -f "$project_root/$filename"{,.sha256}
deactivate
mamba deactivate

# patch nextflow
find /galaxy -type f -name nextflow -print0 2>/dev/null |
    while IFS= read -r -d '' pathname; do
        if ! grep -qx 'unset _JAVA_OPTIONS' "$pathname"; then
            sed -i '/unset JAVA_TOOL_OPTIONS/a unset _JAVA_OPTIONS' "$pathname"
            echo "patched $pathname"
        fi
    done

if [ ! -e "$project_root/scripts" ]; then
    mkdir -m 0755 -p "$project_root/scripts"
fi

cp /root/bin/galaxy-resource-config.py "$project_root/scripts"
chown root:root "$project_root/scripts/galaxy-resource-config.py"
chmod 755 "$project_root/scripts/galaxy-resource-config.py"
