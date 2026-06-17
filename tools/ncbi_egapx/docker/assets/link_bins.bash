#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

executable_dirs="
/img/gp/arch
/img/gp/third-party
/img/dev/gp
/img/dev/toolkit
"

symlinks_dir="/img/gp/bin"

# clear and recreate symlinks dir
test -e "$symlinks_dir" && rm -rf "$symlinks_dir"
mkdir -m 0755 -p "$symlinks_dir"

# find executables and create symlinks
cd "$symlinks_dir"

for executable_dir in $executable_dirs; do
    find "$executable_dir" -type f -print0 2>/dev/null |
        while IFS= read -r -d '' pathname; do
            value=$(file "$pathname" 2>/dev/null | grep "executable" || true)
            if [ -n "$value" ]; then
                filename="${pathname##*/}"
                if [ ! -e "$filename" ]; then
                    ln -s "$pathname" "$filename"
                fi
            fi
        done
done

ln -s /img/gp/third-party "$symlinks_dir/third-party"
