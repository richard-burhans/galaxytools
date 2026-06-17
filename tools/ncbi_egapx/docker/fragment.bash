#!/usr/bin/env bash

pathname="egapx/ui/egapx.py"
if grep -qx '    biomol_str = " AND biomol_transcript\\\[properties\\] "' "$pathname"; then
    sed -i 's/    biomol_str = " AND biomol_transcript\\\[properties\\] "/    biomol_str = " AND biomol_transcript[properties] "/' "$pathname"
    echo "patched $pathname"
fi

