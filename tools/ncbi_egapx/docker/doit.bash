#!/usr/bin/env bash

egapx_github_repo="https://github.com/ncbi/egapx.git"
quay_robot_server="${QUAY_ROBOT_SERVER:-quay.io}"
quay_robot_username="${QUAY_ROBOT_USERNAME:-galaxy+egapx}"
## Provide the quay.io robot token via the environment, e.g.
##   export QUAY_ROBOT_PASSWORD='...'
## Never commit the token to the repo.
: "${QUAY_ROBOT_PASSWORD:?Set QUAY_ROBOT_PASSWORD to the quay.io robot token}"

printf '%s' "$QUAY_ROBOT_PASSWORD" | docker login --username "$quay_robot_username" --password-stdin "$quay_robot_server"

for image_id in $(docker images --format json | jq -r '.ID'); do
    docker rmi --force "$image_id"
done
docker system prune -a -f

docker images

egapx_versions=$(git ls-remote --tags --refs "$egapx_github_repo" \
    | cut -d / -f 3 \
    | sed -e 's/^v//' \
    | sort -u \
    | tr '\n' ' ' \
    | sed -e 's/ $//')

for egapx_version in $egapx_versions; do
    #readme_url="https://raw.githubusercontent.com/ncbi/egapx/refs/tags/v$egapx_version/README.md"
    #curl -sLO "$readme_url"
    #egrep '^- Nextflow' README.md
    #egrep '^- Python' README.md

    # no longer create augmented container for 0.1
    if [[ "$egapx_version" =~ ^0\.1\. ]]; then
        continue
    fi

    # no longer create augmented container for 0.2
    if [[ "$egapx_version" =~ ^0\.2- ]]; then
        continue
    fi

    # no longer create augmented container for 0.3
    if [[ "$egapx_version" =~ ^0\.3\. ]]; then
        continue
    fi

    # no longer create augmented container for 0.4
    if [[ "$egapx_version" =~ ^0\.4\. ]]; then
        continue
    fi

    # no longer create augmented container for 0.5.0
    if [[ "$egapx_version" =~ ^0\.5\.0 ]]; then
        continue
    fi

    echo "$egapx_version"

    docker buildx build --progress=plain --build-arg "EGAPx_TAG=$egapx_version" --tag "quay.io/galaxy/egapx:$egapx_version" .
    docker push "quay.io/galaxy/egapx:$egapx_version"

done

docker logout "$quay_robot_server"
