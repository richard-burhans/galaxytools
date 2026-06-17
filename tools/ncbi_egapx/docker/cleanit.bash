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

