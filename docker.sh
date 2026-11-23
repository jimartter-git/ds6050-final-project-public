#!/usr/bin/env bash
# Run a command in the dev container, using the GPU when Docker can reach one.
# Usage: bash docker.sh [command...]   (no command opens a shell)
set -euo pipefail
cd "$(dirname "$0")"
image=ds6050-penn-action

if ! compgen -G 'data/Penn_Action/labels/*.mat' >/dev/null; then
    echo 'Penn Action labels not found; real-data runs need: bash docker.sh python scripts/download_data.py' >&2
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker compose -f docker/docker-compose.yml build
fi

files=(-f docker/docker-compose.yml)
if docker run --rm --gpus all "$image" python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' >/dev/null 2>&1; then
    files+=(-f docker/docker-compose.gpu.yml)
    echo 'GPU detected: container will use CUDA.' >&2
else
    echo 'No usable GPU: container will use CPU.' >&2
fi

exec docker compose "${files[@]}" run --rm dev "$@"
