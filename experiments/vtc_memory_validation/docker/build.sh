#!/usr/bin/env bash
set -euo pipefail

docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "${docker_dir}/.." && pwd)"
image="${VTC_DOCKER_IMAGE:-vtc-memory-validation:cu129}"

docker build \
  --file "${docker_dir}/Dockerfile" \
  --tag "${image}" \
  "${experiment_dir}"
