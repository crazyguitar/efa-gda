#!/bin/bash

set -exo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
sqsh="${DIR}/cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${DIR}/build/src/gda"

srun --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name efa \
  --mpi=pmix \
  --ntasks-per-node=1 \
  "${binary}"
