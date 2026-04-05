#!/bin/bash
set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Use container-shipped verl (from mldev-images)
python -c "import verl; print(f'verl version: {verl.__version__}')"
pip freeze | grep verl
pip freeze | grep torch
pip freeze | grep transformers

pip install tensordict

# Add EOPD module to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
python -c "from eopd import losses; print('EOPD module imported successfully')"
python -c "from opsd import batch_builder; print('OPSD batch_builder imported successfully')"
