#!/bin/bash
set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Verify pre-installed verl and dependencies
python -c "import verl; print(f'verl version: {verl.__version__}')"
pip freeze | grep verl
pip freeze | grep torch
pip freeze | grep transformers

pip install tensordict

# Add OPD module to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
python -c "from opd import losses; print('OPD module imported successfully')"
