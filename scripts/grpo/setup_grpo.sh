#!/bin/bash
set -x
set -e

# Verify pre-installed verl and dependencies
python -c "import verl; print(f'verl version: {verl.__version__}')"
pip freeze | grep verl
pip freeze | grep torch
pip freeze | grep transformers

pip install tensordict
