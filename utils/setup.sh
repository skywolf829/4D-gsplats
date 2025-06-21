#!/bin/bash

# Setup pypi index urls
#export PIP_EXTRA_INDEX_URL="https://pypi.org/your/pypi/index/simple"

# ask mamba to show folder name instead of tag or abs path for envs in non default location.
mamba config --set env_prompt '({name}) '

mamba env update -q $CONDA_DEBUG_FLAG --prefix .venv/ --file "utils/conda.yaml" || exit -1
mamba activate .venv/ || exit -1
pip install uv
uv pip install -e ".[tests]"

mamba activate .venv/ || exit -1

echo "[setup] done!"
