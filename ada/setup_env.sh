#!/bin/bash
# One-time environment setup on the cluster (run on the LOGIN node from the repo root):
#     bash ada/setup_env.sh
# Needs Python 3.11 (pandas 3 / numpy 2.4). Uses conda if available, else installs Miniconda in ~/miniconda3.
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    if [ -x "$HOME/miniconda3/bin/conda" ]; then
        source "$HOME/miniconda3/bin/activate"
    else
        echo "Installing Miniconda into ~/miniconda3 ..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        source "$HOME/miniconda3/bin/activate"
    fi
fi

eval "$(conda shell.bash hook)"
conda env list | grep -q "^er " || conda create -y -n er python=3.11
conda activate er
pip install --upgrade pip
pip install pandas==3.0.2 numpy==2.4.4 scipy==1.17.1 pyarrow==25.0.1 rapidfuzz==3.14.6 \
            lightgbm==4.7.0 sparse_dot_topn==1.2.0
python -c "import pandas, numpy, scipy, pyarrow, rapidfuzz, lightgbm, sparse_dot_topn; print('env OK')"
