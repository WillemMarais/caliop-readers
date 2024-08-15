#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"/..

# Get the conda path
conda_dirP_str=$(dirname $(dirname $(which conda)))

# If the conda environment already exists, remove it
conda env list \
    | grep "\"${conda_dirP_str}/envs/caliop-readers\"" > /dev/null 2>&1 \
    || conda remove --name caliop-readers --all --yes

# Create environment
# conda create --name caliop-readers "python<3.12" ndcctools pyhdf xarray netcdf4 pip cython -y -c conda-forge
conda create --name caliop-readers "python<3.12" pip jupyterlab -y -c conda-forge

# Activate the cluster-hsrl-processing environment
eval "$(conda shell.bash hook)"
conda activate caliop-readers

# Get the pip path to the new env
miniconda_dirP=$(dirname $(dirname $(which python)))
pip_fileP=${miniconda_dirP}/bin/pip

# # Install caliop-readers
${pip_fileP} install -e ${SCRIPTPATH}
