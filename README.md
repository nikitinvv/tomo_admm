# tomo_admm
Tomography solver with ADMM 


## Set path to nvcc compiler, e.g.

export CUDACXX=/local/cuda-12.1/bin/nvcc

## Create a new conda evnironment and install dependencies

conda create -n tomo_admm -c conda-forge scikit-build swig cupy dxchange

conda activate tomo_admm

## Clone the package

git clone https://github.com/nikitinvv/tomo_admm

## Installation from source

cd tomo_admm; pip install .

## Tests

Check folder tests/:

1) test_adjoint.py - the adjoint test

2) test_tomo.py - reconstruction by the cg method

3) test_admm.py - reconstruction by the cg method
