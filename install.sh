#!/bin/bash

python setup.py build_ext --inplace
python CVF3D/process/utils_CVF/numba_utils_CVF.py