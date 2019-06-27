#!/usr/bin/env bash

cd utils/pyvotkit
python3 setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python3 setup.py build_ext --inplace
cd ../../../
