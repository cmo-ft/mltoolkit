#!/bin/bash
# python -m pip install --user htcondor 

BASEDIR=$(dirname "$BASH_SOURCE")
BASEDIR=$(realpath $BASEDIR)
export PATH=${PATH}:${BASEDIR}/bin/:
export PYTHONPATH=${PYTHONPATH}:${BASEDIR}:
export MLTOOLKIT_PATH=${BASEDIR}
