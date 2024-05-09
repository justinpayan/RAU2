#!/usr/bin/env bash

module load gurobi/1001

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3

python ../src/calc_metrics.py --dset_name $DSET_NAME --alloc_type $ALLOC_TYPE --conf_level $CONF_LEVEL