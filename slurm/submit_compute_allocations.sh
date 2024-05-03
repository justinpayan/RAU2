#!/usr/bin/env bash

module load gurobi/1001

DSET_NAME=$1
ALLOC_TYPE=$2

python ../src/compute_allocations.py --dset_name $DSET_NAME --alloc_type $ALLOC_TYPE