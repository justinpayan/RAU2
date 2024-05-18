#!/usr/bin/env bash

module load gurobi/1001

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3
ADV_USW_METHOD=$4

python ../src/compute_allocations.py --dset_name $DSET_NAME --alloc_type $ALLOC_TYPE --conf_level $CONF_LEVEL --adv_usw_method $ADV_USW_METHOD --mode "time"