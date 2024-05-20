#!/usr/bin/env bash

module load gurobi/1001

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3
NOISE_MULTIPLIER=$4
SEED=$5

python ../src/compute_allocations.py --dset_name $DSET_NAME --alloc_type $ALLOC_TYPE --seed $SEED --n_samples 1000 --conf_level $CONF_LEVEL --noise_multiplier $NOISE_MULTIPLIER --save_with_noise_multiplier 1