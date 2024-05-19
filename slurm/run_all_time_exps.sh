#!/usr/bin/env bash

DSET_NAME="gauss_aamas1"
ALLOC_TYPE="adv_usw"
CONF_LEVEL=0.3
ADV_USW_METHOD="IQP"

./timing_exps.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${ADV_USW_METHOD}