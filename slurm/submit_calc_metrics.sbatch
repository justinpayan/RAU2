#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3

sbatch \
--job-name=cm_${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}_calc_metrics.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}_calc_metrics.err \
--mem=64G \
--nodes=1 \
--ntasks=1 \
--time=0-04:00:00 \
--partition=defq \
./submit_calc_metrics.sh ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
