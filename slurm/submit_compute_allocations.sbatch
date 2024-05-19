#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3

sbatch \
--job-name=${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}_alloc.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}_alloc.err \
--mem=128G \
--nodes=1 \
--ntasks=1 \
--time=1-00:00:00 \
--partition=longq \
./submit_compute_allocations.sh ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
