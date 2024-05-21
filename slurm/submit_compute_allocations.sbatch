#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3
SEED=$4

JN=${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}_${SEED}

sbatch \
--job-name=${JN} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/${JN}_alloc.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/${JN}_alloc.err \
--mem=64G \
--nodes=1 \
--ntasks=1 \
--time=0-12:00:00 \
--partition=defq \
./submit_compute_allocations.sh ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${SEED}
