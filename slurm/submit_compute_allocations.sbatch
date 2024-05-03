#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2

sbatch \
--job-name=${DSET_NAME}_${ALLOC_TYPE} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_alloc.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/${DSET_NAME}_${ALLOC_TYPE}_alloc.err \
--nodes=1 \
--ntasks=1 \
--time=0-6:00:00 \
--partition=defq \
./submit_compute_allocations.sh ${DSET_NAME} ${ALLOC_TYPE}
