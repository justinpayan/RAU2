#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3
NOISE_MULTIPLIER=$4

JN=cm_noise_${DSET_NAME}_${ALLOC_TYPE}_${NOISE_MULTIPLIER}_${CONF_LEVEL}

sbatch \
--job-name=${JN} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/${JN}.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/${JN}.err \
--mem=16G \
--nodes=1 \
--ntasks=1 \
--time=0-04:00:00 \
--partition=defq \
./submit_noise_calc_metrics.sh ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${NOISE_MULTIPLIER}
