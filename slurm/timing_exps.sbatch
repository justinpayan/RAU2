#! /bin/bash

DSET_NAME=$1
ALLOC_TYPE=$2
CONF_LEVEL=$3
ADV_USW_METHOD=$4

JSTR=time_${ADV_USW_METHOD}_${DSET_NAME}_${ALLOC_TYPE}_${CONF_LEVEL}

sbatch \
--job-name=${JSTR} \
--output=/mnt/nfs/scratch1/jpayan/logs/RAU2/$JSTR.out \
--error=/mnt/nfs/scratch1/jpayan/logs/RAU2/$JSTR.err \
--mem=32G \
--nodes=1 \
--ntasks=1 \
--time=0-04:00:00 \
--partition=defq \
./timing_exps.sh ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${ADV_USW_METHOD}
