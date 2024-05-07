#!/usr/bin/env bash

#for DSET_NAME in "ads" "cs" "aamas1" "aamas2" "aamas3"; do
#  for ALLOC_TYPE in "exp_usw_max" "exp_gesw_max"; do
#    ./submit_compute_allocations.sbatch ${DSET_NAME} ${ALLOC_TYPE} 0
#    sleep 1
#  done
#  for ALLOC_TYPE in "cvar_usw" "cvar_gesw"; do
#    for CONF_LEVEL in 0.7 0.8 0.9 0.95; do
#      ./submit_compute_allocations.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
#      sleep 1
#    done
#  done
#done

#DSET_NAME="cs"
#for CONF_LEVEL in 0.7 0.8 0.9 0.95; do
#  ./submit_compute_allocations.sbatch ${DSET_NAME} adv_usw ${CONF_LEVEL}
#  sleep 1
#  ./submit_compute_allocations.sbatch ${DSET_NAME} adv_gesw ${CONF_LEVEL}
#  sleep 1
#done

for DSET_NAME in "ads" "aamas1" "aamas2" "aamas3"; do
  for CONF_LEVEL in 0.7 0.8 0.9 0.95; do
    ./submit_compute_allocations.sbatch ${DSET_NAME} adv_usw ${CONF_LEVEL}
    sleep 1
  done
done