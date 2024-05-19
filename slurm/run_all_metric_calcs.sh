#!/usr/bin/env bash
#
#for DSET_NAME in "cs" "aamas1" "aamas2" "aamas3"; do
#  for ALLOC_TYPE in "exp_usw_max" "exp_gesw_max"; do
#    ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} 0
#    sleep .1
#  done
#  for ALLOC_TYPE in "cvar_usw" "cvar_gesw" "adv_usw" "adv_gesw"; do
#    for CONF_LEVEL in 0.05 0.1 0.2 0.3; do
#      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
#      sleep .1
#    done
#  done
#done

for SEED in {1..50}; do
  for DSET_NAME in "aamas1" "aamas2" "aamas3" "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
    for ALLOC_TYPE in "exp_usw_max" "exp_gesw_max"; do
      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} 0 $SEED
      sleep .1
    done
    for ALLOC_TYPE in "cvar_usw" "cvar_gesw"; do
      CONF_LEVEL=0.01
      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} $SEED
      sleep .1
    done
    for ALLOC_TYPE in "adv_usw" "adv_gesw"; do
      CONF_LEVEL=0.3
      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} $SEED
      sleep .1
    done
  done
done

#for DSET_NAME in "cs" "aamas1" "aamas2" "aamas3"; do
#  for ALLOC_TYPE in "exp_usw_max" "exp_gesw_max"; do
#    ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} 0
#    sleep .1
#  done
#  for ALLOC_TYPE in "cvar_usw" "cvar_gesw" "adv_usw" "adv_gesw"; do
#    for CONF_LEVEL in 0.05 0.1 0.2 0.3; do
#      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
#      sleep .1
#    done
#  done
#done

#
#for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
#  for ALLOC_TYPE in "exp_usw_max" "exp_gesw_max"; do
#    ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} 0
#    sleep .1
#  done
#  for ALLOC_TYPE in "cvar_usw" "cvar_gesw" "adv_usw" "adv_gesw"; do
#    for CONF_LEVEL in 0.05 0.1 0.2 0.3; do
#      ./submit_calc_metrics.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL}
#      sleep .1
#    done
#  done
#done