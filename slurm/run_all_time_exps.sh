#!/usr/bin/env bash

ALLOC_TYPE="adv_usw"
CONF_LEVEL=0.3
for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
  for ADV_METHOD in "IQP" "ProjGD" "SubgradAsc"; do
    ./timing_exps.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${ADV_METHOD}
  done
done

ALLOC_TYPE="adv_gesw"
CONF_LEVEL=0.3
for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
  for ADV_METHOD in "ProjGD" "SubgradAsc"; do
    ./timing_exps.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${ADV_METHOD}
  done
done