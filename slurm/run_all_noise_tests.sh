#!/usr/bin/env bash

for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
  for ALLOC_TYPE in "cvar_usw" "cvar_gesw"; do
    CONF_LEVEL=0.3
    for NOISE_MULTIPLIER in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0; do
      ./noise_tests.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${NOISE_MULTIPLIER}
      sleep .1
    done
  done
done
