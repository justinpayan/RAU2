#!/usr/bin/env bash

for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
  for ALLOC_TYPE in "cvar_usw" "cvar_gesw"; do
    CONF_LEVEL=0.3
    for NOISE_MULTIPLIER in 1.0 100.0 200.0 300.0 400.0 500.0 600.0 700.0 800.0 900.0 1000.0; do
      ./noise_tests.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${NOISE_MULTIPLIER}
      sleep .1
    done
  done
done
