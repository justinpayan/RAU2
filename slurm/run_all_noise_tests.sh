#!/usr/bin/env bash

for DSET_NAME in "gauss_aamas1" "gauss_aamas2" "gauss_aamas3"; do
  for ALLOC_TYPE in "cvar_usw" "cvar_gesw"; do
    CONF_LEVEL=0.01
    for NOISE_MULTIPLIER in 1.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0; do
      for SEED in {1..20}; do
        ./noise_tests.sbatch ${DSET_NAME} ${ALLOC_TYPE} ${CONF_LEVEL} ${NOISE_MULTIPLIER} ${SEED}
        sleep .1
      done
    done
  done
done
