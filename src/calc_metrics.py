import argparse
import numpy as np
import os
import pickle

from metric_code import compute_usw, compute_gesw, compute_cvar_usw, compute_cvar_gesw, \
    compute_adv_usw_linear, compute_adv_gesw_linear, compute_adv_usw_ellipsoidal, compute_adv_gesw_ellipsoidal
from compute_allocations import get_samples, load_dset, dset_name_map, dset_outname_map


def main(args):
    dset_name = args.dset_name
    alloc_type = args.alloc_type
    conf_level = args.conf_level
    noise_multiplier = args.noise_multiplier
    save_with_noise_multiplier = args.save_with_noise_multiplier
    seed = args.seed

    base_dir = "/mnt/nfs/scratch1/jpayan/RAU2"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")

    # small_sample = dset_name.startswith("gauss") and alloc_type == "adv_gesw"
    small_sample = False

    central_estimate, variances, covs_lb, covs_ub, loads, groups, coi_mask, rhs_bd_per_group = load_dset(dset_name, data_dir, seed, small_sample=small_sample)

    print("Loaded dataset %s, loading %s allocation. Conf level %.2f. Seed %d" % (dset_name, alloc_type, conf_level, seed), flush=True)

    fname_base = os.path.join(output_dir, dset_outname_map[dset_name], "%s" % alloc_type)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        fname_base += ("_%.2f" % conf_level)

        # The allocations based on the expected values dont have any variation across noise multipliers, so we didn't save those fresh.
        if save_with_noise_multiplier:
            fname_base += ("_%.2f" % noise_multiplier)

    fname_base += "_%d" % seed

    alloc_fname = fname_base + "_alloc.npy"

    allocation = np.load(alloc_fname)

    # Save it all in a dictionary, print and dump
    metrics_to_values = {}

    # First get the USW and GESW for this allocation under the expected values
    print("Computing USW and GESW on expected values", flush=True)
    usw = compute_usw(allocation, central_estimate)
    metrics_to_values['usw'] = usw

    gesw = compute_gesw(allocation, central_estimate, groups)
    metrics_to_values['gesw'] = gesw


    print("Done with USW/ESW", flush=True)

    # Now do CVaR of USW/GESW at different confidence levels
    metrics_to_values['cvar_usw'] = {}
    metrics_to_values['cvar_gesw'] = {}
    metrics_to_values['adv_usw'] = {}
    metrics_to_values['adv_gesw'] = {}

    conf_levels = [0.01, 0.3]

    value_samples = get_samples(central_estimate, variances, dset_name, num_samples=10000, noise_multiplier=noise_multiplier, seed=seed)
    print(value_samples[10][10, 10])

    for c in conf_levels:
        print("Calculating cvar usw", flush=True)
        metrics_to_values['cvar_usw'][c] = compute_cvar_usw(allocation, value_samples, c)

        print("Calculating cvar gesw", flush=True)
        metrics_to_values['cvar_gesw'][c] = compute_cvar_gesw(allocation, value_samples, groups, c)

        delta = c
        a = 1
        b = 0

        if int(noise_multiplier) == 1:
            print("Calculating adv usw", flush=True)
            if dset_name.startswith("gauss"):
                adv_usw = compute_adv_usw_ellipsoidal(allocation, central_estimate, variances, rhs_bd_per_group[delta], groups)
            else:
                adv_usw = compute_adv_usw_linear(allocation, central_estimate, coi_mask, rhs_bd_per_group[delta], groups, a_val=a, b_val=b)

            print("Calculating adv gesw", flush=True)
            if dset_name.startswith("gauss"):
                adv_gesw = compute_adv_gesw_ellipsoidal(allocation, central_estimate, variances**2, rhs_bd_per_group[delta], groups)
            else:
                adv_gesw = compute_adv_gesw_linear(allocation, central_estimate, coi_mask, rhs_bd_per_group[delta], groups, a_val=a, b_val=b)

            metrics_to_values['adv_usw'][c] = adv_usw
            metrics_to_values['adv_gesw'][c] = adv_gesw

    print(metrics_to_values, flush=True)

    fname_base = os.path.join(output_dir, dset_outname_map[dset_name], "%s" % alloc_type)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        fname_base += ("_%.2f" % conf_level)

    if save_with_noise_multiplier:
        fname_base += ("_%.2f" % noise_multiplier)

    fname_base += "_%d" % seed

    metric_fname = fname_base + "_metrics.pkl"
    pickle.dump(metrics_to_values, open(metric_fname, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")
    parser.add_argument("--conf_level", type=float, default=0.9)
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--save_with_noise_multiplier", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31345)

    args = parser.parse_args()
    main(args)