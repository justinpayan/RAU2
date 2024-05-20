import argparse
import numpy as np
import os
import pickle
import sys

from allocation_code import solve_usw_gurobi, solve_gesw, \
    solve_cvar_usw, solve_cvar_gesw, solve_adv_usw, solve_adv_gesw, solve_cvar_usw_gauss, solve_cvar_gesw_gauss
from scipy.stats import chi2

dset_name_map = {"aamas1": "AAMAS1", "aamas2": "AAMAS2", "aamas3": "AAMAS3",
                 "gauss_aamas1": "AAMAS1", "gauss_aamas2": "AAMAS2", "gauss_aamas3": "AAMAS3",
                 "ads": "Advertising", "cs": "cs"}

dset_outname_map = {"aamas1": "AAMAS1", "aamas2": "AAMAS2", "aamas3": "AAMAS3",
                    "gauss_aamas1": "gAAMAS1", "gauss_aamas2": "gAAMAS2", "gauss_aamas3": "gAAMAS3",
                    "ads": "Advertising", "cs": "cs"}


def load_dset(dset_name, data_dir, seed, mode='compute_alloc', small_sample=False):
    sample_frac = .2
    # if mode == "time" or small_sample:
    #     sample_frac = .1
    rng = np.random.default_rng(seed=seed)

    if dset_name.startswith("aamas"):
        idx = int(dset_name[-1])
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_%d.npy" % idx))
        coi_mask = np.load(os.path.join(data_dir, "AAMAS", "coi_mask_%d.npy" % idx))
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "prob_up_%d.npy" % idx))
        variances = None

        nrevs = central_estimate.shape[0]
        npaps = central_estimate.shape[1]

        chosen_revs = rng.choice(range(nrevs), int(sample_frac * nrevs))
        chosen_paps = rng.choice(range(npaps), int(sample_frac * npaps))

        central_estimate = central_estimate[chosen_revs, :][:, chosen_paps]
        coi_mask = coi_mask[chosen_revs, :][:, chosen_paps]
        groups = groups[chosen_paps]

        cs = [3, 2, 2]
        ls = [15, 15, 10]
        covs_lb = cs[idx - 1] * np.ones(central_estimate.shape[1])
        covs_ub = covs_lb
        loads = ls[idx - 1] * np.ones(central_estimate.shape[0])

        rhs_bd_per_group = pickle.load(open(os.path.join(data_dir, "AAMAS", "delta_to_normal_bd_%d.pkl" % idx), 'rb'))

    elif dset_name.startswith("gauss_aamas"):
        idx = int(dset_name[-1])
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_%d.npy" % idx))
        variances = np.load(os.path.join(data_dir, "AAMAS", "zeta_matrix_%d.npy" % idx))
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_%d.npy" % idx))
        coi_mask = np.load(os.path.join(data_dir, "AAMAS", "coi_mask_%d.npy" % idx))

        nrevs = central_estimate.shape[0]
        npaps = central_estimate.shape[1]

        chosen_revs = rng.choice(range(nrevs), int(sample_frac * nrevs))
        chosen_paps = rng.choice(range(npaps), int(sample_frac * npaps))

        central_estimate = central_estimate[chosen_revs, :][:, chosen_paps]
        coi_mask = coi_mask[chosen_revs, :][:, chosen_paps]
        groups = groups[chosen_paps]
        variances = variances[chosen_revs, :][:, chosen_paps]

        cs = [3, 2, 2]
        ls = [15, 15, 10]
        covs_lb = cs[idx - 1] * np.ones(central_estimate.shape[1])
        covs_ub = covs_lb
        loads = ls[idx - 1] * np.ones(central_estimate.shape[0])

        ngroups = len(set(groups))

        rhs_bd_per_group = {}
        for delta in [.3, .2, .1, .05, .01]:
            rhs_bd_per_group[delta] = []
            for gidx in range(ngroups):
                gmask = np.where(groups == gidx)[0]
                c_value = np.sum(coi_mask[:, gmask])
                rhs_bd_per_group[delta].append(np.sqrt(chi2.ppf(1 - (delta / ngroups), df=c_value)))

    # elif dset_name == "ads":
    #     central_estimate = np.load(os.path.join(data_dir, "Advertising", "mus.npy"))
    #     variances = np.load(os.path.join(data_dir, "Advertising", "sigs.npy"))
    #     coi_mask = np.load(os.path.join(data_dir, "Advertising", "coi_mask.npy"))
    #
    #     covs_lb = np.zeros(central_estimate.shape[1]) # ad campaigns have no lower bounds
    #     covs_ub = 100*np.ones(central_estimate.shape[1])
    #     loads = np.ones(central_estimate.shape[0]) # Each user impression can only have 1 ad campaign
    #     groups = np.load(os.path.join(data_dir, "Advertising", "groups.npy"))
    #
    #     ngroups = len(set(groups))
    #     rhs_bd_per_group = {}
    #     for delta in [.3, .2, .1, .05]:
    #         rhs_bd_per_group[delta] = []
    #         for gidx in range(ngroups):
    #             gmask = np.where(groups == gidx)[0]
    #             c_value = np.sum(coi_mask[:, gmask])
    #             rhs_bd_per_group[delta].append(chi2.ppf(1 - (delta / ngroups), df=c_value))
    #
    # elif dset_name == "cs":
    #     rhs_bd_per_group = pickle.load(open(os.path.join(data_dir, "cs", "delta_to_normal_bd.pkl"), 'rb'))
    #     central_estimate = (np.load(os.path.join(data_dir, "cs", "asst_scores.npy"))+5)/6
    #     coi_mask = np.load(os.path.join(data_dir, "cs", "coi_mask.npy"))
    #
    #     covs_lb = 2 * np.ones(central_estimate.shape[1])
    #     covs_ub = 2 * np.ones(central_estimate.shape[1])
    #     loads = 20 * np.ones(central_estimate.shape[0])
    #     groups = np.load(os.path.join(data_dir, "cs", "groups.npy"))
    #     variances = None

    covs_lb = np.minimum(covs_lb, np.sum(coi_mask, axis=0))

    return central_estimate, variances, covs_lb, covs_ub, loads, groups, coi_mask, rhs_bd_per_group


# If doing on cs or aamas, assume the central_estimate are the parameters of a multivariate Bernoulli
# else, assume Gaussian.
def get_samples(central_estimate, variances, dset_name, num_samples=100, noise_multiplier=1.0, seed=0, paired=False):
    rng = np.random.default_rng(seed=seed)
    if dset_name.startswith("aamas") or dset_name == 'cs':
        samples = [rng.uniform(size=central_estimate.shape) < central_estimate for _ in range(num_samples)]
        return samples
    else:
        std_devs = np.sqrt(variances)

        if paired:
            # samples = [rng.normal(central_estimate, variances*noise_multiplier) for _ in range(num_samples)]
            # left_only = []
            # for s in samples:
            #     left_only.append(central_estimate - np.abs(central_estimate - s))
            # return left_only
            first_half = [rng.normal(central_estimate, std_devs * noise_multiplier) for _ in
                          range((num_samples // 2) + 1)]
            second_half = []
            for s in first_half:
                second_half.append(2 * central_estimate - s)
            return first_half + second_half
        else:
            return [rng.normal(central_estimate, std_devs * noise_multiplier) for _ in range(num_samples)]


def main(args):
    dset_name = args.dset_name
    alloc_type = args.alloc_type
    conf_level = args.conf_level
    adv_method = args.adv_method
    mode = args.mode
    noise_multiplier = args.noise_multiplier
    save_with_noise_multiplier = args.save_with_noise_multiplier
    n_samples = args.n_samples
    seed = args.seed

    base_dir = "/mnt/nfs/scratch1/jpayan/RAU2"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")

    fname_base = os.path.join(output_dir, dset_outname_map[dset_name], "%s" % alloc_type)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        fname_base += ("_%.2f" % conf_level)

    if save_with_noise_multiplier:
        fname_base += ("_%.2f" % noise_multiplier)

    fname_base += "_%d" % seed

    fname = fname_base + "_alloc.npy"
    if not os.path.exists(fname):
        # small_sample = dset_name.startswith("gauss") and alloc_type == "adv_gesw"
        small_sample = False

        central_estimate, variances, covs_lb, covs_ub, loads, groups, coi_mask, rhs_bd_per_group = load_dset(dset_name,
                                                                                                             data_dir,
                                                                                                             seed, mode, small_sample)

        print("Loaded dataset %s, computing %s allocation" % (dset_name, alloc_type), flush=True)

        # If we are wanting exp_usw_max or exp_gesw_max, we can just compute those using the central estimates.
        # Save the results to outputs/{AAMAS, Advertising, cs}
        if alloc_type == "exp_usw_max":
            alloc = solve_usw_gurobi(central_estimate, covs_lb, covs_ub, loads, coi_mask)
        elif alloc_type == "exp_gesw_max":
            alloc = solve_gesw(central_estimate, covs_lb, covs_ub, loads, groups, coi_mask)

        if alloc_type.startswith("cvar"):
            value_samples = get_samples(central_estimate, variances, dset_name, num_samples=n_samples,
                                        noise_multiplier=noise_multiplier, seed=seed, paired=True)
            print(value_samples[10][10, 10])

        if alloc_type == "cvar_usw" or alloc_type == "cvar_gesw":
            # if dset_name.startswith("gauss"):
            #     if alloc_type == "cvar_usw":
            #         alloc = solve_cvar_usw_gauss(central_estimate, variances, covs_lb, covs_ub, loads, conf_level, coi_mask)
            #     elif alloc_type == "cvar_gesw":
            #         alloc = solve_cvar_gesw_gauss(central_estimate, variances, covs_lb, covs_ub, loads, conf_level, groups, coi_mask)
            # else:
            if alloc_type == "cvar_usw":
                alloc = solve_cvar_usw(covs_lb, covs_ub, loads, conf_level, value_samples, coi_mask)
            elif alloc_type == "cvar_gesw":
                alloc = solve_cvar_gesw(covs_lb, covs_ub, loads, conf_level, value_samples, groups, coi_mask)

        timestamps, obj_vals = None, None

        if alloc_type == "adv_usw":
            delta = conf_level
            alloc, timestamps, obj_vals = solve_adv_usw(central_estimate, variances, covs_lb, covs_ub, loads,
                                                        rhs_bd_per_group[delta], coi_mask, groups, method=adv_method)

        elif alloc_type == "adv_gesw":
            delta = conf_level
            if adv_method == "IQP":
                adv_method = "SubgradAsc"
            if variances is not None:
                variances *= variances
            alloc, timestamps, obj_vals = solve_adv_gesw(central_estimate, variances, covs_lb, covs_ub, loads,
                                                         rhs_bd_per_group[delta], coi_mask, groups, method=adv_method)

        if mode == "time" and timestamps is not None:
            print("Saving out timestamps and objective values for iterations")
            timestamp_fname = os.path.join(output_dir, dset_outname_map[dset_name],
                                           "%s_%s_%.2f_timestamps.pkl" % (alloc_type, adv_method, conf_level))
            pickle.dump(timestamps, open(timestamp_fname, 'wb'))
            obj_vals_fname = os.path.join(output_dir, dset_outname_map[dset_name],
                                          "%s_%s_%.2f_obj_vals.pkl" % (alloc_type, adv_method, conf_level))
            pickle.dump(obj_vals, open(obj_vals_fname, 'wb'))

        if mode == "save_alloc":
            print("Saving allocation", flush=True)

            fname_base = os.path.join(output_dir, dset_outname_map[dset_name], "%s" % alloc_type)

            if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
                fname_base += ("_%.2f" % conf_level)

            if save_with_noise_multiplier:
                fname_base += ("_%.2f" % noise_multiplier)

            fname_base += "_%d" % seed

            np.save(fname_base + "_alloc.npy", alloc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")
    parser.add_argument("--conf_level", type=float, default=0.9)
    parser.add_argument("--adv_method", type=str, default="IQP")
    parser.add_argument("--mode", type=str, default='save_alloc')
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--save_with_noise_multiplier", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=31345)

    args = parser.parse_args()
    main(args)
