import argparse
import numpy as np
import os
import pickle
import sys
from allocation_code import solve_usw_gurobi, solve_gesw, solve_cvar_usw, solve_cvar_gesw, solve_adv_usw, solve_adv_gesw

dset_name_map = {"aamas1": "AAMAS1", "aamas2": "AAMAS2", "aamas3": "AAMAS3", "ads": "Advertising", "cs": "cs"}


def load_dset(dset_name, data_dir):

    if dset_name.startswith("aamas"):
        idx = int(dset_name[-1])
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_%d.npy" % idx))
        std_devs = np.load(os.path.join(data_dir, "AAMAS", "zeta_matrix_%d.npy" % idx))
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_%d.npy" % idx))
        coi_mask = np.load(os.path.join(data_dir, "AAMAS", "coi_mask_%d.npy" % idx))

        cs = [3, 2, 2]
        ls = [15, 15, 4]
        covs_lb = cs[idx-1] * np.ones(central_estimate.shape[1])
        covs_ub = covs_lb
        loads = ls[idx-1] * np.ones(central_estimate.shape[0])
        rhs_bd_per_group = None

    elif dset_name == "ads":
        central_estimate = np.load(os.path.join(data_dir, "Advertising", "mus.npy"))
        std_devs = np.load(os.path.join(data_dir, "Advertising", "sigs.npy"))
        coi_mask = np.load(os.path.join(data_dir, "Advertising", "coi_mask.npy"))

        covs_lb = np.zeros(central_estimate.shape[1]) # ad campaigns have no lower bounds
        covs_ub = 100*np.ones(central_estimate.shape[1])
        loads = np.ones(central_estimate.shape[0]) # Each user impression can only have 1 ad campaign
        groups = np.load(os.path.join(data_dir, "Advertising", "groups.npy"))
        rhs_bd_per_group = None

    elif dset_name == "cs":
        rhs_bd_per_group = pickle.load(open(os.path.join(data_dir, "cs", "delta_to_normal_bd.pkl"), 'rb'))
        central_estimate = np.load(os.path.join(data_dir, "cs", "asst_scores.npy"))
        coi_mask = np.load(os.path.join(data_dir, "cs", "coi_mask.npy"))

        covs_lb = 2 * np.ones(central_estimate.shape[1])
        covs_ub = 2 * np.ones(central_estimate.shape[1])
        loads = 20 * np.ones(central_estimate.shape[0])
        groups = np.load(os.path.join(data_dir, "cs", "groups.npy"))
        std_devs = None

    covs_lb = np.minimum(covs_lb, np.sum(coi_mask, axis=0))

    return central_estimate, std_devs, covs_lb, covs_ub, loads, groups, coi_mask, rhs_bd_per_group

# If std_devs is None, assume the central_estimate are the parameters of a multivariate Bernoulli
# else, assume Gaussian.
def get_samples(central_estimate, std_devs, num_samples=10):
    rng = np.random.default_rng(seed=0)
    if std_devs is None:
        p = (central_estimate + 5)/6
        samples = [rng.uniform(size=p.shape) < p for _ in range(num_samples)]
        return [6 * vs - 5 for vs in samples]
    else:
        return [rng.normal(central_estimate, std_devs) for _ in range(num_samples)]


def main(args):
    dset_name = args.dset_name
    alloc_type = args.alloc_type
    conf_level = args.conf_level

    base_dir = "/mnt/nfs/scratch1/jpayan/RAU2"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")

    central_estimate, std_devs, covs_lb, covs_ub, loads, groups, coi_mask, rhs_bd_per_group = load_dset(dset_name, data_dir)

    print("Loaded dataset %s, computing %s allocation" % (alloc_type, dset_name), flush=True)

    # If we are wanting exp_usw_max or exp_gesw_max, we can just compute those using the central estimates.
    # Save the results to outputs/{AAMAS, Advertising, cs}
    if alloc_type == "exp_usw_max":
        alloc = solve_usw_gurobi(central_estimate, covs_lb, covs_ub, loads, coi_mask)
    elif alloc_type == "exp_gesw_max":
        alloc = solve_gesw(central_estimate, covs_lb, covs_ub, loads, groups, coi_mask)

    if alloc_type.startswith("cvar"):
        value_samples = get_samples(central_estimate, std_devs)

    if alloc_type == "cvar_usw":
        alloc = solve_cvar_usw(covs_lb, covs_ub, loads, conf_level, value_samples, coi_mask)
    elif alloc_type == "cvar_gesw":
        alloc = solve_cvar_gesw(covs_lb, covs_ub, loads, conf_level, value_samples, groups, coi_mask)

    if alloc_type == "adv_usw":
        delta = np.round(1-conf_level, decimals=2)
        if std_devs is None:
            central_estimate = (central_estimate + 5) / 6
        alloc = solve_adv_usw(central_estimate, std_devs, covs_lb, covs_ub, loads, rhs_bd_per_group[delta], coi_mask, groups)
    elif alloc_type == "adv_gesw":
        delta = np.round(1-conf_level, decimals=2)
        if std_devs is None:
            central_estimate = (central_estimate + 5) / 6
        alloc = solve_adv_gesw(central_estimate, std_devs, covs_lb, covs_ub, loads, rhs_bd_per_group[delta], coi_mask, groups)

    print("Saving allocation", flush=True)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        np.save(os.path.join(output_dir, dset_name_map[dset_name], "%s_%.2f_alloc.npy" % (alloc_type, conf_level)), alloc)
    else:
        np.save(os.path.join(output_dir, dset_name_map[dset_name], "%s_alloc.npy" % alloc_type), alloc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")
    parser.add_argument("--conf_level", type=float, default=0.9)

    args = parser.parse_args()
    main(args)