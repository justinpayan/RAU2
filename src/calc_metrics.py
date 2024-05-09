import argparse
import numpy as np
import os
import pickle
import sys

from allocation_code import solve_usw_gurobi, solve_gesw, solve_cvar_usw, solve_cvar_gesw, solve_adv_usw, solve_adv_gesw
from metric_code import compute_usw, compute_gesw
from scipy.stats import chi2


dset_name_map = {"aamas1": "AAMAS1", "aamas2": "AAMAS2", "aamas3": "AAMAS3", "ads": "Advertising", "cs": "cs"}


def load_dset(dset_name, data_dir):

    if dset_name.startswith("aamas"):
        idx = int(dset_name[-1])
        # central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_%d.npy" % idx))
        # std_devs = np.load(os.path.join(data_dir, "AAMAS", "zeta_matrix_%d.npy" % idx))
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_%d.npy" % idx))
        coi_mask = np.load(os.path.join(data_dir, "AAMAS", "coi_mask_%d.npy" % idx))
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "prob_up_%d.npy" % idx))
        std_devs = None

        cs = [3, 2, 2]
        ls = [15, 15, 4]
        covs_lb = cs[idx-1] * np.ones(central_estimate.shape[1])
        covs_ub = covs_lb
        loads = ls[idx-1] * np.ones(central_estimate.shape[0])

        # ngroups = len(set(groups))

        rhs_bd_per_group = pickle.load(open(os.path.join(data_dir, "AAMAS", "delta_to_normal_bd_%d.pkl" % idx), 'rb'))
        # rhs_bd_per_group = {}
        # for delta in [.3, .2, .1, .05]:
        #     rhs_bd_per_group[delta] = []
        #     for gidx in range(ngroups):
        #         gmask = np.where(groups == gidx)[0]
        #         c_value = np.sum(coi_mask[:, gmask])
        #         rhs_bd_per_group[delta].append(chi2.ppf(1-(delta/ngroups), df=c_value))

    elif dset_name == "ads":
        central_estimate = np.load(os.path.join(data_dir, "Advertising", "mus.npy"))
        std_devs = np.load(os.path.join(data_dir, "Advertising", "sigs.npy"))
        coi_mask = np.load(os.path.join(data_dir, "Advertising", "coi_mask.npy"))

        covs_lb = np.zeros(central_estimate.shape[1]) # ad campaigns have no lower bounds
        covs_ub = 100*np.ones(central_estimate.shape[1])
        loads = np.ones(central_estimate.shape[0]) # Each user impression can only have 1 ad campaign
        groups = np.load(os.path.join(data_dir, "Advertising", "groups.npy"))

        ngroups = len(set(groups))
        rhs_bd_per_group = {}
        for delta in [.3, .2, .1, .05]:
            rhs_bd_per_group[delta] = []
            for gidx in range(ngroups):
                gmask = np.where(groups == gidx)[0]
                c_value = np.sum(coi_mask[:, gmask])
                rhs_bd_per_group[delta].append(chi2.ppf(1 - (delta / ngroups), df=c_value))

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

    print("Loaded dataset %s, loading %s allocation. Conf level %.2f" % (dset_name, alloc_type, conf_level), flush=True)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        alloc_fname = os.path.join(output_dir, dset_name_map[dset_name], "%s_%.2f_alloc.npy" % (alloc_type, conf_level))
    else:
        alloc_fname = os.path.join(output_dir, dset_name_map[dset_name], "%s_alloc.npy" % alloc_type)

    allocation = np.load(alloc_fname)

    # First get the USW and GESW for this allocation under the expected values
    print("Computing USW and GESW on expected values", flush=True)
    usw = compute_usw(allocation, central_estimate)
    gesw = compute_gesw(allocation, central_estimate, groups)

    print("Done with USW/ESW", flush=True)

    # Save it all in a dictionary, print and dump
    conf_levels = [.7, .8, .9, .95]
    metrics_to_values = {}
    metrics_to_values['usw'] = usw
    metrics_to_values['gesw'] = gesw

    metrics_to_values['cvar_usw'] = {}
    metrics_to_values['cvar_gesw'] = {}
    metrics_to_values['adv_usw'] = {}
    metrics_to_values['adv_gesw'] = {}

    for c in conf_levels:
        metrics_to_values['cvar_usw'][c] = 0.0
        metrics_to_values['cvar_gesw'][c] = 0.0
        metrics_to_values['adv_usw'][c] = 0.0
        metrics_to_values['adv_gesw'][c] = 0.0

    print(metrics_to_values, flush=True)

    if alloc_type.startswith("cvar") or alloc_type.startswith("adv"):
        metric_fname = os.path.join(output_dir, dset_name_map[dset_name], "%s_%.2f_metrics.pkl" % (alloc_type, conf_level))
    else:
        metric_fname = os.path.join(output_dir, dset_name_map[dset_name], "%s_metrics.pkl" % alloc_type)

    pickle.dump(metrics_to_values, open(metric_fname, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")
    parser.add_argument("--conf_level", type=float, default=0.9)

    args = parser.parse_args()
    main(args)