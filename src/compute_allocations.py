import argparse
import numpy as np
import os
import pickle
import sys
from allocation_code import solve_usw_gurobi, solve_gesw

dset_name_map = {"aamas1": "AAMAS1", "aamas2": "AAMAS2", "aamas3": "AAMAS3", "ads": "Advertising", "cs": "cs"}


def load_dset(dset_name, data_dir):

    if dset_name == "aamas1":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_1.npy"))
        covs_lb = 3 * np.ones(central_estimate.shape[1])
        covs_ub = 3 * np.ones(central_estimate.shape[1])
        loads = 10 * np.ones(central_estimate.shape[0])
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_1.npy"))

    elif dset_name == "aamas2":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_2.npy"))
        covs_lb = 3 * np.ones(central_estimate.shape[1])
        covs_ub = 3 * np.ones(central_estimate.shape[1])
        loads = 10 * np.ones(central_estimate.shape[0])
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_2.npy"))

    elif dset_name == "aamas3":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_3.npy"))
        covs_lb = 3 * np.ones(central_estimate.shape[1])
        covs_ub = 3 * np.ones(central_estimate.shape[1])
        loads = 4 * np.ones(central_estimate.shape[0])
        groups = np.load(os.path.join(data_dir, "AAMAS", "groups_3.npy"))

    elif dset_name == "ads":
        central_estimate = np.load(os.path.join(data_dir, "Advertising", "mus.npy"))
        covs_lb = np.zeros(central_estimate.shape[1]) # ad campaigns have no lower bounds
        covs_ub = 100*np.ones(central_estimate.shape[1])
        loads = np.ones(central_estimate.shape[0]) # Each user impression can only have 1 ad campaign
        groups = np.load(os.path.join(data_dir, "Advertising", "groups.npy"))

    elif dset_name == "cs":
        central_estimate = np.load(os.path.join(data_dir, "cs", "asst_scores.npy"))
        covs_lb = 2 * np.ones(central_estimate.shape[1])
        covs_ub = 2 * np.ones(central_estimate.shape[1])
        loads = 13 * np.ones(central_estimate.shape[0])
        groups = np.load(os.path.join(data_dir, "cs", "groups.npy"))

    return central_estimate, covs_lb, covs_ub, loads, groups

def main(args):
    dset_name = args.dset_name
    alloc_type = args.alloc_type

    base_dir = "/mnt/nfs/scratch1/jpayan/RAU2"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")

    central_estimate, covs_lb, covs_ub, loads, groups = load_dset(dset_name, data_dir)

    print("Loaded dataset %s, computing %s allocation" % (alloc_type, dset_name), flush=True)

    # If we are wanting exp_usw_max or exp_gesw_max, we can just compute those using the central estimates.
    # Save the results to outputs/{AAMAS, Advertising, cs}
    if alloc_type == "exp_usw_max":
        _, alloc = solve_usw_gurobi(central_estimate, covs_lb, covs_ub, loads)
    elif alloc_type == "exp_gesw_max":
        alloc = solve_gesw(central_estimate, covs_lb, covs_ub, loads, groups)

    print("Saving allocation", flush=True)

    np.save(os.path.join(output_dir, dset_name_map[dset_name], "%s_alloc.npy" % alloc_type), alloc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")

    args = parser.parse_args()
    main(args)