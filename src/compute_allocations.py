import argparse
import numpy as np
import os
import pickle
import sys
sys.path.append("/mnt/nfs/scratch1/jpayan/RAU")
from solve_usw import solve_usw_gurobi


def main(args):
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")

    dset_name = args.dset_name
    alloc_type = args.alloc_type

    base_dir = "/mnt/nfs/scratch1/jpayan/RAU2"
    data_dir = os.path.join(base_dir, "data")

    if dset_name == "aamas1":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_1.npy"))
        cov, load =
    elif dset_name == "aamas2":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_2.npy"))
    elif dset_name == "aamas3":
        central_estimate = np.load(os.path.join(data_dir, "AAMAS", "mu_matrix_3.npy"))
    elif dset_name == "ads":
        central_estimate = np.load(os.path.join(data_dir, "Advertising", "mus.npy"))
    elif dset_name == "cs":
        central_estimate = np.load(os.path.join(data_dir, "cs", "asst_scores.npy"))

    # If we are wanting exp_usw_max or exp_gesw_max, we can just compute those using the central estimates.
    # Save the results to outputs/{AAMAS, Advertising, cs}
    solve_usw_gurobi(central_estimate, covs, loads)

    # TODO: We need to actually implement the group egalitarian maximizer, but I think I have in a NB somewhere.









    # base_dir = "/mnt/nfs/scratch1/jpayan/predictive_expert_assignment"
    #
    # # Open up the folder and load in the asst_scores, covs, loads, kp_matching_scores,
    # # user_rep_scores
    # data_dir = os.path.join(base_dir, "data", "%s.stackexchange.com" % topic, "npy")
    # asst_scores = np.load(os.path.join(data_dir, "asst_scores.npy"))
    # asst_scores_badges = np.load(os.path.join(data_dir, "asst_scores_badges.npy"))
    # asst_scores_user_embs = np.load(os.path.join(data_dir, "asst_scores_user_embs.npy"))
    #
    # covs = pickle.load(open(os.path.join(data_dir, "covs.pkl"), 'rb'))
    # loads = pickle.load(open(os.path.join(data_dir, "loads.pkl"), 'rb'))
    # user_rep_scores = np.load(os.path.join(data_dir, "user_rep_scores.npy"))
    # kp_matching_scores = np.load(os.path.join(data_dir, "kp_matching_scores.npy"))
    # sim_scores = np.load(os.path.join(data_dir, "sim_scores.npy"))
    #
    # # Select a subset of users and questions to work with (maybe like .5 or .6
    # # fraction of each?)
    # rng = np.random.default_rng(seed=seed)
    # num_e, num_q = asst_scores.shape
    # frac = .6
    # chosen_experts = rng.choice(range(num_e), size=int(frac*num_e), replace=False)
    # chosen_queries = rng.choice(range(num_q), size=int(frac*num_q), replace=False)
    #
    # alloc_fname = os.path.join(data_dir, "alloc_%d.npy" % seed)
    #
    # if not os.path.exists(alloc_fname):
    #     np.save(os.path.join(data_dir, "chosen_experts_%d.npy" % seed), chosen_experts)
    #     np.save(os.path.join(data_dir, "chosen_queries_%d.npy" % seed), chosen_queries)
    #
    #     asst_scores = asst_scores[chosen_experts, :]
    #     asst_scores = asst_scores[:, chosen_queries]
    #     asst_scores_badges = asst_scores_badges[chosen_experts, :]
    #     asst_scores_badges = asst_scores_badges[:, chosen_queries]
    #     asst_scores_user_embs = asst_scores_user_embs[chosen_experts, :]
    #     asst_scores_user_embs = asst_scores_user_embs[:, chosen_queries]
    #     covs = covs[chosen_queries]
    #     loads = loads[chosen_experts]
    #     user_rep_scores = user_rep_scores[chosen_experts, :]
    #     user_rep_scores = user_rep_scores[:, chosen_queries]
    #     kp_matching_scores = kp_matching_scores[chosen_experts, :]
    #     kp_matching_scores = kp_matching_scores[:, chosen_queries]
    #     sim_scores = sim_scores[chosen_experts, :]
    #     sim_scores = sim_scores[:, chosen_queries]
    #
    #     # Now just compute all the assignments, and save them out.
    #     print("Data loaded. Starting allocations", flush=True)
    #     est_usw, alloc = solve_usw_gurobi(asst_scores, covs, loads)
    #     print("Finished with pred asst, est_usw is ", est_usw, flush=True)
    #     np.save(os.path.join(data_dir, "alloc_%d.npy" % seed), alloc)
    #
    #     est_usw, alloc = solve_usw_gurobi(asst_scores_badges, covs, loads)
    #     print("Finished with pred asst with badges, est_usw is ", est_usw, flush=True)
    #     np.save(os.path.join(data_dir, "alloc_badges_%d.npy" % seed), alloc)
    #
    #     est_usw, alloc = solve_usw_gurobi(asst_scores_user_embs, covs, loads)
    #     print("Finished with pred asst with user embs, est_usw is ", est_usw, flush=True)
    #     np.save(os.path.join(data_dir, "alloc_user_embs_%d.npy" % seed), alloc)
    #
    #     # for lam in np.arange(0, 1.01, .1):
    #     for lam in range(11):
    #         print("Starting on lambda=", lam, flush=True)
    #         lambda_val = lam*.1
    #         non_pred_scores = lambda_val * user_rep_scores / np.max(user_rep_scores)
    #         # non_pred_scores += (1 - lambda_val) * kp_matching_scores / np.max(kp_matching_scores)
    #         non_pred_scores += (1 - lambda_val) * sim_scores / np.max(sim_scores)
    #
    #         _, alloc_non_pred = solve_usw_gurobi(non_pred_scores, covs, loads)
    #         np.save(os.path.join(data_dir, "alloc_non_pred_%d_%d.npy" % (lam, seed)), alloc_non_pred)
    #
    #     for ridx in range(100):
    #         print("Starting on ridx=", ridx, flush=True)
    #         rand_scores = rng.uniform(0, 1, size=alloc.shape)
    #         _, alloc_rand = solve_usw_gurobi(rand_scores, covs, loads)
    #         np.save(os.path.join(data_dir, "alloc_rand_%d_%d.npy" % (ridx, seed)), alloc_rand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="aamas1")
    parser.add_argument("--alloc_type", type=str, default="exp_usw_max")

    args = parser.parse_args()
    main(args)