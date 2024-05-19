import cvxpy as cp
import numpy as np
import torch

from collections import Counter
import gurobipy as gp
from gurobipy import Model, multidict, GRB
from scipy.stats import norm

import random
import time

random.seed(10)
np.random.seed(10)
torch.manual_seed(0)


def solve_usw_gurobi(affinity_scores, covs_lb, covs_ub, loads, coi_mask):
    m = Model("TPMS")

    alloc = m.addMVar(affinity_scores.shape, vtype=GRB.BINARY, name='alloc')

    m.addConstr(alloc.sum(axis=0) >= covs_lb)
    m.addConstr(alloc.sum(axis=0) <= covs_ub)
    m.addConstr(alloc.sum(axis=1) <= loads)
    m.addConstr(alloc <= coi_mask)

    obj = (alloc * affinity_scores).sum()
    m.setObjective(obj, GRB.MAXIMIZE)

    m.optimize()

    return alloc.x


def solve_gesw(affinity_scores, covs_lb, covs_ub, loads, groups, coi_mask):
    num_groups = len(set(groups))
    group_indicators = []
    group_size = Counter(groups)
    for g in range(num_groups):
        group_indicators.append(np.zeros(affinity_scores.shape))
    for idx, g in enumerate(groups):
        group_indicators[g][:, idx] = 1 / group_size[g]
    group_indicators = [cp.reshape(gi, (gi.size, 1)) for gi in group_indicators]

    alloc = cp.Variable(affinity_scores.shape, boolean=True)
    y = cp.Variable()

    flat_alloc = cp.reshape(alloc, (alloc.size, 1))
    groups_stacked = cp.hstack(group_indicators)
    flat_alloc_per_group = cp.multiply(flat_alloc, groups_stacked)
    coeffs = cp.reshape(affinity_scores, (1, affinity_scores.size))
    inner_prods_per_group = coeffs @ flat_alloc_per_group
    obj = y

    constr = [cp.sum(alloc, axis=0) >= covs_lb, cp.sum(alloc, axis=0) <= covs_ub,
              cp.sum(alloc, axis=1) <= loads, y <= inner_prods_per_group, alloc <= coi_mask]
    # We used to have y >= 0, but I think this makes the model infeasible sometimes, and anyway it isn't necessary

    gesw_problem = cp.Problem(cp.Maximize(obj), constr)

    gesw_problem.solve(verbose=True, solver='GUROBI')

    return alloc.value


def solve_cvar_usw(covs_lb, covs_ub, loads, conf_level, value_samples, coi_mask):
    alloc = cp.Variable((loads.size, covs_lb.size), boolean=True)
    alpha = cp.Variable()
    beta = 1 - conf_level
    num_samples = len(value_samples)
    # Beta is the cvar level for the RISK. So at .99, that means we are minimizing the conditional expectation
    # of the highest 1% of RISK scores, or rather, maximizing the CE of the lowest 1% of GAIN scores.

    flat_alloc = cp.reshape(alloc, (1, alloc.size))
    flat_value_samples = [cp.reshape(vs, (vs.size, 1)) for vs in value_samples]
    inner_prods = [flat_alloc @ vs for vs in flat_value_samples]
    summands = [cp.pos(-1 * ip - alpha) for ip in inner_prods]
    obj = cp.sum(summands)
    obj = alpha + obj / (num_samples * (1 - beta))

    constr = [cp.sum(alloc, axis=0) >= covs_lb,
              cp.sum(alloc, axis=0) <= covs_ub,
              cp.sum(alloc, axis=1) <= loads,
              alloc <= coi_mask]

    cvar_usw_problem = cp.Problem(cp.Minimize(obj), constr)

    cvar_usw_problem.solve(verbose=True, solver='GUROBI', mipgap=0.05, timelimit=7200)

    return alloc.value


def solve_cvar_gesw(covs_lb, covs_ub, loads, conf_level, value_samples, groups, coi_mask):
    shape_tup = (loads.size, covs_lb.size)
    num_groups = len(set(groups))
    group_indicators = []
    group_size = Counter(groups)
    for g in range(num_groups):
        group_indicators.append(np.zeros(shape_tup))
    for idx, g in enumerate(groups):
        group_indicators[g][:, idx] = 1 / group_size[g]
    group_indicators = [cp.reshape(gi, (gi.size, 1)) for gi in group_indicators]

    num_samples = len(value_samples)
    gesw_alloc = cp.Variable(shape_tup, boolean=True)
    alpha = cp.Variable()
    beta = 1 - conf_level
    y = cp.Variable((num_samples, 1))

    # Beta is the cvar level for the RISK. So at .99, that means we are minimizing the conditional expectation
    # of the highest 1% of RISK scores, or rather, maximizing the CE of the lowest 1% of GAIN scores.

    flat_alloc = cp.reshape(gesw_alloc, (gesw_alloc.size, 1))

    groups_stacked = cp.hstack(group_indicators)
    flat_alloc_per_group = cp.multiply(flat_alloc, groups_stacked)
    flat_vote_samples = cp.vstack([cp.reshape(vs, (1, vs.size)) for vs in value_samples])

    inner_prods_per_group = flat_vote_samples @ flat_alloc_per_group  # (num_samples x mn) @ (mn x num_groups) = (num_samples x num_groups)

    obj = cp.sum(y)
    obj = alpha + obj / (num_samples * (1 - beta))

    constr = [cp.sum(gesw_alloc, axis=0) >= covs_lb, cp.sum(gesw_alloc, axis=0) <= covs_ub,
              cp.sum(gesw_alloc, axis=1) <= loads, gesw_alloc <= coi_mask, y >= 0,
              y >= -1 * inner_prods_per_group - alpha]

    cvar_gesw_problem = cp.Problem(cp.Minimize(obj), constr)
    # model.setParam('MIPGap', 0.05)

    cvar_gesw_problem.solve(verbose=True, solver='GUROBI', mipgap=0.05, timelimit=7200)

    return gesw_alloc.value


def solve_cvar_usw_gauss(mu_matrix, sigma_matrix, covs_lb, covs_ub, loads, conf_level, coi_mask):
    m = Model("TPMS")

    alloc = m.addMVar(mu_matrix.shape, vtype=GRB.BINARY, name='alloc')

    m.addConstr(alloc.sum(axis=0) >= covs_lb)
    m.addConstr(alloc.sum(axis=0) <= covs_ub)
    m.addConstr(alloc.sum(axis=1) <= loads)
    m.addConstr(alloc <= coi_mask)

    aux = m.addVar(lb=0)

    m.addConstr(aux ** 2 == (alloc * sigma_matrix * alloc).sum())

    frac = norm.pdf(norm.ppf(conf_level)) / (1 - conf_level)
    obj = (alloc * mu_matrix).sum() - frac * aux
    m.setObjective(obj, GRB.MAXIMIZE)

    m.optimize()

    return alloc.x


def solve_cvar_gesw_gauss(mu_matrix, sigma_matrix, covs_lb, covs_ub, loads, conf_level, groups, coi_mask):
    m = Model("TPMS")

    alloc = m.addMVar(mu_matrix.shape, vtype=GRB.BINARY, name='alloc')

    m.addConstr(alloc.sum(axis=0) >= covs_lb)
    m.addConstr(alloc.sum(axis=0) <= covs_ub)
    m.addConstr(alloc.sum(axis=1) <= loads)
    m.addConstr(alloc <= coi_mask)

    n_groups = len(set(groups))

    t = m.addVar()
    obj = t

    for gidx in range(n_groups):
        gmask = np.where(groups == gidx)[0]
        a = alloc[:, gmask]
        sd = sigma_matrix[:, gmask]
        mu = mu_matrix[:, gmask]
        grpsize = gmask.size

        lhs = 1 / grpsize
        lhs *= (a * mu).sum()
        lhs -= t
        lhs = lhs * lhs
        rhs = (a * sd * a).sum()
        rhs *= (norm.pdf(norm.ppf(conf_level)) / (grpsize * (1 - conf_level))) ** 2
        m.addConstr(lhs >= rhs)

    m.setObjective(obj, GRB.MAXIMIZE)

    m.optimize()

    return alloc.x


def prep_groups(central_estimate, variances, covs_lb, covs_ub, coi_mask, groups):
    n_groups = len(set(groups))
    a = 1
    b = 0

    ce_l = []
    covs_lb_l = []
    covs_ub_l = []
    coi_mask_l = []
    var_l = []

    a_l = a * np.ones(n_groups)
    b_l = b * np.ones(n_groups)

    for gidx in range(n_groups):
        gmask = np.where(groups == gidx)[0]

        ce_l.append(central_estimate[:, gmask])
        if variances is not None:
            var_l.append(variances[:, gmask])
        covs_lb_l.append(covs_lb[gmask])
        covs_ub_l.append(covs_ub[gmask])
        coi_mask_l.append(coi_mask[:, gmask])

    return a_l, b_l, ce_l, var_l, covs_lb_l, covs_ub_l, coi_mask_l


def solve_adv_usw(central_estimate, variances, covs_lb, covs_ub, loads, rhs_bd_per_group, coi_mask, groups,
                  method="IQP"):
    a_l, b_l, ce_l, var_l, covs_lb_l, covs_ub_l, coi_mask_l = \
        prep_groups(central_estimate, variances, covs_lb, covs_ub, coi_mask, groups)

    timestamps, obj_vals = None, None

    print("Solving adv usw using method ", method)

    if variances is None:
        print("Actually, it is linear, solving QP directly")

        # This is the model based on cross-entropy loss, so we'll use the linear function
        group_allocs, _ = compute_group_utilitarian_linear(a_l, b_l, ce_l, coi_mask_l,
                                                           rhs_bd_per_group, loads, covs_lb_l, covs_ub_l)
    else:
        if method == "IQP":
            obj = UtilitarianAlternation(ce_l, covs_lb_l, covs_ub_l, loads, [v.flatten() for v in var_l],
                                         rhs_bd_per_group, coi_mask_l)
            _, group_allocs, _, timestamps, obj_vals = obj.iterative_optimization()
        elif method == "ProjGD":
            step_size = 1e-1
            obj = ComputeUtilitarianQuadraticProj(ce_l, covs_lb_l, covs_ub_l, coi_mask_l, loads,
                                                  [v.flatten() for v in var_l], rhs_bd_per_group, step_size)
            group_allocs, _, _, timestamps, obj_vals = obj.gradient_descent()
        elif method == "SubgradAsc":
            group_allocs, timestamps, obj_vals = subgrad_ascent_util_ellipsoid(ce_l, covs_lb_l, covs_ub_l, loads, var_l,
                                                                               rhs_bd_per_group)

    # Stitch together group_allocs into a single allocation and return it
    final_alloc = np.zeros_like(central_estimate)
    for gidx in range(len(set(groups))):
        gmask = np.where(groups == gidx)[0]
        final_alloc[:, gmask] = group_allocs[gidx].reshape(final_alloc[:, gmask].shape)
    return final_alloc, timestamps, obj_vals


def solve_adv_gesw(central_estimate, variances, covs_lb, covs_ub, loads, rhs_bd_per_group, coi_mask, groups,
                   method="ProjGD"):
    print("Solving adv gesw using method ", method)

    a_l, b_l, ce_l, var_l, covs_lb_l, covs_ub_l, coi_mask_l = \
        prep_groups(central_estimate, variances, covs_lb, covs_ub, coi_mask, groups)
    timestamps = None
    obj_vals = None
    if variances is None:
        print("Actually, it is linear, solving QP directly")
        # This is the model based on cross-entropy loss, so we'll use the linear function
        group_allocs, _ = compute_group_egal_linear(a_l, b_l, ce_l, coi_mask_l,
                                                    rhs_bd_per_group, loads, covs_lb_l, covs_ub_l)
    else:
        if method == "ProjGD":
            step_size = 1e-1
            egalObject = ComputeGroupEgalitarianQuadraticProj(ce_l, covs_lb_l, covs_ub_l, coi_mask_l, loads,
                                                              [v.flatten() for v in var_l], rhs_bd_per_group, step_size)
            group_allocs, _, _, timestamps, obj_vals = egalObject.gradient_descent()
        elif method == "SubgradAsc":
            group_allocs, timestamps, obj_vals = subgrad_ascent_egal_ellipsoid(ce_l, covs_lb_l, covs_ub_l, loads, var_l,
                                                                               rhs_bd_per_group)

    # Stitch together group_allocs into a single allocation and return it
    final_alloc = np.zeros_like(central_estimate)
    for gidx in range(len(set(groups))):
        gmask = np.where(groups == gidx)[0]
        final_alloc[:, gmask] = group_allocs[gidx].reshape(final_alloc[:, gmask].shape)
    return final_alloc, timestamps, obj_vals


def compute_group_utilitarian_linear(a_l, b_l, phat_l, C_l, rhs_bd_per_group, loads, covs_lb_l, covs_ub_l, milp=False):
    ngroups = len(phat_l)
    model = gp.Model()

    e_vals = []
    c_vals = []
    f_vals = []
    x_vals = []
    Allocs = []

    eps = 1e-6

    for gdx in range(ngroups):
        print("starting with group ", gdx)
        n_agents = phat_l[gdx].shape[0]
        n_items = phat_l[gdx].shape[1]
        phat = phat_l[gdx].flatten()
        C = C_l[gdx].flatten()
        covs_lb = covs_lb_l[gdx].flatten()
        covs_ub = covs_ub_l[gdx].flatten()

        A_multiplier = (a_l[gdx] - b_l[gdx])
        if milp == False:
            A = model.addMVar(len(phat_l[gdx].flatten()), lb=0, ub=1, vtype=gp.GRB.CONTINUOUS,
                              name='Alloc' + str(gdx))
        else:
            A = model.addMVar(len(phat_l[gdx].flatten()), lb=0, ub=1, vtype=gp.GRB.INTEGER, name='Alloc' + str(gdx))
        Allocs.append(A)

        log_p_phat = np.log(phat).flatten()
        log_one_minus_phat = np.log(1 - phat).flatten()

        rhs_bd = rhs_bd_per_group[gdx]

        mn = int(n_agents * n_items)
        # mn = np.sum(C)
        c_val = np.sum(C)

        e = -1.0 * (c_val * rhs_bd + np.sum(C * log_one_minus_phat))
        neg_ones = -1 * np.ones(mn)

        c = np.vstack((np.array([e]).reshape(1, 1), neg_ones.flatten().reshape(-1, 1))).flatten()
        f = C * (log_p_phat - log_one_minus_phat).flatten()

        x = model.addMVar(mn + 1, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="pval")
        e_vals.append(e)
        c_vals.append(c)
        f_vals.append(f)
        x_vals.append(x)

        model.addConstrs(A[i] <= C[i] for i in range(mn))

        model.addConstrs(gp.quicksum(A[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                         range(n_items))

        model.addConstrs(gp.quicksum(A[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                         range(n_items))

        model.addConstrs((f[jdx] * x[0] - x[jdx + 1] <= A_multiplier * A[jdx] for jdx in range(mn)),
                         name='ctr' + str(gdx))

    load_sum = model.addMVar(loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS, name='load_sum')

    model.addConstrs(load_sum[idx] == gp.quicksum(
        Allocs[gdx][idx * phat_l[gdx].shape[1]:(idx + 1) * (phat_l[gdx].shape[1])].sum() for gdx in range(ngroups)) for
                     idx in range(loads.size))
    total_agents = loads.size
    model.addConstrs(load_sum[idx] <= loads[idx] for idx in range(total_agents))

    model.setObjective(gp.quicksum(c_vals[idx] @ x_vals[idx] for idx in range(ngroups)), gp.GRB.MAXIMIZE)
    model.setParam('OutputFlag', 1)

    model.optimize()
    final_allocs = []
    for idx in range(ngroups):
        final_allocs.append(Allocs[idx].X)

    obj = model.getObjective()

    return final_allocs, obj.getValue()


def compute_group_egal_linear(a_l, b_l, phat_l, C_l, rhs_bd_per_group, loads, covs_lb_l, covs_ub_l, milp=False):
    ngroups = len(phat_l)
    model = gp.Model()

    t = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='t')

    e_vals = []
    c_vals = []
    f_vals = []
    x_vals = []
    Allocs = []

    for gdx in range(ngroups):
        n_agents = phat_l[gdx].shape[0]
        n_items = phat_l[gdx].shape[1]
        phat = phat_l[gdx].flatten()
        C = C_l[gdx].flatten()
        covs_lb = covs_lb_l[gdx].flatten()
        covs_ub = covs_ub_l[gdx].flatten()

        A_multiplier = (a_l[gdx] - b_l[gdx])
        if milp == False:
            A = model.addMVar(len(phat_l[gdx].flatten()), lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='Alloc' + str(gdx))
        else:
            A = model.addMVar(len(phat_l[gdx].flatten()), lb=0, ub=1, vtype=gp.GRB.INTEGER, name='Alloc' + str(gdx))
        Allocs.append(A)

        eps = 1e-6

        log_p_phat = np.log(phat).flatten()
        log_one_minus_phat = np.log(1 - phat).flatten()
        rhs_bd = rhs_bd_per_group[gdx]

        mn = int(n_agents * n_items)
        c_val = np.sum(C)

        e = -1.0 * (c_val * rhs_bd + np.sum(C * log_one_minus_phat))
        neg_ones = -1 * np.ones(mn)
        c = np.vstack((np.array([e]).reshape(1, 1), neg_ones.flatten().reshape(-1, 1))).flatten()
        f = C * (log_p_phat - log_one_minus_phat).flatten()

        x = model.addMVar(mn + 1, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="pval")
        e_vals.append(e)
        c_vals.append(c)
        f_vals.append(f)
        x_vals.append(x)

        model.addConstrs(A[i] <= C[i] for i in range(mn))

        model.addConstrs(gp.quicksum(A[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                         range(n_items))

        model.addConstrs(gp.quicksum(A[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                         range(n_items))

        model.addConstrs((f[jdx] * x[0] - x[jdx + 1] <= A_multiplier * A[jdx] / n_items for jdx in range(mn)),
                         name='ctr' + str(gdx))
        model.addConstr(t <= c @ x, name='min_w' + str(gdx))

    load_sum = model.addMVar(loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS, name='load_sum')

    model.addConstrs(load_sum[idx] == gp.quicksum(
        Allocs[gdx][idx * phat_l[gdx].shape[1]:(idx + 1) * (phat_l[gdx].shape[1])].sum() for gdx in range(ngroups)) for
                     idx in range(loads.size))
    total_agents = loads.size
    model.addConstrs(load_sum[idx] <= loads[idx] for idx in range(total_agents))

    model.setObjective(t, gp.GRB.MAXIMIZE)
    model.setParam('OutputFlag', 1)

    model.optimize()
    final_allocs = []
    for idx in range(ngroups):
        final_allocs.append(Allocs[idx].X)

    obj = model.getObjective()

    return final_allocs, obj.getValue()


class UtilitarianAlternation():
    def __init__(self, mu_list, covs_lb_list, covs_ub_list, loads, Sigma_list, rad_list, coi_mask_list, n_iter=1000,
                 integer=False):

        self.mu_list = mu_list
        self.Sigma_list = Sigma_list
        self.rad_list = rad_list
        print("self.rad_list: ", self.rad_list)
        self.covs_ub_list = covs_ub_list
        self.covs_lb_list = covs_lb_list
        self.loads = loads
        self.coi_mask_list = coi_mask_list
        self.n_iter = n_iter
        self.integer = integer

        # For logging
        self.iter_timestamps = []
        self.iter_obj_vals = []

        self.ngroups = len(self.mu_list)
        self.nA_list = []
        self.nI_list = []
        for idx in range(self.ngroups):
            nA = self.mu_list[idx].shape[0]
            nI = self.mu_list[idx].shape[1]
            self.nA_list.append(nA)
            self.nI_list.append(nI)

        self.lamda = np.random.randint(1, 2, self.ngroups)
        self.betas = None

    def optimize_lambda(self, allocs, betas):
        lamdas = []
        for gdx in range(self.ngroups):
            A = allocs[gdx].flatten()
            beta = betas[gdx].flatten()
            temp = (A - beta).reshape(1, -1)
            x = np.sum((temp.flatten() ** 2) * self.Sigma_list[gdx].flatten())
            y = 4 * (self.rad_list[gdx] ** 2) + 1e-10
            lamda = np.abs(np.sqrt(x / (y)))

            lamdas.append(lamda)
        lamdas = np.array(lamdas)
        print("got lambda", lamdas)

        return lamdas

    def compute_welfare(self, allocs, betas, lamdas):
        welfare_util = 0.0
        for gdx in range(self.ngroups):
            A = allocs[gdx].flatten()
            beta = betas[gdx].flatten()
            lamda = lamdas[gdx]
            temp = (A - beta).reshape(1, -1)
            welfare = np.dot((A - beta), self.mu_list[gdx].flatten()) - np.sum(
                (temp.flatten() ** 2) * self.Sigma_list[gdx].flatten()) / (4 * lamda) - lamda * self.rad_list[gdx] ** 2
            welfare_util += welfare
        return welfare_util

    def iterative_optimization(self, niters=1000, eps=1e-5, group_welfare=False):
        welfare = None
        prev_welfare = None
        allocs = None
        betas = None
        self.group_welfare = group_welfare

        start_time = time.time()

        for iter in range(niters):
            allocs, betas = self.optimize_a_beta()

            lamda = self.optimize_lambda(allocs, betas)
            self.lamda = np.array(lamda)

            new_welfare = self.compute_welfare(allocs, betas, lamda)
            if prev_welfare is None:
                prev_welfare = new_welfare
            else:
                prev_welfare = welfare
            welfare = new_welfare
            if iter != 0 and np.abs(prev_welfare - welfare) < eps:
                print("got welfare", welfare)
                break
            self.iter_timestamps.append(time.time() - start_time)
            self.iter_obj_vals.append(welfare)
            print(f"Iter: {iter} Utilitarian welfare: {welfare}")
        return welfare, allocs, betas, self.iter_timestamps, self.iter_obj_vals

    def optimize_a_beta(self):
        ngroups = len(self.mu_list)
        beta_list = []
        alloc_list = []
        temp_list = []
        zeta_list = []
        model = gp.Model()

        for gdx in range(ngroups):
            tpms = np.array(self.mu_list[gdx])
            n_reviewers = int(tpms.shape[0])
            n_papers = int(tpms.shape[1])

            assert (np.all(self.covs_ub_list[gdx] <= n_reviewers))

            num = int(n_reviewers * n_papers)

            beta_g = model.addMVar(num, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta" + str(gdx))

            zeta_g = model.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                   name="zeta" + str(gdx))

            beta_list.append(beta_g)

            if self.integer:
                alloc_g = model.addMVar(num, lb=0, ub=1, vtype=gp.GRB.INTEGER, name="alloc" + str(gdx))
            else:
                alloc_g = model.addMVar(num, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="alloc" + str(gdx))

            alloc_list.append(alloc_g)
            zeta_list.append(zeta_g)

            zeros = np.zeros(num)
            model.addConstr(beta_g >= zeros, name='c8' + str(gdx))
            n_agents = tpms.shape[0]
            n_items = tpms.shape[1]
            covs_ub = self.covs_ub_list[gdx]
            covs_lb = self.covs_lb_list[gdx]

            model.addConstrs(
                gp.quicksum(alloc_g[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                range(n_items))

            model.addConstrs(
                gp.quicksum(alloc_g[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                range(n_items))

            if self.group_welfare:
                model.addConstr(zeta_g == ((1.0) / (n_items)) * alloc_g - beta_g)
            else:
                model.addConstr(zeta_g == (alloc_g - beta_g))

            model.addConstr(alloc_g <= self.coi_mask_list[gdx].flatten())

        load_sum = model.addMVar(self.loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS,
                                 name='load_sum')

        model.addConstrs(load_sum[idx] == gp.quicksum(
            alloc_list[gdx][idx * self.mu_list[gdx].shape[1]:(idx + 1) * (self.mu_list[gdx].shape[1])].sum() for gdx in
            range(ngroups))
                         for idx in range(self.loads.size))

        model.addConstr(load_sum <= self.loads, name='load_constr')

        model.setObjective(gp.quicksum((zeta_list[gdx]) @ self.mu_list[gdx].flatten() -
                                       gp.quicksum(
                                           zeta_list[gdx][jdx] * self.Sigma_list[gdx][jdx] * zeta_list[gdx][jdx] * (
                                                       1 / (4 * self.lamda[gdx])) for jdx in
                                           range(len(self.Sigma_list[gdx]))) - self.lamda[gdx] * self.rad_list[gdx] ** 2
                                       for
                                       gdx in range(ngroups)), gp.GRB.MAXIMIZE)
        # model.setParam('OutputFlag', 1)

        model.setParam('MIPGap', 0.05)

        model.optimize()
        print("objective", model.ObjVal)

        allocs = []
        betas = []
        for g in range(ngroups):
            print("Group %d parameters: " % g)
            alloc_v = alloc_list[g].X

            beta_v = beta_list[g].X
            allocs.append(np.array(list(alloc_v)).reshape(self.mu_list[g].shape))
            betas.append(np.array(list(beta_v)).reshape(self.mu_list[g].shape))
            print("sum Beta_v: ", np.sum(beta_v))
            print("sum zeta_v: ", np.sum(zeta_list[g].X))
            print("small Beta_v: ", beta_v[:100])
            print("small zeta_v: ", zeta_list[g].X[:100])
            print("Beta_v: ", beta_v)
            print("zeta_v: ", zeta_list[g].X)

        model.dispose()

        del model
        # self.mu_list = allocs
        # self.betas = betas
        return allocs, betas


class ComputeUtilitarianQuadraticProj():
    def __init__(self, mu_list, covs_lb_l, covs_ub_l, coi_mask_l, loads, Sigma_list, rad_list, step_size, n_iter=1000):

        self.mu_list = mu_list
        self.Sigma_list = Sigma_list
        self.rad_list = rad_list
        self.covs_lb_list = covs_lb_l
        self.covs_ub_list = covs_ub_l
        self.coi_mask_list = coi_mask_l
        self.loads = loads
        self.step_size = step_size
        self.n_iter = n_iter
        self.timestamps = []
        self.objective_vals = []

        self.eta = .1

        self.ngroups = len(self.mu_list)
        self.nA_list = []
        self.nI_list = []
        for idx in range(self.ngroups):
            nA = self.mu_list[idx].shape[0]
            nI = self.mu_list[idx].shape[1]
            self.nA_list.append(nA)
            self.nI_list.append(nI)

        self.beta_list = [torch.zeros(self.mu_list[idx].shape) for idx in range(self.ngroups)]
        self.A_list = [torch.zeros(self.mu_list[idx].shape) for idx in range(self.ngroups)]

        self.lamda = np.zeros(self.ngroups)
        self.convert_to_tensors()

    def convert_to_tensors(self):
        self.mu_tl = []
        self.A_tl = []
        self.beta_tns = []
        self.Lamda_tns = None
        self.sigma_tns = []
        self.coi_tns = []
        self.Lamda_tns = torch.rand(self.ngroups, requires_grad=True)

        params = []
        params.append(self.Lamda_tns)

        for gdx in range(self.ngroups):
            self.mu_tl.append(torch.Tensor(self.mu_list[gdx]))
            self.beta_tns.append(torch.rand(self.beta_list[gdx].shape, requires_grad=True))
            self.A_tl.append(torch.rand(self.A_list[gdx].shape, requires_grad=True))
            self.sigma_tns.append(torch.Tensor(np.diag(self.Sigma_list[gdx].flatten())))
            self.coi_tns.append(torch.Tensor(self.coi_mask_list[gdx]))
            params.append(self.A_tl[gdx])
            params.append(self.beta_tns[gdx])

        self.optimizer = torch.optim.Adam(params, lr=self.step_size)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max')

    def welfare(self):
        welfares = []
        for gdx in range(self.ngroups):
            Ag = self.A_tl[gdx].flatten()
            Bg = self.beta_tns[gdx].flatten()
            Vg = self.mu_tl[gdx].flatten()
            Sigma_g = self.sigma_tns[gdx]
            Cg = self.coi_tns[gdx].flatten()
            term1 = torch.sum((Cg * (Ag - Bg)).flatten() * Vg.flatten())
            temp = (Cg * (Ag - Bg)).reshape(-1, 1)
            term2 = -(torch.mm(torch.mm(temp.t(), Sigma_g), temp)) / (4 * (self.Lamda_tns[gdx] + 1e-5))
            term3 = -self.Lamda_tns[gdx] * self.rad_list[gdx] ** 2
            w = (term1 + term2 + term3)
            welfares.append(w.detach().cpu().numpy())

        return welfares

    def func(self):
        # term_sum = 0.0
        terms = torch.zeros(self.ngroups)
        for gdx in range(self.ngroups):
            Ag = self.A_tl[gdx].flatten()
            Bg = self.beta_tns[gdx].flatten()
            Vg = self.mu_tl[gdx].flatten()
            Cg = self.coi_tns[gdx].flatten()
            Sigma_g = self.sigma_tns[gdx]
            term1 = torch.sum((Cg * (Ag - Bg)).flatten() * Vg.flatten())
            temp = (Cg * (Ag - Bg)).reshape(-1, 1)
            print(temp)
            term2 = -(torch.mm(torch.mm(temp.t(), Sigma_g), temp)) / (4 * (self.Lamda_tns[gdx] + 1e-3))
            term3 = -self.Lamda_tns[gdx] * self.rad_list[gdx]
            print(term1, term2, term3)
            # term = torch.exp(-1 * self.eta * (term1 + term2 + term3))
            # print(term)
            # term_sum = term_sum + term
            terms[gdx] += term1 + term2[0, 0] + term3
            print()
        return -1 * torch.sum(terms)
        # soft_min = (-1.0 / self.eta) * torch.log((1.0 / self.ngroups) * term_sum)
        # return -soft_min

    def gradient_descent(self):
        loss_BGD = []

        start_time = time.time()

        for i in range(self.n_iter):
            loss = self.func()
            print(f"Iter {i} Loss {loss}")
            # storing the calculated loss in a list
            loss_BGD.append(loss.item())
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            self.optimizer.step()
            for idx in range(self.ngroups):
                self.A_tl[idx].grad.data.zero_()
                self.beta_tns[idx].grad.data.zero_()
            self.Lamda_tns.grad.data.zero_()
            projected_A, projected_beta, projected_lamda = self.projection(self.A_tl, self.beta_tns, self.Lamda_tns)
            for idx in range(self.ngroups):
                self.A_tl[idx].data = torch.Tensor(projected_A[idx])
                self.beta_tns[idx].data = torch.Tensor(projected_beta[idx])
            self.Lamda_tns.data = torch.Tensor(projected_lamda)
            welfares = self.welfare()
            sum_w = np.sum(welfares)
            self.scheduler.step(sum_w)

            self.timestamps.append(time.time() - start_time)
            self.objective_vals.append(sum_w)

            print(f'Iter: {i}, \tLoss: {loss.item()} Worst welfare: {worst_w} Welfare sum: {sum_w}')

        return self.convert_to_numpy(self.A_tl), self.beta_tns, self.Lamda_tns, self.timestamps, self.objective_vals

    def convert_to_numpy(self, X_list):
        n = len(X_list)
        X_new = []
        for idx in range(n):
            if isinstance(X_list[idx], np.ndarray) == False:
                X_new.append(X_list[idx].detach().cpu().numpy())
            else:
                X_new.append(X_list[idx])
        return X_new

    def projection(self, A_vals, beta_vals, lamda_vals):

        beta_vals = self.convert_to_numpy(beta_vals)
        A_vals = self.convert_to_numpy(A_vals)
        if isinstance(lamda_vals, np.ndarray) == False:
            lamda_vals = lamda_vals.detach().cpu().numpy()
        model = gp.Model()

        A_diffs = []
        A_abss = []
        As = []
        g = len(A_vals)

        m = len(lamda_vals)

        for i in range(g):

            A_g = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=1, vtype=gp.GRB.CONTINUOUS, name='A_g' + str(i))
            A_diff = model.addMVar(len(A_vals[i].flatten()), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                   vtype=gp.GRB.CONTINUOUS,
                                   name='A_g' + str(i))
            A_abs = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                  name='A_g' + str(i))

            mn = len(A_vals[i].flatten())
            As.append(A_g)
            A_diffs.append(A_diff)

            Aval = A_vals[i].flatten()
            for jdx in range(mn):
                model.addConstr(A_diff[jdx] == Aval[jdx] - A_g[jdx], name='c1' + str(i) + str(jdx))
                model.addConstr(A_abs[jdx] == gp.abs_(A_diff[jdx]), name='c1' + str(i) + str(jdx))

            n_agents = self.A_list[i].shape[0]
            n_items = self.A_list[i].shape[1]
            covs_lb = self.covs_lb_list[i].flatten()
            covs_ub = self.covs_ub_list[i].flatten()
            C = self.coi_mask_list[i].flatten()

            model.addConstrs(A_g[i] <= C[i] for i in range(mn))

            model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                             range(n_items))
            model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                             range(n_items))

            A_abss.append(A_abs)

        load_sum = model.addMVar(self.loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS,
                                 name='load_sum')

        model.addConstrs(load_sum[idx] == gp.quicksum(
            As[gdx][idx * self.mu_list[gdx].shape[1]:(idx + 1) * (self.mu_list[gdx].shape[1])].sum() for gdx in
            range(self.ngroups)) for
                         idx in range(self.loads.size))
        total_agents = self.loads.size
        model.addConstrs(load_sum[idx] <= self.loads[idx] for idx in range(total_agents))

        model.setObjective(gp.quicksum(
            gp.quicksum(A_abss[jdx][idx] ** 2 for idx in range(len(self.A_list[jdx].flatten()))) for jdx in range(g)),
            gp.GRB.MINIMIZE)
        model.setParam('OutputFlag', 1)
        model.setParam("TimeLimit", 600)

        model.optimize()
        projected_As = []
        projected_betas = []

        for idx in range(g):
            A = np.array(As[idx].X).reshape(A_vals[idx].shape)
            beta = np.clip(beta_vals[idx], a_min=0, a_max=np.inf)
            projected_As.append(A)
            projected_betas.append(beta)
        projected_lamda = np.clip(lamda_vals, a_min=0, a_max=np.inf)
        return projected_As, projected_betas, projected_lamda

    def compute_gradient(self):

        output = self.func()
        output.backward()
        A_grads = []
        beta_grads = []
        for gdx in range(self.ngroups):
            A_grads.append(self.A_tl[gdx].grad)
            beta_grads.append(self.beta_tns[gdx].grad)
            lamda_grads = self.lamda_tns.grad
        return A_grads, beta_grads, lamda_grads


class ComputeGroupEgalitarianQuadraticProj():
    def __init__(self, mu_list, covs_lb_l, covs_ub_l, coi_mask_l, loads, Sigma_list, rad_list, step_size, n_iter=1000):

        self.mu_list = mu_list
        self.Sigma_list = Sigma_list
        self.rad_list = rad_list
        self.covs_lb_list = covs_lb_l
        self.covs_ub_list = covs_ub_l
        self.coi_mask_list = coi_mask_l
        self.loads = loads
        self.step_size = step_size
        self.n_iter = n_iter
        self.timestamps = []
        self.obj_vals = []

        self.eta = .1

        self.ngroups = len(self.mu_list)
        self.nA_list = []
        self.nI_list = []
        for idx in range(self.ngroups):
            nA = self.mu_list[idx].shape[0]
            nI = self.mu_list[idx].shape[1]
            self.nA_list.append(nA)
            self.nI_list.append(nI)

        self.beta_list = [torch.zeros(self.mu_list[idx].shape) for idx in range(self.ngroups)]
        self.A_list = [torch.zeros(self.mu_list[idx].shape) for idx in range(self.ngroups)]

        self.lamda = np.zeros(self.ngroups)
        self.convert_to_tensors()

    def convert_to_tensors(self):
        self.mu_tl = []
        self.A_tl = []
        self.beta_tns = []
        self.Lamda_tns = None
        self.sigma_tns = []
        self.coi_tns = []
        self.Lamda_tns = torch.rand(self.ngroups, requires_grad=True)

        params = []
        params.append(self.Lamda_tns)

        for gdx in range(self.ngroups):
            self.mu_tl.append(torch.Tensor(self.mu_list[gdx]))
            self.beta_tns.append(torch.rand(self.beta_list[gdx].shape, requires_grad=True))
            self.A_tl.append(torch.rand(self.A_list[gdx].shape, requires_grad=True))
            self.sigma_tns.append(torch.Tensor(np.diag(self.Sigma_list[gdx].flatten())))
            self.coi_tns.append(torch.Tensor(self.coi_mask_list[gdx]))
            params.append(self.A_tl[gdx])
            params.append(self.beta_tns[gdx])

        self.optimizer = torch.optim.Adam(params, lr=self.step_size)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max')

    def welfare(self):
        welfares = []
        for gdx in range(self.ngroups):
            Ag = self.A_tl[gdx].flatten()
            gsize = self.A_tl[gdx].shape[1]
            Bg = self.beta_tns[gdx].flatten()
            Vg = self.mu_tl[gdx].flatten()
            Sigma_g = self.sigma_tns[gdx]
            Cg = self.coi_tns[gdx].flatten()
            term1 = torch.sum((Cg * (Ag/gsize - Bg)).flatten() * Vg.flatten())
            temp = (Cg * (Ag/gsize - Bg)).reshape(-1, 1)
            term2 = -(torch.mm(torch.mm(temp.t(), Sigma_g), temp)) / (4 * (self.Lamda_tns[gdx] + 1e-5))
            term3 = -self.Lamda_tns[gdx] * self.rad_list[gdx] ** 2
            w = (term1 + term2 + term3)
            welfares.append(w.detach().cpu().numpy())

        return welfares

    def func(self):

        # term_sum = 0.0
        terms = torch.zeros(self.ngroups)
        for gdx in range(self.ngroups):
            Ag = self.A_tl[gdx].flatten()
            gsize = self.A_tl[gdx].shape[1]

            Bg = self.beta_tns[gdx].flatten()
            Vg = self.mu_tl[gdx].flatten()
            Cg = self.coi_tns[gdx].flatten()
            Sigma_g = self.sigma_tns[gdx]
            term1 = torch.sum((Cg * (Ag/gsize - Bg)).flatten() * Vg.flatten())
            temp = (Cg * (Ag/gsize - Bg)).reshape(-1, 1)
            print(temp)
            term2 = -(torch.mm(torch.mm(temp.t(), Sigma_g), temp)) / (4 * (self.Lamda_tns[gdx] + 1e-3))
            term3 = -self.Lamda_tns[gdx] * self.rad_list[gdx]
            print(term1, term2, term3)
            # term = torch.exp(-1 * self.eta * (term1 + term2 + term3))
            # print(term)
            # term_sum = term_sum + term
            terms[gdx] += term1 + term2[0, 0] + term3
            print()
        return -1 * torch.min(terms)
        # soft_min = (-1.0 / self.eta) * torch.log((1.0 / self.ngroups) * term_sum)
        # return -soft_min

    def gradient_descent(self):
        loss_BGD = []

        st = time.time()

        for i in range(self.n_iter):
            loss = self.func()
            print(f"Iter {i} Loss {loss}")
            # storing the calculated loss in a list
            loss_BGD.append(loss.item())
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            self.optimizer.step()
            for idx in range(self.ngroups):
                self.A_tl[idx].grad.data.zero_()
                self.beta_tns[idx].grad.data.zero_()
            self.Lamda_tns.grad.data.zero_()
            projected_A, projected_beta, projected_lamda = self.projection(self.A_tl, self.beta_tns, self.Lamda_tns)
            for idx in range(self.ngroups):
                self.A_tl[idx].data = torch.Tensor(projected_A[idx])
                self.beta_tns[idx].data = torch.Tensor(projected_beta[idx])
            self.Lamda_tns.data = torch.Tensor(projected_lamda)
            welfares = self.welfare()
            worst_w = np.min(welfares)
            sum_w = np.sum(welfares)
            self.scheduler.step(worst_w)
            print(f'Iter: {i}, \tLoss: {loss.item()} Worst welfare: {worst_w} Welfare sum: {sum_w}')
            self.timestamps.append(time.time() - st)
            self.obj_vals.append(worst_w)

        return self.A_tl, self.beta_tns, self.Lamda_tns, self.timestamps, self.obj_vals

    def convert_to_numpy(self, X_list):
        n = len(X_list)
        X_new = []
        for idx in range(n):
            if isinstance(X_list[idx], np.ndarray) == False:
                X_new.append(X_list[idx].detach().cpu().numpy())
            else:
                X_new.append(X_list[idx])
        return X_new

    def projection(self, A_vals, beta_vals, lamda_vals):
        beta_vals = self.convert_to_numpy(beta_vals)
        A_vals = self.convert_to_numpy(A_vals)
        if isinstance(lamda_vals, np.ndarray) == False:
            lamda_vals = lamda_vals.detach().cpu().numpy()
        model = gp.Model()

        A_diffs = []
        A_abss = []
        As = []
        g = len(A_vals)

        m = len(lamda_vals)

        for i in range(g):

            A_g = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=1, vtype=gp.GRB.CONTINUOUS, name='A_g' + str(i))

            A_diff = model.addMVar(len(A_vals[i].flatten()), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                   vtype=gp.GRB.CONTINUOUS,
                                   name='A_g' + str(i))
            A_abs = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                  name='A_g' + str(i))

            mn = len(A_vals[i].flatten())
            As.append(A_g)
            A_diffs.append(A_diff)

            Aval = A_vals[i].flatten()
            for jdx in range(mn):
                model.addConstr(A_diff[jdx] == Aval[jdx] - A_g[jdx], name='c1' + str(i) + str(jdx))
                model.addConstr(A_abs[jdx] == gp.abs_(A_diff[jdx]), name='c1' + str(i) + str(jdx))

            n_agents = self.A_list[i].shape[0]
            n_items = self.A_list[i].shape[1]
            covs_lb = self.covs_lb_list[i].flatten()
            covs_ub = self.covs_ub_list[i].flatten()
            C = self.coi_mask_list[i].flatten()

            model.addConstrs(A_g[i] <= C[i] for i in range(mn))

            model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                             range(n_items))
            model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                             range(n_items))

            A_abss.append(A_abs)

        load_sum = model.addMVar(self.loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS,
                                 name='load_sum')

        model.addConstrs(load_sum[idx] == gp.quicksum(
            As[gdx][idx * self.mu_list[gdx].shape[1]:(idx + 1) * (self.mu_list[gdx].shape[1])].sum() for gdx in
            range(self.ngroups)) for
                         idx in range(self.loads.size))
        total_agents = self.loads.size
        model.addConstrs(load_sum[idx] <= self.loads[idx] for idx in range(total_agents))

        model.setObjective(gp.quicksum(
            gp.quicksum(A_abss[jdx][idx] ** 2 for idx in range(len(self.A_list[jdx].flatten()))) for jdx in range(g)),
            gp.GRB.MINIMIZE)
        model.setParam('OutputFlag', 1)
        model.setParam("TimeLimit", 600)

        model.optimize()
        projected_As = []
        projected_betas = []
        projected_lamda = None

        for idx in range(g):
            A = np.array(As[idx].X).reshape(A_vals[idx].shape)
            beta = np.clip(beta_vals[idx], a_min=0, a_max=np.inf)
            projected_As.append(A)
            projected_betas.append(beta)
        projected_lamda = np.clip(lamda_vals, a_min=0, a_max=np.inf)
        return projected_As, projected_betas, projected_lamda

    def compute_gradient(self):

        output = self.func()
        output.backward()
        A_grads = []
        beta_grads = []
        for gdx in range(self.ngroups):
            A_grads.append(self.A_tl[gdx].grad)
            beta_grads.append(self.beta_tns[gdx].grad)
            lamda_grads = self.lamda_tns.grad
        return A_grads, beta_grads, lamda_grads


def project_to_feasible(group_allocs, covs_lb_l, covs_ub_l, loads):
    A_vals = group_allocs

    model = gp.Model()

    A_diffs = []
    A_abss = []
    As = []
    g = len(A_vals)

    for i in range(g):
        A_g = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=1, vtype=gp.GRB.CONTINUOUS, name='A_g' + str(i))

        A_diff = model.addMVar(len(A_vals[i].flatten()), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                               vtype=gp.GRB.CONTINUOUS,
                               name='A_g' + str(i))

        A_abs = model.addMVar(len(A_vals[i].flatten()), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                              name='A_g' + str(i))

        mn = len(A_vals[i].flatten())
        As.append(A_g)
        A_diffs.append(A_diff)

        Aval = A_vals[i].flatten()
        for jdx in range(mn):
            model.addConstr(A_diff[jdx] == Aval[jdx] - A_g[jdx], name='c1' + str(i) + str(jdx))
            model.addConstr(A_abs[jdx] == gp.abs_(A_diff[jdx]), name='c1' + str(i) + str(jdx))

        n_agents = A_vals[i].shape[0]
        n_items = A_vals[i].shape[1]
        covs_lb = covs_lb_l[i].flatten()
        covs_ub = covs_ub_l[i].flatten()

        model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) >= covs_lb[idx] for idx in
                         range(n_items))
        model.addConstrs(gp.quicksum(A_g[jdx * n_items + idx] for jdx in range(n_agents)) <= covs_ub[idx] for idx in
                         range(n_items))

        A_abss.append(A_abs)

    load_sum = model.addMVar(loads.size, lb=0, ub=gp.GRB.INFINITY, obj=0.0, vtype=gp.GRB.CONTINUOUS,
                             name='load_sum')

    model.addConstrs(load_sum[idx] == gp.quicksum(
        As[gdx][idx * A_vals[gdx].shape[1]:(idx + 1) * (A_vals[gdx].shape[1])].sum() for gdx in
        range(g)) for
                     idx in range(loads.size))
    total_agents = loads.size
    model.addConstrs(load_sum[idx] <= loads[idx] for idx in range(total_agents))

    model.setObjective(gp.quicksum(gp.quicksum(
        A_abss[jdx][idx] ** 2 for idx in range(len(A_vals[jdx].flatten()))) for jdx
                                   in range(g)), gp.GRB.MINIMIZE)
    model.setParam('OutputFlag', 1)
    model.setParam("TimeLimit", 600)

    model.optimize()
    projected_As = []

    for idx in range(g):
        A = np.array(As[idx].X).reshape(A_vals[idx].shape)
        projected_As.append(A)
    return projected_As


def get_worst_case_usw(group_allocs, group_mus, group_variances, rhs_bd_per_group):
    print(rhs_bd_per_group)
    m = gp.Model()

    ngroups = len(group_allocs)

    obj_terms = []

    vs = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)

        a = group_allocs[gidx]
        ce = group_mus[gidx]
        var = group_variances[gidx]
        rhs_bd = rhs_bd_per_group[gidx]

        v = m.addMVar(ce.shape)

        m.addConstr(((v - ce) * (1 / var) * (v - ce)).sum() <= rhs_bd ** 2)

        m.addConstr(v >= 0)
        obj_terms.append((a * v).sum())
        vs.append(v)
    obj = gp.quicksum(t for t in obj_terms)
    m.setObjective(obj)
    m.optimize()
    m.setParam('OutputFlag', 1)

    return obj.getValue(), [v.X for v in vs]


def subgrad_ascent_util_ellipsoid(mu_list, covs_lb_l, covs_ub_l, loads, Sigma_list, rad_list):
    group_allocs = [np.clip(np.random.randn(mu.shape[0], mu.shape[1]), 0, 1) for mu in mu_list]

    global_opt_obj = -np.inf
    prev_obj_val = -np.inf
    global_opt_alloc = [ga.copy() for ga in group_allocs]

    t = 0
    converged = False
    max_iter = 1000

    iter_timestamps = []
    iter_obj_vals = []

    st = time.time()

    while not converged and t < max_iter:
        # Project to the set of feasible allocations
        print("Projecting to feasible: %s elapsed" % (time.time() - st))
        group_allocs = project_to_feasible(group_allocs, covs_lb_l, covs_ub_l, loads)

        # Compute the worst-case V matrix
        print("Computing worst case V matrix")
        print("%s elapsed" % (time.time() - st))
        obj_val, worst_vs = get_worst_case_usw(group_allocs, mu_list, Sigma_list, rad_list)

        if obj_val > global_opt_obj:
            global_opt_obj = obj_val
            global_opt_alloc = group_allocs
        t += 1

        converged = (np.abs(prev_obj_val - obj_val) < 1e-3)

        prev_obj_val = obj_val

        rate = 1 / (t + 1)
        group_allocs = [a + rate * v for a, v in zip(group_allocs, worst_vs)]

        if t % 1 == 0:
            print("Step %d" % t)
            print("Obj value: ", obj_val)
            print("%s elapsed" % (time.time() - st))

        iter_timestamps.append(time.time() - st)
        iter_obj_vals.append(obj_val)

    return global_opt_alloc, iter_timestamps, iter_obj_vals


def get_worst_case_gesw(group_allocs, group_mus, group_variances, rhs_bd_per_group):
    print(rhs_bd_per_group)
    m = gp.Model()

    ngroups = len(group_allocs)

    gesw = m.addVar()
    aux_vars = m.addVars(ngroups, vtype=gp.GRB.CONTINUOUS)

    obj_terms = []

    vs = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)

        a = group_allocs[gidx]
        ce = group_mus[gidx]
        var = group_variances[gidx]
        rhs_bd = rhs_bd_per_group[gidx]
        grpsize = ce.shape[1]

        v = m.addMVar(ce.shape)

        m.addConstr(((v - ce) * (1 / var) * (v - ce)).sum() <= rhs_bd ** 2)

        m.addConstr(v >= 0)
        m.addConstr(aux_vars[gidx] == (a * v).sum() / grpsize)

        vs.append(v)
    m.addConstr(gesw == gp.min_(aux_vars))
    m.setObjective(gesw)
    m.optimize()
    m.setParam('OutputFlag', 1)

    worst_group = np.argmin([av.X for av in aux_vars])

    return gesw.getValue(), [v.X for v in vs], worst_group


def subgrad_ascent_egal_ellipsoid(mu_list, covs_lb_l, covs_ub_l, loads, Sigma_list, rad_list):
    group_allocs = [np.clip(np.random.randn(mu.shape[0], mu.shape[1]), 0, 1) for mu in mu_list]

    global_opt_obj = -np.inf
    global_opt_alloc = [ga.copy() for ga in group_allocs]

    t = 0
    converged = False
    max_iter = 1000

    iter_timestamps = []
    iter_obj_vals = []

    st = time.time()

    while not converged and t < max_iter:
        # Project to the set of feasible allocations
        print("Projecting to feasible: %s elapsed" % (time.time() - st))
        group_allocs = project_to_feasible(group_allocs, covs_lb_l, covs_ub_l, loads)

        # Compute the worst-case V matrix
        print("Computing worst case V matrix")
        print("%s elapsed" % (time.time() - st))
        obj_val, worst_vs, worst_group = get_worst_case_gesw(group_allocs, mu_list, Sigma_list, rad_list)

        if obj_val > global_opt_obj:
            global_opt_obj = obj_val
            global_opt_alloc = group_allocs
        t += 1

        converged = (np.abs(prev_obj_val - obj_val) < 1e-3)

        prev_obj_val = obj_val

        rate = 1 / (t + 1)
        group_allocs[worst_group] += rate * worst_vs[worst_group]

        if t % 1 == 0:
            print("Step %d" % t)
            print("Obj value: ", obj_val)
            print("%s elapsed" % (time.time() - st))

        iter_timestamps.append(time.time() - st)
        iter_obj_vals.append(obj_val)

    return global_opt_alloc, iter_timestamps, iter_obj_vals


def check_ellipsoid(Sigma, mu, x, rsquared):
    temp = (x - mu).reshape(-1, 1)
    temp1 = np.matmul(temp.transpose(), Sigma)
    temp2 = np.matmul(temp1.reshape(1, -1), temp)

    if temp2.flatten()[0] <= rsquared:
        return True
    else:
        return False


def softtime(model, where):
    softlimit = 5
    gaplimit = 0.05
    if where == gp.GRB.Callback.MIP:
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
        objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)

        if runtime > softlimit and gap < gaplimit:
            model.terminate()


def run():
    n = np.random.randint(10, 100)

    n_reviewers = 10
    n_papers = 10
    n = n_reviewers * n_papers
    c = np.random.uniform(0.1, 1, n)
    k = np.random.uniform(0.1, 1, n)
    ksquared = k * k
    sigma = np.eye(n) * ksquared
    mu = np.random.uniform(0.1, 1, n)
    coi_mask = np.oneslike(mu)
    p = np.random.rand()
    df = np.random.randint(1, 10)
    from scipy.stats import chi2
    rsquared = chi2.ppf(p, df=df)
    loads = np.ones(n_reviewers) * n_papers
    covs = np.random.randint(1, n_reviewers, n_papers)

    std_devs = np.sqrt(np.diag(sigma))
    mu = mu.reshape((n_reviewers, n_papers))
    ngroups = 3
    k = int(n_reviewers / ngroups)
    step_size = 1e-1
    mu_list = [mu[:, 0:k], mu[:, k:2 * k], mu[:, 2 * k:]]
    coi_list = [coi_mask[:, 0:k], coi_mask[:, k:2 * k], coi_mask[:, 2 * k:]]

    covs_list = [np.random.randint(1, n_reviewers, mu_list[0].shape[1]),
                 np.random.randint(1, n_reviewers, mu_list[1].shape[1]),
                 np.random.randint(1, n_reviewers, mu_list[2].shape[1])]
    loads_list = loads

    rad_list = [rsquared for x in range(ngroups)]

    Sigma_list = [np.random.uniform(0.1, 1, len(mu_list[idx].flatten())) for idx in range(ngroups)]

    # egalObject =    ComputeGroupEgalitarianQuadratic(mu_list, covs_list, loads_list, Sigma_list, rad_list, eta, step_size, n_iter=1000)
    # egalObject.gradient_descent()

    egalObject = ComputeGroupEgalitarianQuadraticProj(mu_list, covs_list, covs_list, coi_list, loads_list, Sigma_list,
                                                      rad_list, step_size, n_iter=1000)
    egalObject.gradient_descent()

    # Util = UtilitarianAlternation(mu_list, covs_list, covs_list, loads_list, Sigma_list, rad_list, integer=False)
    # alloc, _, _, iter_times, iter_objs = Util.iterative_optimization(group_welfare=False)
    # print(iter_times, iter_objs)


if __name__ == '__main__':
    run()
