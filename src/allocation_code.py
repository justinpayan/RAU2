import cvxpy as cp
import numpy as np

from collections import Counter
from gurobipy import Model, multidict, GRB


def solve_usw_gurobi(affinity_scores, covs_lb, covs_ub, loads, coi_mask):
    m = Model("TPMS")

    alloc = m.addMVar(affinity_scores.shape, vtype=GRB.BINARY, name='alloc')

    m.addConstr(alloc.sum(axis=0) >= covs_lb)
    m.addConstr(alloc.sum(axis=0) <= covs_ub)
    m.addConstr(alloc.sum(axis=1) <= loads)
    m.addConstr(alloc <= coi_mask)

    obj = (alloc*affinity_scores).sum()
    m.setObjective(obj, GRB.MAXIMIZE)

    m.optimize()

    return alloc.value


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
              cp.sum(alloc, axis=1) <= loads, y >= 0, y <= inner_prods_per_group, alloc <= coi_mask]

    gesw_problem = cp.Problem(cp.Maximize(obj), constr)

    gesw_problem.solve(verbose=True, solver='GUROBI')

    return alloc.value


def solve_cvar_usw(covs_lb, covs_ub, loads, conf_level, value_samples, coi_mask):
    alloc = cp.Variable((loads.size, covs_lb.size), boolean=True)
    alpha = cp.Variable()
    beta = conf_level
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

    cvar_usw_problem.solve(verbose=True, solver='GUROBI', reoptimize=True)

    return alloc.value


def solve_cvar_gesw(covs_lb, covs_ub, loads, conf_level, value_samples, groups, coi_mask):
    shape_tup = (covs_lb.size, loads.size)
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
    beta = conf_level
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

    cvar_gesw_problem.solve(verbose=True, solver='GUROBI')

    return gesw_alloc.value