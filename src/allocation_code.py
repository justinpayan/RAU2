import cvxpy as cp
import numpy as np
import re

from collections import Counter
from gurobipy import Model, multidict, GRB

def create_multidict(pra):
    d = {}
    for rev in range(pra.shape[0]):
        for paper in range(pra.shape[1]):
            d[(paper, rev)] = pra[rev, paper]
    return multidict(d)


def add_vars_to_model(m, paper_rev_pairs):
    x = m.addVars(paper_rev_pairs, name="assign", vtype=GRB.BINARY)  # The binary assignment variables
    return x


def add_constrs_to_model(m, x, covs_lb, covs_ub, loads):
    papers = range(covs_lb.shape[0])
    revs = range(loads.shape[0])

    m.addConstrs((x.sum(paper, '*') <= covs_ub[paper] for paper in papers), 'covs')  # Paper coverage constraints
    m.addConstrs((x.sum(paper, '*') >= covs_lb[paper] for paper in papers), 'covs')  # Paper coverage constraints

    m.addConstrs((x.sum('*', rev) <= loads[rev] for rev in revs), 'loads_ub')  # Reviewer load constraints


def convert_to_mat(m, num_papers, num_revs):
    alloc = np.zeros((num_revs, num_papers))
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[r, p] = 1
    return alloc


def solve_usw_gurobi(affinity_scores, covs_lb, covs_ub, loads):
    paper_rev_pairs, pras = create_multidict(affinity_scores)

    m = Model("TPMS")

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs_lb, covs_ub, loads)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    m.optimize()

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_mat(m, covs_lb.shape[0], loads.shape[0])

    return m.objVal, alloc


def solve_gesw(affinity_scores, covs_lb, covs_ub, loads, groups):
    num_groups = len(set(groups))
    group_indicators = []
    group_size = Counter(groups)
    for g in range(num_groups):
        group_indicators.append(np.zeros(affinity_scores.shape))
    for idx, g in enumerate(groups):
        group_indicators[g][:, idx] = 1 / group_size[g]
    group_indicators = [cp.reshape(gi, (gi.size, 1)) for gi in group_indicators]

    max_gesw_expected_alloc = cp.Variable(affinity_scores.shape, boolean=True)
    y = cp.Variable()

    flat_alloc = cp.reshape(max_gesw_expected_alloc, (max_gesw_expected_alloc.size, 1))
    groups_stacked = cp.hstack(group_indicators)
    flat_alloc_per_group = cp.multiply(flat_alloc, groups_stacked)
    coeffs = cp.reshape(affinity_scores, (1, affinity_scores.size))
    inner_prods_per_group = coeffs @ flat_alloc_per_group
    obj = y

    constr = [cp.sum(max_gesw_expected_alloc, axis=0) >= covs_lb, cp.sum(max_gesw_expected_alloc, axis=0) <= covs_ub,
              cp.sum(max_gesw_expected_alloc, axis=1) <= loads, y >= 0, y <= inner_prods_per_group]

    gesw_problem = cp.Problem(cp.Maximize(obj), constr)

    gesw_problem.solve(verbose=True, solver='GUROBI')

    return max_gesw_expected_alloc.value


def solve_cvar_usw(covs_lb, covs_ub, loads, conf_level, value_samples):
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
              cp.sum(alloc, axis=1) <= loads]

    cvar_usw_problem = cp.Problem(cp.Minimize(obj), constr)

    cvar_usw_problem.solve(verbose=True, solver='GUROBI', reoptimize=True)

    return alloc.value


def solve_cvar_gesw(covs_lb, covs_ub, loads, conf_level, value_samples, groups):
    return None
    # alloc = cp.Variable((loads.size, covs_lb.size), boolean=True)
    # alpha = cp.Variable()
    # beta = conf_level
    # num_samples = len(value_samples)
    # # Beta is the cvar level for the RISK. So at .99, that means we are minimizing the conditional expectation
    # # of the highest 1% of RISK scores, or rather, maximizing the CE of the lowest 1% of GAIN scores.
    #
    # flat_alloc = cp.reshape(alloc, (1, alloc.size))
    # flat_value_samples = [cp.reshape(vs, (vs.size, 1)) for vs in value_samples]
    # inner_prods = [flat_alloc @ vs for vs in flat_value_samples]
    # summands = [cp.pos(-1 * ip - alpha) for ip in inner_prods]
    # obj = cp.sum(summands)
    # obj = alpha + obj / (num_samples * (1 - beta))
    #
    # constr = [cp.sum(alloc, axis=0) >= covs_lb,
    #           cp.sum(alloc, axis=0) <= covs_ub,
    #           cp.sum(alloc, axis=1) <= loads]
    #
    # cvar_usw_problem = cp.Problem(cp.Minimize(obj), constr)
    #
    # cvar_usw_problem.solve(verbose=True, solver='GUROBI', reoptimize=True)
    #
    # return alloc.value