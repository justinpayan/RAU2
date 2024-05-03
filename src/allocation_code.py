import numpy as np
import re

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


def add_constrs_to_model(m, x, covs, loads):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    m.addConstrs((x.sum(paper, '*') >= covs[paper] for paper in papers), 'covs')  # Paper coverage constraints
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


def solve_usw_gurobi(affinity_scores, covs, loads):
    paper_rev_pairs, pras = create_multidict(affinity_scores)

    m = Model("TPMS")

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs, loads)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    m.optimize()

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_mat(m, covs.shape[0], loads.shape[0])

    return m.objVal, alloc