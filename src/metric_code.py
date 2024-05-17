
import gurobipy as gp
import numpy as np


def compute_usw(allocation, central_estimate):
    return np.sum(allocation * central_estimate)/allocation.shape[1]

def compute_gesw(allocation, central_estimate, groups):
    ngroups = len(set(groups))
    welfares = []
    for gidx in range(ngroups):
        gmask = np.where(groups == gidx)[0]
        welfares.append(compute_usw(allocation[:, gmask], central_estimate[:, gmask]))
    return np.min(welfares)

def compute_cvar_usw(allocation, value_samples, conf_level):
    usws = []
    for vs in value_samples:
        usws.append(compute_usw(allocation, vs))
    cutoff = int(len(usws)*conf_level)
    return np.mean(sorted(usws)[:cutoff])

def compute_cvar_gesw(allocation, value_samples, groups, conf_level):
    gesws = []
    for vs in value_samples:
        gesws.append(compute_gesw(allocation, vs, groups))
    cutoff = int(len(gesws)*conf_level)
    return np.mean(sorted(gesws)[:cutoff])


def compute_adv_usw_linear(allocation, central_estimate, coi_mask, rhs_bd_per_group, groups, a_val=1, b_val=0):
    m = gp.Model()

    ngroups = len(set(groups))

    obj_terms = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)
        gmask = np.where(groups == gidx)[0]

        a = allocation[:, gmask]
        ce = central_estimate[:, gmask]
        cm = coi_mask[:, gmask]
        rhs_bd = rhs_bd_per_group[gidx]

        v = m.addMVar(ce.shape)
        aux = m.addMVar(ce.shape)

        x = np.log(1 - ce)
        y = np.log(ce)

        c_times_x_minus_y = cm * (x - y)
        c_times_x = cm * x

        lhs = aux.sum()

        m.addConstr(aux == v * c_times_x_minus_y - c_times_x)
        m.addConstr(lhs <= np.sum(cm) * rhs_bd)
        m.addConstr(v >= 0)
        m.addConstr(v <= 1)
        obj_terms.append((a * v).sum())
    obj = gp.quicksum(t for t in obj_terms)
    m.setObjective(obj)
    m.optimize()
    m.setParam('OutputFlag', 1)

    return ((a_val-b_val)*obj.getValue() + b_val*np.sum(allocation))/allocation.shape[1]


def compute_adv_gesw_linear(allocation, central_estimate, coi_mask, rhs_bd_per_group, groups, a_val=1, b_val=0):
    m = gp.Model()

    ngroups = len(set(groups))

    gesw = m.addVar()
    aux_vars = m.addVars(ngroups, vtype=gp.GRB.CONTINUOUS)

    grpsizes = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)
        gmask = np.where(groups == gidx)[0]
        grpsize = gmask.shape[0]
        grpsizes.append(grpsize)

        a = allocation[:, gmask]
        ce = central_estimate[:, gmask]
        nagents, nitems = ce.shape
        cm = coi_mask[:, gmask]
        rhs_bd = rhs_bd_per_group[gidx]

        v = m.addMVar(ce.shape)
        aux = m.addMVar(ce.shape)

        x = np.log(1 - ce)
        y = np.log(ce)

        c_times_x_minus_y = cm * (x - y)
        c_times_x = cm * x

        lhs = aux.sum()

        m.addConstr(aux == v * c_times_x_minus_y - c_times_x)
        m.addConstr(lhs <= np.sum(cm) * rhs_bd)
        m.addConstr(v >= 0)
        m.addConstr(v <= 1)
        m.addConstr(aux_vars[gidx] == (a * v).sum()/grpsize)

    m.addConstr(gesw == gp.min_(aux_vars))

    m.setObjective(gesw)
    m.optimize()
    m.setParam('OutputFlag', 1)

    return (a_val-b_val)*gesw.X + b_val

def compute_adv_usw_ellipsoidal(allocation, central_estimate, std_devs, rhs_bd_per_group, groups):
    m = gp.Model()

    ngroups = len(set(groups))

    obj_terms = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)
        gmask = np.where(groups == gidx)[0]

        a = allocation[:, gmask]
        ce = central_estimate[:, gmask]
        sd = std_devs[:, gmask]
        rhs_bd = rhs_bd_per_group[gidx]

        v = m.addMVar(ce.shape)

        m.addConstr(((v - ce)*sd*(v-ce)).sum() <= rhs_bd**2)

        m.addConstr(v >= 0)
        obj_terms.append((a * v).sum())
    obj = gp.quicksum(t for t in obj_terms)
    m.setObjective(obj)
    m.optimize()
    m.setParam('OutputFlag', 1)

    return obj.getValue()/allocation.shape[1]


def compute_adv_gesw_ellipsoidal(allocation, central_estimate, std_devs, rhs_bd_per_group, groups):
    m = gp.Model()

    ngroups = len(set(groups))

    gesw = m.addVar()
    aux_vars = m.addVars(ngroups, vtype=gp.GRB.CONTINUOUS)

    grpsizes = []

    for gidx in range(ngroups):
        print("setting up group ", gidx)
        gmask = np.where(groups == gidx)[0]
        grpsize = gmask.shape[0]
        grpsizes.append(grpsize)

        a = allocation[:, gmask]
        ce = central_estimate[:, gmask]
        sd = std_devs[:, gmask]
        rhs_bd = rhs_bd_per_group[gidx]

        v = m.addMVar(ce.shape)

        m.addConstr(((v - ce)*sd*(v-ce)).sum() <= rhs_bd**2)
        m.addConstr(v >= 0)
        m.addConstr(aux_vars[gidx] == (a * v).sum()/grpsize)

    m.addConstr(gesw == gp.min_(aux_vars))

    m.setObjective(gesw)
    m.optimize()
    m.setParam('OutputFlag', 1)

    return gesw.X