import numpy as np


def compute_usw(allocation, central_estimate):
    return np.sum(allocation * central_estimate)

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

def compute_adv_usw_linear(central_estimate, coi_mask, rhs_bd_per_group, groups):

    return 0.0

def compute_adv_gesw_linear(central_estimate, coi_mask, rhs_bd_per_group, groups):
    return 0.0