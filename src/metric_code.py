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