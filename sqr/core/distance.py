import numpy as np
import networkx as nx
import itertools
import pandas as pd

# crude distance measures:
# maximal distance across bounding rectangle

mean_col = 'mean'

def euclidian(arr):
    return np.sqrt(np.sum(np.power(arr, 2)))

def unweighted(group):
    dist_hor_ver = group[['e','n']].apply(lambda c: c.max()-c.min()+1, axis=0)
    return euclidian(dist_hor_ver)

def weighted(group):
    pop_weight = group.pop_weight.iloc[0] # same for all in group - choose first
    dist_hor_ver = group[['e','n']].apply(lambda c: c.max()-c.min()+1, axis=0)
    return euclidian(dist_hor_ver) * pop_weight

def unweighted_pop(group_index, df):
    return unweighted(df.loc[group_index]) * df.loc[group_index, mean_col].sum()
