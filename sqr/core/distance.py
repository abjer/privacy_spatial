import numpy as np
import networkx as nx
import itertools
import pandas as pd


from sqr.core.config import mean_cols, minimum_cols

# crude distance measures:
# maximal distance across bounding rectangle

def euclidian(arr):
    return np.sqrt(np.sum(np.power(arr, 2)))

def unweighted(group):
    dist_hor_ver = group[['e','n']].apply(lambda c: c.max()-c.min()+1, axis=0)
    return euclidian(dist_hor_ver)

def weighted_pers(group):    
    pers_weight = group.pers_weight.iloc[0] # same for all in group - choose first
    dist_hor_ver = group[['e','n']].apply(lambda c: c.max()-c.min()+1, axis=0)
    return euclidian(dist_hor_ver) * pers_weight


def weighted_hh(group):    
    hh_weight = group.hh_weight.iloc[0] # same for all in group - choose first
    dist_hor_ver = group[['e','n']].apply(lambda c: c.max()-c.min()+1, axis=0)
    return euclidian(dist_hor_ver) * hh_weight


# def unweighted_pop_old(group_index, df):
#     return unweighted(df.loc[group_index]) * df.loc[group_index, mean_col].sum()

def total_distance(group_index, df):
    a, b = df[mean_cols].mean()
    ratio_pers_hh = a/b 
    
    dist = unweighted(df.loc[group_index])
    
    pers_count = df.loc[group_index, mean_cols[0]].sum()
    hh_count = df.loc[group_index, mean_cols[0]].sum()
    
    return (pers_count * 0.5 + hh_count * ratio_pers_hh * 0.5) * dist
