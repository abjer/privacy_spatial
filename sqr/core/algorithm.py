import itertools

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.neighbors import KDTree

import sqr.core.network
import sqr.core.scoring

from sqr.core.config import years, years_num, years, years_hh, years_pers, minimum_cols

def check_pop(df, idxs, base_years):   
    '''
    Checks whether population constraints are satisfied for the set 
    of indices in the DataFrame, in the years specified.
    
    Parameters
    -----------
    df: pd.DataFrame
        This dataframe must contain cell level population data.
    
    idxs: 1-d array
        Contains indices for subsample of dataframe.
        
    base_years: 1-d array
        Contains years as strings.
    '''
    
    check = True
    for label, threshold in  [('hh', 50),('pers', 100)]:
        check_years = [f'{c}_{label}' for c in base_years]
        count = \
            df.loc[idxs, check_years]\
                .dropna(how='all',axis=1)\
                .sum(axis=0)\
                .min()
        check &= (count >=  threshold)
    return check

def cell_expansion(i, unassigned, df, G, max_group_attempts,
                   year_cols, check_connectivity):
    '''
    This function begins with a single individual and tries
    to attach it to neighboring cells. This collection
    is evaluated repeatedly. If it's value exceeds 100
    the operation terminates succesfully with a novel group.

    Note that the search is limited to neighbors as specified
    in the graph neighboring tile cells.
    '''


    # initialize group
    current_group = np.array([i])
    current_group_neighbors = np.array([i])
    temp_unassigned = np.setdiff1d(unassigned, [i])
    group_attempts = 0

    while True:

        # choose next cell
        i = np.random.choice(current_group_neighbors)
        feasible_neighbors = np.intersect1d(temp_unassigned, list(G.neighbors(i)))
        if feasible_neighbors.size > 0:
            j = np.random.choice(feasible_neighbors)
        else:
            current_group_neighbors = np.setdiff1d(current_group_neighbors, [i])

            if group_attempts >= max_group_attempts:
                return False, unassigned, None

            if current_group_neighbors.size==0:
                return False, unassigned, None

            group_attempts += 1
            continue

        next_group = np.union1d(current_group,[j])

        temp_unassigned = np.setdiff1d(temp_unassigned, [j])

        if check_pop(df, next_group, base_years = year_cols): #Rettet

            if check_connectivity:
                # check if null observation is necessary, otherwise remove
                if len(next_group) > 1:
                    group_df = df.loc[next_group]

                    if (group_df[minimum_cols].isnull().min(1)).sum() > 0:
                        next_group, removal = remove_null(next_group, df=df, G=G) 
                        temp_unassigned = np.union1d(temp_unassigned, removal)

            return True, temp_unassigned, next_group

        else:
            current_group = next_group

            current_group_neighbors = np.union1d(current_group_neighbors, [j])

            if group_attempts >= max_group_attempts:
                return False, unassigned, None

            group_attempts += 1




def remove_null(group, df, G): 
    '''
    This function attempts to remove all null entries
    if they are not critical for connecting the group
    '''

    sub_G = G.subgraph(group)
    group_df = df.loc[group]
 
    

    empty = group_df[minimum_cols].isnull().min(1)
    X_e = group_df.loc[empty, ['e','n']].values
    X_ne = group_df.loc[~empty, ['e','n']].values

    tree = KDTree(X_ne)
    dists = [4,3,2,1]
    has_neighbors = [list(map(lambda l: len(l)>0, tree.query_radius(X_e, d))) for d in dists]

    sorted_neighbors = \
        pd.DataFrame(np.array(has_neighbors).T, group_df[empty].index, dists)\
            .sort_values(by=dists)
    removal = []

    for idx_null in sorted_neighbors.index:
        g_n = list(set(group)-set([idx_null]))
        if sqr.core.network.group_connect(g_n, sub_G):
            removal += [idx_null]
            group = np.setdiff1d(group,[idx_null])


    return group, removal


def assign_selfsufficient(partition, df, unassigned_pop): #Rettet
    '''
    Remove cells with at least 100 inhabitants. These become
    groups with a single unique member, i.e. themselves.
    '''
    
    # check if cell is self-sufficient
    check_pers = df[minimum_cols[0]]>=100
    check_hh = df[minimum_cols[1]]>=50

    selfsufficient = df[check_pers & check_hh].index

    unassigned_selfsuff = np.intersect1d(selfsufficient, unassigned_pop)
    partition += [[cell] for cell in unassigned_selfsuff]
    unassigned_pop = np.setdiff1d(unassigned_pop, unassigned_selfsuff)

    return partition, unassigned_pop

def assign_remaining(partition, unassigned_pop, df, G, year_cols):
    '''
    All unassigned cells are attempted to be merged with other
    close cells.
    '''

    for i in unassigned_pop:

        optimals = {}
        neighbor_groups = [g for g in partition if len(set(G.neighbors(i)) & set(g))>0]

        for group in neighbor_groups:
            if check_pop(df, np.union1d(group,[i]), base_years=year_cols):
                optimals[partition.index(group)] = \
                    sqr.core.scoring.group_collapse(group, [i], df)

        optimals = pd.Series(optimals)

        if optimals.max()>0:
            partition[int(optimals.idxmax())] += [i]
            unassigned_pop = np.setdiff1d(unassigned_pop, [i])

    return partition, unassigned_pop
