import itertools

import numpy as np
import pandas as pd
import networkx as nx

import sqr.core.network
import sqr.core.scoring

year_cols_all = list(map(str,range(1986,2016)))

mean_col = 'mean'
minimum_col='minimum'


def check_pop(df, group, check_year_cols = year_cols_all):
        select = group,check_year_cols
        min_total = df.loc[select]\
                .dropna(how='all',axis=1)\
                .sum(axis=0)\
                .min()
        return min_total >= 100

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

        if check_pop(df, next_group, check_year_cols = year_cols): #Rettet

            if check_connectivity:
                # check if null observation is necessary, otherwise remove
                if len(next_group) > 1:
                    group_df = df.loc[next_group]

                    if (group_df[minimum_col].isnull()).sum() > 0:
                        next_group, removal = remove_null(next_group, df=df, G=G) #Rettet
                        temp_unassigned = np.union1d(temp_unassigned,removal)

            return True, temp_unassigned, next_group

        else:
            current_group = next_group

            current_group_neighbors = np.union1d(current_group_neighbors, [j])

            if group_attempts >= max_group_attempts:
                return False, unassigned, None

            group_attempts += 1




def remove_null(group, df, G): #Rettet
    '''
    This function attempts to remove all null entries
    if they are not critical for connecting the group
    '''

    sub_G = G.subgraph(group)
    group_df = df.loc[group][minimum_col] #Rettet
    empty = group_df.isnull().to_dict()

    empty_dists = [(i,min([(nx.shortest_path_length(sub_G, i, j)) \
             for j in group if not empty[j]])) \
             for i in group if empty[i]]


    empty_group_by_dist = pd.Series(dict(empty_dists)).sort_values(ascending=False).index

    removal = []

    for idx_null in empty_group_by_dist:
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

    selfsufficient = df[df[minimum_col]>=100].index

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
            if check_pop(df, np.union1d(group,[i]), check_year_cols=year_cols):
                optimals[partition.index(group)] = \
                    sqr.core.scoring.group_collapse(group, [i], df)

        optimals = pd.Series(optimals)

        if optimals.max()>0:
            partition[int(optimals.idxmax())] += [i]
            unassigned_pop = np.setdiff1d(unassigned_pop, [i])

    return partition, unassigned_pop
