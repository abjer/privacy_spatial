import copy
import itertools

import numpy as np
import pandas as pd

from sqr.core.algorithm import cell_expansion, assign_selfsufficient, assign_remaining
from sqr.core.config import years, years_hh, years_pers, minimum_cols

def run_assignment(df, G, max_total_attempts,
                   max_group_attempts = 50,
                   in_partition = [],
                   fill_missing = True,
                   year_cols = years,
                   check_connectivity = True):
    """
    Function calculates a partion of the data supplied. The function calls
    subroutines to assign cells to one another in order to fulfill
    the requirement of minimum 100 inhabitants.

    Key arguments:
    df: Pandas dataframe with index corresponding to graph G
    G: NetworkX Graph with index corresponding to df. Contains neighbors
    minimum_cols: Columns of df with minimum (not nan)

    pre_include: If True then cells incircled by partition gets included in partition
    fill_missing: If True algorithm attempts to include non-assigned cells where the do least harm
    missing_lim: Limit of amount in cell before attemting to include

    """

    # initalize nodes and partition
    total_attempts = 0

    partition = copy.deepcopy(in_partition)

    assigned = list(itertools.chain(*partition))
    unassigned = np.setdiff1d(list(G.nodes()), assigned)

    has_pop = df[df[minimum_cols].notnull().max(1)].index
    unassigned_pop =  np.setdiff1d(has_pop, assigned)

    if unassigned_pop.size != 0:

        while True:

            # initialize starting cell
            i = np.random.choice(unassigned_pop)

            # pass if empty cell
            # if pd.isnull(df.loc[i,minimum_col]):
            #     total_attempts += 1
            #     if total_attempts > max_total_attempts:
            #         break
            #     continue

            # check if cell is self-sufficient
            check_pers = df.loc[i,minimum_cols[0]]>=100
            check_hh = df.loc[i,minimum_cols[1]]>=50
            if check_pers & check_hh:
                partition += [[i]]
                unassigned = np.setdiff1d(unassigned, [i])
                unassigned_pop = np.setdiff1d(unassigned_pop, [i])
                if unassigned_pop.size == 0:
                    break


                continue

            # attempt local cell expansion for merging cells
            
            check_solution, unassigned, group = \
                cell_expansion(i=i,
                               unassigned=unassigned,
                               df=df,
                               G=G,
                               max_group_attempts=max_group_attempts,
                               year_cols=year_cols,
                               check_connectivity=check_connectivity)

            if check_solution:
                unassigned_pop = np.setdiff1d(unassigned_pop, group)
                partition+=[list(group)]

            if (unassigned_pop.size == 0) or (total_attempts > max_total_attempts):
                break

            total_attempts += 1

        # assign cells with 100 as self sufficient groups
        partition, unassigned_pop = \
            assign_selfsufficient(partition=partition,
                                  unassigned_pop=unassigned_pop,
                                  df=df)


        if fill_missing: #Assign missing where they do least harm
            partition, unassigned_pop = \
                assign_remaining(partition=partition,
                                 unassigned_pop=unassigned_pop,
                                 G=G,
                                 df=df,
                                 year_cols = year_cols)

    return partition #[[int(i) for i in g] for g in partition]
