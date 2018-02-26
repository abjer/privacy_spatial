import copy
import numpy as np
import pandas as pd
import itertools
import sqr.core.algorithm

minimum_col = 'minimum'

def run_assignment(df, G, max_total_attempts,
                   max_group_attempts = 50,
                   in_partition = [],
                   fill_missing = True,
                   year_cols = sqr.core.algorithm.year_cols_all,
                   check_connectivity = True):
    """
    Function calculates a partion of the data supplied. The function calls
    subroutines to assign cells to one another in order to fulfill
    the requirement of minimum 100 inhabitants.

    Key arguments:
    df: Pandas dataframe with index corresponding to graph G
    G: NetworkX Graph with index corresponding to df. Contains neighbors
    minimum_col: Column of df with minimum (not nan)

    pre_include: If True then cells incircled by partition gets included in partition
    fill_missing: If True algorithm attempts to include non-assigned cells where the do least harm
    missing_lim: Limit of amount in cell before attemting to include

    """

    # initalize nodes and partition
    total_attempts = 0

    partition = copy.deepcopy(in_partition)

    assigned = list(itertools.chain(*partition))
    unassigned = np.setdiff1d(list(G.nodes()), assigned)

    has_pop = df[df.minimum.notnull()].index
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
            if df.loc[i,minimum_col]>=100: #rettet
                partition += [[i]]
                unassigned = np.setdiff1d(unassigned, [i])
                unassigned_pop = np.setdiff1d(unassigned_pop, [i])
                if unassigned_pop.size == 0:
                    break


                continue

            # attempt local cell expansion for merging cells
            check_solution, unassigned, group = \
                sqr.core.algorithm.cell_expansion(i=i,
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
            sqr.core.algorithm.assign_selfsufficient(partition=partition,
                                                     unassigned_pop=unassigned_pop,
                                                     df=df)


        if fill_missing: #Assign missing where they do least harm
            partition, unassigned_pop = \
                sqr.core.algorithm.assign_remaining(partition=partition,
                                                    unassigned_pop=unassigned_pop,
                                                    G=G,
                                                    df=df,
                                                    year_cols = year_cols)

    return partition #[[int(i) for i in g] for g in partition]
