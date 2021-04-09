import random
import itertools
from copy import deepcopy

import pandas as pd
import numpy as np

import sqr.core.network
import sqr.core.algorithm
import sqr.core.distance
from sqr.core.config import mean_cols, minimum_cols, years

def internal_trades(df,G,in_partition,p_lookup,
                    check_cells, unassigned_all,
                    remove_missing=True,
                    output_status=False):
    '''
    Tries to reassign neighborign cell:
    1) by assigning them to other unassigned cells;
    2) by assigning them into the group of one another.
    The exchanges are made if they lower the average distance.
    '''


    local_partition = deepcopy(in_partition)

    trade_counter = {'self-sufficient':0, 'unassigned switch':0, 'assigned switch':0}

    has_pop = lambda i: df.loc[i,minimum_cols].notnull().max()

    is_relevant = lambda i: i in check_cells

    # initalize new assignments and checks
    new_assigned = []

    new_check_cells = set([])


    redundant = []
    for p in filter(lambda p: len(p)>1, local_partition):
        for i in filter(has_pop, filter(is_relevant, p)):

            if sqr.core.algorithm.check_pop(df, np.setdiff1d(p,i), years):
                redundant += [int(i)]

    removed=[]

    for i in redundant:

        if i not in removed:

            p_idx_i = p_lookup[i]
            p_i = local_partition[p_idx_i]



            if sqr.core.algorithm.check_pop(df, np.setdiff1d(p_i,i), years):

                # check if self-sufficient
                if sqr.core.algorithm.check_pop(df, [i], years):
                    if sqr.core.network.group_connect([j for j in p_i if j!=i], G=G):
                        #print('selfsuff',i,p_i)
                        trade_counter['self-sufficient'] += 1
                        p_i.remove(i)
                        p_lookup[i] = len(local_partition)
                        local_partition += [[i]]
                        removed += [i]

                        new_check_cells |= update_check_cells(G, [i]+p_i)

                        continue


                #check for matches with others
                candidates = [i for i in G.neighbors(i) if i not in (p_i+removed)]
                i_unmatched = True

                # check for matches with unassigned
                unassigned = list(np.intersect1d(candidates, unassigned_all))
                unassigned_nnull =  filter(has_pop, unassigned)

                for m in unassigned_nnull:
                    if i_unmatched:
                        if sqr.core.algorithm.check_pop(df, [i]+[m], years):
                            if sqr.core.network.group_connect([j for j in p_i if j!=i], G=G):
                                #print('unassgined',i,m,p_i)
                                trade_counter['unassigned switch'] += 1
                                p_i.remove(i)
                                p_lookup[i] = len(local_partition)
                                p_lookup[m] = len(local_partition)
                                local_partition += [[i,m]]
                                removed += [i,m]
                                i_unmatched = False



                                new_assigned.append(m)

                                new_check_cells |= update_check_cells(G, [i,m]+p_i)



                #Check assigned matches
                assigned = list(np.setdiff1d(candidates, unassigned))
                assigned_nnull = filter(has_pop, assigned)

                for m in assigned_nnull:
                    if i_unmatched:

                        p_m = local_partition[p_lookup[m]]


                        if sqr.core.algorithm.check_pop(df, [i]+p_m, years):
                            pminusi = [j for j in p_i if j!=i]
                            if sqr.core.network.group_connect(pminusi, G=G):

                                pminusi = pminusi

                                dist_old = sqr.core.distance.total_distance(p_i, df) + \
                                           sqr.core.distance.total_distance(p_m, df)
                                dist_new = sqr.core.distance.total_distance(pminusi, df) + \
                                           sqr.core.distance.total_distance([i] + p_m, df)

                                if dist_new<dist_old: #perform realignment if weighted distance is decreased
                                    check_prior = sqr.core.algorithm.check_pop(df, pminusi, years)
                                    check_next = sqr.core.algorithm.check_pop(df, [i] + p_m, years)

                                    if check_prior and check_next:
                                        #print('switch',i,m,p_i,p_m, pminusi)
                                        trade_counter['assigned switch'] += 1

                                        local_partition[p_lookup[i]].remove(i)
                                        local_partition[p_lookup[m]].append(i)
                                        p_lookup[i] = p_lookup[m]

                                        removed += [i]
                                        i_unmatched = False

                                        new_check_cells |= update_check_cells(G, p_i+p_m)



    if remove_missing:
        for (p_idx,p) in [(idx,p) for idx,p in enumerate(local_partition) if len(p)>1]:
            # check if null observation is necessary, otherwise remove

            group_df = df.loc[p]

            if group_df[minimum_cols].isnull().max(1).sum() > 0:
                new_group, _ = \
                    sqr.core.algorithm.remove_null(p, df=df, G=G)

                local_partition[p_idx] = list(map(int,new_group))
    print (trade_counter)

    out_unassigned = list(set(unassigned_all)-set(new_assigned))

    output = (local_partition, p_lookup, list(new_check_cells), out_unassigned)

    return output, trade_counter


def internal_trades_iterative(df, G, partition,
                              remove_missing = True,
                              num_iter = 25):
    '''
    This functions iteratively applies the function 'internal_trades'
    until no more improving trades are possible.
    '''


    iter_trade = [list(map(int,p)) for p in partition]

    unassigned = list(set(map(int,df.index))-set(itertools.chain(*partition)))
    random.shuffle(unassigned)
    
    p_lookup = dict([(int(i), p_idx) for (p_idx, p) in enumerate(iter_trade) for i in p])
    p_lookup.update(dict(zip(unassigned,[None]*len(iter_trade))))

    check_cells = df.index.tolist()

    for _ in range(num_iter):
        output_data, output_info =\
                  internal_trades(df = df ,
                                  G = G,
                                  in_partition = iter_trade,
                                  p_lookup = p_lookup,
                                  unassigned_all = unassigned,
                                  check_cells = check_cells,
                                  remove_missing=remove_missing,
                                  output_status=True)

        (iter_trade, p_lookup, check_cells, unassigned) = output_data

        if sum(output_info.values()) == 0:
            return iter_trade


    return iter_trade


def update_check_cells(G, group):

    new_checks = set([])

    for i in group:
        new_checks.add(i)
        for j in G.neighbors(i):
            new_checks.add(j)

    return new_checks
