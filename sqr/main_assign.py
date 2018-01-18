import time
import copy
import json

import pandas as pd
import numpy as np

import sqr.base_assign
import sqr.trade_assign
import sqr.core.scoring

data_cols = ['mun_idx','partition','unweighted_dist','weighted_dist',
             'count_nonzero_null', 'pop_nonzero_null',
             'finish_ts', 'delta_t', 'repetitions', 'trade']

minimum_col = 'minimum'

def get_assignment(input_tuple, export =True, trade_iter_count = 25):
    (idx, df, small_G, big_G, pop_density, trade_assign) = input_tuple

    # try:

    local_df = df.copy()
    local_G = copy.deepcopy(small_G)

    start = time.time()
    count_pop = df[df.minimum.notnull()].shape[0]
    max_attempts = count_pop / 5
    max_group_attempts = 50
    repetitions = get_repetitions(pop_density)


    assign_iter = []

    print('year_assign')
    # apply base assign on cells popping up with new construction
    year_tuples = get_newconstruction_frames(df)

    np.random.shuffle(year_tuples)

    for [year_cols,year_obs,year_df] in year_tuples:
        old_assign_iter = assign_iter

        assign_iter = \
            sqr.base_assign.run_assignment(df = year_df,
                                           G = big_G.subgraph(year_df.index),
                                           max_total_attempts = year_obs*2,
                                           max_group_attempts = 50,
                                           in_partition = assign_iter,
                                           year_cols = year_cols,
                                           check_connectivity = False)

        print(min(year_cols),max(year_cols),year_obs, [g for g in assign_iter if g not in old_assign_iter])

    print(len(assign_iter))


    print('persistent_assign')
    # iterartively apply base assign on persistent cells
    for rep in range(repetitions+1):
        assign_iter = \
            sqr.base_assign.run_assignment(df=local_df,
                                           G=local_G,
                                           max_total_attempts = max_attempts,
                                           max_group_attempts = max_group_attempts*(10**rep),
                                           in_partition = assign_iter)


    if trade_assign:
        print('trade_assign')
        assign_iter = \
            sqr.trade_assign.internal_trades_iterative(local_df, local_G, assign_iter,
                                                       num_iter = trade_iter_count)



    if len(assign_iter) > 0:
        end = time.time()
        partition = [[int(i) for i in g] for g in assign_iter]
        evaluated = sqr.core.scoring.partition(partition, local_df)
        output = [[idx]+ evaluated + [end, end-start, repetitions, trade_assign]]
        out_file = 'data/temp_output/%s.csv' % str(end).replace('.','-')

        out = pd.DataFrame(data=output,
                     columns=data_cols)
        if export == True:
            out.to_csv(out_file, index=False)
        else:
            return out

    # with open('data/temp_output/%s.json' % str(end).replace('.','-'), 'w') as f:
    #     f.write(json.dumps(output))

    # except Exception as e:
    #     with open('data/temp_output_error/%s.txt' % str(int(time.time())), 'w') as f:
    #         f.write(str(e))

pop_thresholds = [25,5,1,0]
def get_repetitions(pop_density):
    for idx,threshold in enumerate(pop_thresholds):
        if pop_density >= threshold:
            return idx


def get_newconstruction_frames(df):
    '''
    Return list of DataFrames which format
    the input data  to only contain data for a
    specific year with new construction, if the cells
    with new construction permits a separate solution,
    i.e. have more than 100 inhabitants collectively
    in the construction year.
    '''


    years = df.inhabit_year.dropna().astype(int).unique()

    year_tuples = []

    for year in sorted(filter(lambda y: y!=1986.0,years)):

        year_cols = list(map(str,range(year,2016)))
        year_select = (df.inhabit_year==year)
        df_pop = df.loc[year_select,year_cols+['n','e','mean','minimum']]

        check_feasible = \
            sqr.core.algorithm.check_pop(df_pop,df_pop.index,check_year_cols=year_cols)

        if check_feasible:
            year_tuples.append([year_cols,
                                year_select.sum(),
                                df_pop])

    return year_tuples
