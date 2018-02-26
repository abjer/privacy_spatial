import itertools

import pandas as pd
import numpy as np

import sqr.core.distance as dist

mean_col = 'mean'

def partition(partition, df):
    '''
    Format the partition by annotating various
    summary statistics and values
    '''

    temp_df = pd.DataFrame(df[[mean_col,'e','n']])
    for idx,group in enumerate(partition):
        temp_df.loc[group,'assignment'] = idx
        temp_df.loc[group,'pop_weight'] = temp_df.loc[group,mean_col].sum()/\
                                          temp_df.loc[list(itertools.chain(*partition)),mean_col].sum()

    unweighted_distance = temp_df.groupby('assignment').apply(dist.unweighted).mean()
    weighted_distance = temp_df.groupby('assignment').apply(dist.weighted).sum()

    nonzero_null = (temp_df.assignment.isnull()) & (temp_df[mean_col].notnull())

    return [partition,
            float(unweighted_distance),
            float(weighted_distance),
            int(nonzero_null.sum()),
            int(temp_df[nonzero_null][mean_col].sum())]


# evalulating merge of cell into collection
alpha = 1/10

def group_collapse(group, extra, df):
    dist_new = dist.unweighted(df.loc[group+extra])
    dist_old = dist.unweighted(df.loc[group])

    ratio_dist = dist_new / dist_old

    pop_new = df.loc[group+extra][mean_col].sum()
    pop_old = df.loc[group][mean_col].sum()

    ratio_pop = pop_new / pop_old

    return ratio_pop - alpha * ratio_dist

beta = 1/100

def partition_score(weighted_dist, pop_na, pop_total, beta = beta):
    '''
    Value function to determine how
    well the partition performs.
    '''

    return  -pop_na/pop_total - beta * weighted_dist


dom_cols = ['weighted_dist','pop_nonzero_null']

def get_undominated(df):
    '''
    Retrieves the points in the input DataFrame
    which are not strictly dominating in all
    dimensions of 'dom_cols'.
    '''

    undominated_candidates = df.index.tolist()

    undominated = []

    for idx in undominated_candidates:
        others = undominated_candidates[idx+1:]+undominated
        indices = range(len(others))
        is_undominated = True
        for other_idx in indices:
            other = others[other_idx]
            if (df.loc[other][dom_cols] < df.loc[idx][dom_cols]).min():
                is_undominated = False
                break


        if is_undominated:
            undominated.append(idx)

    return df.loc[undominated]



# def get_undominated_alt(df):
#
#     def check_dom(row , df = df):
# #         print(df.index, row.name)
#         other = df.loc[np.setdiff1d(df.index,row.name)]
#         return other.apply(lambda orow: (orow[dom_cols] < df.loc[row.name, dom_cols]).min(), axis=1).max()
#
#     out = df[~df.apply(check_dom, axis=1)]
#
#     return out
