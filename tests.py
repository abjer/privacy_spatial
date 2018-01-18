import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry
import networkx as nx
from nose.tools import assert_raises, assert_true

import sqr.core.shape
import sqr.core.distance
import sqr.core.network

import sqr.base_assign
import sqr.core.algorithm
import sqr.trade_assign
import sqr.core.scoring




test_data = {0:[0,1,30],
             1:[1,2,30],
             2:[1,1,30],
             3:[1,3,np.nan],
             4:[1,4,25],
             5:[2,3,np.nan],
             6:[2,5,np.nan],
             7:[0,0,1],
             8:[3,1,101],
             9:[1,0,10],
            10:[0,4,33],
            11:[0,5,34],
            12:[1,5,33]}

test_df = pd.DataFrame(test_data).T
test_df.columns = ['n','e','minimum']

get_label = lambda row: sqr.core.shape.coord2label(e=row.e, n=row.n)
test_df['KN100mDK'] = test_df.apply(get_label,axis=1)

test_edgelist = [(0,2),(1,2),(1,3),(3,4),(3,5),(0,7),
                 (2,9),(10,11),(11,12),(10,4),(12,4)]
test_G = nx.Graph()
test_G.add_nodes_from(test_data.keys())
test_G.add_edges_from(test_edgelist)


test_gdf = sqr.core.shape.make_gdf_square_data(test_df)


# test for 'shape' module

def test_labelling():

    # tests for shape functions
    # test coord2label
    assert_true(test_df['KN100mDK'].iloc[:3].tolist()==['100m_0_1', '100m_1_2', '100m_1_1'])


def test_shape():
    # test 'make_gdf_square_data'
    cell_geom = shapely.geometry.box(*[100,0,200,100])
    test_gdf_mini = sqr.core.shape.make_gdf_square_data(test_df.iloc[:1])

    assert_true(cell_geom.equals(test_gdf_mini.iloc[0].geometry))


# test for 'distance' module

def test_dist_euclidian():

    # test distance measures
    # test 'euclidian_dist'
    assert_true(sqr.core.distance.euclidian([1,2])==np.sqrt(5))

def test_dist_unweighted():
    # test 'unweighted_dist'
    assert_true(sqr.core.distance.unweighted(test_df.iloc[:2])==2*np.sqrt(2))


def test_dist_unweighted_pop():
    # test 'unweighted_dist_pop'
    dist_pop = sqr.core.distance.unweighted_pop([0,1], test_df.iloc[:2], col='minimum')
    assert_true(dist_pop==2*np.sqrt(2)*60)


# test for 'network' module

def test_duplication():
    assert_true(sqr.core.network.partition_surjective([[0,1],[2,3]]))
    assert_true(not sqr.core.network.partition_surjective([[0,1],[1,2]]))

def test_connectivity_group():
    # check connectivity group
    assert_true(not sqr.core.network.group_connect(group=range(2), G=test_G))
    assert_true(not sqr.core.network.group_connect(group=range(7), G=test_G))
    assert_true(sqr.core.network.group_connect(group=range(3), G=test_G))
    assert_true(sqr.core.network.group_connect(group=range(6), G=test_G))

def test_local_edges():
    # check local edges
    edges_true = pd.DataFrame([[(0, 0), (0, 1)], [(0, 0), (1, 0)]])
    edges_test = pd.DataFrame(sqr.core.network.local_edges(0, 0, max_dist = 1))
    assert_true(edges_true.equals(edges_test))

    edges_true_onehalf = pd.DataFrame([[(0, 0), (0, 1)], [(0, 0), (1, 0)],[(0, 0), (1, 1)]])
    edges_test_onehalf = pd.DataFrame(sqr.core.network.local_edges(0, 0, max_dist = 1.5))
    assert_true(edges_true_onehalf.equals(edges_test_onehalf))

def test_local_network():
    # check network constructed
    graph_true = pd.DataFrame(test_G.subgraph(range(3)).edges())
    graph_test = pd.DataFrame(sqr.core.network.local_graph(test_df.iloc[:3]).edges())
    assert_true(graph_true.equals(graph_test))

    assert_raises(ValueError, sqr.core.network.local_graph, test_df.iloc[:2])



# test for 'base_core' module

def test_remove_null():
    # check 'remove null'
    remove_params = {'group':np.array(range(6)), 'df':test_df, 'col':'minimum', 'G':test_G}
    test_remove = sqr.core.algorithm.remove_null(**remove_params)

    assert_true(np.array_equal(test_remove[0], range(5)))
    assert_true(np.array_equal(test_remove[1], [5]))

def test_cell_expansion():
    # test 'cell_expansion'

    # expansion to cells in range(0,5)
    args_0 = {'i':0, 'unassigned':range(7), 'df':test_df, 'G':test_G, 'col':'minimum', 'max_group_attempts':50}
    test_cellexp_0 = sqr.core.algorithm.cell_expansion(**args_0)

    assert_true(test_cellexp_0[0])
    assert_true(np.array_equal(test_cellexp_0[1], range(5,7)))
    assert_true(np.array_equal(test_cellexp_0[2], range(5)))

    # null expansion from disconnected cell
    args_6 = {'i':6, 'unassigned':range(7), 'df':test_df, 'G':test_G, 'col':'minimum', 'max_group_attempts':50}
    test_cellexp_6 = sqr.core.algorithm.cell_expansion(**args_6)

    assert_true(test_cellexp_6[0]==False)


def test_selfsufficient():
    selfsuff_arg = {'partition':[list(range(5))], 'df':test_df, 'col':'minimum', 'unassigned':range(5,9)}
    selfsuff_test = sqr.core.algorithm.assign_selfsufficient(**selfsuff_arg)

    assert_true(np.array_equal(selfsuff_test[0][0],range(5)))
    assert_true(np.array_equal(selfsuff_test[0][1],[8]))
    assert_true(np.array_equal(selfsuff_test[1],range(5,8)))

def test_remain_assign():
    # no inclusion
    remain_arg = {'partition':[list(range(5))], 'df':test_df, 'col':'minimum', 'G':test_G, 'unassigned':range(5,10)}
    remain_test = sqr.core.algorithm.assign_remaining(**remain_arg)
    assert_true(np.array_equal(remain_test[0],[list(range(5))+[7,9]]))
    assert_true(np.array_equal(remain_test[1],[5,6,8]))

    # with inclusion
    remain_arg_2 = {'partition':[[0,1,2,9]], 'df':test_df, 'col':'minimum', 'G':test_G, 'unassigned':range(5,9)}
    remain_test_2 = sqr.core.algorithm.assign_remaining(**remain_arg_2)
    assert_true(np.array_equal(remain_test_2[0],[[0,1,2,9,7]]))


def test_base_assign():
    base_assign_params = {'df':test_df.iloc[:6],
                          'G':test_G.subgraph(range(6)),
                          'max_total_attempts':1}

    # test failure
    np.random.seed(1)
    output = sqr.base_assign.run_assignment(**base_assign_params)
    assert_true(output==[])

    # success where cell 5 must be removed as it is null
    np.random.seed(2)
    output = sqr.base_assign.run_assignment(**base_assign_params)
    assert_true(output[0]==[0,1,2,3,4])

    # test that only cell 8 is assigned to fixed partition
    base_assign_params_2 = {'df':test_df.iloc[:9],
                            'in_partition':[[0, 1, 2, 3, 4]],
                            'G':test_G.subgraph(range(6)),
                            'max_total_attempts':1}

    output_2 = sqr.base_assign.run_assignment(**base_assign_params_2)

    assert_true(output_2[0] == [0,1,2,3,4])
    assert_true(output_2[1] == [8])

    # test that both cell 8 and 9 are assigned
    base_assign_params_3 = {'df':test_df.iloc[:10],
                            'in_partition':[[0, 1, 2, 3, 4]],
                            'G':test_G.subgraph(range(10)),
                            'max_total_attempts':1}

    output_3 = sqr.base_assign.run_assignment(**base_assign_params_3)
    assert_true(output_3[0] == [0,1,2,3,4,7,9])
    assert_true(output_3[1] == [8])

def test_internal_trade():
    in_part = [[0,1,2,3,4,9],
               [10,11,12]]

    trade_test = sqr.trade_assign.internal_trades(partition=in_part,
                                                               df=test_df,
                                                               G=test_G)

    assert_true(np.array_equal(trade_test[0],[0,1,2,9]))
    assert_true(np.array_equal(trade_test[1],[10,11,12,4]))




# test for preprocessing

test_df2 = test_df.copy()
test_df2[['e','n']] = test_df2[['e','n']] + 10
test_df2['KN100mDK'] = test_df2.apply(get_label,axis=1)
pre_df = pd.concat([test_df,test_df2])

remain = [(n,e)
             for n in range(int(test_df2.n.max())+1)
             for e in range(int(test_df2.e.max())+1)
             if [n,e] not in pre_df[['n','e']].astype(int).values.tolist()]

remain_df = pd.DataFrame(remain, columns =['n','e'])
remain_df['minimum'] = np.nan
remain_df['KN100mDK'] = remain_df.apply(get_label,axis=1)

pre_df = pre_df.append(remain_df, ignore_index=True)

pre_gdf = sqr.core.shape.make_gdf_square_data(pre_df)

pre_gdf[['n','e']] = pre_gdf[['n','e']].astype(int)


def test_pre_partition_network():
    pre_gdf['partition'] = pre_gdf.index.to_series().apply(lambda idx: 1 if idx<13 else (0 if idx<26 else np.nan))
    sub_df = pre_gdf[pre_gdf.minimum.notnull()]

    true_part = sub_df.partition.astype(int)
    test_part = sqr.core.network.get_communities(sub_df)

    assert_true(true_part.shape[0]==test_part.shape[0])
    assert_true((true_part!=test_part).sum()==0 )



def test_voronoi():
    selection = test_gdf[test_gdf.minimum.notnull()].iloc[:5]
    true_geom = shapely.geometry.Polygon([[100,0],[100,100],[200,100],[300,0]])
    test_geom = sqr.core.shape.get_voronoi_series(selection).iloc[0]

    assert_true(true_geom.equals(test_geom))


def test_domination():
    dom_cols = ['weighted_dist','pop_nonzero_null']
    dom_data = [[2,1],[1,0],[0,1],[0,1],[0.1,1.2]]
    dom_df = pd.DataFrame(dom_data,columns=dom_cols)

    test_dom = sqr.core.scoring.get_undominated(dom_df)
    true_dom = pd.DataFrame([[1,0],[0,1],[0,1]],columns=dom_cols,index=[1,2,3]).astype(float)
    assert_true(test_dom.equals(true_dom))
