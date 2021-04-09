import itertools

import community
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.neighbors import KDTree
from sqr.core.distance import euclidian
from sqr.core.config import cell_label

# function for checking that groups are connected
def group_connect(group, G):
    '''
    Check whether a given group's subgraph is connected
    '''

    sub_G = G.subgraph(group)
    return nx.is_connected(sub_G)

def partition_connect(partition, G):
    '''
    Check whether each group in the partition is connected.
    '''

    return min([check_connected_group(p, G) for p in partition])


def partition_surjective(partition,show=False):
    '''
    This functions whether each cell has at most one
    group of cells which it is assigned to.
    That is, whether there is no duplicates.
    '''

    overlap = [np.intersect1d(partition[sp1_idx], partition[sp2_idx]).tolist() \
                 for sp1_idx in range(len(partition)) \
                 for sp2_idx in range(sp1_idx)]
    if show==False:
        return len(set(itertools.chain(*overlap)))==0

    if show==True:
        return set(itertools.chain(*overlap))


def local_graph(df, cell_var=cell_label, max_dist=1):
    '''
    Outputs a graph where nodes are input cells in DataFrame
    and edges are between cells within 'max_dist'.

    Raises error when empty graph (i.e. length zero edgelist).
    '''
    

    square_labels = df[cell_var].str[5:]

    int_dict = square_labels.reset_index().set_index(cell_var)['index'].to_dict()
    
    X = df[['e','n']].values
    tree = KDTree(X)
    neighbors = tree.query_radius(X, max_dist)    
    edgelist = \
        np.array([(i, j) for i in range(len(X)) for j in neighbors[i] if j!=i])
    edgelist_idx = \
        np.concatenate([df.iloc[edgelist[:,c]].index.values.reshape(-1,1)
                        for c in [0,1]],axis=1)
    
    if len(edgelist)>0:


        G = nx.Graph()
        G.add_nodes_from(df.index)
        G.add_edges_from(edgelist_idx)

        return G

    else:
        raise ValueError('Empty network')



def get_communities(sub_gdf, max_dist = 10):
    '''
    Returns communities from network of local cells.

    The network is constructed with a specification
    of the maximal distance between adjacent cells.

    Subsequently the Louvain algorithm is applied.
    '''

    G = local_graph(sub_gdf, max_dist=max_dist)
    partition = community.best_partition(G)

    return pd.Series(list(partition.values()), sub_gdf.index)\
             .rename('assignment')
