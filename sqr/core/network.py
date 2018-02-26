import itertools

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import louvain

from sqr.core.distance import euclidian

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


def local_edges(p, q, max_dist = 1):
    
    local = range(int(max_dist)+1)

    check_dist_trivial = lambda i,j: (euclidian([i,j])<=max_dist and (i,j)!=(0,0))

    neighbors = [[(p,q),(i+p,j+q)] for i in local for j in local if check_dist_trivial(i,j)]

    return neighbors


def local_graph(df, cell_var='KN100mDK', max_dist=1, output_class='networkx'):
    '''
    Outputs a graph where nodes are input cells in DataFrame
    and edges are between cells within 'max_dist'.

    Raises error when empty graph (i.e. length zero edgelist).
    '''
    
    #TODO: implement in sklearn's radius neighbor

    square_labels = df[cell_var].str[5:]

    int_dict = square_labels.reset_index().set_index('KN100mDK')['index'].to_dict()

    e_max,e_min = df.e.max(), df.e.min()
    n_max,n_min = df.n.max(), df.n.min()

    edges = [local_edges(p, q, max_dist) for [p,q] in df[['n','e']].values.tolist()]
    edges_chain = list(itertools.chain(*edges))
    edgelist = pd.DataFrame([('%i_%i' % p[0], '%i_%i' % p[1]) for p in edges_chain],columns=['i','j'])
    edgelist = edgelist[edgelist.i.isin(square_labels) & edgelist.j.isin(square_labels)]

    if len(edgelist)>0:

        edgelist = edgelist.applymap(lambda cell: int_dict[cell]).values.tolist()

        if output_class == 'networkx':
            G = nx.Graph()
            G.add_nodes_from(df.index.tolist())
            G.add_edges_from(edgelist)
        elif output_class == 'igraph':
            lookup = dict(zip(df.index,range(len(df))))
            G = ig.Graph()
            G.add_vertices(range(len(df)))
            def min_pop(i,j): return min(df.loc[i,'minimum'], df.loc[j,'minimum'])

            edgelist_idx = [(lookup[i],lookup[j],min_pop(i,j)) for i,j in edgelist]
            G.add_edges([e[:2] for e in edgelist_idx])
            G.es["weight"] = [e[2] for e in edgelist_idx]
        else:
            raise ValueError('Unsupported graph format')
        return G

    else:
        raise ValueError('Empty network')



def get_communities(sub_df, max_dist = 10):
    '''
    Returns communities from network of local cells.

    The network is constructed with a specification
    of the maximal distance between adjacent cells.

    Subsequently the Louvain algorithm is applied.
    '''

    # sub_df = df[df.minimum.notnull()]

    G = local_graph(sub_df, max_dist=max_dist, output_class='igraph')

    partition = louvain.find_partition(G, louvain.ModularityVertexPartition, weights='weight')

    return pd.Series(partition.membership, index=sub_df.index)\
             .rename('assignment')
