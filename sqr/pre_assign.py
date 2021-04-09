import pandas as pd
import geopandas as gpd

from shapely.ops import unary_union
from sqr.core.shape import get_voronoi_series, find_neighbor_shapes
from sqr.core.network import get_communities
from sqr.core.config import years_hh, years_pers

def pre_partition_area(gdf, origin_geom):
    '''
    Makes a pre partition of the area shape into sub-areas
    of shapes.
    '''

    has_pop_and_pers = gdf.minimum_pers.notnull() & gdf.minimum_hh.notnull()

    sub_gdf = gdf[has_pop_and_pers].copy()
    
    # get assignment and voronoi shape
    voronoi_geo = get_voronoi_series(sub_gdf, origin_geom)
    
    sub_gdf['assignment'] = get_communities(sub_gdf)

    sub_gdf = gpd.GeoDataFrame(geometry=voronoi_geo, data=sub_gdf)

    # get assignment information and geometry
    gb_assign = sub_gdf.groupby('assignment')
    
    pers =  gb_assign\
                .apply(lambda g: g[years_pers].dropna(axis=1,how='all').sum(0).min())\
                .rename('count_pers')
    
    hh =  gb_assign\
                .apply(lambda g: g[years_hh].dropna(axis=1,how='all').sum(0).min())\
                .rename('count_hh')
    cell =  gb_assign.apply(lambda g: g.shape[0]).rename('count_cells')
    info = pd.concat([pers,hh,cell], axis=1)
    info['cells'] = gb_assign.apply(lambda g: list(g.index))
    geoms = gb_assign.geometry.apply(lambda g: g.unary_union)

    df_pre = gpd.GeoDataFrame(data=info, geometry=geoms)

    return df_pre

def merge_insufficient(in_gdf):
    '''
    Merge partition with insufficient population
    onto their nearby neighboring partition
    shapes
    '''


    gdf = in_gdf.copy()

    insuff = ((gdf.count_pers<100)|(gdf.count_hh<50)).to_dict()

    overlap = find_neighbor_shapes(gdf)
    overlap = overlap[overlap.idx1.apply(lambda i: insuff[i])]

    overlap['other_insuff'] = overlap.idx2.apply(lambda i: insuff[i])

    gb_idx = overlap.groupby('idx1')
    neighbor_suff = (~gb_idx.other_insuff.min())
    match_to_neighbor = neighbor_suff.sort_values().index

    info_cols = ['count_pers', 'count_hh','count_cells','cells']
    optimals = {}

    geoms = gdf.geometry.to_dict()

    for idx in match_to_neighbor:
        opt = gb_idx\
                .get_group(idx)\
                .sort_values(['other_insuff','overlap_area'],ascending=[1,0])\
                .iloc[0]\
                .idx2

        if opt in optimals:
            opt = optimals[opt]

        optimals[idx] = opt

        geoms[opt] = unary_union([geoms[opt],geoms[idx]])

        gdf.loc[opt,info_cols] += gdf.loc[idx,info_cols].values

    df = gdf\
            .drop(match_to_neighbor, axis=0)\
            .drop('geometry', axis=1)

    geos = gpd.GeoSeries(geoms, crs=gdf.crs)

    out_gdf = gpd.GeoDataFrame(data=df, geometry=geos)

    return df


def assign_cells_partition(part_gdf, cells_gdf):
    '''
    Merge a GeoDataFrame of partitions onto a
    DataFrame containing cells and information.
    '''

    assign = {}

    for idx in part_gdf.index:
        geom = part_gdf.loc[idx].geometry

        assign[idx] = set(cells_gdf[cells_gdf.intersects(geom)].index)


    part_neighbors = find_neighbor_shapes(part_gdf, overlap=False)\
                        .values\
                        .tolist()

    for i1,i2 in part_neighbors:


        intersects = list(assign[i1] & assign[i2])

        cells_gdf_sub = cells_gdf.loc[intersects]

        intersect_1 = cells_gdf_sub.geometry.intersection(part_gdf.loc[i1].geometry).area
        intersect_2 = cells_gdf_sub.geometry.intersection(part_gdf.loc[i2].geometry).area

        i1_more = intersect_1>intersect_2

        assign[i1] -= set(cells_gdf_sub[~i1_more].index)
        assign[i2] -= set(cells_gdf_sub[i1_more].index)


    errors = []

    for (i1,i2) in part_neighbors:
        if len(assign[i1] & assign[i2])>0:
            errors+= [(i1,i2)]

    if len(errors)>0:
        raise ValueError('Non surjective assignment')

    series = [(idx,pd.Series(list(assign[idx]))) for idx in assign.keys()]

    assignment = pd.concat(dict(series), axis=0)\
                    .reset_index()\
                    .drop('level_1',axis=1)\
                    .rename(columns = {'level_0':'assignment'})\
                    .set_index(0)

    return assignment
