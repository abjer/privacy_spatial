import geopandas as gpd
import pandas as pd

def read_parse_mun():
    kommuner = gpd.read_file('data/shape/KOMMUNE.shp')

    mun_pop_min = {}
    mun_pop_avg = {}
    mun_cell_count = {}

    for idx in kommuner.index:
        mun_data = \
            pd.read_hdf('data/parsed/sqr_mun.hdf', key='sqidx%i'% idx)
        mun_pop_min[idx] = mun_data.minimum.sum()    
        mun_pop_avg[idx] = mun_data['mean'].sum()    
        mun_cell_count[idx] = mun_data.shape[0]

    kommuner['minimum_total'] = pd.Series(mun_pop_min)
    kommuner['mean_total'] = pd.Series(mun_pop_avg)
    kommuner['cell_count'] = pd.Series(mun_cell_count)
    kommuner['to_assign'] = kommuner.minimum_total>=100
    kommuner['density'] = kommuner['mean_total']/kommuner['cell_count']
    
    return kommuner