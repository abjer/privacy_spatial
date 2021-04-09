import geopandas as gpd
import pandas as pd

from sqr.core.config import dummy_mun_codes_4char, dummy_compute

def read_parse_mun(dummy_compute = dummy_compute, compute_stats=True):
    gdf = gpd.read_file('data/shape/KOMMUNE.shp').to_crs(epsg=25832)\
    
    if dummy_compute:
        gdf = gdf[gdf.KOMKODE.isin(dummy_mun_codes_4char)].copy()
    
    # statistics for sub shapes of municipality 
    if compute_stats:        
        for idx in gdf.index:
            mun_data = \
                pd.read_hdf('data/parsed/sqr_mun.hdf', key='sqidx%i'% idx)
            gdf.loc[idx,'minimum_total_pers'] = mun_data.minimum_pers.sum()    
            gdf.loc[idx,'mean_total_pers'] = mun_data.mean_pers.sum()    
            gdf.loc[idx,'minimum_total_hh'] = mun_data.minimum_hh.sum()    
            gdf.loc[idx,'mean_total_hh'] = mun_data.mean_hh.sum()            
            gdf.loc[idx,'cell_count'] = mun_data.shape[0]
    
        # whether or not to process
        gdf['to_assign'] = \
            (gdf.minimum_total_pers>=100) & (gdf.minimum_total_hh>=50)

        # density measure
        gdf['density_pers'] = gdf['mean_total_pers']/gdf['cell_count']
        gdf['density_hh'] = gdf['mean_total_hh']/gdf['cell_count']
        
    return gdf
