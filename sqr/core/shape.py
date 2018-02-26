import os

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import fiona.crs
from scipy.spatial import Voronoi


# coordinates and shapes
def coord2label(n,e):
    return '100m_%i_%i' % (n,e)

def label2coord(l):
    return [int(c) for c in l.split('_')[1:]]


# shape function

def make_gdf_square_data(df, polygons = True):
    '''
    Either converts cell labels into polygons (default)
    or into points.
    '''

    if polygons:
        geom = df.apply(polygon_from_north_east, axis=1)
        return gpd.GeoDataFrame(df, geometry = geom, crs=fiona.crs.from_epsg(25832))

    else:
        geom = df.apply(lambda row: Point(row.e*100, row.n*100), axis=1)
        return gpd.GeoDataFrame(df, geometry = geom, crs=fiona.crs.from_epsg(25832))

def polygon_from_north_east(row):
    '''
    Returns the polygon associated with label of northing
    and easting.
    '''
    bbox = [row.e*100, row.n*100, row.e*100+100, row.n*100+100]
    return shapely.geometry.box(*bbox)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
        
    Copied from https://gist.github.com/pv/8036995
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_voronoi_series(df, origin_geom=None):
    '''
    Returns a series of Voronoi shapely objects contruced
    from the centroids.    
    '''

    if origin_geom == None:
        desc = (df[['e','n']].describe().loc[['min','max']]*100).T
        desc['max'] = desc['max'] +100
        origin_geom = shapely.geometry.box(*(desc.T.stack()))


    e = df.geometry.apply(lambda p: p.centroid.coords.xy[0][0])
    n = df.geometry.apply(lambda p: p.centroid.coords.xy[1][0])
    points = pd.concat({'e':e,'n':n},axis=1)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    poly_list = [shapely.geometry.Polygon(vertices[region]) for region in regions]

    poly_list = [pol.intersection(origin_geom) for pol in poly_list]


    return gpd.GeoSeries(poly_list, index=points.index, name='vor_poly')

def find_neighbor_shapes(gdf, buff_intersect = 100, overlap=True, buff_overlap = 100):
    geo_buff = gdf.geometry.buffer(buff_intersect)
    intersect = [(i,j) for i in gdf.index for j in gdf.index if (i!=j and geo_buff[i].intersects(geo_buff[j]))]

    if overlap:
        geo_buff2 = gdf.geometry.buffer(buff_overlap)
        overlap = [(i,j,geo_buff2[i].intersection(geo_buff2[j]).area/10**4) for (i,j) in intersect]
        return pd.DataFrame(overlap, columns = ['idx1', 'idx2', 'overlap_area'])

    return pd.DataFrame(intersect, columns = ['idx1', 'idx2'])
