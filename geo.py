"""Spatial join of property coordinates to CA School District Areas 2024-25.

Only district *identity* columns are kept from the boundary file (CDSCode, name, type, county).
The file also ships enrollment/ethnicity/poverty demographics; they are dropped on load on purpose
(they would act as protected-class proxies and are not property characteristics).

The join is a static geographic attribute lookup that never touches ClosePrice, so doing it once on
the full table before the chronological split is not leakage.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BOUNDARY_PATH = Path('data/external/ca_school_districts_2024_25/DistrictAreas2425.shp')
SOURCE_URL = 'https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/shapefile?layers=0'
PROJECTED_CRS = 'EPSG:3310'            # California Albers, metres: valid for area ranking and joins
KEEP_COLUMNS = ['CDSCode', 'DistrictNa', 'DistrictTy', 'CountyName']
CA_LAT = (32.4, 42.1)                  # California bounding box used to validate coordinates
CA_LON = (-124.5, -114.1)
UNKNOWN = 'Unknown'


def load_districts(path=BOUNDARY_PATH):
    """Read boundaries, keep identity columns, repair invalid geometry, project to EPSG:3310."""
    gdf = gpd.read_file(path)[KEEP_COLUMNS + ['geometry']]
    n_invalid = int((~gdf.is_valid).sum())
    gdf['geometry'] = gdf.geometry.buffer(0)           # standard repair for self-intersections
    gdf = gdf.to_crs(PROJECTED_CRS)
    gdf['area_m2'] = gdf.geometry.area
    gdf.attrs['n_invalid_repaired'] = n_invalid
    return gdf.reset_index(drop=True)


def valid_coordinate_mask(df):
    """True where Latitude/Longitude are present, nonzero and inside the California bounding box."""
    lat, lon = df['Latitude'], df['Longitude']
    return (lat.between(*CA_LAT) & lon.between(*CA_LON) & lat.ne(0) & lon.ne(0)).fillna(False)


def _match_one_type(points, districts, district_type):
    """Best polygon of one district type per point (tie rule: smallest area). Index = point index."""
    polys = districts.loc[districts['DistrictTy'].eq(district_type), ['CDSCode', 'DistrictNa', 'area_m2', 'geometry']]
    hit = gpd.sjoin(points, polys, how='inner', predicate='within')
    missed = points.index.difference(hit.index)
    if len(missed):                                    # points exactly on a shared boundary
        hit = pd.concat([hit, gpd.sjoin(points.loc[missed], polys, how='inner', predicate='intersects')])
    hit = hit.sort_values('area_m2').loc[lambda d: ~d.index.duplicated(keep='first')]
    return hit[['CDSCode', 'DistrictNa']]


def assign_school_districts(df, districts=None):
    """Return df with SchoolDistrict(+Name), HighSchoolDistrict and match flags added.

    SchoolDistrict is the primary categorical feature: Unified if the point is in one, else
    Elementary, else High. HighSchoolDistrict is the High-type polygon where available
    (elementary-district homes sit inside a separate high-school district). Unmatched -> 'Unknown'.
    Row count and order are preserved (asserted).
    """
    districts = load_districts() if districts is None else districts
    n_before = len(df)
    out = df.copy()
    ok = valid_coordinate_mask(out)
    pts = gpd.GeoDataFrame(index=out.index[ok],
                           geometry=gpd.points_from_xy(out.loc[ok, 'Longitude'], out.loc[ok, 'Latitude']),
                           crs='EPSG:4326').to_crs(PROJECTED_CRS)   # reproject points to polygon CRS
    by_type = {t: _match_one_type(pts, districts, t) for t in ['Unified', 'Elementary', 'High']}

    primary = by_type['Unified'].reindex(out.index)
    primary_type = pd.Series(np.where(primary['CDSCode'].notna(), 'Unified', None), index=out.index, dtype=object)
    for t in ['Elementary', 'High']:
        cand = by_type[t].reindex(out.index)
        fill = primary['CDSCode'].isna() & cand['CDSCode'].notna()
        primary = primary.where(~fill, cand)
        primary_type = primary_type.mask(fill, t)
    out['SchoolDistrict'] = primary['CDSCode'].fillna(UNKNOWN).astype(object)
    out['SchoolDistrictName'] = primary['DistrictNa'].fillna(UNKNOWN).astype(object)
    out['SchoolDistrictType'] = primary_type.fillna(UNKNOWN).astype(object)
    out['HighSchoolDistrict'] = by_type['High']['CDSCode'].reindex(out.index).fillna(UNKNOWN).astype(object)
    out['coord_valid'] = ok.values

    assert len(out) == n_before and out.index.equals(df.index), 'spatial join changed the row count/order'
    return out


def match_report(joined):
    """Match-rate summary with unmatched reasons (null coords / outside CA box / in CA but no polygon)."""
    n = len(joined)
    matched = joined['SchoolDistrict'].ne(UNKNOWN)
    null_coord = joined['Latitude'].isna() | joined['Longitude'].isna()
    outside = ~joined['coord_valid'] & ~null_coord
    no_poly = joined['coord_valid'] & ~matched
    return pd.DataFrame({
        'rows': [n, int(matched.sum()), int((~matched).sum()), int(null_coord.sum()), int(outside.sum()), int(no_poly.sum())],
        'pct': [100.0, *(100 * np.array([matched.sum(), (~matched).sum(), null_coord.sum(), outside.sum(), no_poly.sum()]) / n)],
    }, index=['total', 'matched', 'unmatched', '  null lat/lon', '  outside CA bbox / zero', '  inside CA, no polygon'])
