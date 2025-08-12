import geopandas as gpd
import pandas as pd
import sys
import json
import numpy as np
import rasterio as rs
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


def calculate_adjusted_area(repr_raster, prj_districts, error_dict, outfile):
    '''
    Requires raster that has been reprj to meter CRS, a shapefile
    for the 26 priority districts, and a dictionary containing error
    statistics for each land use class
    Calculates area assessment in ha for each land use class in each
    district and adjusts assessments based on model error
    '''
    
    zonal_stats = []
    
    with rs.open(repr_raster) as src:
        for _, district in prj_districts.iterrows():
            out_image, out_transform = mask(src, [district.geometry], crop=True)
            district_mask = out_image[0] 
            unique, counts = np.unique(district_mask, return_counts=True)
            land_use_stats = dict(zip(unique, counts))

            # Calculate pixel size to convert to hectares
            pixel_width = src.transform[0]  # X resolution
            pixel_height = -src.transform[4]  # Y resolution
            px_size = pixel_width * pixel_height
            land_use_stats = {k: v * (px_size / 10000) for k, v in land_use_stats.items()}
            
            # Add district name
            land_use_stats['district'] = district.ADM2_EN
            zonal_stats.append(land_use_stats)

    df = pd.DataFrame(zonal_stats)
    df = df.round(2).rename(columns={
        0: "Background",
        1: "Monoculture",
        2: "Agroforestry",
        3: "Natural",
        255: "No data"
    })
    
    # Adjust for error
    for land_use_class, stats in error_dict.items():
        if isinstance(stats, dict) and 'adj' in stats:
            if land_use_class in df.columns:
                df[land_use_class] = (df[land_use_class] * stats['adj']).round()

    df.to_csv(outfile, index=False)
    return df

def area_assessment_table(input_f, output_f, include_summary_row=True):
    '''
    Takes in a CSV of area assessment calculations and creates
    a structured table for publication. Rounds values to the nearest 10 ha,
    includes a total area column based on the 'No data' category,
    and adds regional summary rows.

    Parameters:
    - input_f (str): Path to input CSV
    - output_f (str): Path to output CSV
    - include_summary_row (bool): Whether to append summary row for each region
    '''
    import geopandas as gpd
    import pandas as pd

    north = gpd.read_file('../../data/shapefiles/pd_north.shp')
    east = gpd.read_file('../../data/shapefiles/pd_east.shp')
    west = gpd.read_file('../../data/shapefiles/pd_west.shp')
    main_dir = '../../data/area_assessments/'
    df = pd.read_csv(main_dir + input_f)

    district_region = {
        district: 'north' for district in north.ADM2_EN
    }
    district_region.update({
        district: 'south' for district in east.ADM2_EN
    })
    district_region.update({
        district: 'south' for district in west.ADM2_EN
    })
    df['Zone'] = df['district'].map(district_region)

   # Rename columns for clarity
    df = df.rename(columns={
        'Monoculture': 'Monoculture (ha)',
        'Agroforestry': 'Agroforestry (ha)',
        'Natural': 'Natural (ha)',
    })

    #df['Total Area (ha)'] = df[['Monoculture (ha)', 'Agroforestry (ha)', 'Natural (ha)', 'Background', 'No data']].sum(axis=1)
    df_pub = df[['Zone', 'district', 'Monoculture (ha)', 'Agroforestry (ha)', 'Natural (ha)', #'Total Area (ha)'
                 ]]
    df_pub = df_pub.sort_values(by='Zone', ascending=True).reset_index(drop=True)

    totals = df_pub[['Monoculture (ha)', 'Agroforestry (ha)', 'Natural (ha)']].sum()
    print("\nTotal area across all 26 districts (ha):")
    print(totals)

    # Add summary rows per region
    if include_summary_row:
        region_totals = df.groupby('Zone')[['Monoculture (ha)', 'Agroforestry (ha)', 'Natural (ha)']].sum()
        summary_rows = []
        for region, row in region_totals.iterrows():
            summary_rows.append({
                'Zone': region.upper(),
                'district': 'TOTAL',
                'Monoculture (ha)': row['Monoculture (ha)'],
                'Agroforestry (ha)': row['Agroforestry (ha)'],
                'Natural (ha)': row['Natural (ha)'],
                #'Total Area (ha)': row['Total Area (ha)']
            })
        df_pub = pd.concat([df_pub, pd.DataFrame(summary_rows)], ignore_index=True)

    # Final step: round all area columns to nearest 10 hectares
    area_cols = ['Monoculture (ha)', 'Agroforestry (ha)', 'Natural (ha)', #'Total Area (ha)'
                 ]
    df_pub[area_cols] = df_pub[area_cols].round(-1)

    # Save output
    df_pub.to_csv(main_dir + output_f, index=False)
    return df_pub