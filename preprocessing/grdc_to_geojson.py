# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:27:18 2021

@author: mvand
"""


from pathlib import Path 
import pandas as pd 
import numpy as np 
from pprint import pprint 
import pyproj 
import geojson 

#%% Define functions

def reproject_coordinates(src_coords, src_epsg, dst_epsg=int(4326)):
    
    '''
    Function that transforms coordinates - assumes input is a matrix 
    of 2 columns and n rows, with first column containing x/lon coordinates,
    and second column with y/lat coordinates - returns a 2D array with first 
    '''

    if type(src_coords) != np.ndarray or  len(src_coords.shape) < 2:
        src_coords = np.array([src_coords])
         
    ## check input 
    assert src_coords.shape[1] >= 2, '[ERROR] coordinates are not pairwise'
    assert type(src_epsg) == int, '[ERROR] source epsg code (src_epsg) is not an integer'
    assert type(src_epsg) == int, '[ERROR] destination epsg code (dst_epsg) is not an integer' 
    
    
    ## extract x and y coordinates as vectors 
    src_x = src_coords[:,0]
    src_y = src_coords[:,1]
    
    ## define transformer 
    transformer = pyproj.Transformer.from_crs('epsg:{}'.format(int(src_epsg)),
                                             'epsg:{}'.format(dst_epsg))
    
    ## transform vectors & return 
    dst_y, dst_x = transformer.transform(src_y, src_x)
    return np.array([dst_x, dst_y]).transpose()


def read_gauge_data(fn, transform = False, src_proj = int(4326), dst_proj = int(3035),
                    out_dir = None):
    
    '''
    Function that reads GRDC gauge data - 
    and returns a geojson file:
    
    fn         path to (csv) datafile 
    transform  transform coordinate system 
    '''
    
    out_df = None
    meta = {} 

    output_cols = ['date', 'time', 'value', 'quantity', 
                   'epsg', 'nan_val', 'loc_id', 'y', 
                   'x', 'upArea', 'unit']
        
    if not Path(fn).exists():
        print('[ERROR] file not found \n{} does not exist'.format(fn))
        
    ## get gauge id nr from filename             
    gauge_id_nr = fn.name.split('_')[0]
    
    ## open file 
    ##  other options for encoding:
    ##  encoding = 'mbcs'(windows only - not for linux) # encoding = 'ansi'
		##  tried: ansi (does not work on linux?), utf-8, ascii, cp500 
    df = pd.read_csv(fn, skiprows=36, delimiter=';', encoding = 'cp850') 
    
    ## extract data 
    discharge = df[' Value'].values 
    dt_dates = pd.to_datetime(df['YYYY-MM-DD'], yearfirst=True, 
                              format='%Y-%m-%d') 


    ## get metadata 
    byte_df = open(fn, encoding='cp850')
    lines = byte_df.readlines()[:36]
    
    # temp_df['loc_id'] = gauge_id_nr 
    meta['loc_id'] = gauge_id_nr 
    
    for line in lines:
        vals = line.split(' ')

        if 'Station:' in vals:
            meta['loc_name'] = vals[-1].replace('\n', '').lower()

        if 'missing' in vals:
            meta['nan'] = float(vals[-1])

        if 'Latitude' in vals:
            meta['lat'] = float(vals[-1])

        if 'Longitude' in vals:
            meta['lon'] = float(vals[-1])

        if 'Unit' in vals:
            meta['unit'] = vals[-1].replace('\n', '')

        if 'area' in vals:
            meta['upArea(km2)'] = float(vals[-1])
        
        if 'Altitude' in vals:
            elevation = float(vals[-1]) 
            # if elevation <= -999.:
                # elevation = np.nan 
                
            meta['elevation'] = elevation 

        if 'Content:' in vals:
            meta['description'] = ' '.join( vals[-3:] ).replace('\n', '').lower()
      
    
    # discharge[discharge == meta['nan']] = np.nan
        
    ## close meta file 
    byte_df.close()
    lines = None 
    
    ## add observations 
    meta['time'] = list( dt_dates.dt.strftime('%Y-%m-%d') )
    meta['obs'] = list( discharge )
    meta['n_obs'] = np.count_nonzero(~np.isnan(discharge)) 
    meta['proj'] = 4326
    
    ### transform data for geo-formatting  
    
    point = geojson.Point( (meta['lon'], meta['lat']) ) 
    
    ### transfer data from meta to properties
    properties = {} 
    property_keys = ['time', 'obs', 'n_obs', 'description',
                     'elevation', 'loc_id', 'loc_name',
                     'proj']
    
    for key in property_keys:
        properties[key] = meta[key]
               
    if transform:
        X,Y = reproject_coordinates(np.array([meta['lon'], meta['lat']]),
                                    src_epsg = src_proj, dst_epsg = dst_proj)[0]
        
        properties['X'] = X
        properties['Y'] = Y
        properties['proj_XY'] = dst_proj
     
    if out_dir == None:
        return geojson.Feature(geometry=point, 
                               properties=properties )
    else:
        fn_file = '{}.geojson'.format( fn.stem ) 
        out_fn = out_dir / fn_file 
            
        features = [] 
        features.append( 
            geojson.Feature(
                geometry=point,
                properties=properties ))       
        
        feature_collection = geojson.FeatureCollection(features)
        
        with open(out_fn, 'w') as f:
            geojson.dump(feature_collection, f)
        

if __name__ == '__main__':
    
    base_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\dev")
    gauge_dir = base_dir / "V1_gauge_obs"
    output_dir = base_dir / "V1_gauge_obs_geojson"

    src_files = [file for file in gauge_dir.glob('*')] 
    
    ## TO CREATE SEPARATE FILES 
    # for file in src_files:
    #     try:
    #         read_gauge_data(file, out_dir = output_dir, transform = True)
    #     except:
              ## fails if no observations 
    #         print('fail: ', file) 
    
    ## TO CREATE A SINGLE FILE 
    collect_features = [] 
    for file in src_files:
        try:
            feature = read_gauge_data(file, transform = True) 
            collect_features.append(feature) 
        except:
            ## fails if no observations 
            print('failed: ', file) 
    
    print(len(collect_features))
    feature_collection = geojson.FeatureCollection(collect_features) 
    output_file = output_dir / 'V1_gauge_observations.geojson' 
    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)














