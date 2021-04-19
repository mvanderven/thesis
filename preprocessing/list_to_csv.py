# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:59:12 2021

@author: mvand
"""

#%% Load modules 
import pandas as pd 
from pathlib import Path 
import pyproj 
import numpy as np 

#%% Get file paths 

data_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data") 
gauge_data = data_dir / 'V1' 
match_data = data_dir / "V1_grdc_efas_selection_20210319.csv"


#%% Functions 

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


def prep_gauge_processing_file(fn_matches, gauge_dir, out_dir = None, out_fn = 'cartesius_prep.csv', save_file = False):
    
    ## create output 
    out_df = pd.DataFrame()
    
    ## load match data 
    df_matches = pd.read_csv(fn_matches) 
    
    ## columns to extract 
    save_cols = ['updated_GRDC_ID', 'LisfloodX', 'LisfloodY'] 
    out_df[save_cols] = df_matches[save_cols] 
    
    ## format dataframe 
    out_df = out_df.rename(columns={
                                'updated_GRDC_ID': 'gauge_ID',
                                'LisfloodX': 'Lisflood_X',
                                'LisfloodY': 'Lisflood_Y'
                            })    
    out_df = out_df.set_index('gauge_ID')
        
    ## loop through gauge_dir 
    ## extract metadata:
    ##      lat,lon 
    ##      station upArea 
    ##      start date, end date 
    for file in gauge_dir.glob('*.txt'):
        
        ## get gauge id from file name 
        gauge_id = int( file.name.split('_')[0] )
                
        if gauge_id in out_df.index:
            
            sub_dict = {}
            
            ## read first lines to get metadata 
            lines = open(file).readlines()[:36]
                        
            for line in lines: 

                if 'Latitude' in line:
                    out_df.loc[gauge_id, 'gauge_lat'] = float(line.split(' ')[-1])
                
                if 'Longitude' in line: 
                    out_df.loc[gauge_id, 'gauge_lon'] = float(line.split(' ')[-1]) 
                
                if 'Time series' in line:
                    
                    t_range = line.split('    ')[-1].split(' - ') 
                    
                    if len(t_range) > 1:
                                        
                        t_start = t_range[0] 
                        start_year, start_month = [int(t) for t in t_start.split('-')]
                    
                        t_end = t_range[-1] 
                        end_year, end_month = [int(t) for t in t_end.split('-')]
                        
                        out_df.loc[gauge_id, 'start_year'] = start_year
                        out_df.loc[gauge_id, 'start_month'] = start_month
                        out_df.loc[gauge_id, 'end_year'] = end_year
                        out_df.loc[gauge_id, 'end_month'] = end_month
               
                if 'Catchment area' in line:
                    out_df.loc[gauge_id, 'upArea'] = float(line.split(' ')[-1])
    
    
    ## reproject gauge lat/lon coordinates 
    ## to EFAS coordinates (epsg:3039)            
    reproj_XY = reproject_coordinates(out_df[['gauge_lon', 'gauge_lat']].values, 
                                      src_epsg = 4326, dst_epsg=3035) 
    
    out_df['proj_X'] = reproj_XY[:,0] 
    out_df['proj_Y'] = reproj_XY[:,1] 
    
    ## drop any rows with missing values 
    out_df = out_df.dropna(axis = 0, how = 'any') 
    
    ## filter on end year 
    out_df = out_df[ out_df['end_year'] >= 1991] 
    
    
    ## analyse distances 
    
                
    ## return filename or dataframe 
    if save_file:
        if out_dir != None:
            out_fn = Path(out_dir) / out_fn 
        out_df.to_csv(out_fn)
        return out_fn 
    else:
        return out_df 


#%% Run 

if __name__ == '__main__':
    
    fn = prep_gauge_processing_file(match_data, gauge_data, out_dir = data_dir)
    
    

    
    
    
    
    
    
    