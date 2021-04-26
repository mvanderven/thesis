# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:21:29 2021

@author: mvand
"""

import pandas as pd  
import xarray as xr     
import time 

def load_efas(files): 

    ## get file names in dir 
    ## test run 
    # files = [file for file in efas_dir.glob('EFAS_1991_*.nc')] 
    
    ## final run
    # files = [file for file in efas_dir.glob('*.nc')] 
    
    ## load list 
    ds_out = xr.open_mfdataset(files, chunks = {"time": 30}) 
    
    ## rename variables for consistency
    # ds_out = ds_out.rename(name_dict={  'latitude': 'lat',
    #                                     'longitude': 'lon'})
    
    return ds_out 


def buffer_search(gauge_id, df, ds, buffer_size = 2, cell_size = 5000., var = 'dis24'): 
        
    ## create empty output dataframe
    out_df = pd.DataFrame()
        
    gauge_X, gauge_Y = df.loc[gauge_id, ['proj_X', 'proj_Y']]
    
    ## find closest cell center form (gauge_X, gauge_Y)
    center_search = ds.sel( {'x': gauge_X,
                             'y': gauge_Y},
                           method = 'nearest') 
    
    ## extract x and y from search results 
    center_cell_X = center_search.x.values 
    center_cell_Y = center_search.y.values 
    
    ## create a searching area 
    ## based on center_cell XY and buffer size 
    mask_x = ( ds['x'] >=  center_cell_X - (1.1*buffer_size*cell_size) ) & ( ds['x'] <=  center_cell_X + (1.1*buffer_size*cell_size) )
    mask_y = ( ds['y'] >= center_cell_Y - (1.1*buffer_size*cell_size) ) & ( ds['y'] <= center_cell_Y + (1.1*buffer_size*cell_size) ) 
    
    ## execute mask search 
    ds_buffer = ds.where( mask_x & mask_y, drop=True)
    
    ## get coordinates 
    buffer_x = ds_buffer['x'].values 
    buffer_y = ds_buffer['y'].values 
    
    ## slowest part ? read data into memory? 
    ## convert xarray to dataframe 
    df_buffer = ds_buffer.to_dataframe() 
    
    ## result is a multi-index array 
    ## could try to go for non-multi index array?
    # print(df_buffer.head())

    for i in range(len(buffer_x)):
        for j in range(len(buffer_y)):
            
            ## get specific cell from df_buffer
            df_cell = df_buffer.loc[:, buffer_x[i], buffer_y[j]][[var, 'lat', 'lon', 'upArea']]
            df_cell = df_cell.reset_index() 
            
            ## add metadata 
            df_cell['gauge'] = gauge_id 
            df_cell['ID'] = 'cell_{}_{}{}'.format(gauge_id, i,j)
            df_cell['x'] = buffer_x[i] 
            df_cell['y'] = buffer_y[j]
            df_cell = df_cell.set_index('ID')
            
            ## append to output dataframe 
            out_df = out_df.append(df_cell)
            
    return out_df 

def resample_efas(efas_dir, dir_out): 

    ## get file names in dir 
    files = [file for file in efas_dir.glob('*.nc')]  
    
    assert dir_out.exists(), '[ERROR] output dir does not exist'
    
    for file in files:
        
        print(file.name)
        time_resample = time.time() 
        
        ds_out = xr.open_dataset(file) 

        ds_out = ds_out.resample(time='1D').mean()
        
        ds_out = ds_out.rename(name_dict={    'latitude': 'lat',
                                               'longitude': 'lon',
                                               'dis06': 'dis24'}) 
        
        fn_out = '_'.join( file.name.split('_')[:-1])+ '_24h.nc'
        dst_out = dir_out / fn_out 
        
        clevel = 5
        ds_out.to_netcdf(dst_out, encoding={'dis24': {'zlib': True, 'complevel': clevel},
                                            'time': {'zlib': True, 'complevel': clevel},
                                            'x': {'zlib': True, 'complevel': clevel},
                                            'y': {'zlib': True, 'complevel': clevel},
                                            'lat': {'zlib': True, 'complevel': clevel},
                                            'lon': {'zlib': True, 'complevel': clevel},
                                            'upArea': {'zlib': True, 'complevel': clevel},
                                            'lambert_azimuthal_equal_area': {'zlib': True, 'complevel': clevel},
                                            'land_binary_mask': {'zlib': True, 'complevel': clevel}})
        
        print('Resampling finished in {:.2f} minutes\n'.format( (time.time() - time_resample)/60. ))

    return dir_out 




