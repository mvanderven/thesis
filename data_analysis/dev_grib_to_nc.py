# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:33:20 2021

@author: mvand
"""

from pathlib import Path 
import xarray as xr 
import numpy as np 
import pandas as pd 

## to import own files 
import sys 
sys.path.append(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\code\data_analysis")
import dev_utils

def grib_to_nc(fn_list, months, out_fn = None):
    
    Q = []
    time = [] 
    
    i = 1 
    itot = len(fn_list)
    
    for fn in fn_list:
        
        print('Open file {} of {}'.format(i, itot))
        i += 1 
        
        ## check if file exists 
        assert Path(fn).exists(), '[ERROR] file {} not found'.format(fn)
        
        ## open dataset 
        ds = xr.open_dataset(fn, engine='cfgrib')
        ds_attr = ds.attrs 
                
        ## check variables 
        var_list = list(ds.variables)
        
        ## get latitude/longitude vars 
        lat = ds['latitude'].data 
        lon = ds['longitude'].data 
        
        if 'time' in var_list:
            
            ## extract time 
            time_i = ds['time'].data 
            
            ## collect in list 
            time.append(time_i)
            
            
        if 'dis24' in var_list:
            
            ## extract Q 
            Q_i = ds['dis24'].data 
            Q_attr = ds['dis24'].attrs 
            
            ## collect in list 
            for layer in Q_i:
                Q.append(layer)
    
    ## time to list 
    time = [t for t_list in time for t in t_list]
    dt_time = pd.DatetimeIndex(time) 
    
    print(time)
        
    ## Q to array 
    Q = np.array(Q) 
            
    for i in range(len(out_fn)): 
        
        file_name = out_fn[i]
        print(file_name)
        
        ixs = np.where( dt_time.month==months[i] )[0]
        
        dates = time[ixs[0]:(ixs[-1]+1)]
        sub_Q = Q[ixs[0]:(ixs[-1]+1)] 
        
        
        ds_out = xr.Dataset(
            
                {
                    'dis24': (["time", "lat", "lon"], sub_Q, Q_attr ) 
                },
                coords = {
                    "lat": (lat),
                    "lon": (lon),
                    "time": (dates)
                    },
                attrs = ds_attr
            )
        
        ds_out.to_netcdf(file_name)
    
    return out_fn 

## set data path 
# data_path = Path(r'C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\playground') 

## get files and folders 
# grib_files_dict = dev_utils.get_file_paths(data_path, 'grib', long_name=True) 
# keys_grib = list( grib_files_dict.keys() )
# test_grib = grib_files_dict[ keys_grib[1] ]


# out_fn = [
#     r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\playground\GLOFAS_2007\glofas_200710.nc", 
#     r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\playground\GLOFAS_2007\glofas_200711.nc",
#     r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\playground\GLOFAS_2007\glofas_200712.nc",
#     ]

# test_list = [test_grib[3]]
# months = [10,11,12]

# grib_to_nc(test_list, months, out_fn)



