# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:39:05 2020

@author: mvand
"""

from pathlib import Path 
import xarray as xr 
import numpy as np 
import pandas as pd 
import pyproj 


def reproject_coordinates(src_coords, src_epsg, dst_epsg=int(4326)):
    
    '''
    Function that transforms coordinates - assumes input is a matrix 
    of n columns and 2 rows, with first column containing x coordinates,
    and second column with y coordinates 
    '''
    
    ## check input 
    assert src_coords.shape[1] >= 2, '[ERROR] too little columns - can not unpack x y coordinates'
    assert type(src_epsg) == int, '[ERROR] source epsg code (src_epsg) is not an integer'
    assert type(src_epsg) == int, '[ERROR] destination epsg code (dst_epsg) is not an integer'  
    
    ## define transformer 
    transformer = pyproj.Transformer.from_crs('epsg:{}'.format(int(src_epsg)),
                                             'epsg:{}'.format(dst_epsg))
    
    ## extract x and y coordinates as vectors 
    src_x = src_coords[:,0]
    src_y = src_coords[:,1]
    
    ## transform vectors & return 
    return transformer.transform(src_x, src_y)

def get_file_paths(base_dir, file_ext='*', long_name=False):
    
    ## create empty dictionary 
    path_dict = {}
    
    ## loop through sub-directories of base_dir 
    for sub_dir in Path(base_dir).rglob("**/*"):
        
        ## create empty list for files in sub-directories
        sub_files = []
        
        ## if sub_dir is sub-directory
        if sub_dir.is_dir():
            
            ## loop through sub-directories 
            ## optional: look only for specific file types 
            for sub_file in sub_dir.rglob("*.{}".format(file_ext)):
                
                ## add files to list 
                if long_name:
                    sub_files.append(sub_file)
                else:
                    sub_files.append(sub_file.name)
        
            ## add sub-directory and files to dictionary   
            if len(sub_files) >= 1:
                path_dict[sub_dir.name] = sub_files
    
    ## return dictionary 
    return path_dict 

def open_efas(fn, dT='24', return_all = False):
    
    delta_t = ['06', '24']
    
    assert dT in delta_t, '[ERROR] dT of {} doest not match "06" or "24"'.format(dT)
    assert Path(fn).exists(), '[ERROR] file {} not found'.format(fn)
    
    ## open dataset 
    ds = xr.open_dataset(fn) 
    ## check variables 
    var_list = list(ds.variables)
    
    time = ds['time']
    lat = ds['latitude']
    lon = ds['longitude']
    
    ## check discharge data time 
    Q_dT = 'dis{}'.format(dT)
    assert Q_dT in var_list, '[ERROR] dT does not match with dT in file'
    
    ## discharge data and (static) upstream area data in one file 
    Q = ds[Q_dT]
    A = ds['upArea']
    
    ## other data - incompatible with 
    ## GLOFAS data 
    if return_all:
        y_coords = ds['y']
        x_coords = ds['x']
        d_t = ds['step']
        surface = ds['surface']
        valid_time = ds['valid_time']
    
        return Q, A, lat, lon, time, x_coords, y_coords, d_t, surface, valid_time
    
    else:
        return Q, A, lat, lon 




def open_glofas(fn_list):
    Q = [] 
    A = [] 
    time = []
    
    for fn in fn_list:
        ## check file 
        assert Path(fn).exists(), '[ERROR] file {} not found'.format(fn)
        
        ## open dataset 
        ds = xr.open_dataset(fn) 
        ## check variables 
        var_list = list(ds.variables)
        
        ## get lat/lon vars 
        lat_i = ds['lat']
        lon_i = ds['lon']
        
        lat_attr = lat_i.attrs 
        lon_attr = lon_i.attrs 

        ##  GLOFAS discharge data
        if 'time' in var_list:
            time_i = ds['time'].data[0]
            Q_i = ds['dis24'].data[0]
            Qattr = ds['dis24'].attrs 
           
            ## accumulate discharge data 
            ## with time identity 
            Q.append(Q_i)
            time.append(time_i)
                        
        ## GLOFAS static upstream area data 
        if 'upArea' in var_list:
            A = ds['upArea']
    
    ## create new DataArray with all discharge data 
    Q = np.array(Q)
    Q = xr.DataArray(Q, coords = [time, lat_i, lon_i], dims=["time", "lat", "lon"], attrs = Qattr, name='dis24')
    
    ## create new DataArrays for lon and lat 
    lat, lon = np.meshgrid( lon_i, lat_i)
    lat = xr.DataArray(lat, name = 'latitude', dims=["y", "x"], attrs = lat_attr)
    lon = xr.DataArray(lon, name = 'longitude', dims=["y", "x"], attrs = lon_attr)
    
    return Q, A, lat, lon


    

def open_data(fn_list, dtype='GLOFAS', dT='24', return_efas = False):
    
    ## set options 
    dtypes = ['glofas', 'efas']
    delta_t = ['06', '24']
    
    ## check input 
    assert dtype.lower() in dtypes, '[ERROR] dtype {} not found - should be {}'.format(dtype, dtypes)
    assert dT in delta_t, '[ERROR] dT of {} doest not match "06" or "24"'.format(dT)
    
    Q = [] 
    A = [] 
    time = []
             
    for fn in fn_list:
        ## check file 
        assert Path(fn).exists(), '[ERROR] file {} not found'.format(fn)
    
        ## open dataset 
        ds = xr.open_dataset(fn) 
        ## check variables 
        var_list = list(ds.variables)
        
        ## extract data for EFAS
        if 'efas' in dtype.lower():
            
            time = ds['time']
            lat = ds['latitude']
            lon = ds['longitude']
            
            ## check discharge data time 
            Q_dT = 'dis{}'.format(dT)
            assert Q_dT in var_list, '[ERROR] dT does not match with dT in file'
            
            ## discharge data and (static) upstream area data in one file 
            Q = ds[Q_dT]
            A = ds['upArea']
            
            ## other data - incompatible with 
            ## GLOFAS data 
            # if return_efas:
            #     y_coords = ds['y']
            #     x_coords = ds['x']
            #     d_t = ds['step']
            #     surface = ds['surface']
            #     valid_time = ds['valid_time']
            
            #     return Q, A, lat, lon, time, x_coords, y_coords, d_t, surface, valid_time
            # else:
            
            # return lat, lon, time, Q, A
            return Q, A, lat, lon 
            
        ## extract data for GLOFAS 
        if 'glofas' in dtype.lower():
                
            lat_i = ds['lat']
            lon_i = ds['lon']
            
            lat_attr = lat_i.attrs 
            lon_attr = lon_i.attrs 

            ##  GLOFAS discharge data
            if 'time' in var_list:
                time_i = ds['time'].data[0]
                Q_i = ds['dis24'].data[0]
                Qattr = ds['dis24'].attrs 
               
                ## accumulate discharge data 
                ## with time identity 
                Q.append(Q_i)
                time.append(time_i)
                            
            ## GLOFAS static upstream area data 
            if 'upArea' in var_list:
                A = ds['upArea']
    
    ## create new DataArray with all discharge data 
    Q = np.array(Q)
    Q = xr.DataArray(Q, coords = [time, lat_i, lon_i], dims=["time", "lat", "lon"], attrs = Qattr, name='dis24')
    
    lat, lon = np.meshgrid( lon_i, lat_i)
    lat = xr.DataArray(lat, name = 'latitude', dims=["y", "x"], attrs = lat_attr)
    lon = xr.DataArray(lon, name = 'longitude', dims=["y", "x"], attrs = lon_attr)
    
    return Q, A, lat, lon


def read_gauge_data(fn, dtype = 'rws', transform = False, src_proj = None, dst_proj = None):
    
    '''
    Function that reads different types of gauge data - 
    and returns a dataframe with same set/order of columns:
    [loc_id, quantity, unit, date, time, value, epsg, x, y]
    
    fn         path to (csv) datafile 
    dtype      datasource - file will be opened accordingly 
    transform  transform coordinate system 
    '''
    
    
    dtypes = ['rws']
    output_cols = ['loc_id', 'quantity', 'unit', 'date', 
                   'time', 'value', 'epsg', 'X', 'Y']

    assert Path(fn).exists(), '[ERROR] file not found'
    assert dtype.lower() in dtypes, '[ERROR] datasource {} not found'.format(dtype)
    
    if 'rws' in dtype.lower():
        df = pd.read_csv(fn, sep='[;,;;,;;;,;;;;]', header = 0, 
                         index_col = False, engine='python')
        
        
        use_cols = ['MEETPUNT_IDENTIFICATIE', 'GROOTHEID_ CODE', 
                    'EENHEID_CODE', 'WAARNEMINGDATUM', 
                    'WAARNEMINGTIJD', 'NUMERIEKEWAARDE', 'EPSG', 
                    'X', 'Y']
        
        out_df = df[use_cols]
        out_df.columns = output_cols 
        
    return out_df













