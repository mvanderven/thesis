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


def open_grib(fn):
    
    assert Path(fn).exists(), '[ERROR] file {} not found'.format(fn)
    ds = xr.open_dataset(fn, engine='cfgrib')
    
    print(ds)
    
    return

def open_efas(fn, dT='24'):
    
    if type(fn) is list:
        fn = fn[0]
        
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
    
    x = ds['x']
    y = ds['y']
    
    ## check discharge data time 
    Q_dT = 'dis{}'.format(dT)
    assert Q_dT in var_list, '[ERROR] dT does not match with dT in file'
    
    ## discharge data and (static) upstream area data in one file 
    Q = ds[Q_dT]
    A = ds['upArea']
       
    #### return new dataarray/dataset 
    ds_out = xr.Dataset(
                    
                    {
                        Q_dT: (["time", "y", "x"], Q.data, Q.attrs),
                        "upArea": (["y", "x"], A.data, A.attrs)
                    },
                coords = {
                    "lon": (["y", "x"], lon.data, lon.attrs),
                    "lat": (["y", "x"], lat.data, lat.attrs),
                    "x"  : (x.data),
                    "y"  : (y.data),
                    "time": time.data
                },
                attrs = ds.attrs)

    return ds_out 

def open_glofas(fn_list):
    Q = [] 
    time = []
    A = []
    
    data_dict = {}
    
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
        
        ##  GLOFAS discharge data
        if 'time' in var_list:
            
            ## extract time 
            time_i = ds['time'].data 
            
            ## extract Q 
            Q_i = ds['dis24'].data 
            
            Q_attr = ds['dis24'].attrs 
            
            ## get dataset attributes 
            data_attr = ds.attrs
           
            ## collect discharge data 
            ## with time identity 
            for layer in Q_i:
                Q.append(layer)
            
            for t_step in time_i:
                time.append(t_step)
                    
        ## GLOFAS static upstream area data 
        if 'upArea' in var_list:
            A = ds['upArea']
            A_attr = A.attrs 

    ## reshape Q to array 
    Q = np.array(Q)
    
    ## if present- add data to output file 
    if len(Q) > 0:
        data_dict['dis24'] = (["time", "lat", "lon"], Q, Q_attr)

    if len(A) > 0:
        data_dict['upArea'] = (["lat", "lon"], A.data, A_attr)
        
    ## create new output dataset
    ds_out = xr.Dataset(
        data_dict,
        coords = {
            "lon":(lon_i),
            "lat":(lat_i),
            "time":(time)
        },
        attrs = data_attr       
    )
    return  ds_out 

def resample_gauge_data(df, column, target_dt='D'):
    ## make sure datetime index 
    df.index = pd.to_datetime( df.index.values)
    
    ## resample 
    df[column] = df[column].resample(target_dt).mean()   
    
    ## set index correct 
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    ## return new dataframe 
    return df.dropna()



def read_gauge_data(fn_list, dtype = 'grdc', transform = False, src_proj = None, dst_proj = None, resample24hr=False):
    
    '''
    Function that reads different types of gauge data - 
    and returns a dataframe with same set/order of columns:
    [loc_id, quantity, unit, date, time, value, epsg, x, y]
    
    fn         path to (csv) datafile 
    dtype      datasource - file will be opened accordingly 
    transform  transform coordinate system 
    '''
    
    out_df = None
    meta = {}
    
    dtypes = ['rws', 'grdc']
    
#     output_cols = ['loc_id', 'quantity', 'unit', 'date', 
#                    'time', 'value', 'epsg', 'X', 'Y']
    
    output_cols = ['date', 'time', 'value', 'quantity', 
                   'epsg', 'nan_val', 'loc_id', 'Y', 
                   'X', 'upArea', 'unit']
    
    out_df = pd.DataFrame(columns=output_cols)
    df_len = 0 
    
    assert dtype.lower() in dtypes, '[ERROR] datasource {} not found'.format(dtype)
    
    if 'rws' in dtype.lower():
        
        for fn in fn_list:
            
            assert Path(fn).exists(), '[ERROR] file not found'
        
            ## create output dataframe 
            df = pd.read_csv(fn, sep='[;]', header = 0, 
                             index_col = False, engine='python')

            use_cols = ['MEETPUNT_IDENTIFICATIE', 'GROOTHEID_ CODE', 
                        'EENHEID_CODE', 'WAARNEMINGDATUM', 
                        'WAARNEMINGTIJD', 'NUMERIEKEWAARDE', 'EPSG', 
                        'X', 'Y']

            out_df = df[use_cols]
            out_df.columns = output_cols 

            ## check measurements for nan values        
            Q_vals = out_df['value']
            X_vals = out_df['X']
            Y_vals = out_df['Y']

            ## convert string to float and fill NaN values 
            Q_vals = np.array([s.replace(',','.') for s in Q_vals])
            Q_vals = Q_vals.astype(np.float)
            Q_vals[Q_vals >= 32767.] = np.nan 

            ## convert string to float 
            X_vals = np.array([s.replace(',','.') for s in X_vals])
            X_vals = X_vals.astype(np.float)

            ## convert string to float 
            Y_vals = np.array([s.replace(',','.') for s in Y_vals])
            Y_vals = Y_vals.astype(np.float)

            ## update dataframe 
            out_df = out_df.drop(columns=['value', 'X', 'Y'])
            out_df['value'] = Q_vals
            out_df['X'] = X_vals
            out_df['Y'] = Y_vals

            ## aggregate date and time - set as index 
            ## set format same as glofas/efas date format 
            datetime_series = out_df[['date', 'time']].agg(' '.join, axis=1)
            out_df.index = pd.to_datetime( datetime_series, 
                           format='%d-%m-%Y %H:%M:%S'
                           ).dt.strftime('%Y-%m-%d %H:%M:%S')

            if resample24hr:
                out_df = resample_gauge_data(out_df, 'value', target_dt='D')

            ## create metadata 
            meta['nan'] = [np.nan] * 3 
            meta['loc'] = np.unique( out_df['loc_id'].values )
            meta['lat'] = np.unique( out_df['Y'].values)
            meta['lon'] = np.unique( out_df['X'].values)
            meta['upArea'] = [np.nan] * 3 
            meta['description'] = [ np.unique(df['GROOTHEID_OMSCHRIJVING'].values)[0] ] * 3
            meta['unit'] = [ np.unique(out_df['unit'].values)[0] ] * 3   
            meta['epsg'] = [ int( np.unique(out_df['epsg'].values)[0] ) ] * 3 
            meta['start_date'] = str(datetime_series.iloc[0])
            meta['end_date'] = str(datetime_series.iloc[-1])
     
    if 'grdc' in dtype.lower():
        
        for fn in fn_list:   
            
            assert Path(fn).exists(), '[ERROR] file not found'
            
            ## open file 
            ##  other options for encoding:
            ## encoding = 'ascii'  # encoding = 'mbcs' # encoding = 'ansi'
            df = pd.read_csv(fn, skiprows=36, delimiter=';', encoding = 'ansi') 
        
            ## extract data 
            discharge = df[' Value'].values 
            dt_dates = pd.to_datetime(df['YYYY-MM-DD'], yearfirst=True, 
                                      format='%Y-%m-%d')

            ## what time
            dt_time =  pd.Series(['00:00:00']*len(dt_dates))
            # dt_time =  pd.Series(['12:00:00']*len(dt_dates))

            ## create output dataframe 
            temp_df = pd.DataFrame( {'date':dt_dates, 'time':dt_time, 
                                    'value':discharge})

            ## get metadata 
            byte_df = open(fn)
            lines = byte_df.readlines()[:36]
            
            for line in lines:
                vals = line.split(' ')

                if 'Station:' in vals:
                    meta['loc'] = vals[-1].replace('\n', '').lower()
                    temp_df['loc_id'] = meta['loc']

                if 'missing' in vals:
                    meta['nan'] = float(vals[-1])
                    temp_df['nan_val'] = float(vals[-1])

                if 'Latitude' in vals:
                    meta['lat'] = float(vals[-1])
                    temp_df['Y'] = meta['lat']

                if 'Longitude' in vals:
                    meta['lon'] = float(vals[-1])
                    temp_df['X'] = meta['lon']

                if 'Unit' in vals:
                    meta['unit'] = vals[-1].replace('\n', '')
                    temp_df['unit'] = meta['unit']

                if 'area' in vals:
                    meta['upArea(km2)'] = float(vals[-1])
                    temp_df['upArea'] = meta['upArea(km2)']

                if 'Content:' in vals:
                    meta['description'] = ' '.join( vals[-3:] ).replace('\n', '').lower()

                temp_df['quantity'] = 'Q'
                temp_df['epsg'] = 4326

                meta['start_date'] = str(dt_dates.iloc[0])
                meta['end_date'] = str(dt_dates.iloc[-1])

            ## set nan values 
            temp_df.loc[ temp_df['value'] == meta['nan'], 'value' ] = np.nan        

            ## close meta file 
            byte_df.close()
            lines = None 

            ## aggregate date and time - set as index 
            ## set format same as glofas/efas date format 
            temp_df.index = pd.to_datetime( temp_df['date'],
                                                format='%Y-%m-%dT%H:%M:%S.%f' 
                                               ).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            out_df = out_df.append(temp_df)
    return out_df, meta












