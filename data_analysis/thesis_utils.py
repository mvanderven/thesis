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
from tqdm import tqdm
import warnings
import datetime 

import matplotlib.pyplot as plt 

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
    
    # print(ds)
    
    return ds 

def open_efas(fn, dT='24', do_resample_24h = False):
    
    if type(fn) is list:
        ds_out = xr.open_mfdataset(fn, chunks={"time": 100})#, 
                                                # 'x': 100,
                                                # 'y': 100}) 
        
        if do_resample_24h:
            ds_out = ds_out.resample(time="1D").mean() 
        
        ## rename some variables 
        ds_out = ds_out.rename(name_dict={  'latitude': 'lat',
                                            'longitude': 'lon',
                                            'dis06': 'dis24'})
        return ds_out 
        
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
    df.index = pd.to_datetime(df.index.values)
    
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
                   'epsg', 'nan_val', 'loc_id', 'y', 
                   'x', 'upArea', 'unit']
    
    out_df = pd.DataFrame(columns=output_cols)
    # df_len = 0 
    
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
            X_vals = out_df['x']
            Y_vals = out_df['y']

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
            out_df = out_df.drop(columns=['value', 'x', 'y'])
            out_df['value'] = Q_vals
            out_df['x'] = X_vals
            out_df['y'] = Y_vals

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
            meta['lat'] = np.unique( out_df['y'].values)
            meta['lon'] = np.unique( out_df['x'].values)
            meta['upArea'] = [np.nan] * 3 
            meta['description'] = [ np.unique(df['GROOTHEID_OMSCHRIJVING'].values)[0] ] * 3
            meta['unit'] = [ np.unique(out_df['unit'].values)[0] ] * 3   
            meta['epsg'] = [ int( np.unique(out_df['epsg'].values)[0] ) ] * 3 
            meta['start_date'] = str(datetime_series.iloc[0])
            meta['end_date'] = str(datetime_series.iloc[-1])
     
    if 'grdc' in dtype.lower():
        
        for fn in fn_list:   
            
            assert Path(fn).exists(), '[ERROR] file not found'
            
            ## get gauge id nr from filename             
            gauge_id_nr = fn.name.split('_')[0]
            
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
            
            temp_df['loc_id'] = gauge_id_nr 
            meta['loc_id'] = gauge_id_nr 
            
            for line in lines:
                vals = line.split(' ')

                if 'Station:' in vals:
                    meta['loc_name'] = vals[-1].replace('\n', '').lower()
                    temp_df['loc_id_name'] = meta['loc_name']

                if 'missing' in vals:
                    meta['nan'] = float(vals[-1])
                    temp_df['nan_val'] = float(vals[-1])

                if 'Latitude' in vals:
                    meta['lat'] = float(vals[-1])
                    # temp_df['y'] = meta['lat']
                    temp_df['lat'] = meta['lat']

                if 'Longitude' in vals:
                    meta['lon'] = float(vals[-1])
                    # temp_df['x'] = meta['lon']
                    temp_df['lon'] = meta['lon']

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

                # meta['start_date'] = str(dt_dates.iloc[0])
                # meta['end_date'] = str(dt_dates.iloc[-1])

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


def search_ds(ds, search_dict, return_df = False):
    
    ## get search keys 
    keys = list(search_dict.keys())
    
    ## variables of interest 
    voi = ['dis24', 'dis06', 'upArea', 'lat', 'latitude', 'lon', 'longitude']
    
    ## coordinates of interest 
    coi = ['x', 'y']
      
    ## loop through queries 
    for key in keys:
        sub_keys = search_dict[key].keys() 
        
        assert 'query' in sub_keys, '[ERROR] no search query found for {}'.format(key)
        
        if not 'method' in sub_keys:
            method = None 
        else:
            method = search_dict[key]['method']
        
        if not 'tolerance' in sub_keys:
            tolerance = None 
        else:
            tolerance = search_dict[key]['tolerance']
        
        query = search_dict[key]['query']
        subset = ds.sel(query, method=method, tolerance = tolerance)
    
    ## create output dataframe instead of xarray dataset 
    if return_df:
        ix = subset.time.values
        
        out_df = pd.DataFrame(index=ix)

        data_vars = list( subset.data_vars )
        # coord_vars = list(subset.coords)
        
        for var in data_vars: 
            if var in voi:
                out_df[var] = subset[var].data 
                
        coord_vars = subset.coords.keys() 
        for coord in coord_vars:
            if coord in coi:
                out_df[coord] = subset[coord].values                
        return out_df
    
    return subset

def iterative_pixel_search(ds, location, init_x, init_y, 
                           cell_size_x, cell_size_y, buffer_size, 
                           cols, coords=['time'], locs=['x','y']):
    
    ## find center coordinates 
    center_search = ds.sel( {locs[0]: init_x,
                           locs[1]: init_y},
                         method = 'nearest')
    
    ## extract center coordiantes 
    cx = center_search.x.values 
    cy = center_search.y.values   
    
    ## create buffer area 
    mask_x = ( ds[locs[0]] >= cx - (1.1*buffer_size*cell_size_x) ) & ( ds[locs[0]] <= cx + (1.1*buffer_size*cell_size_x) )
    mask_y = ( ds[locs[1]] >= cy - (1.1*buffer_size*cell_size_y) ) & ( ds[locs[1]] <= cy + (1.1*buffer_size*cell_size_y) )
    
    ## extract buffer 
    buffer_ds = ds.where( mask_x & mask_y, drop=True)
    
    ## get coordinates 
    x_cells = buffer_ds[locs[0]].values 
    y_cells = buffer_ds[locs[1]].values 
        
    ## convert xarray dataset to dataframe 
    buffer_df = buffer_ds.to_dataframe()  
    buffer_cols = buffer_df.columns 
    
    ## create empty output dataframe 
    out_df = pd.DataFrame()    
    ## set iteration counter to 0 
    n_iter = 0 
    
    ## start looping through all cells in buffer 
    for i in range(len(x_cells)):
        for j in range(len(y_cells)):
            
            ## search results 
            sub_df = buffer_df.loc[ :, x_cells[i], y_cells[j]  ]
            
            ## create empty dataframe 
            _df = pd.DataFrame()
            
            ## dates 
            sub_ix = sub_df.index    
            _df['date'] = sub_ix 
             
            ## get timeseries values 
            for col in cols:
                _df[col] = sub_df[col].values  

            ## get coordinates 
            for col in ['lat', 'lon']:
                if col in buffer_cols:
                    _df[col] = sub_df[col].values
            
            ## set other metadata 
            _df['match_gauge'] = location 
            _df['iter_id'] = n_iter 
            _df['x'] = x_cells[i] 
            _df['y'] = y_cells[j]
            n_iter += 1 
            
            ## append dataframe to output dataframe 
            out_df = out_df.append(_df)
    return out_df 


def buffer_search(ds, df_gauges, cell_size_X, cell_size_Y,
                  buffer_size, cols = ['dis24', 'upArea'], coords=['time'],
                  save_csv = False, save_dir = None):
    
    out_df = pd.DataFrame() 
    
    ## ignore warning that come with loading 
    ## data into dataframe 
    warnings.filterwarnings('ignore')
    
    gauge_locations = df_gauges['loc_id'].unique() 
    
    ## start loop trough gauge locations 
    for i in tqdm(range(len(gauge_locations))):
        loc= gauge_locations[i]
        
        ## get gauge X and Y  
        data_gauge = df_gauges[ df_gauges['loc_id'] == loc]
        gauge_X = data_gauge['x'].unique()[0] 
        gauge_Y = data_gauge['y'].unique()[0]
    
        ##buffer search
        df_buffer = iterative_pixel_search(     ds,
                                                loc,
                                                init_x = gauge_X,
                                                init_y = gauge_Y,
                                                cell_size_x = cell_size_X,
                                                cell_size_y = cell_size_Y,
                                                buffer_size = buffer_size,
                                                cols = cols,
                                                coords = coords) 
        
        out_df = out_df.append(df_buffer)
 
    ## reset display of warning messages
    warnings.filterwarnings('default')
    
    ## save results? 
    if save_csv:
        fn_out = 'buffer_search_{}_buffer_size_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'), buffer_size )
        if save_dir is not None:
            fn_out = Path(save_dir) / fn_out 
        out_df.to_csv(fn_out, index=False)
        return out_df, fn_out
    
    return out_df 


def match_label(df_features, df_match, 
                gauge_id, feature_X, feature_Y,
                match_id, match_X, match_Y,
                tol = 5000.): 
       
    n_found = 0  
    n_not_available = 0 
    n_not_found= 0 
    
    rejected_x = [] 
    rejected_y = [] 
    id_to_remove = [] 
    
    ## get gauge IDs from labelled set 
    match_ids = df_match[match_id].astype('str').values
        
    ## filter feature_ids based on gauge match 
    for m_id in match_ids:
        
        ## get subset of df with features - without the gauge data 
        subset_features = df_features[ (df_features[gauge_id] == m_id) & (df_features['is_gauge'] == 0)] 
        
        ## extract the gauge from features 
        gauge_data = df_features[ (df_features[gauge_id] == m_id) & (df_features['is_gauge'] == 1)] 
                
        if len(subset_features) > 0:

            ## get gauge specific labelled data 
            loc_df = df_match[ df_match[match_id] == int(m_id)] 
            
            ## get Lisflood XY coordinates & matching lat/lon coordinates 
            ecmwf_X, ecmwf_Y, ecmwf_lat, ecmwf_lon = loc_df[[match_X, match_Y, 'StationLat', 'StationLon']].values[0]
            
            ## get mapped Lisflood XY coordinates 
            x_mapped, y_mapped = loc_df[[match_X, match_Y]].values[0]
            
            ## get gauge data - as gauge directly placed on model 
            grdc_X, grdc_Y, grdc_lat, grdc_lon = gauge_data[[feature_X, feature_Y, 'lat', 'lon']].values[0]

            ## collect data and calculate distances 
            sort_df = pd.DataFrame() 
            sort_df['X'] = subset_features[feature_X]
            sort_df['Y'] = subset_features[feature_Y]
            sort_df['dX'] = sort_df['X'] - x_mapped
            sort_df['dY'] = sort_df['Y'] - y_mapped
            sort_df['X_abs'] = sort_df['dX'].abs() 
            sort_df['Y_abs'] = sort_df['dY'].abs() 
                     
            ## sort based on absolute distances 
            sorted_df = sort_df.sort_values(by=['X_abs', 'Y_abs']) 
            
            ## get ID index of best match 
            matched_ID = sorted_df.index[0]

            ## check match - extract dX and dY 
            dX = sorted_df.loc[matched_ID, 'X_abs']
            dY = sorted_df.loc[matched_ID, 'Y_abs']
            

            ## if tolerance smaller than size of grid cell: accept match 
            if (dX <= tol)  & (dY <= tol):
                df_features.loc[matched_ID, 'match_obs'] = 1
                n_found += 1 
                
            ## else: reject match
            else:
                n_not_found += 1 
                rejected_x.append(dX) 
                rejected_y.append(dY) 
                # id_to_remove.append(m_id)
                                                
                print('Gauge id: ', m_id)
                print('XY  cell distance (ecmwf minus grdc): {:.3f}, {:.3f}'.format((ecmwf_X-grdc_X)/tol, (ecmwf_Y-grdc_Y)/tol ) )
                print('\n')        
            

        else:
            n_not_available += 1 
            
    ## remove not found from dataset 
    # ix_to_remove = df_features.index[ df_features['match'].isin(id_to_remove) ]
    # df_features = df_features.drop( index=ix_to_remove ) 
    
    ## show overview 
    print('\n\n[OVERVIEW] \n {:.2f}% ({}) found \n {:.2f}% ({}) not found \n {:.2f}% ({}) gauge not available'.format( (n_found/len(match_ids))*100, n_found,                                      
                                                                                                                 (n_not_found/len(match_ids))*100, n_not_found,
                                                                                                                 (n_not_available/len(match_ids))*100, n_not_available) )
    
    ## fill non-one values with zeroes 
    df_features['match_obs']  = df_features['match_obs'].fillna(0)

    ## final check - only gauges with a match remain 
    for match_id in df_features['match'].unique():
        check_sum = df_features[ (df_features['match'] == match_id)]
        if not check_sum['match_obs'].sum() > 0:
            df_features = df_features.drop(index = check_sum.index)

    return df_features 


def calc_similarity_vector(df, gauge_id_col, type_col, label_col, feature_cols, method = 'euclidian'): 
    
    methods = ['euclidian'] # to implement??: ['cosine'] 
    
    assert method.lower() in methods, '[ERROR] selected method {} not in available methods: {}'.format(method.lower(), methods) 
    
    ### loop through df 
    ### calculate distance to observation features for each potential match 
    ### according to selected method (default = euclidian)
    ### save similarity vectors in new dataframe 
    df_SV = pd.DataFrame() 
    
    ## get unique identifiers  
    gauge_ids = df[gauge_id_col].unique() 
    
    for gauge_id in gauge_ids: 
        
        ## get subset of single gauge 
        gauge_set = df[ df[gauge_id_col] == gauge_id] 
        
        ## split into observations and simulations 
        obs_features = gauge_set[ gauge_set[type_col] == 1 ]
        sim_features = gauge_set[ gauge_set[type_col] == 0 ]
        
        ## get target values/labels and index 
        labels = sim_features[label_col]
        ix = sim_features.index
        
        ## calculate distance according to method 
        if method == 'euclidian':
            distance = ((sim_features[feature_cols].values - obs_features[feature_cols].values)**2)**0.5
        
        ## create dataframe with results 
        _df = pd.DataFrame(distance, columns=feature_cols) 
        _df = _df.set_index(ix)
        _df['target'] = labels 
        
        ## add to similarity vector dataframe 
        df_SV = df_SV.append(_df)
    return df_SV

def norm_scaler():
    return 



