# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:45:08 2021

@author: mvand
"""
import numpy as np 
import pandas as pd 
# import xarray as xr 
# import cfgrib
from pathlib import Path 
# from pprint import pprint
# import pyproj
import datetime
# import matplotlib.pyplot as plt 
# import cartopy 
import time 

## import own functions  --> check bar in top right hand to directory of files
import thesis_utils as utils 
import thesis_plotter as plotter 
import thesis_signatures 

#%% Collect file paths 

start_time_overall = time.time()

print('Collect file names')

home_dir = Path.home()

## set data paths 
model_data = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\model_data")
gauge_data = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\training_data")

## model data paths 
model_data_dict = utils.get_file_paths(model_data, 'nc', long_name=True)
keys = list(model_data_dict.keys())

keys_efas = [key for key in keys if 'EFAS' in key]

## gauge data - GRDC
gauge_data_dict = utils.get_file_paths(gauge_data, 'txt', long_name=True)
gauge_keys = list( gauge_data_dict.keys() )
print('File names loaded \n')


#%% Load EFAS data 

print('Load EFAS data')
# efas_dir = model_data_dict[keys_efas[1]]
efas_dir = model_data_dict[keys_efas[0]]
ds_efas = utils.open_efas(efas_dir, dT='06', do_resample_24h = True)
print('EFAS data loaded \n')

#%% Load gauge data 

print('Load gauge data')
start_time = time.time()


## load gauge data 
gauge_file_names = gauge_data_dict[gauge_keys[0]] 
gauge_data_grdc, meta_grdc = utils.read_gauge_data( gauge_file_names, dtype='grdc')

print(gauge_data_grdc.head())

print("--- {:.2f} minutes ---".format( (time.time() - start_time)/60.) )


#%% Reproject gauge coordinates

### sort data based on loc_id 
sorted_gauge_data = gauge_data_grdc[['loc_id', 'lat', 'lon']].groupby(by='loc_id')
## get lat,lon coordinates 
lat_coords = sorted_gauge_data['lat'].mean()
lon_coords = sorted_gauge_data['lon'].mean()

## get gauge identifiers 
gauge_locs = lat_coords.index 

## reproject coordinates 
coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = utils.reproject_coordinates(coords_4326, 4326, 3035)
 
## add reprojected coordinates to gauge dataframe
for i in range(len(gauge_locs)):
    gauge_id = gauge_locs[i]
    gauge_data_grdc.loc[ gauge_data_grdc['loc_id'] == gauge_id, 'x'] = coords_3035[i,0]
    gauge_data_grdc.loc[ gauge_data_grdc['loc_id'] == gauge_id, 'y'] = coords_3035[i,1]

print('Gauge data loaded \n')

#%% Apply time search to model data

print('Execute time search')
## set up time of search query 
start_date = '1991-01-01'
end_date = '1991-12-31'

T0 = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
T1 = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d') 

time_search = {
    'time': {
            'query': {'time':slice(T0,T1)},
            'method': None
            }}

efas_time = utils.search_ds(ds_efas, time_search, return_df = False)
print('Time search done \n')

#%% Apply time search to gauge data 

## crop gauge timeseries to dates 
gauge_time = gauge_data_grdc[ (gauge_data_grdc['date'] >= T0) & (gauge_data_grdc['date'] <= T1)]

#%% Drop nan-values from gauge time series 

## check if nan-values for gauges 
## in considered time period 
## if total gauge period is nan - gauge is thus dropped 
gauge_time = gauge_time.dropna(subset=['value'])

## give overview of stations dropped 
updated_gauge_locs = gauge_time['loc_id'].unique() 
gauges_dropped = 0 

for gauge in gauge_locs:
    if gauge not in updated_gauge_locs:
        print('[INFO] gauge {} dropped - no data in period of interest'.format(gauge))
        gauges_dropped += 1 
        
print('[INFO] {} gauges dropped - {} remaining\n'.format(gauges_dropped, len(updated_gauge_locs)))

## update list with gauge locations 
gauge_locs = updated_gauge_locs 

## release total gauge memory (?)
# gauge_data_grdc = None 


#%% Load or create buffer 

load_buffer_results = False

#%% Buffer analysis in model data 

if not load_buffer_results:
    
    print('Start buffer search\n')
    
    buffer_size = 1
    cell_size_efas = 5000       # m2 
    cell_size_glofas = 0.1      # degrees lat/lon
    
    ## TEST 
    collect_efas, fn_save_results = utils.buffer_search(
                                       efas_time, gauge_locs,
                                       coords_3035[:,0], coords_3035[:,1],
                                       cell_size_efas, cell_size_efas, buffer_size,
                                       save_csv=True, save_dir = model_data)
    
    print('Buffer search done \n')

#%% Load previously created buffer results 

if load_buffer_results: 
    print('Load buffer results \n')
    fn_save_step = model_data / 'save_buffer_search.csv'
    collect_efas = pd.read_csv(fn_save_step, index_col=0).astype({'match_gauge':'str'})

#%% Show buffer results 

## release glofas and efas xarray from memory (large memory)
# ds_efas = None

print(collect_efas.head())

#%% Collect model and gauge time series 

## combine efas and gauge time series to
## a df with unique timeseries in each column 
## and a df with coordinates of each gauge + iterations 
collect_timeseries, collect_locations = thesis_signatures.reshape_data(gauge_time, 
                                                                       collect_efas,
                                                                       gauge_locs, 
                                                                       var='dis24',
                                                                       T1 = end_date)

#%% Show collected data
print(collect_timeseries.head())


#%% Check for missing values in model simulation data 
missing_cols = collect_timeseries.columns[ collect_timeseries.isnull().any()].tolist() 

## minimum percentage of availabel data over period of interest 
max_percentage = 80. 

for col in missing_cols:
    n_missing = collect_timeseries[col].isnull().sum() 
    
    p_missing = (n_missing/len(collect_timeseries))*100
    n_remaining = len(collect_timeseries)-n_missing
    print('{} missing {} values -- {:.2f} % of total, {} remaining'.format(col,
                                                                           n_missing,
                                                                           p_missing,
                                                                           n_remaining) )
    
    if p_missing > max_percentage:
        print('\tMissing percentage is too large - removed from analysis')
        collect_timeseries = collect_timeseries.drop(columns=[col])  
        collect_locations = collect_locations.drop(index=[col])
        # collect_timeseries = collect_timeseries.drop(labels=[col], axis=1)

#%% Signature calculation 

### SIGNATURES 
calc_features = [
                    'normal', 
                    'log', 
                    'gev', 
                    'gamma', 
                    'n-acorr',
                    'n-ccorr',
                    'fdc-q', 'fdc-slope', 'lf-ratio',
                    'bf-index', 
                    'dld', 
                    'rld', 
                    'rbf',
                    'src'
                   ]

efas_feature_table = thesis_signatures.calc_features(collect_timeseries, 
                                                     collect_locations,
                                                     features=calc_features,
                                                     n_lag=[1,2,5], 
                                                     n_cross = [0, 1, 5],
                                                     var='dis06',
                                                     T_end = end_date)

#%% Show results signature calculation 
print(efas_feature_table.head())

#%% Missing values? 

print('\n Check for missing values:')
check_cols = [
    'Nm-all', 'Lmy-all', 'Lmx-all', 'Gu-all', 'Gk-all',
    'Ns-all', 'Lsy-all', 'Lsx-all', 'Ga-all', 'Gt-all',
    'N-gof-all', 'L-gof-all', 'Gev-gof-all', 'G-gof-all',
    'alag-1-all', 'alag-2-all', 'alag-5-all',
    'clag-0-all', 'clag-1-all', 'clag-5-all',
    'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
    'fdcQ-95-all', 'fdcQ-99-all',
    'fdcS-all', 'lf-all', 
    'bfi-all', 'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all'
    ]

print(efas_feature_table[check_cols].isnull().sum())

#%% Identify rows with missing values
missing_rows = efas_feature_table[ efas_feature_table[check_cols].isnull().any(axis=1) ].index.to_list()
print(missing_rows)

if len(missing_rows) >= 1:
    ## drop rows with missing values 
    efas_feature_table = efas_feature_table.drop(index=missing_rows)  
    print('Remaining missing values: ', efas_feature_table[check_cols].isnull().sum().sum()  )

#%% Save feature table values 

features_fn = gauge_data / 'unlabelled_features_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'))
efas_feature_table.to_csv(features_fn, index=False)
print('[INFO] features saved as csv:\n{}'.format(features_fn))                                                               

#%% Display feature cross-correlation

## all columns 
# cols_analysis = efas_feature_table.columns 

## all columns - sorted 
cols_analysis = [
    'x', 'lon', 'y', 'lat',
    'Nm-all', 'Lmy-all', 'Lmx-all', 'Gu-all', 'Gk-all',
    'Ns-all', 'Lsy-all', 'Lsx-all', 'Ga-all', 'Gt-all',
    'N-gof-all', 'L-gof-all', 'Gev-gof-all', 'G-gof-all',
    'alag-1-all', 'alag-2-all', 'alag-5-all',
    'clag-0-all', 'clag-1-all', 'clag-5-all',
    'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
    'fdcQ-95-all', 'fdcQ-99-all',
    'fdcS-all', 'lf-all', 
    'bfi-all', 'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all'
    ]

fig = plotter.display_cross_correlation(efas_feature_table, cols_analysis)


#%%  Label data-set 

### label with dataset 
labelled_fn = gauge_data / "grdc_efas_selection_20210218-1.csv" 
print(labelled_fn.exists())


#%% Show total time duration 
print("--- {:.2f}s seconds ---".format(time.time() - start_time_overall))





