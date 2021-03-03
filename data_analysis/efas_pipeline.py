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
import matplotlib.pyplot as plt 
# import cartopy 
import time 

## import own functions  --> check bar in top right hand to directory of files
import thesis_utils as utils 
import thesis_plotter as plotter 
import thesis_signatures 

#%% 

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


#%%

print('Load EFAS data')
efas_dir = model_data_dict[keys_efas[0]]
ds_efas = utils.open_efas(efas_dir, dT='06')
print('EFAS data loaded \n')

#%%
print('Load gauge data')
start_time = time.time()


## load gauge data 
gauge_file_names = gauge_data_dict[gauge_keys[0]] 
gauge_data_grdc, meta_grdc = utils.read_gauge_data( gauge_file_names, dtype='grdc')

print(gauge_data_grdc.head())

print("--- {:.2f} minutes ---".format( (time.time() - start_time)/60.) )


#%% 

### sort data based on loc_id 
sorted_gauge_data = gauge_data_grdc[['loc_id', 'lat', 'lon']].groupby(by='loc_id')
lat_coords = sorted_gauge_data['lat'].mean()
lon_coords = sorted_gauge_data['lon'].mean()

gauge_locs = lat_coords.index 

coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = utils.reproject_coordinates(coords_4326, 4326, 3035)
print('Gauge data loaded \n')


#%% 

print('Execute time search')
## set up time of search query 
start_date = '1991-01-01'
end_date = '1991-01-31'

T0 = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
T1 = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d') 

time_search = {
    'time': {
            'query': {'time':slice(T0,T1)},
            'method': None
            }}

efas_time = utils.search_ds(ds_efas, time_search, return_df = False)
print('Time search done \n')

#%% 

## crop gauge timeseries to dates 
gauge_time = gauge_data_grdc[ (gauge_data_grdc['date'] >= T0) & (gauge_data_grdc['date'] <= T1)]

#%%
## check if nan-values for gauges 
## in considered time period 
## if total gauge period is nan - gauge is thus dropped 
gauge_time = gauge_time.dropna(subset=['value'])

## release total gauge memory (?)
# gauge_data_grdc = None 

#%%

print('Start buffer search')
## set up location search query 
## refine buffer analysis ? 

## buffer size = 1 --> returns 9 pixels --> 1 ring buffer 
n_iter = 3 
cell_size_efas = 5000       # m2 
cell_size_glofas = 0.1      # degrees lat/lon


## buffer size = 2 --> returns 25 pixels --> 2 ring buffer 
# n_iter = 5
# cell_size_efas = 5000 * n_iter * 0.4
# cell_size_glofas = 5000 * n_iter * 0.4 

collect_glofas = pd.DataFrame() 
collect_efas = pd.DataFrame()

for i in range(len(gauge_locs)):
    loc= gauge_locs[i]

    ## efas buffer search
    efas_buffer = utils.iterative_pixel_search(efas_time,
                                                    loc,
                                                    init_x = coords_3035[i][0],
                                                    init_y = coords_3035[i][1],
                                                    x_tol = cell_size_efas, 
                                                    y_tol = cell_size_efas,
                                                    n_iter = n_iter,
                                                    cols = ['dis24', 'upArea'],
                                                    coords = ['time']) 
    
    collect_efas = collect_efas.append(efas_buffer)
    
print('Buffer search done \n')


## release glofas and efas xarray from memory (large memory)
# ds_efas = None
 

#%% 

## resample timeseries from 6 hourly to 24 hourly 

collect_efas_resampled = utils.resample_gauge_data(collect_efas, 'dis06', target_dt='D')

#%% 

## combine efas and gauge time series to
## a df with unique timeseries in each column 
## and a df with coordinates of each gauge + iterations 
collect_timeseries, collect_locations = thesis_signatures.reshape_data(gauge_time, 
                                                                       collect_efas_resampled,
                                                                       gauge_locs, 
                                                                       var='dis06',
                                                                       T1 = end_date)

#%% 

### SIGNATURES 
calc_features = [
                    'normal', 
                    # 'log', 
                    # 'gev', 
                    # 'gamma', 
                    # 'n-acorr', 
                    # 'n-ccorr',
                    # 'fdc-q', 'fdc-slope', 'lf-ratio',
                    # 'bf-index', 
                    # 'dld', 
                    # 'rld', 
                    # 'rbf',
                    # 'src'
                   ]

efas_feature_table = thesis_signatures.calc_features(collect_timeseries, 
                                                     collect_locations,
                                                     features=calc_features,
                                                     n_lag=[1,2,5], 
                                                     n_cross = [0, 1, 5],
                                                     var='dis06',
                                                     T_end = end_date)

#%% 

print(efas_feature_table)

#%% 

import seaborn as sns 

cols_analysis = ['Nm-all', 'Ns-all', 'N-gof-all', 'Lmy-all',
        'Lsy-all', 'Lmx-all', 'Lsx-all', 'L-gof-all', 'Gu-all', 'Ga-all',
        'Gev-gof-all', 'Gk-all', 'Gt-all', 'G-gof-all', 'alag-1-all',
        'alag-2-all', 'alag-5-all', 'clag-0-all', 'clag-1-all', 'clag-5-all',
        'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
        'fdcQ-95-all', 'fdcQ-99-all', 'fdcS-all', 'lf-all', 'bfi-all',
        'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all']

plt.figure()
plt.title('Cross-correlation')
sns.heatmap(efas_feature_table[cols_analysis].corr(), vmin=-1, vmax=1, cmap='bwr' )
plt.show() 

#%%  

### label with dataset 
labelled_fn = gauge_data / "grdc_efas_selection_20210218-1.csv" 
print(labelled_fn.exists())



#%% 
print("--- {:.2f}s seconds ---".format(time.time() - start_time_overall))





