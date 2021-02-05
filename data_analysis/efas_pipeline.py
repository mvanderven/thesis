# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:45:08 2021

@author: mvand
"""
import numpy as np 
import pandas as pd 
import xarray as xr 
import cfgrib
from pathlib import Path 
from pprint import pprint
import pyproj
import datetime
import matplotlib.pyplot as plt 
import cartopy 

## import own functions  --> check bar in top right hand to directory of files
import dev_utils
import dev_plotter 
import dev_signatures 

#%% 

print('Collect file names')
## set data paths 
model_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")
gauge_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\gauge_data") 

## model data paths 
model_data_dict = dev_utils.get_file_paths(model_data, 'nc', long_name=True)
keys = list(model_data_dict.keys())

keys_efas = [key for key in keys if 'EFAS' in key]

## gauge data - GRDC
gauge_data_dict = dev_utils.get_file_paths(gauge_data, 'txt', long_name=True)
gauge_keys = list( gauge_data_dict.keys() )
print('File names loaded \n')

#%%

print('Load EFAS data')
efas_dir = model_data_dict[keys_efas[2]]
ds_efas = dev_utils.open_efas(efas_dir)
print('EFAS data loaded \n')

#%%

print('Load gauge data')
## load gauge data 
gauge_file_names = gauge_data_dict[gauge_keys[0]] 
gauge_data_grdc, meta_grdc = dev_utils.read_gauge_data( gauge_file_names, dtype='grdc')
 
gauge_locs = gauge_data_grdc['loc_id'].unique()
lat_coords = gauge_data_grdc['lat'].unique()
lon_coords = gauge_data_grdc['lon'].unique() 

coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = dev_utils.reproject_coordinates(coords_4326, 4326, 3035)
print('Gauge data loaded \n')


#%% 

print('Execute time search')
## set up time of search query 
start_date = '2007-01-01'
end_date = '2007-12-31'

T0 = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
T1 = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d') 

time_search = {
    'time': {
            'query': {'time':slice(T0,T1)},
            'method': None
            }}

efas_time = dev_utils.search_ds(ds_efas, time_search, return_df = False)
print('Time search done \n')

#%% 

## crop gauge timeseries to dates 
gauge_time = gauge_data_grdc[ (gauge_data_grdc['date'] >= T0) & (gauge_data_grdc['date'] <= T1)]

## release total gauge memory (?)
gauge_data_grdc = None 

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
    efas_buffer = dev_utils.iterative_pixel_search(efas_time,
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
ds_efas = None


#%% 

## now: 2 dataframes with model data around gauges 
## gauge timeseries with corresponding values 

subset_locations = ['andernach', 'nettegut', 'friedrichsthal'] 

gauge_set = gauge_time[gauge_time['loc_id'].isin(subset_locations)]
efas_set = collect_efas[ collect_efas['match_gauge'].isin(subset_locations)]


#%% 

show_efas_search = dev_plotter.dashboard(efas_time, efas_set, gauge_set, subset_locations)
 
#%% 

### SIGNATURES 
calc_features = [
                    'normal', 
                   #  'log', 
                   #    'gev', 
                   #  'gamma', 
                   #  'n-acorr', 
                   #  'n-ccorr',
                    # 'fdc-q', 'fdc-slope', 'lf-ratio',
                    'bf-index', 'dld', 'rld', 'rbf'
                   ]

efas_feature_table = dev_signatures.calc_features(gauge_time, collect_efas, gauge_locs, features=calc_features,
                                                  n_lag=[1,2,5], n_cross = [0, 1, 5])


















