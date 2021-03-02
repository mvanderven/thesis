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

## import own functions  --> check bar in top right hand to directory of files
import thesis_utils as utils 
import thesis_plotter as plotter 
import thesis_signatures 

#%% 

print('Collect file names')

## set data paths 
model_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")
gauge_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\gauge_data") 

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
efas_dir = model_data_dict[keys_efas[2]]
ds_efas = utils.open_efas(efas_dir)
print('EFAS data loaded \n')

#%%

print('Load gauge data')
## load gauge data 
gauge_file_names = gauge_data_dict[gauge_keys[0]] 
gauge_data_grdc, meta_grdc = utils.read_gauge_data( gauge_file_names, dtype='grdc')

print(gauge_data_grdc)


#%% 

gauge_locs = gauge_data_grdc['loc_id'].unique()
lat_coords = gauge_data_grdc['lat'].unique()
lon_coords = gauge_data_grdc['lon'].unique() 

coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = utils.reproject_coordinates(coords_4326, 4326, 3035)
print('Gauge data loaded \n')


#%% 

print('Execute time search')
## set up time of search query 
start_date = '2006-01-01'
end_date = '2006-12-31'

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
ds_efas = None

 
#%% 

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

efas_feature_table = thesis_signatures.calc_features(gauge_time, collect_efas, 
                                                     gauge_locs, 
                                                     features=calc_features,
                                                     n_lag=[1,2,5], 
                                                     n_cross = [0, 1, 5])

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















