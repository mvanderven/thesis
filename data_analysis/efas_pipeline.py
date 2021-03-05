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
from tqdm import tqdm 
import warnings 

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
efas_dir = model_data_dict[keys_efas[1]]
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
lat_coords = sorted_gauge_data['lat'].mean()
lon_coords = sorted_gauge_data['lon'].mean()

gauge_locs = lat_coords.index 

coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = utils.reproject_coordinates(coords_4326, 4326, 3035)
print('Gauge data loaded \n')


#%% Apply time search to model data

print('Execute time search')
## set up time of search query 
start_date = '1991-01-01'
end_date = '1991-03-31'

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


#%% 

load_buffer_results = True

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
    
    
    
    # def buffer_search(ds, gauge_locations, X0, Y0, cell_size_X, cell_size_Y,
    #                   buffer_size, cols = ['dis24', 'upArea'], coords=['time']):
    
    # collect_efas = pd.DataFrame()
    
    ## ignore warning that come with loading 
    ## data into dataframe 
    # warnings.filterwarnings('ignore')
    
    # for i in tqdm(range(len(gauge_locs))):
        # loc= gauge_locs[i]
    
        # ## efas buffer search
        # efas_buffer = utils.iterative_pixel_search(     efas_time,
        #                                                 loc,
        #                                                 init_x = coords_3035[i][0],
        #                                                 init_y = coords_3035[i][1],
        #                                                 cell_size_x = cell_size_efas,
        #                                                 cell_size_y = cell_size_efas,
        #                                                 buffer_size = buffer_size,
        #                                                 cols = ['dis24', 'upArea'],
        #                                                 coords = ['time']) 
        
        # collect_efas = collect_efas.append(efas_buffer)
        
    ## reset display of warning messages
    # warnings.filterwarnings('default')
    
    print('Buffer search done \n')

#%% 

if load_buffer_results: 
    fn_save_step = model_data / 'save_buffer_search.csv'
    collect_efas = pd.read_csv(fn_save_step, index_col=0)


#%% 

## release glofas and efas xarray from memory (large memory)
# ds_efas = None
 
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
min_percentage = 10. 

for col in missing_cols:
    n_missing = collect_timeseries[col].isnull().sum() 
    
    p_missing = (n_missing/len(collect_timeseries))*100
    n_remaining = len(collect_timeseries)-n_missing
    print('{} missing {} values -- {:.2f} % of total, {} remaining'.format(col,
                                                                           n_missing,
                                                                           p_missing,
                                                                           n_remaining) )
    
    if p_missing > min_percentage:
        print('Missing percentage is too large - remove from analysis')
        collect_timeseries = collect_timeseries.drop(columns=[col], axis=1) 
    

## TO DO -- set a limited minimum of required data points for gauge observations 
## if gauge observations do not meet this limit 
## exclude gauge + simulations from the analysis 

#%% Signature calculation 

### SIGNATURES 
calc_features = [
                    'normal', 
                    # 'log', 
                    # 'gev', 
                    # 'gamma', 
                    # 'n-acorr',          ## fix nan-issues - remaining length after nan removal too small
                    # 'n-ccorr',                        ## fix nan-issues 
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

#%% Show results signature calculation 

print(efas_feature_table)

#%% Display feature cross-correlation

import seaborn as sns 

cols_analysis = efas_feature_table.columns 

# cols_analysis = ['Nm-all', 'Ns-all', 'N-gof-all', 'Lmy-all',
#         'Lsy-all', 'Lmx-all', 'Lsx-all', 'L-gof-all', 'Gu-all', 'Ga-all',
#         'Gev-gof-all', 'Gk-all', 'Gt-all', 'G-gof-all', 'alag-1-all',
#         'alag-2-all', 'alag-5-all', 'clag-0-all', 'clag-1-all', 'clag-5-all',
#         'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
#         'fdcQ-95-all', 'fdcQ-99-all', 'fdcS-all', 'lf-all', 'bfi-all',
#         'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all']

plt.figure()
plt.title('Cross-correlation')
sns.heatmap(efas_feature_table[cols_analysis].corr(), vmin=-1, vmax=1, cmap='bwr' )
plt.show() 

#%%  Label data-set 

### label with dataset 
labelled_fn = gauge_data / "grdc_efas_selection_20210218-1.csv" 
print(labelled_fn.exists())



#%% Show total time duration 
print("--- {:.2f}s seconds ---".format(time.time() - start_time_overall))





