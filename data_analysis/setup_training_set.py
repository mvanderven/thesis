# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:45:08 2021

@author: mvand
"""
import numpy as np 
import pandas as pd 
from pathlib import Path 
import datetime
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
# efas_dir = model_data_dict[keys_efas[1]] ## three months 
efas_dir = model_data_dict[keys_efas[0]]   ## one year 

ds_efas = utils.open_efas(efas_dir, dT='06', do_resample_24h = True)
print('EFAS data loaded \n')

#%% Load gauge data 

print('Load gauge data')
start_time = time.time()

## load gauge data 
# gauge_file_names = gauge_data_dict[gauge_keys[0]]  ## V0 ~ 200 gauges 
gauge_file_names = gauge_data_dict[gauge_keys[1]]  ## V1 ~ 802 gauges 

## get a sub-sample of gauge_file_names 
n_samples = 0 
if n_samples > 0:
    gauge_file_names = np.random.choice(gauge_file_names, n_samples) 
    
## load data 
gauge_data_grdc, meta_grdc = utils.read_gauge_data( gauge_file_names, dtype='grdc')

# #%% Repoject gauge coordinates 
coords_4326 = np.array( [gauge_data_grdc['lon'].values, gauge_data_grdc['lat'].values]).transpose()
coords_3035 = utils.reproject_coordinates(coords_4326, 4326, 3035) 

gauge_data_grdc['x'] = coords_3035[:,0]
gauge_data_grdc['y'] = coords_3035[:,1]

gauge_locs = gauge_data_grdc['loc_id'].unique()

print("--- {:.2f} minutes ---\n".format( (time.time() - start_time)/60.) )
print('Gauge data loaded \n')
print(gauge_data_grdc.head())

#%% Set up time boundaries 

start_date = '1991-01-01'
end_date = '2020-12-31'

T0 = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
T1 = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d') 

#%% EFAS time search 

print('Execute time search')

time_search = {
    'time': {
            'query': {'time':slice(T0,T1)},
            'method': None
            }}

efas_time = utils.search_ds(ds_efas, time_search, return_df = False)

print('Time search done \n')

#%% Apply time search to gauge data 

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

buffer_size = 2
cell_size_efas = 5000       # m2 
cell_size_glofas = 0.1      # degrees lat/lon

if not load_buffer_results:
    ## do this per year --> append results per defined period (for PC runs?)
    print('Start buffer search\n')
    


    collect_efas, fn_save_results = utils.buffer_search(
    # collect_efas = utils.buffer_search(
                                       efas_time, gauge_time,
                                       cell_size_efas, cell_size_efas, buffer_size,
                                       save_csv=True, save_dir = model_data)
    
    print('Buffer search done \n')

if load_buffer_results: 
    print('Load buffer results \n')
    fn_save_step = model_data / "buffer_search_20210310_buffer_size_2.csv"
    collect_efas = pd.read_csv(fn_save_step).astype({'match_gauge':'str'})

#%% Show buffer results 

## release glofas and efas xarray from memory (large memory)
# ds_efas = None

print(collect_efas.head())

#%% Collect model and gauge time series 

## combine efas and gauge time series to
## a df with unique timeseries in each column 

## and a second df with coordinates of each gauge + iterations 
print('Combine gauge and model timeseries')
start_time = time.time() 

collect_timeseries, collect_locations = thesis_signatures.reshape_data(gauge_time, 
                                                                       collect_efas,
                                                                       gauge_locs, 
                                                                       var='dis24',
                                                                       T1 = end_date)

fn_timeseries = model_data / 'collect_ts_obs_mod_{}_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'), 
                                                                    buffer_size) 

fn_locations = model_data / 'collect_loc_obs_mod_{}_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'),
                                                                    buffer_size)

collect_timeseries.to_csv(fn_timeseries) 
collect_locations.to_csv(fn_locations)




print("--- {:.2f} minutes ---\n\n".format( (time.time() - start_time)/60.) )

#%% Show collected data

print(collect_timeseries.head())


#%% Remove missing values in model simulation data 
# missing_series = collect_timeseries.columns[ collect_timeseries.isnull().any()].tolist() 

# collect_timeseries = collect_timeseries.drop(columns=missing_series) 
# collect_locations = collect_locations.drop(index=missing_series)

# n_drop = len(missing_series)
# n_after_drop = collect_timeseries.shape[1]
# print('{} simulations dropped - {} remaining for analysis \n'.format(n_drop, n_after_drop))

#%% Save cleaned up timeseries and location dataframes 

fn_timeseries = model_data / 'collect_ts_obs_mod_{}_{}-filtered.csv'.format( datetime.datetime.today().strftime('%Y%m%d'), 
                                                                    buffer_size) 

fn_locations = model_data / 'collect_loc_obs_mod_{}_{}-filtered.csv'.format( datetime.datetime.today().strftime('%Y%m%d'),
                                                                    buffer_size)

collect_timeseries.to_csv(fn_timeseries) 
collect_locations.to_csv(fn_locations)


#%% Signature calculation 

calc_features = True 

if calc_features:

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
                                                         n_lag=[1], 
                                                         n_cross = [0, 1],
                                                         T_end = end_date)

if not calc_features: 
    efas_feature_table_fn = gauge_data / "unlabelled_features_20210310_buffer_2.csv" 
    efas_feature_table = pd.read_csv(efas_feature_table_fn) 

#%% Show results signature calculation 
print(efas_feature_table.head())

#%% Missing values? 

print('\n[INFO] check for missing values:')

check_cols = [
    'Nm-all', 'Lm-all', 'Gu-all', 'Gk-all',
    'Ns-all', 'Ls-all', 'Ga-all', 'Gt-all',
    'N-gof-all', 'L-gof-all', 'Gev-gof-all', 'G-gof-all',
    'alag-1-all',
    'clag-0-all', 'clag-1-all',
    'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
    'fdcQ-95-all', 'fdcQ-99-all',
    'fdcS-all', 'lf-all', 
    'bfi-all', 'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all'
    ]

# print(efas_feature_table[check_cols].isnull().sum())

#%% Identify rows with missing values
missing_rows = efas_feature_table[ efas_feature_table[check_cols].isnull().any(axis=1) ].index.to_list()

if len(missing_rows) >= 1:
    ## drop rows with missing values 
    efas_feature_table = efas_feature_table.drop(index=missing_rows)  
    print('Remaining missing values: ', efas_feature_table[check_cols].isnull().sum().sum()  )

#%% Save feature table values 
features_fn = gauge_data / 'unlabelled_features_{}_buffer_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'),
                                                                        buffer_size)
efas_feature_table.to_csv(features_fn)
print('[INFO] features saved as csv:\n{}'.format(features_fn))       

#%% 

efas_feature_table = pd.read_csv(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data\unlabelled_features_20210310_buffer_2.csv",
                                 index_col=0)                                                        

#%% Display feature cross-correlation

## all columns 
# cols_analysis = efas_feature_table.columns 

## all columns - sorted 
cols_analysis = [
    'x', 'lon', 'y', 'lat',
    'Nm-all', 'Lm-all', 'Gu-all', 'Gk-all',
    'Ns-all', 'Ls-all', 'Ga-all', 'Gt-all',
    'N-gof-all', 'L-gof-all', 'Gev-gof-all', 'G-gof-all',
    'alag-1-all',
    'clag-0-all', 'clag-1-all',
    'fdcQ-1-all', 'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
    'fdcQ-95-all', 'fdcQ-99-all',
    'fdcS-all', 'lf-all', 
    'bfi-all', 'dld-all', 'rld-all', 'rbf-all', 's_rc-all', 'T0_rc-all'
    ]

fig = plotter.display_cross_correlation(efas_feature_table, cols_analysis)


#%%  Load labelled dataset 
labelled_fn = gauge_data / "V0-grdc_efas_selection_20210309-1.csv" 
# labelled_fn = gauge_data / "V1_grdc_efas_selection_20210319.csv"
df_labels = pd.read_csv(labelled_fn)

### COLUMNS OF INTEREST 
##             grdc station id      provided lat,   provided lon 
label_cols = ['updated_GRDC_ID',   'StationLat',   'StationLon',   
              
##            Lisflood mapped X/Y-coord 
              'StationX', 'StationY', 
              
##           lon/lat coords based on proximity 
             'snap_lon', 'snap_lat',  
             
##           Lisflood X/Y pixel based on lat/lon coords 
             'snap_X', 'snap_Y'  ] 

#%% Create labelled feature set 

features_matched = utils.match_label(efas_feature_table, df_labels, 
                                     'match', 'x', 'y',
                                     'updated_GRDC_ID', 'StationX', 'StationY')

#%% 

cell_matches = features_matched[ features_matched['match_obs']==1.]

#%% 
features_labelled_fn = gauge_data / 'labelled_features_{}_buffer_{}.csv'.format( datetime.datetime.today().strftime('%Y%m%d'),
                                                                        buffer_size)
## index = ID, so save index 
features_matched.to_csv(features_labelled_fn) 

# load_fn = gauge_data / "labelled_features_20210310_buffer_2.csv"
# features_matched = pd.read_csv(load_fn, index_col=0)
                                
#%% Calculate similarity vectors 

similarity_vectors = utils.calc_similarity_vector(features_matched, 
                                                  'match','is_gauge', 'match_obs',
                                                  feature_cols = cols_analysis,
                                                   method = 'euclidian' )

#%% Save set 

similarity_fn = gauge_data / 'similarity_vector_labelled_buffer_{}-{}.csv'.format(buffer_size,
                                                                              datetime.datetime.today().strftime('%Y%m%d'))

similarity_vectors.to_csv(similarity_fn)

#%% Show total time duration 
print("--- {:.2f}s seconds ---".format(time.time() - start_time_overall))





