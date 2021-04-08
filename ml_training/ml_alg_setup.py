# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:37:08 2021

@author: mvand
"""

#%% Load modules 

import pandas as pd  
from pathlib import Path 

#%% Load utils 

import ml_utils_update as utils 

#%% Set data paths 

data_dir = Path.home() / Path(r"Documents\Master EE\Year 4\Thesis\data")

model_dir = data_dir / Path(r"model_data")
train_dir = data_dir / Path(r"training_data") 
geo_dir   = data_dir / Path(r"river_network_DRT\data\DRT\upscaled_global_hydrography\by_HydroSHEDS_Hydro1k\shapefiles\global_16th")

#%% Set file names 

training_fn = train_dir / "similarity_vector_labelled_buffer_2-20210311.csv"

timeseries_fn = model_dir / "collect_ts_obs_mod_20210315_2.csv"
locations_fn =  model_dir / "collect_loc_obs_mod_20210315_2.csv"

river_network_fn = geo_dir / r"DraingeLine_16th_10.shp" 


#%% Load labelled similarity vector set 

df_target = pd.read_csv(training_fn, index_col = 0) 

print('-----'*10)
print('Load similarity vector dataset')
print('-----'*10)
print(df_target.head())

#%% Load labelled similarity vector set 

df_timeseries = pd.read_csv(timeseries_fn, index_col = 0) 

print('-----'*10)
print('Load timeseries dataset')
print('-----'*10)
print(df_timeseries.head()) 

#%% Test: drop features
df_target.drop(columns=['lat', 'lon'], inplace=True)

#%% Extract gauge names to get gauge IDs 

idx = pd.Series(df_target.index.values).str.split('_', expand=True).values 
## add as separate column
df_target['gauge_id'] = idx[:,1]

#%% Start cross validation procedures 

df_cv = utils.Kfold_CV(df_target, 'gauge_id', 'target', 
                        methods = ['LogisticRegressor-1', 'LogisticRegressor-2', 'nc'], 
                        df_ts = df_timeseries,
                        k=5) 

#%% After processing 

df_cv['p_true'] = (df_cv['n_true'] / df_cv['n_sample'])*100

for method in df_cv['method'].unique():
    df_method = df_cv[ df_cv['method'] == method] 
    print(method)
    print(df_method[['n_true', 'n_false', 'n_out', 'n_sample', 'p_true', 'acc', 'prec', 'rec', 'f1']].sum()) 
    print(df_method[['n_true', 'n_false', 'n_out', 'n_sample', 'p_true', 'acc', 'prec', 'rec', 'f1']].describe())
    print()


#%% Start cross validation procedures 

df_cv = utils.Kfold_CV(df_target, 'gauge_id', 'target', 
                        methods = ['LogisticRegressor-1', 'LogisticRegressor-2', 'nc'], 
                        df_ts = df_timeseries,
                        k=10) 

#%% After processing 

df_cv['p_true'] = (df_cv['n_true'] / df_cv['n_sample'])*100

for method in df_cv['method'].unique():
    df_method = df_cv[ df_cv['method'] == method] 
    print(method)
    print(df_method[['n_true', 'n_false', 'n_out', 'n_sample', 'p_true', 'acc', 'prec', 'rec', 'f1']].sum()) 
    print(df_method[['n_true', 'n_false', 'n_out', 'n_sample', 'p_true', 'acc', 'prec', 'rec', 'f1']].describe())
    print()

























