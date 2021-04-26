# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:11:15 2021

@author: mvand
"""

#%% Import modules

import pandas as pd 
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt

import thesis_utils as utils 
import thesis_plotter as plotter 
import thesis_signatures 

#%% Set path files 

dir_timeseries = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")
fn_ts = dir_timeseries / "efas_timeseries_1991_1992-test.csv" 

dir_gauges = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data") 
fn_gauges = dir_gauges / "V1_grdc_efas_selection-cartesius-snapped.csv" 

#%% Load data 

## timeseries per gauge 
df_model = pd.read_csv(fn_ts, index_col = 0) 

## gauge metadata 
df_gauges = pd.read_csv(fn_gauges, index_col=0)  

#%% Load gauge timeseries data 

do_sample = True
gauge_ids = df_gauges.index.values 

## sample?
if do_sample:
    n_samples = 3
    gauge_ids = df_gauges.sample(n=n_samples).index.values 

dir_v1 = dir_gauges/ 'V1' 

gauge_files = [ dir_v1 /'{}_Q_Day.Cmd.txt'.format(gauge_id) for gauge_id in gauge_ids  ]


## load gauge data 
gauge_data, meta_grdc = utils.read_gauge_data( gauge_files, dtype='grdc')

## filter time 
gauge_data = gauge_data[ (gauge_data['date'] >= '1991') &  (gauge_data['date'] < '1992')].copy()
# gauge_data = gauge_data[ gauge_data['date'] >= '1991'].copy()

#%% Filter model data based on selected gauge_ids 

df_model = df_model[ df_model['gauge'].isin(gauge_ids) ] 


#%% Reshape df_ts to dataframe with unique observations as columns 
##  and resample from 6hrly to 24 hrly 

def rows_to_cols(df, id_col, index_col, target_col, resample_24hr = False):
    
    out_df = pd.DataFrame()
    
    unique_ids = df[id_col].unique() 
    
    for target_id in unique_ids:
        
        _df = df[ df[id_col] == target_id ] 
        col_names = _df.index.unique() 
        
        for col_name in col_names:
            col_df = _df[ _df.index == col_name ] 
            col_df = col_df.set_index(index_col) 
            
            out_df[col_name] = col_df[target_col]
    
    if resample_24hr:
        out_df.index = pd.to_datetime( out_df.index  )
        out_df = out_df.resample('D').mean()
        
    return out_df  

df_simulations = rows_to_cols( df_model, 'gauge', 'time', 'dis06',
                        resample_24hr = True) 


#%% Calculate timeseries signatures 

df_features = thesis_signatures.calc_signatures(gauge_data, df_simulations,
                                                time_window = ['all', 'seasonal'], # 'monthly'],
                                                features = ['normal','bf-index', 'dld', 'rld', 'rbf', 'src']) 

#%% 

print(df_features.isnull().sum()) 

    
    
    






































