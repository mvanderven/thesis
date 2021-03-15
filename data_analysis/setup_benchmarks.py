# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:19:16 2021

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


#%% Approaches 

'''
Benchmark performance using naive approaches. Select model pixel matching with observations
using the following methods:
    (1) nearest cell - place observation on cell closest to observation coordinates
    (2) min(RMSE) - in buffer, timeseries with smallest RMSE(observations,model) 
    
    ?? min(RMSE) for features, max(NSE) timeseries? 
    
Method 1
- labelled locations 
- smallest X and Y 

OR 
- load x-array 
- extract coordinates of nearest cell 
- check cell with labelled cells 

Metod 2 
- labelled locations 
- timeseries from cell within buffer 

'''


#%% Set paths & load data 

home_dir = Path.home()

## set data paths 
model_data = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\model_data")
gauge_data = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\training_data")

print('Load data')
## load data for nearest cell
labelled_features_fn = gauge_data / "similarity_vector_labelled_buffer_2-20210311.csv"
df_lf = pd.read_csv(labelled_features_fn, index_col=0) 

## load data for min(RMSE)
timeseries_fn = model_data / "collect_ts_obs_mod_20210315_2.csv"
locations_fn =  model_data / "collect_loc_obs_mod_20210315_2.csv"
df_ts = pd.read_csv(timeseries_fn, index_col=0) 
df_loc = pd.read_csv(locations_fn, index_col=0)

#%% Benchmark: nearest cell 

def benchmark_nearest_cell(df, x_col, y_col, target_col): 
    
    n_correct = 0   
    n_gauges = 0 
    
    ## get gauge ids     
    idx = pd.Series(df.index.values).str.split('_', expand=True).values 
    ## add as separate columns 
    df['gauge_id'] = idx[:,1] 
    
    ## add 'distance' column   
    df['distance'] = ((df[x_col]**2) + (df[y_col]**2))**0.5
    
    ## create y_hat column for storing results 
    target_col_hat = '{}_hat'.format(target_col)
    df[target_col_hat] = 0 
    
    # ## loop through all single gauges 
    for gauge_id in df['gauge_id'].unique():
        
        df_gauge = df[ df['gauge_id'] == gauge_id] 
        y_gauge = df_gauge[target_col] 
        
        ## check if a match is found in the buffer 
        if y_gauge.sum() > 0:
            n_gauges += 1 
            
            ## get ix of smallest distance column  
            y_hat = df_gauge['distance'].idxmin()
            
            ## set minimum value to 1 
            df.loc[y_hat, target_col_hat ] = 1 
            
            ## get accuracy    
            y = y_gauge.idxmax() 
                
            ## if correct, count as success 
            if y == y_hat:
                n_correct += 1

    print()
    print('-----'*10)        
    print('Total found: {} ({:.2f}%)'.format( n_correct, (n_correct/n_gauges)*100) )
    print('-----'*10)    
    
    return df 

#%% Benchmark: min(RMSE)

from sklearn.metrics import mean_squared_error 

def benchmark_rmse(df_timeseries, df_locations, df_labels, T0 = '1991-01-01', T1 = '2020-12-31'): 
    
    ## get gauge IDs             
    gauge_idx = df_locations['gauge_id'].unique() 

    ## prepare df_labels:
    ## extract gauge id      
    idx = pd.Series(df_labels.index.values).str.split('_', expand=True).values 
    ## add as separate column
    df_labels['gauge_id'] = idx[:,1] 
    
    ## create emtpy dataframe for output 
    df_benchmark = pd.DataFrame()
    
    ## set counters 
    n_gauges = 0
    n_correct = 0 
    
    for gauge_ix in gauge_idx:
        
        col_ix = df_locations[ df_locations['gauge_id'] == gauge_ix].index.values 
        
        sub_df = df_timeseries[col_ix] 
        gauge_col = [col for col in sub_df.columns if 'gauge' in col] 
                
        if len(gauge_col) > 0:
            n_gauges += 1 
            model_cols = [col for col in sub_df.columns if not 'gauge' in col ] 
                                    
            ### for each entry, calculate RMSE of
            ### observation timeseries and model timeseries 
            _df = pd.DataFrame() 
            
            for col in model_cols:
                calc_rmse = mean_squared_error( sub_df[gauge_col], sub_df[col], squared = False)
                
                _df.loc[col, 'rmse'] = calc_rmse
                _df.loc[col, 'gauge'] = str(gauge_ix)
            
            ## after calculating rmse - find smallest RMSE, flag as 1 
            _df['y_hat'] = 0 
            y_hat_ix = _df['rmse'].idxmin() 
            _df.loc[y_hat_ix, 'y_hat'] = 1 
            
            ## append true value (y) 
            sub_y = df_labels[ df_labels['gauge_id'] == str(gauge_ix)] 
            y = sub_y['target'] 

            ## check if labelled 
            if y.sum() > 0:
                _df['y'] = y                 
                y_flag = y.idxmax() 
                
                ## check prediction
                if y_flag == y_hat_ix:
                    n_correct += 1 
                            
            ## append to df_benchmark 
            df_benchmark = df_benchmark.append(_df)
    
    print()
    print('-----'*10)        
    print('Total found: {} ({:.2f}%)'.format( n_correct, (n_correct/n_gauges)*100) )
    print('-----'*10)      
    
    return df_benchmark 

#%% 

benchmark_nearest_cell(df_lf, 'x', 'y', 'target')
benchmark_rmse(df_ts, df_loc, df_lf, T1 = '1991-12-31')






























