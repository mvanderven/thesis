# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:00:48 2021

@author: mvand
"""

import pandas as pd 
from pathlib import Path

import signatures_utils as s_utils

import dask.dataframe as dd 
import pathos as pa 

import time 

#%% Define pathos functions 

def run_parallel():
    
    #### 1 - PATHS 
    ## cartesius environment 
    ... 
    
    ## laptop test 
    dir_simulations = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")
    
    dir_gauges = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data") 
    fn_gauges = dir_gauges / "V1_grdc_efas_selection-cartesius-snapped.csv" 
    
    
    #### 2 - LOAD MODEL DATA  
    print('[INFO] load model data')
    
    files = [file for file in dir_simulations.glob('efas_timeseries_*_buffer_4.csv')] 
    
    time_dask = time.time()
    df_model = dd.read_csv(files).compute()
    print('Dask: {:.2f} minutes'.format( (time.time() - time_dask)/60. ))
    
    # time_pd = time.time()
    # df_model = pd.DataFrame() 
    # for file in files:
    #     print('[INFO] load ',file.name)
    #     _df = pd.read_csv(file, index_col = 0) 
    #     df_model = df_model.append(_df)
    # print('pd: {:.2f} minutes'.format( (time.time() - time_pd)/60. )) 
    
    
    ## third try 
    time_test = time.time() 
    df_model = pd.concat(map(pd.read_csv, files))
    print('concat: {:.2f} minutes'.format( (time.time() - time_test)/60. )) 
    
    print('[INFO] model data loaded')
    # print(df_model)
    # print()
    # print(df_model.info())    
    
    #### 3 - LOAD GAUGE METADATA
    df_gauges = pd.read_csv(fn_gauges, index_col=0) 
    ## extract list of gauges 
    gauge_ids = df_gauges.index.values  
    
    return 






#%% Run if main 

if __name__ == '__main__':
    run_parallel()
    

















