# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:52:31 2021

@author: mvand
"""

#%% Import modules 
import pandas as pd  
from pathlib import Path 
import time 
import pathos as pa 

from cartesius_utils import load_efas, buffer_search

#%% Set file paths 

fn = 'V1_grdc_efas_selection-cartesius.csv'
efas_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data\EFAS_6h")
# efas_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data\EFAS_6h_test")


#%% Define function 

def run_parallel(fn, efas_dir):
    
    ## load data 
    df = pd.read_csv(fn, index_col = 0) 
    ds = load_efas(efas_dir) 
    
    ## set function aprameters 
    test_ix = df.index.values
    df_list = [df]*len(test_ix) 
    ds_list = [ds]*len(test_ix) 
    
    print('Start parallel processing')
    time_parallel = time.time() 
    
    ## create pool
    p = pa.pools.ParallelPool(nodes=4) 
    
    ## map 
    results_parallel = p.map(buffer_search, test_ix, df_list, ds_list) 
    
    print('{:.2f} minutes\n'.format( (time.time() - time_parallel)/60. )) 
    
    ## collect data
    ## reorder results_parallel
    df_collect = pd.DataFrame() 
    for df_result in results_parallel:
        df_collect = df_collect.append(df_result) 
    return df_collect 

#%% Run 

if __name__ == '__main__':
    
    df = run_parallel(fn, efas_dir)
    
    print(df)
    print('\n########\n') 
    print(df.info())
    















