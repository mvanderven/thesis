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

from cartesius_utils import load_efas, buffer_search, resample_efas

#%% Set file paths 

fn = 'V1_grdc_efas_selection-cartesius.csv'

# efas_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data\EFAS_6h") 
efas_dir = Path(r"F:\thesis_data\EFAS_6h")

#%% Define function 

def run_parallel(fn, efas_dir, n_nodes = 4, buffer_size = 2):
    
    ## load data 
    df = pd.read_csv(fn, index_col = 0) 
    ds = load_efas(efas_dir) 
    
    ## set function aprameters 
    test_ix = df.index.values
    df_list = [df]*len(test_ix) 
    ds_list = [ds]*len(test_ix) 
    buffer_list = [buffer_size]*len(test_ix)
    
    print('Start parallel processing')
    time_parallel = time.time() 
    
    ## create pool
    p = pa.pools.ParallelPool(nodes=n_nodes) 
    
    ## map 
    results_parallel = p.map(buffer_search, test_ix, df_list, ds_list, buffer_list) 
    
    print('Processing finished in {:.2f} minutes\n'.format( (time.time() - time_parallel)/60. )) 
    
    ## collect data
    ## reorder results_parallel
    df_collect = pd.DataFrame() 
    
    for df_result in results_parallel:
        df_collect = df_collect.append(df_result) 
    
    return df_collect 

#%% Run 

if __name__ == '__main__':
    
    ## START
    ## copy TAR file from home dir to scratch 
 
    ## extract TAR file?
    print('[INFO] Start resampling')
    ds = resample_efas(efas_dir, dir_out = Path(r"F:\thesis_data\EFAS_24h"))
    
    ## run in parallel     
    ## process per year?
    # df = run_parallel(fn, efas_dir)
    
    
    ## save output 
    # print('\nSave output\n')
    # df.to_csv(r"F:\thesis_data\efas_timeseries_1991-1992_chunk_1000.csv")
    
    ## remove files from scratch / move back to home dir

    
    ## FINISH
    
    # print(df)
    # print('\n########\n') 
    # print(df.info())
    















