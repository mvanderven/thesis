# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:00:48 2021

@author: mvand
"""

import pandas as pd 
from pathlib import Path

from signatures_utils import pa_calc_signatures
import pathos as pa 

import time 

#%% Define pathos functions 

def run_parallel(n_nodes=2):
    
    year_start = 1991 
    # for testing
    year_end = 1996 
    
    # year_end = 2021
    #### 1 - PATHS 
    ## cartesius environment 
    ... 
    
    ## laptop test 
    input_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\dev_input") 
    
    fn_gauges = input_dir / "V1_grdc_efas_selection-cartesius-snapped.csv" 
    dir_gauges = input_dir / 'V1'
    
    #### 2 - SET MODEL DATA  
    files = [file for file in input_dir.glob('efas_timeseries_*_buffer_4.csv')] 
        
    #### 3 - LOAD GAUGE METADATA
    df_gauges = pd.read_csv(fn_gauges, index_col=0) 
    
    ## extract list of gauges 
    # gauge_ids = df_gauges.index.values  
    ## TEST 
    gauge_ids = df_gauges.sample(n=2).index.values 
    
    #### 4 - SETUP PATHOS POOL
    p = pa.pools.ParallelPool(nodes=n_nodes) 
    
    #### 5 - SET RUN PARAMETERS
    fn_simulations = [files] * len(gauge_ids)
    gauge_dirs = [dir_gauges] * len(gauge_ids)

    #### 6 - RUN POOL
    print('[START] parellel run')
    time_parallel = time.time()
    results_parallel = p.map(pa_calc_signatures, gauge_ids, fn_simulations, gauge_dirs) 
    print('[END] parellel run - finished in {:.2f} minutes'.format( (time.time()-time_parallel)/60.  ))
    #### 7 - COLLECT RESULTS 
    return  pd.concat(results_parallel)



#%% Run if main 

if __name__ == '__main__':
    
    time_total = time.time() 
    
    results = run_parallel() 
    # print(results)
    # print(results.info())

    output_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\dev_input") 
    fn_out = output_dir / 'test.csv'
    results.to_csv(fn_out)
    
    print('Total time: {:.2f} minutes'.format( (time.time() - time_total)/60. ))

















