# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:47:19 2021

@author: mvand
"""


#%% Import modules 

import pandas as pd 
from pathlib import Path 
from sklearn.model_selection import train_test_split
import numpy as np 

#%% Paths 

training_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data") 

fn_labels = training_dir / "V1_grdc_efas_selection-cartesius-snapped.csv" 

signature_dir = training_dir / "signatures_V1_output"

#%% Split func 

def split_test_set(fn, out_name, test_size = 0.15, out_dir = None):
    
    ## read file 
    df = pd.read_csv(fn, index_col = 0) 
    
    ## split
    df_train_val, df_test = train_test_split(df, test_size = test_size)

    ## filenames 
    if out_dir == None:
        out_dir = fn.parent 
    
    fn_out_train_val = out_dir / '{}_TRAIN_VAL.csv'.format(out_name)
    fn_out_test = out_dir / '{}_TEST.csv'.format(out_name) 
    
    ## save files 
    df_train_val.to_csv(fn_out_train_val)
    df_test.to_csv(fn_out_test)

    return fn_out_train_val, fn_out_test

#%% Move feature files to directories 

def create_feature_files(fn, dir_name, src_dir): 
    
    ## create target directory 
    stem = fn.parent 
    target_dir = stem / dir_name 
    if not target_dir.exists():
        target_dir.mkdir() 
    
    assert target_dir.exists(), '[ERROR] target directory does not exist'
    
    ## copy files from source directory 
    assert src_dir.exists(), '[ERROR] source directory does not exist - cannot copy files' 
    
    df = pd.read_csv(fn, index_col = 0)  
    idx = df.index.values  
    
    src_fns = ['signatures_{}.csv'.format(ix) for ix in idx] 
    
    fns_failed = []
    for src_fn in src_fns:
        source = src_dir / src_fn 
        target = target_dir / src_fn 
        
        if source.exists():
            target.write_bytes(source.read_bytes()) 
        else:
            fns_failed.append(source)
    
    print('[INFO] copying failed for {} file(s) - source file does not exist'.format(len(fns_failed)))
      
    return target_dir, fns_failed





#%% Combine all feature files into one 

def concatenate_files(file_list, output_name): 
    
    df_concat = pd.concat(map(pd.read_csv, file_list)) 
    df_concat = df_concat.set_index('ID')
        
    gauge_idx = pd.Series(df_concat.index.values).str.split('_', expand=True).values[:,1]
    
    df_concat['gauge'] = gauge_idx 
   
    df_concat.to_csv(output_name)
    
    if output_name.exists():
        return df_concat  
    else:
        return -1 


#%% RUN STEPS 

## split based on gauge ID 
# train_val, test = split_test_set(fn_labels, 'V1_grdc_efas_selection')

## move training/validation files to new directory
# train_val_dir, fn_failed = create_feature_files(train_val, 'V1_TRAIN_VAL',
                                     # signature_dir)

## move test files to new directory
# test_dir, test_failed = create_feature_files(test, 'V1_TEST', signature_dir)

## concatenate all loose files 
# signature_files = [file for file in signature_dir.glob('signatures_loop_*.csv') ]
# df_concact = concatenate_files(signature_files, signature_dir/'signatures_all.csv') 

# train_val_files = [file for file in train_val_dir.glob('signatures_*.csv')] 
# df_train_val = concatenate_files(train_val_files, train_val_dir/'signatures_TRAIN_VAL.csv')

# test_files = [file for file in test_dir.glob('signatures_*.csv')]  
# df_test = concatenate_files(test_files, test_dir / 'signatures_TEST.csv')









