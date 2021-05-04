# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:31:56 2021

@author: mvand
"""

#%% Import modules 

import pandas as pd 
from pathlib import Path 

#%% Set paths 

src_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\dev") 
trg_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data")
fn = src_dir / "efas_timeseries_1991_buffer_4.csv" 

#%% Get cell metadata 

def extract_metadata(fn, unique_id = 'ID',
                     cols_of_interest = ['ID', 'lat', 'lon', 'upArea',
                                         'gauge', 'x', 'y'],
                     out_dir = None, save = False):
    
    ## load data 
    df = pd.read_csv(fn)
    
    ## get columns of interest 
    df_select = df[cols_of_interest]  
    
    ## drop duplicates 
    out_df = df_select.drop_duplicates(ignore_index=True) 
    
    ## save output 
    if save:
        if out_dir == None:
            out_df.to_csv('cell_metadata.csv', index=False) 
        else:
            out_df.to_csv(out_dir / 'V1_cell_metadata.csv', index=False)
    
    ## return output 
    return out_df 

# df_meta = extract_metadata(fn, out_dir = trg_dir, save=True)  

#%% Split metadata
 

fn_metadata = trg_dir / "V1_cell_metadata.csv"
df_meta = pd.read_csv(fn_metadata) 

train_val_fn = trg_dir / 'V1_grdc_efas_selection_TRAIN_VAL.csv'
df_train_val = pd.read_csv(train_val_fn) 

test_fn = trg_dir / "V1_grdc_efas_selection_TEST.csv"
df_test = pd.read_csv(test_fn) 

def split_metadata(df_meta, df, id_meta = 'gauge', id_col = 'gauge_ID'):
    unique_id = df[id_col].unique() 
    return df_meta[ df_meta[id_meta].isin(unique_id)] 

df_meta_train_val = split_metadata(df_meta, df_train_val)
# df_meta_train_val.to_csv( trg_dir / 'V1_metadata_TRAIN_VAL.csv', index=False )


df_meta_test = split_metadata(df_meta, df_test)
# df_meta_test.to_csv( trg_dir / 'V1_metadata_TEST.csv', index=False )
























