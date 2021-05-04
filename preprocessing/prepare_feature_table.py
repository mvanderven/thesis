# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:29:38 2021

@author: mvand
"""

## modules 
import pandas as pd 
from pathlib import Path 
import matplotlib.pyplot as plt 

#%% Paths 

data_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data_S1") 

fn_signatures       = data_dir / 'V1_S1_signatures_TRAIN_VAL.csv'
fn_gauge_metadata   = data_dir / 'V1_S1_grdc_efas_selection_TRAIN_VAL.csv'
fn_cell_metadata    = data_dir / 'V1_S1_metadata_TRAIN_VAL.csv'  

fn_features = data_dir / "V1_S1_features_TRAIN_VAL.csv"
fn_similarity_vector = data_dir / "V1_S1_similarity_TRAIN_VAL.csv"

#%% Prepare feature table 

def feature_table(fn_gauge_meta, fn_cell_meta, fn_signatures):
    
    ## load data 
    df_gauge_meta = pd.read_csv(fn_gauge_meta, index_col=0) 
    df_cell_meta  = pd.read_csv(fn_cell_meta, index_col=0) 
    df_signatures = pd.read_csv(fn_signatures, index_col=0) 
        
    # columns to get from cell_meta and gauge_meta 
    cols_of_interest = ["x", "y", "lat", "lon", "upArea"]
    
    ## rename gauge_meta columns for easy finding
    df_gauge_meta = df_gauge_meta.rename(columns = {"Lisflood_X": "x",
                                                    "Lisflood_Y": "y",
                                                    "gauge_lat": "lat",
                                                    "gauge_lon": "lon"})
    
    ## gauge upArea [km2] vs efas upArea [m2] --> convert 
    df_gauge_meta["upArea"] = df_gauge_meta["upArea"] * 10**6 
    
    df_features = pd.DataFrame()
    
    ## go over gauges - collect required data from all files 
    for gauge_id in df_signatures['gauge'].unique():
        
        ## filter data 
        signature_set = df_signatures[ df_signatures['gauge'] == gauge_id] 
        cell_metadata = df_cell_meta[ df_cell_meta['gauge'] == gauge_id] 
        gauge_metadata = df_gauge_meta.loc[gauge_id] 
        
        ## save all signatures in feature table 
        _df = signature_set.copy() 
        
        ## get files from cell_metadata 
        cell_data = cell_metadata[cols_of_interest]  
        _df = _df.join(cell_data)  
        
        gauge_data = gauge_metadata[cols_of_interest] 
        gauge_ix = 'gauge_{}'.format(gauge_id)
        _df.loc[ gauge_ix, cols_of_interest] = gauge_data 
        
        df_features = df_features.append(_df)
    
    return df_features

#%% Calculate similarity vector  (label?) 


def calc_similarity_vector(feature_table, id_col = 'gauge'):
    
    df_similarity = pd.DataFrame() 
    
    cols = feature_table.columns 
    calc_cols = [col for col in cols if not 'gauge' in col]
    
    for u_id in feature_table[id_col].unique():
        
        sub_df = feature_table[ feature_table[id_col] == u_id] 
       
        ## get gauge data 
        gauge_row = sub_df.loc['gauge_{}'.format(u_id), calc_cols]
        
        ## get dataframe of cells 
        cell_rows = [row for row in sub_df.index if not 'gauge' in row] 
        df_cells = sub_df.loc[cell_rows, calc_cols]
        
        ## similarity 
        similarity_vectors = df_cells.values - gauge_row.values 
        
        ## save as dataframe         
        _df = pd.DataFrame(data=similarity_vectors, index=cell_rows, columns=calc_cols)
        _df['gauge'] = u_id
        
        ## try to add target value 
        _df['target'] = 0 
        
        gauge_X, gauge_Y = gauge_row[['x', 'y']].values  
        target_row = df_cells[ (df_cells['x'] == gauge_X) & (df_cells['y'] == gauge_Y) ] 
        
        if len(target_row) > 0:
            _df.loc[target_row.index, 'target'] = 1
                     
        ## save outptu 
        df_similarity = df_similarity.append(_df)
        
    return df_similarity

#%% Run to create feature table 

# df_features = feature_table(fn_gauge_metadata, fn_cell_metadata, fn_signatures)
# df_features.to_csv(fn_features)

#%% Load feature data 

df_features = pd.read_csv(fn_features, index_col=0)

#%% Run to calculate similarity scores 

# df_similarity = calc_similarity_vector(df_features)
# df_similarity.to_csv(fn_similarity_vector)

#%% Load similarity data  

df_similarity = pd.read_csv(fn_similarity_vector, index_col=0)

#%% Analyse similarity data 

import numpy as np
from sklearn.preprocessing import MinMaxScaler

plot_cols = ['Nm-all', 'Ns-all', 'N-gof-all', 'Lm-all', 'Ls-all', 'L-gof-all',
       'Gu-all', 'Ga-all', 'Gev-gof-all', 'Gk-all', 'Gt-all', 'G-gof-all',
       'alag-1-all', 'clag-0-all', 'clag-1-all', 'fdcQ-1-all', 'fdcQ-2-all',
       'fdcQ-5-all', 'fdcQ-10-all', 'fdcQ-50-all', 'fdcQ-90-all',
       'fdcQ-95-all', 'fdcQ-99-all', 'fdcS-all', 'lf-all', 'bfi-all',
       'dld-all', 'rld-all', 'rbf-all', 'b_rc-all', 'a_rc-all', 'upArea']

## based on plots - log transform: 
# skewed_cols = ['clag-0-all', 'clag-1-all', 'fdcQ-1-all'] 
skewed_cols=[]


plt.figure(figsize=(14,10)) 
n_rows = 4 
n_cols = 8 


for i, col in enumerate(plot_cols):
    
    ## scale data 
    data = df_similarity[col].values 
    vmin, vmax = np.nanmin(data), np.nanmax(data) 
        
    if (vmax-vmin)!=0.:
        data = (data-vmin) / (vmax - vmin) 
    
    if col in skewed_cols:
        data = np.log(data+10e-6) 

        
    plt.subplot(n_rows, n_cols, int(i+1))
    plt.title(col) 
    plt.hist(data, bins=50)

plt.show()





























