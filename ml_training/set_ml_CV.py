# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:28:17 2021

@author: mvand
""" 

#%% load modules 

import pandas as pd  
from pathlib import Path 
import numpy as np 

# import matplotlib.pyplot as plt 

# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import MinMaxScaler

import ml_utils as utils 

#%% paths 

home_dir = Path.home()

## set data paths 
model_dir = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\model_data")
training_dir = home_dir / Path(r"Documents\Master EE\Year 4\Thesis\data\training_data")

training_set = training_dir / "similarity_vector_labelled_buffer_2-20210311.csv"

#%% load dataset 

df = pd.read_csv(training_set, index_col = 0) 

print('-----'*10)
print('Load dataset')
print('-----'*10)
print(df.head())


#%% Test: drop features
df.drop(columns=['lat', 'lon'], inplace=True) 

#%% Prepar to divide dataset based on number of gauges 
## 70% for training
## 15% for testing
## 15% for validation 

## To this end, first get unique gauge     
idx = pd.Series(df.index.values).str.split('_', expand=True).values 
## add as separate columns 
df['gauge_id'] = idx[:,1]

unique_gauges = np.unique(idx) 

## check if all buffers contain one positive sample  
for gauge in unique_gauges:
    
    check_df = df[df['gauge_id']==gauge]

    ## if sum is smaller than one, wronglabels 
    if check_df['target'].sum() < 0.5:        
        df.drop(index=check_df.index, inplace=True)


## update unique gauges 
unique_gauges = df['gauge_id'].unique()


#%% Get k_fold classification results 

print('-----'*10)
print('K fold Cross Validation')
print('-----'*10)

df_results = utils.k_fold_CV(df, k=10) 
df_prob = df_results[ df_results['cat'] == 'prob'] 
print(df_prob.describe())

#%% Leave one out CV 
print()
print('-----'*10)
print('Leave One Out Cross Validation')
print('-----'*10)

df_LOOCV = utils.k_fold_CV(df, k=len(unique_gauges)) 
df_prob = df_LOOCV[ df_LOOCV['cat'] == 'prob'] 
print(df_prob.describe())










