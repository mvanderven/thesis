# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:37:31 2021

@author: mvand
"""

#%% load modules 

import pandas as pd  
from pathlib import Path 
import numpy as np 

from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

import ml_utils as utils 

#%% paths 

training_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data")

training_set = training_dir / "similarity_vector_labelled_buffer_2-20210311.csv"

#%% load dataset 

df = pd.read_csv(training_set, index_col = 0) 

print('-----'*10)
print('Load dataset')
print('-----'*10)
print(df.head())


#%% Divide dataset based on number of gauges 
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

print(df.head())

## update unique gauges 
unique_gauges = df['gauge_id'].unique()

#%% Determine gauges to be used in training and testing, or for validation 

## further split train test into validation set 
gauge_train_test, gauge_validation = train_test_split( unique_gauges, test_size = 0.15)

df_train_test = df[ df['gauge_id'].isin( gauge_train_test)]
df_validate = df[ df['gauge_id'].isin(gauge_validation)] #.drop('gauge_id', axis=1)

#%% subsample train+test dataset to get balanced dataset 

target_col = 'target'
df_sampled = utils.subsample(df_train_test, target_col, n_frac = 1)

#%% split dataset into training, test and validation set 

X_train, X_val, y_train, y_val = train_test_split( df_sampled.drop([ 'gauge_id', target_col], axis=1), 
                                                    df_sampled[target_col],
                                                    test_size = 0.2)


#%% scale/normalize training set & apply to validation set 

sc = MinMaxScaler([0,1]) 

## train scaler 
X_train = sc.fit_transform(X_train) 

## apply on validation data 
X_val = sc.transform(X_val)

#%% train model 
print()
print('-----'*10)
print('Train logistic regressor')
print('-----'*10)

lr_model = utils.train_logistic_regressor(X_train, y_train)

## training performance 
y_hat_train = lr_model.predict(X_train)

## validation performance 
y_hat_val = lr_model.predict(X_val)

#%% Analyse performance 

print()
print('-----'*10)
print('Validation classification report')
print('-----'*10)
print(classification_report(y_val, y_hat_val)) 

#%% Plot confusion matrix 

utils.plot_confusion_matrix(y_train, y_hat_train, name = 'Training score' )
utils.plot_confusion_matrix(y_val, y_hat_val, name = 'Test score' )

#%% Validate results in buffer format 

utils.buffer_validation(df_validate, target_col, lr_model, sc)

#%% Compare with benchmark results 

## Nearest cell method outperforms LogisticRegression model 
utils.benchmark_nearest_cell(df_validate, 'x', 'y', target_col)

#%% Analyse coefficients 

lr_coefs = lr_model.coef_ 
coef_names = df.drop([target_col, 'gauge_id'], axis=1).columns

df_coef = pd.DataFrame(lr_coefs[0], index = coef_names, columns=['Coefficients'])

# print(df_coef)















