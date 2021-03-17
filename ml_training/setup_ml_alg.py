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
print('-----'*10)

#%% Plot confusion matrix 

# utils.plot_confusion_matrix(y_train, y_hat_train, name = 'Training score' )
# utils.plot_confusion_matrix(y_val, y_hat_val, name = 'Test score' )

#%% Validate results in buffer format 

df_val, val_true, val_guess, val_false = utils.buffer_validation(df_validate, target_col, lr_model, sc)

#%% Benchmark Nearest Cell 

## Nearest cell method outperforms LogisticRegression model 
df_nc, nc_true, nc_false = utils.benchmark_nearest_cell(df_validate, 'x', 'y', target_col)

#%% Benchmark RMSE 

## load data for min(RMSE)
timeseries_fn = model_dir / "collect_ts_obs_mod_20210315_2.csv"
locations_fn =  model_dir / "collect_loc_obs_mod_20210315_2.csv"
df_ts = pd.read_csv(timeseries_fn, index_col=0) 
df_loc = pd.read_csv(locations_fn, index_col=0)

#%% 
## Algorithm and nearest cell benchmark outperfrom RMSE
df_rmse, rmse_true, rmse_false = utils.benchmark_skill_score(df_ts, df_loc, df_validate) #, T1 = '1991-12-31')

#%% 
## NSE
df_nse, nse_true, nse_false = utils.benchmark_skill_score(df_ts, df_loc, df_validate, method='nse') #, T1 = '1991-12-31')

#%% Show confusion plots for validation

# utils.plot_confusion_matrix(df_val['target'], df_val['y_hat'], name = 'Validation score' )
# utils.plot_confusion_matrix(df_val['target'], df_val['y_hat_prob'], name = 'Validation proba score' )
# utils.plot_confusion_matrix(df_nc['target'], df_nc['target_hat'], name = 'Benchmark nearest cell score' )
# utils.plot_confusion_matrix(df_rmse['y'], df_rmse['y_hat'], name = 'Benchmark RMSE score' )

#%% Analyse coefficients 

lr_coefs = lr_model.coef_ 
coef_names = df.drop([target_col, 'gauge_id'], axis=1).columns

df_coef = pd.DataFrame(lr_coefs[0], index = coef_names, columns=['Coefficients'])
# print()
# print('-----'*10)
# print('Logistic regression coefficients')
# print('-----'*10)
# print(df_coef)

#%% Show validation results in a grid view 

# utils.grid_viewer(df_val, 'target', 'y_hat', gauge_ids = val_true, plot_title = 'Validation true')
# utils.grid_viewer(df_val, 'target', 'y_hat', gauge_ids = val_guess, plot_title = 'Validation guess')
# utils.grid_viewer(df_val, 'target', 'y_hat', gauge_ids = val_false, plot_title = 'Validation false')

# utils.grid_viewer(df_nc, 'target', 'target_hat', gauge_ids = nc_true, plot_title = 'Benchmark nearest cell true')
# utils.grid_viewer(df_nc, 'target', 'target_hat', gauge_ids = nc_false, plot_title = 'Benchmark nearest cell false')

# utils.grid_viewer(df_rmse, 'y', 'y_hat', gauge_ids = rmse_true, plot_title = 'Benchmark RMSE true')
# utils.grid_viewer(df_rmse, 'y', 'y_hat', gauge_ids = rmse_false, plot_title = 'Benchmark RMSE false')

# utils.grid_viewer(df_nse, 'y', 'y_hat', gauge_ids = nse_true, plot_title = 'Benchmark NSE true')
# utils.grid_viewer(df_nse, 'y', 'y_hat', gauge_ids = nse_false, plot_title = 'Benchmark NSE false')

#%% Show validation results in grid with varying backgrounds 

# utils.grid_view_param(df_val, 'target', 'y_hat', param_col = 'Nm-all', 
#                       gauge_ids = val_true, buffer_size=2, plot_title= 'Validation true',
#                       cmap='Blues')

# utils.grid_view_param(df_val, 'target', 'y_hat', param_col = 'Nm-all', 
#                       gauge_ids = val_guess, buffer_size=2, plot_title= 'Validation guess',
#                       cmap='Blues')

# utils.grid_view_param(df_val, 'target', 'y_hat', param_col = 'Nm-all', 
#                       gauge_ids = val_false, buffer_size=2, plot_title= 'Validation false',
#                       cmap='Blues')

#%% Show NC benchmark results  with varying backgrounds 

utils.grid_view_param(df_nc, 'target', 'target_hat', param_col = 'Nm-all', 
                      gauge_ids = nc_true, buffer_size=2, plot_title= 'Benchmark NC true',
                      cmap='Blues')

utils.grid_view_param(df_nc, 'target', 'target_hat', param_col = 'Nm-all', 
                      gauge_ids = nc_false, buffer_size=2, plot_title= 'Benchmark NC false',
                      cmap='Blues')



#%% Show NSE benchmark results  with varying backgrounds 

utils.grid_view_param(df_nse, 'y', 'y_hat', param_col = 'nse', 
                      gauge_ids = nse_true, buffer_size=2, plot_title= 'Benchmark nse true',
                      cmap='Blues')

utils.grid_view_param(df_nse, 'y', 'y_hat', param_col = 'nse', 
                      gauge_ids = nse_false, buffer_size=2, plot_title= 'Benchmark nse false',
                      cmap='Blues')


#%% Show RMSE benchmark results  with varying backgrounds 

utils.grid_view_param(df_rmse, 'y', 'y_hat', param_col = 'rmse', 
                      gauge_ids = rmse_true, buffer_size=2, plot_title= 'Benchmark rmse true',
                      cmap='Blues')

utils.grid_view_param(df_rmse, 'y', 'y_hat', param_col = 'rmse', 
                      gauge_ids = rmse_false, buffer_size=2, plot_title= 'Benchmark rmse false',
                      cmap='Blues')










