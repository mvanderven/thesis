# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:37:31 2021

@author: mvand
"""

#%% load modules 

import pandas as pd  
from pathlib import Path 

from sklearn.metrics import classification_report #, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

#%% optional: subsample dataset to get balanced training/validation set 

target_col = 'target'
df_sampled = utils.subsample(df, target_col, n_frac = 1)

#%% split dataset into training and validation set 

X_train, X_val, y_train, y_val = train_test_split( df_sampled.drop([target_col], axis=1), 
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

utils.buffer_validation(df, target_col, lr_model, sc)


#%% Analyse coefficients 

lr_coefs = lr_model.coef_ 
coef_names = df.drop([target_col, 'gauge_id'], axis=1).columns

df_coef = pd.DataFrame(lr_coefs[0], index = coef_names, columns=['Coefficients'])

# print(df_coef)















