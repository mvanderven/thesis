# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:37:31 2021

@author: mvand
"""

#%% load modules 

import pandas as pd 
# import numpy as np 
from pathlib import Path 
import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import 

#%% paths 

training_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\training_data")

training_set = training_dir / "similarity_vector_labelled_buffer_2-20210311.csv"

subsample = True

#%% load dataset 

df = pd.read_csv(training_set, index_col = 0) 

print('-----'*10)
print('Load dataset')
print('-----'*10)
print(df.head())

#%% optional: subsample dataset to get balanced training/validation set 

target_col = 'target'

if subsample is True:
    print('Subsample dataset') 
    df_1 = df[ df[target_col] == 1] 
    df_0 = df[ df[target_col] != 1] 
    
    ## get length of values = 1 
    n1 = len(df_1)
    
    ## combine all 1 values with an equal number of 0's 
    df = df_1.append( df_0.sample(n=n1)  )
   
#%% split dataset into training and validation set 

X_train, X_val, y_train, y_val = train_test_split( df.drop([target_col], axis=1),
                                                   df[target_col],
                                                   test_size = 0.2)

#%% scale/normalize training set & apply to validation set 

sc = MinMaxScaler([0,1]) 

## train scaler 
X_train = sc.fit_transform(X_train) 

## apply on validation data 
X_val = sc.transform(X_val)

#%% train model 

## model setup 
lr = LogisticRegression()

## train model 
lr.fit(X_train, y_train) 

## training performance 
y_hat_train = lr.predict(X_train)

## validation performance 
y_hat_val = lr.predict(X_val)

#%% analyse performance 

print()
print('-----'*10)
print('Validation classification report')
print('-----'*10)

print(classification_report(y_val, y_hat_val)) 

#%% 

fig = plt.figure(figsize=(12,8)) 


val_acc = accuracy_score(y_val, y_hat_val)
val_prec = precision_score(y_val, y_hat_val)
val_rec = recall_score(y_val, y_hat_val) 
val_f1 = f1_score(y_val, y_hat_val)

train_acc = accuracy_score(y_train, y_hat_train)
train_prec = precision_score(y_train, y_hat_train)
train_rec = recall_score(y_train, y_hat_train) 
train_f1 = f1_score(y_train, y_hat_train)


plt.subplot(211)
sns.heatmap( pd.DataFrame(confusion_matrix(y_train, y_hat_train)),
                    annot=True, cmap='binary')
plt.ylabel('True class');
plt.xlabel('Predicted class');
plt.title('Training \n accuracy: {:.2f} precision: {:.2f} recall:  {:.2f} f1: {:.2f}'.format(train_acc,
                                                                                                    train_prec,
                                                                                                    train_rec,
                                                                                                    train_f1));

plt.subplot(212)
sns.heatmap(  pd.DataFrame(confusion_matrix(y_val, y_hat_val)),
            annot=True, cmap='binary')
plt.ylabel('True class');
plt.xlabel('Predicted class');
plt.title('Validation \n accuracy: {:.2f} precision: {:.2f} recall:  {:.2f} f1: {:.2f}'.format(val_acc,
                                                                                                    val_prec,
                                                                                                    val_rec,
                                                                                                    val_f1));

plt.tight_layout() 
plt.show()




