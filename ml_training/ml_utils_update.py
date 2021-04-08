# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:36:35 2021

@author: mvand
"""

#%% Load moduls

import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import cartopy as cp 
import cartopy.feature as cfeature


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

import random 
# import pyproj 

#%% Define functions 

def subsample(df, target_col, target_value = 1., n_frac = 1.):
    
    df_1 = df[ df[target_col] == target_value] 
    df_0 = df[ df[target_col] != target_value] 
        
    n1 = min( 1, int(len(df_1)*n_frac ), len(df_0) ) 
    
    ## combine all 1 values with an equal number of 0's 
    return df_1.append( df_0.sample(n=n1)  )


def LR_buffer_prediction(model, scaler, df, id_col, y_col, y_hat_col, method = 1, na_value = -99):
    
    n_correct = 0
    n_missed  = 0 
    n_absent  = 0
    
    df.loc[df.index, y_hat_col] = na_value 
    drop_cols = [id_col, y_col, y_hat_col]
    
    if method == 2:
        df.loc[df.index, 'p0'] = na_value 
        df.loc[df.index, 'p1'] = na_value
        drop_cols = [id_col, y_col, y_hat_col, 'p0', 'p1']
    
    for gauge in df[id_col].unique():
        
        ## get buffer 
        df_gauge = df[ df[id_col] == gauge] 
        
        ## prepare features 
        X = df_gauge.drop( columns = drop_cols)
        X = scaler.transform(X) 
        
        ## create empty dataframe for gauge buffer
        _df = pd.DataFrame() 
        _df['ID'] = df_gauge.index 
        _df = _df.set_index('ID')
        _df[y_col] = df_gauge[y_col] 
        
        ## predict with trained algorithm
        ## according to selected method         
        if method == 1:
            ## predict y_hat 
            y_hat = model.predict(X) 
        
        if method == 2:
            y_prob = model.predict_proba(X) 
            p0, p1 = y_prob[:,0], y_prob[:,1]
            
            y_hat = np.zeros(len(_df)) 

            p1_max = np.argmax(p1) 
            ## confidence score
            if p1[p1_max] > p0[p1_max]:
                y_hat[ np.argmax(p1) ] = 1 
            
            _df['p0'] = p0
            _df['p1'] = p1 
            
        ## add y_hat to dataframe 
        _df[y_hat_col] = y_hat 
        
        ## check prediction
        ## and note score 
        y_ix = np.where( _df[y_col] == 1.)[0] 
        y_hat_ix = np.where( y_hat == 1.)[0] 
        
        if len(y_ix) > 0 and len(y_hat_ix) > 0:
        
            if y_ix[0] in y_hat_ix:
                n_correct += 1 
            else:
                n_missed += 1
                
        if len(y_ix) == 0:
            n_absent += 1 
            if len(y_hat_ix) == 0:
                n_correct += 1 
            else:
                n_missed += 1 
        
        for col in _df.columns:
            df.loc[ _df.index, col ] = _df[col]
     
    return df, n_correct, n_missed, n_absent

def benchmark_buffer_prediction(df, df_ts, id_col, y_col, method='rmse', dx = 'x', dy = 'y'): 

    n_correct = 0
    n_missed  = 0 
    n_absent  = 0    
    
    hat_col = '{}_hat'.format(y_col) 
    method_col = method.lower() 
    
    ## get timesieres column names 
    ts_cols = df_ts.columns 
    
    for gauge in df[id_col].unique():
        
        df_gauge = df[ df[id_col] == gauge] 
        
        gauge_ts_col = 'gauge_{}'.format(gauge) 
        
        ## create empty dataframe for results 
        _df = pd.DataFrame()
        
        _df['ID'] = df_gauge.index 
        _df = _df.set_index('ID') 
        _df[y_col] = df_gauge[y_col] 
        _df[hat_col] = 0

        
        if method.lower() in ['rmse', 'nse']:
                    
            if gauge_ts_col in ts_cols:
                ## get gauge time series 
                obs_ts = df_ts[gauge_ts_col].values 
                obs_avg = np.mean(obs_ts) 
                
                for sim_id in df_gauge.index:
                    sim_ts = df_ts[sim_id].values 
                                                        
                    if method.lower() == 'rmse':
                        _df.loc[sim_id, method_col] = mean_squared_error( obs_ts, sim_ts, squared=False )  
                        
                    if method.lower() == 'nse':
                        _df.loc[sim_id, method_col] = 1 - ((sim_ts - obs_ts)**2).sum() / ( ((obs_ts-obs_avg)**2).sum() )  
        
        if method.lower() in ['nc']:
            ## calculate distance
            _df[method_col] = (df_gauge[dx]**2 + df_gauge[dy]**2)**0.5 
        
        if method_col in _df.columns:       
            ## classify cell 
            if method.lower() in ['rmse', 'nc']:
                y_hat_idx = _df[method_col].idxmin()
            if method.lower() == 'nse': 
                y_hat_idx = _df[method_col].idxmax()
            
        _df.loc[y_hat_idx, hat_col] = 1 
        
        ## check prediction 
        y_idx = _df[y_col].idxmax() 
        y_ix = np.where( _df[y_col] == 1)[0] 
        
        if len(y_ix) > 0:
            if y_idx == y_hat_idx:
                n_correct += 1 
            else:
                n_missed += 1
        else:
            n_absent += 1
            n_missed += 1 
                
        for col in [hat_col, method_col]:
            if col in _df.columns:
                df.loc[_df.index, col] = _df[col]
    
    return df, n_correct, n_missed, n_absent


def Kfold_CV(df_labels, id_col, y_col, k = 5, methods = ['LogisticRegressor'], df_ts = None):
    
    ## check method
    ml_alg_options = ['LogisticRegressor-1', 'LogisticRegressor-2'] 
    benchmark_options = ['RMSE', 'NSE', 'NC']
    method_options = ml_alg_options+benchmark_options 
    
    ## check number K
    assert k <= df_labels[id_col].nunique(), '[ERROR] k={} too large for n={}'.format( k, df_labels[id_col].nunique() )
    
    print('-----'*10)
    print('{} fold Cross Validation'.format(k))
    print('-----'*10)
    
    ## get list with gauge IDs
    gauge_ids = df_labels[id_col].unique() 
    random.shuffle(gauge_ids) 
    
    ## divide in K groups 
    split_gauges = np.array_split(gauge_ids, k) 
    
    ## collect performance 
    df_results = pd.DataFrame() 
    
    for method in methods:
        
        print('Method:\t {}'.format(method))
        
        if method.lower() in [option.lower() for option in method_options]:
    
            ## split CV approach based on method 
            if method.lower() in [option.lower() for option in ml_alg_options]:
                
                ## loop through split_gauges - each time holding a different set of gauges out 
                for i in range(k):
            
                    ## holdout a sublist for validation
                    ## from different groups 
                    gauges_test = split_gauges[i] 
                    
                    if i == 0:
                        test = split_gauges[1:]
                    elif i == int(k-1):
                        test = split_gauges[:-1] 
                    else:
                        test = split_gauges[:i] + split_gauges[i+1:]
                    
                    gauges_train_val = [item for sublist in test for item in sublist] 
                    
                    ## split gauges_train_val into train and validation set 
                    gauges_train, gauges_val = train_test_split(gauges_train_val, test_size= 1./(k-1) ) 
                    
                    ## extract dataframes 
                    df_train = df_labels[ df_labels[id_col].isin(gauges_train) ].copy() 
                    df_val   = df_labels[ df_labels[id_col].isin(gauges_val) ].copy()
                    df_test  = df_labels[ df_labels[id_col].isin(gauges_test) ].copy()
                    
                    ## subsample df_train  
                    for gauge in gauges_train:
                        sub_df = df_train[ df_train[id_col] == gauge] 
                        sub_ix = sub_df.index 
                        
                        if sub_df[y_col].nunique() > 1:
          
                            ix_1 = sub_df[ sub_df[y_col] == 1].index  
                            ix_0 = sub_df[ sub_df[y_col] != 1].sample(n=1).index 
                            
                            ix_to_keep = [ix_1, ix_0] 
                        
                        else:
                            
                            ix_to_keep = sub_df[ sub_df[y_col] != 1].sample(n=2).index 
                        
                        drop_ix = [ ix for ix in sub_ix if ix not in ix_to_keep] 
                        df_train = df_train.drop(index=drop_ix) 
                    
                    ## split X-y sets 
                    X_train = df_train.drop(columns = [id_col, y_col])
                    y_train = df_train[y_col]
                                            
                    ## check data distribution
        
                    ## scale values in df_train 
                    sc = MinMaxScaler([0,1]) 
                    ## train scaler 
                    X_train = sc.fit_transform(X_train) 
                    
                    ## train model
                    lr = LogisticRegression() 
                    lr.fit(X_train, y_train)
                    
                    ## predict 
                    hat_col = '{}_hat'.format(y_col)
                    
                    ## y_hat_training
                    y_hat_train = lr.predict(X_train) 
                    df_train.loc[df_train.index, hat_col ] = y_hat_train
                    
                    ## asses performance 
                    if method.lower() == 'logisticregressor-1':
                                            
                        ## BUFFER validation 
                        ## y_hat_validation 
                        # df_val, val_true, val_false, val_absent = LR_buffer_prediction(lr, sc, df_val, 
                        #                                                                id_col, y_col, hat_col) 
                                        
                        ## y_hat_test
                        df_test, test_true, test_false, test_absent = LR_buffer_prediction(lr, sc, df_test, 
                                                                                           id_col, y_col, hat_col)
                                     
         
                    if method.lower() == 'logisticregressor-2':
                                            
                        ## BUFFER validation 
                        ## y_hat_validation 
                        # df_val, val_true, val_false, val_absent = LR_buffer_prediction(lr, sc, df_val, 
                        #                                                                id_col, y_col, hat_col,
                        #                                                                method = 2) 
                                                    
                        ## y_hat_test
                        df_test, test_true, test_false, test_absent = LR_buffer_prediction(lr, sc, df_test, 
                                                                                          id_col, y_col, hat_col,
                                                                                          method = 2)   
                                     
                    
                    result_dict = {'n_it': [int(i+1)],
                                   'method': [method.lower()], 
                                   'n_true': [test_true], 
                                   'n_false': [test_false], 
                                   'n_out': [test_absent], 
                                   'n_sample': [len(gauges_test)],
                                   'acc': [  accuracy_score(df_test[y_col], df_test[hat_col]) ],
                                   'prec': [ precision_score(df_test[y_col], df_test[hat_col]) ],
                                   'rec': [ recall_score(df_test[y_col], df_test[hat_col]) ],
                                   'f1': [ f1_score(df_test[y_col], df_test[hat_col]) ]}
                                        
                    df_results = df_results.append(pd.DataFrame(result_dict), ignore_index=True)     
                        
                                                       
            if method.lower() in [option.lower() for option in benchmark_options]: 
                
                assert df_ts is not None, '[ERROR] timeseries data necessary for {} calculation'.format(method)
                
                for i in range(k):
                    
                    ## get test set 
                    gauges_test = split_gauges[i] 
                    
                    ## get labels 
                    df_test = df_labels[ df_labels[id_col].isin(gauges_test) ].copy() 
                    
                    ## benchmark classification                     
                    df_test, test_true, test_false, test_absent = benchmark_buffer_prediction(df_test, df_ts, 
                                                                                              id_col, y_col, 
                                                                                              method=method.lower(), 
                                                                                              dx = 'x', dy = 'y')
                    
                    hat_col = '{}_hat'.format(y_col)
                    ## calc acc, precision, recall, f1
                    
                    result_dict = {'n_it': [int(i+1)], 
                                   'method': [method.lower()], 
                                   'n_true': [test_true], 
                                   'n_false': [test_false], 
                                   'n_out': [test_absent], 
                                   'n_sample': [len(gauges_test)],
                                   'acc': [  accuracy_score(df_test[y_col], df_test[hat_col]) ],
                                   'prec': [ precision_score(df_test[y_col], df_test[hat_col]) ],
                                   'rec': [ recall_score(df_test[y_col], df_test[hat_col]) ],
                                   'f1': [ f1_score(df_test[y_col], df_test[hat_col]) ]}
                    
                    df_results = df_results.append(pd.DataFrame(result_dict), ignore_index=True)
                            
    return df_results 











































