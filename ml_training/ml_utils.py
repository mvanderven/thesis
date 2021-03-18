# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:47:23 2021

@author: mvand
"""

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


def subsample(df, target_col, target_value = 1., n_frac = 1.):
    
    df_1 = df[ df[target_col] == target_value] 
    df_0 = df[ df[target_col] != target_value] 
        
    n1 = min( int(len(df_1)*n_frac ), len(df_0) )
    
    ## combine all 1 values with an equal number of 0's 
    return df_1.append( df_0.sample(n=n1)  )


def plot_confusion_matrix(y, y_hat, name = 'Validation score' ):
    
    ## calculate performance scores 
    acc = accuracy_score(y, y_hat)
    prec = precision_score(y, y_hat)
    rec = recall_score(y, y_hat) 
    f1 = f1_score(y, y_hat)
    
    ## create figure 
    plt.figure(figsize=(6,4))
    
    sns.heatmap( pd.DataFrame(confusion_matrix(y, y_hat)),
                    annot=True, cmap='binary', cbar=False)
    plt.ylabel('True class');
    plt.xlabel('Predicted class');
    plt.title('{} (n={}) \n accuracy: {:.2f} || precision: {:.2f} || recall: {:.2f} || f1: {:.2f}'.format(name, len(y),
                                                                                                          acc, prec,
                                                                                                          rec,f1));
                                                                              
    plt.show()
    
    return 



def train_logistic_regressor(X_train, y_train):
    
    ## model setup 
    lr = LogisticRegression(multi_class='multinomial')
    
    ## train model 
    lr.fit(X_train, y_train) 
    
    return lr 



def buffer_validation(df, target_col, model, scaler, verbose = 1 ): 
    
    ## set counters 
    n_gauges = 0 
    n_correct = 0 
    n_guess = 0 
    
    ## create empty list 
    id_true = [] 
    id_guess = [] 
    id_false = [] 
    
    ## copy data for output 
    return_df = df.copy() 
    
    if not 'gauge_id' in return_df.columns:
        # get gauge ids     
        idx = pd.Series(return_df.index.values).str.split('_', expand=True).values 
        # add as separate columns 
        return_df['gauge_id'] = idx[:,1]
        df['gauge_id'] = idx[:,1]
        
    
    for gauge_id in return_df['gauge_id'].unique():
        
        df_gauge = df[df['gauge_id'] == gauge_id] 
        
        ## extract features - scale 
        X = df_gauge.drop([target_col, 'gauge_id'], axis=1) 
        X = scaler.transform(X)
        
        ## extract target value 
        y = df_gauge[target_col] 
        
        if y.sum() > 0:
            
            n_gauges += 1 
            
            ## predicts too many 1s, but can be used to analyse potential matches
            ## predicts multiple potential cell matches 
            y_hat = model.predict(X) 
            
            ## get propabilities of each sample belonging to each class 
            y_prob = model.predict_proba(X) 
            ## get probabilities per calss 
            p0, p1 = y_prob[:,0], y_prob[:,1] 
            y_hat_prob = np.zeros(len(y)) 
            ## get maximum propability for class 1 
            y_hat_prob[ np.argmax(p1) ] = 1 
            
            
            ## add results to return_df 
            return_df.loc[ df_gauge.index, 'y_hat' ] = y_hat 
            return_df.loc[ df_gauge.index, 'y_hat_prob'] = y_hat_prob 
            return_df.loc[ df_gauge.index, 'p0'] = p0 
            return_df.loc[ df_gauge.index, 'p1'] = p1 
                        
            ## check if highest p1 is correct
            if np.all(y==y_hat_prob):
                n_correct += 1 
                id_true.append(gauge_id)
                n_guess += 1 
                
            else:
                ## create df to analyse results 
                # buffer_df = pd.DataFrame({'y': y, 'y_hat': y_hat, 'p0': p0, 'p1': p1})
                
                ## check y_hat - see if other potential cells match 
                ix_y_hat = np.where(y_hat==1)[0] 

                if np.argmax(y) in ix_y_hat:
                    n_guess += 1 
                    id_guess.append(gauge_id)
                    
                    ## show all results in y_hat == 1 
                    # print( buffer_df[buffer_df['y_hat']==1.] )      
                                
                ## wrong results 
                else:
                    id_false.append(gauge_id)
                    # print(buffer_df)
    
    if verbose == 1:
        print()
        print('-----'*10)    
        print('Logistic Regression')    
        print('Total found: {}/{} ({:.2f}%)'.format( n_correct, n_gauges, (n_correct/n_gauges)*100) )
        print('Found in guess-range: {}/{} ({:.2f}%)'.format( n_guess, n_gauges, (n_guess/n_gauges)*100)  )
        print('In total: {}/{} ({:.2f}%) guessed'.format( n_guess, 
                                                          n_gauges,
                                                          (n_guess/n_gauges)*100))
        print('-----'*10)    
    return return_df, id_true, id_guess, id_false


def benchmark_nearest_cell(df_val, x_col, y_col, target_col): 
    
    ## set counters 
    n_correct = 0   
    n_gauges = 0 
    
    ## set empty lsit
    id_true = [] 
    id_false = [] 
    
    ## copy input data for output 
    df = df_val.copy()
    
    if not 'gauge_id' in df.columns:
        # get gauge ids     
        idx = pd.Series(df.index.values).str.split('_', expand=True).values 
        # add as separate columns 
        df['gauge_id'] = idx[:,1]
    
    ## add 'distance' column   
    df['distance'] = ((df[x_col]**2) + (df[y_col]**2))**0.5
    
    ## create y_hat column for storing results 
    target_col_hat = '{}_hat'.format(target_col)
    
    ## loop through all single gauges 
    for gauge_id in df['gauge_id'].unique():
        
        df_gauge = df[ df['gauge_id'] == gauge_id] 
        y_gauge = df_gauge[target_col] 
        
        ## check if a match is found in the buffer 
        if y_gauge.sum() > 0:
            n_gauges += 1 
            
            ## get ix of smallest distance column  
            y_hat = df_gauge['distance'].idxmin() 

            ## set minimum value to 1 
            df.loc[df_gauge.index, target_col_hat ] = 0 
            df.loc[y_hat, target_col_hat] = 1 
            
            ## get accuracy from labelled target column 
            y = y_gauge.idxmax() 
                
            ## if correct, count as success 
            if y == y_hat:
                n_correct += 1
                id_true.append(gauge_id)
            else:
                id_false.append(gauge_id)

    print()
    print('-----'*10)  
    print('Benchmark: naive nearest cell')      
    print('Total found: {}/{} ({:.2f}%)'.format( n_correct, n_gauges, (n_correct/n_gauges)*100) )
    print('-----'*10)    
    
    return df, id_true, id_false 

def benchmark_skill_score(df_timeseries, df_locations, df_labels, method = 'rmse'): #, T0 = '1991-01-01', T1 = '2020-12-31'): 
    
    method = method.lower()
    available_methods = ['rmse', 'nse'] 
    assert method in available_methods, '[ERROR] method {} not available - select from "{}"'.format(method, available_methods)
    
    ## get gauge IDs             
    gauge_idx = df_labels['gauge_id'].unique()
    
    ## create dataframe for output 
    out_df = df_labels.copy()
    
    ## set counters 
    n_gauges = 0
    n_correct = 0 
    
    ## empy lists to collect results 
    id_true = [] 
    id_false = [] 
    
    for gauge_ix in gauge_idx:
        n_gauges += 1 
        col_ix = df_locations[ df_locations['gauge_id'] == int(gauge_ix)].index.values 
        
        sub_df = df_timeseries[col_ix] 
        gauge_col = [col for col in sub_df.columns if 'gauge' in col] 

        if len(gauge_col) > 0:
            # n_gauges += 1 
            model_cols = [col for col in sub_df.columns if not 'gauge' in col ] 
                                    
            ### for each entry, calculate RMSE of
            ### observation timeseries and model timeseries 
            _df = pd.DataFrame() 
            
            ## extract observation data as arrays 
            Q_obs = sub_df[gauge_col].values[:,0]
            Q_obs_avg = Q_obs.mean() 
            
            for col in model_cols:
                
                Q_sim = sub_df[col].values 
                                
                if method == 'rmse':
                    calc_index = mean_squared_error( Q_obs, Q_sim, squared=False )

                if method == 'nse':
                    calc_index =  1 - ((Q_sim - Q_obs)**2).sum() / ( ((Q_obs-Q_obs_avg)**2).sum() )
                    
                _df.loc[col, method] = calc_index
                _df.loc[col, 'gauge_id'] = str(gauge_ix)
                        
            _df['y_hat'] = 0 
            if method == 'rmse':
                y_hat_ix = _df[method].idxmin() 
            
            if method == 'nse':
                ## find largest NSE, flag as 1
                y_hat_ix = _df[method].idxmax() 
            
            _df.loc[y_hat_ix, 'y_hat'] = 1 
                
            ## append true value (y) 
            sub_y = df_labels[ df_labels['gauge_id'] == str(gauge_ix)] 
            y = sub_y['target'] 

            ## check if labelled 
            if y.sum() > 0:
                _df['y'] = y                 
                y_flag = y.idxmax() 
                
                ## check prediction
                if y_flag == y_hat_ix:
                    n_correct += 1 
                    id_true.append(gauge_ix)
                else:
                    id_false.append(gauge_ix)
                            
            ## place results in df  
            out_df.loc[ _df.index, method ] = _df[method]
            out_df.loc[ _df.index, 'y_hat'] = _df['y_hat'] 
                   
    print()
    print('-----'*10) 
    print('Benchmark: {}'.format(method))            
    print('Total found: {} ({:.2f}%)'.format( n_correct, (n_correct/n_gauges)*100) )
    print('-----'*10)      
    
    return out_df, id_true, id_false


def grid_viewer(df, y_col, y_hat_col, gauge_ids = None, buffer_size=2, plot_title=None, cmap = 'binary'):
    
    if not 'gauge_id' in df.columns:
        # get gauge ids     
        idx = pd.Series(df.index.values).str.split('_', expand=True).values 
        # add as separate columns 
        df['gauge_id'] = idx[:,1]

    if gauge_ids is None:
        gauge_ids = df['gauge_id'].unique()

    n_gauges = len(gauge_ids) 

        
    fig = plt.figure(figsize=(6,8))
    
    for j in range(len(gauge_ids)):
        gauge_id = gauge_ids[j]
        
        df_gauge = df[ df['gauge_id'] == gauge_id] 
        df_ix = df_gauge.index.values 
        
        ## create empty buffer grid for display 
        buffer_grid = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
        buffer_label = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
        buffer_guess = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
        
        for i in range(len(df_gauge)):
            ix = int( df_ix[i].split('_')[-1] )
            buffer_grid[ix] = df_gauge.loc[df_ix[i], y_hat_col] 
            buffer_label[ix] = df_gauge.loc[df_ix[i], y_col]
            
            if 'y_hat_prob' in df.columns:
                buffer_guess[ix] = df_gauge.loc[df_ix[i], 'y_hat_prob']
        
        buffer_grid = buffer_grid.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) ))
        buffer_label = buffer_label.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) ))
        row_label, col_label = np.where(buffer_label == buffer_label.max())
        
        if 'y_hat_prob' in df.columns:
            buffer_guess = buffer_guess.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) ))
            row_guess, col_guess = np.where(buffer_guess == buffer_guess.max())
        
        
        if n_gauges <= 10:
            ax = fig.add_subplot( int((n_gauges)), 1, int(j+1) )
        else:
            ax = fig.add_subplot( int(((n_gauges)/3)+1), 3, int(j+1) )
            
        ax.pcolormesh(buffer_grid, cmap='binary_r', edgecolors='white')
        ax.set_title('Gauge ID-{}'.format(gauge_id))
        ax.plot(col_label[0]+0.45, row_label[0]+0.55, marker = 'o', color='red', markersize=4)
        if 'y_hat_prob' in df.columns:
            ax.plot(col_guess[0]+0.45, row_guess[0]+0.5, marker = 'x', color='blue')
        ax.axes.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    if plot_title is not None:
        fig.suptitle(str(plot_title))
    plt.tight_layout()   
    plt.show()
    return 

def grid_view_param(df, y_col, y_hat_col, param_col, gauge_ids = None, buffer_size=2, plot_title=None, cmap='binary_r'):

    if not 'gauge_id' in df.columns:
        # get gauge ids     
        idx = pd.Series(df.index.values).str.split('_', expand=True).values 
        # add as separate columns 
        df['gauge_id'] = idx[:,1]

    if gauge_ids is None:
        gauge_ids = df['gauge_id'].unique()

    n_gauges = len(gauge_ids) 

        
    fig = plt.figure(figsize=(6,8))
    
    for j in range(len(gauge_ids)):
        gauge_id = gauge_ids[j]
        
        df_gauge = df[ df['gauge_id'] == gauge_id] 
        df_ix = df_gauge.index.values 
        
        ## create empty buffer grid for display 
        buffer_param = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
        buffer_y     = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
        buffer_y_hat = np.zeros(( int(1+(2*buffer_size))*int(1+(2*buffer_size)) ))
                
        for i in range(len(df_gauge)):
            ix = int( df_ix[i].split('_')[-1] ) 
            buffer_param[ix] = df_gauge.loc[ df_ix[i], param_col ]
            buffer_y[ix]     = df_gauge.loc[ df_ix[i], y_col] 
            buffer_y_hat[ix] = df_gauge.loc[ df_ix[i], y_hat_col]
            
        
        buffer_param = buffer_param.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) ))
        buffer_y     = buffer_y.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) ))
        row_y, col_y = np.where(buffer_y == buffer_y.max()) 
                
        buffer_y_hat = buffer_y_hat.reshape(( int(1+(2*buffer_size)), int(1+(2*buffer_size)) )) 
        row_y_hat, col_y_hat = np.where( buffer_y_hat == buffer_y_hat.max() )
        
        
        if n_gauges <= 10:
            ax = fig.add_subplot( int((n_gauges)), 1, int(j+1) )
        else:
            ax = fig.add_subplot( int(((n_gauges)/3)+1), 3, int(j+1) )
            
        im=ax.pcolormesh(buffer_param, cmap=cmap, edgecolors='white')
        ax.set_title('Gauge ID-{}'.format(gauge_id))
        ax.plot(col_y + 0.45, row_y + 0.55, marker = 'o', color='red', linestyle='none') 
        ax.plot(col_y_hat + 0.45, row_y_hat + 0.55, marker = 'x', color='orange', linestyle='none')
        
        cbar = fig.colorbar(im,ax=ax, shrink=0.9)
        cbar.ax.set_ylabel(param_col)

        ax.axes.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    if plot_title is not None:
        fig.suptitle(str(plot_title))
    plt.tight_layout()   
    plt.show()
    return 

def plot_locations(df_locations, x = 'x', y='y', gauge_labels=None, label_key = None, 
                   plot_title=None): 
    
    ## select gauge data only 
    df_gauges = df_locations[ df_locations.index.str.contains('gauge') ]
    
    ## FIGURE    
    fig = plt.figure(figsize=(10,6)) 
    
    ax = fig.add_subplot(111, projection = cp.crs.PlateCarree()) 
    
    x_min, x_max = df_gauges[x].min(), df_gauges[x].max()
    y_min, y_max = df_gauges[y].min(), df_gauges[y].max()
        
    ## BACKGROUND
    ## imagery
    rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m') 
    ax.add_feature(cp.feature.OCEAN)
    ax.add_feature(cp.feature.LAND, edgecolor='black')
    ax.add_feature(cp.feature.BORDERS)
    ax.add_feature(rivers_50m, facecolor='None', edgecolor='b') 
    
    wd = 10.
    ax.set_xlim( max(-180, x_min - wd), min(180, x_max + wd) )
    ax.set_ylim( max(-90, y_min - wd),  min(90, y_max + wd) )
    
    ## GAUGE LOCATIONS 
    if gauge_labels is None:
        ax.plot( df_gauges[x], df_gauges[y], linestyle='none', marker='.', color='r', markersize=4)
        
        if plot_title is not None:
            ax.set_title('{} (n={})'.format(plot_title, len(df_gauges)))    
    else:
        n_items = 0 
        if label_key in df_gauges.columns:
            for k in gauge_labels.keys():
                subset = df_gauges[ df_gauges[label_key].isin( gauge_labels[k] )] 
                n_items += len(subset)
                ax.plot( subset[x], subset[y], linestyle='none', marker = '.', markersize=10, label = k)                
            
            ax.legend() 
            
            if plot_title is not None:
                ax.set_title('{} (n={})'.format(plot_title, n_items))
    
        else:
            print('Label not found')
            
     
    plt.show()
    return

def plot_timeseries(df, df_label, show_list = None, plot_title=None, label_list = None): 
    
    if show_list is not None:
        df_label = df_label[ df_label['gauge_id'].isin(show_list)] 
    
    if show_list is None:
        show_list = df_label['gauge_id'].unique()
    
    ts_cols = df.columns  
    
    
    fig = plt.figure(figsize=(12,9)) 

    for i in range(len(show_list)):
        gauge_id = show_list[i]

        labels = df_label[ df_label['gauge_id'] == gauge_id]['y_hat_prob'] 
        
        cols = [col for col in ts_cols if str(gauge_id) in col]         
        collect_ts = df[cols]  
        
                
        if len(show_list) < 6:
            ax = fig.add_subplot( len(show_list), 1, int(i+1) )
        else:
            ax = fig.add_subplot( int((len(show_list)/3)+1), 3, int(i+1) )
        
        for col in cols:
            
            if 'gauge' in col:
                ax.plot(collect_ts[col].values, color='k') 
            
            else:
                if labels.loc[col] == 0:
                    ax.plot(collect_ts[col].values, color='grey', lw=0.5) 
                else:
                    ax.plot(collect_ts[col].values, color='r', lw=0.5)
                    
        ax.set_title('Gauge ID-{}'.format(gauge_id)) 
        ax.set_xlim(0, 365 ) 
        ax.grid()
        
    plt.tight_layout()
    
    return 


def k_fold_CV(df, k = 10, target_col='target'): 
    
    unique_gauges = df['gauge_id'].unique() 
    
    assert k <= len(unique_gauges), '[ERROR] k={} too large for n={}'.format(k, len(unique_gauges))
    
    k_train = [] 
    k_test = [] 
    k_val = [] 
    k_prob = []  
    
    
    ### shuffle list 
    random.shuffle(unique_gauges)
    
    ### divide in k-fold sublists 
    split_gauges = np.array_split(unique_gauges, k) 
        
    for i in range(k): 
        
        ## holdout a sublist for validation
        ## from different groups 
        gauges_validation = split_gauges[i]
        if i == 0:
            test = split_gauges[1:]
        elif i == int(k-1):
            test = split_gauges[:-1] 
        else:
            test = split_gauges[:i] + split_gauges[i+1:]
        
        gauges_train_test = [item for sublist in test for item in sublist]
        
        ## create new dfs 
        df_train_test = df[ df['gauge_id'].isin(gauges_train_test) ] 
        df_validation = df[ df['gauge_id'].isin(gauges_validation) ] 
        
        ## subsample df_train_test  
        ## HERE variation issue -- different 0 cells are selected with subsampling 
        df_train_test_sample = subsample(df_train_test, target_col, n_frac = 1) 
        
        ## split df_train_test_sample into train and test set 
        X_train, X_test, y_train, y_test = train_test_split( df_train_test_sample.drop([ 'gauge_id', target_col], axis=1), 
                                                            df_train_test_sample[target_col],
                                                            test_size = 0.15) 
              
        ## normalize X_train, apply on X_val 
        sc = MinMaxScaler([0,1]) 
        ## train scaler 
        X_train = sc.fit_transform(X_train) 
        ## apply on validation data 
        X_test = sc.transform(X_test) 
        
        ## train model 
        lr_model = train_logistic_regressor(X_train, y_train) 
        
        ## training performance 
        y_hat_train = lr_model.predict(X_train)
        ## validation performance 
        y_hat_test = lr_model.predict(X_test) 
        
        ## run on validation data 
        df_val, val_true, val_guess, val_false = buffer_validation(df_validation, target_col, lr_model, sc, verbose=0) 
        
        y_val = df_val['target'] 
        y_hat_val = df_val['y_hat']
        y_hat_prob = df_val['y_hat_prob']
        
        ## collect results 
        k_train.append([ accuracy_score(y_train, y_hat_train), precision_score(y_train, y_hat_train), 
                        recall_score(y_train, y_hat_train), f1_score(y_train, y_hat_train) ])
        
        k_test.append([accuracy_score(y_test, y_hat_test), precision_score(y_test, y_hat_test), 
                        recall_score(y_test, y_hat_test), f1_score(y_test, y_hat_test)]) 
        
        k_val.append([accuracy_score(y_val, y_hat_val), precision_score(y_val, y_hat_val), 
                        recall_score(y_val, y_hat_val), f1_score(y_val, y_hat_val)]) 
        
        k_prob.append([accuracy_score(y_val, y_hat_prob), precision_score(y_val, y_hat_prob), 
                        recall_score(y_val, y_hat_prob), f1_score(y_val, y_hat_prob)])
        
    k_train = np.array(k_train)
    k_test = np.array(k_test)
    k_val = np.array(k_val)
    k_prob= np.array(k_prob)     
    k_results = np.concatenate((k_train, k_test, k_val, k_prob)) 
      
    cols = ['acc', 'prec', 'rec', 'f1']
    ix = ['train'] * k + ['test'] * k + ['val'] * k + ['prob'] * k
    df = pd.DataFrame(k_results, columns = cols)  
    df['cat'] = ix
    
    return df 





















