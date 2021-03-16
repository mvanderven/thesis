# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:47:23 2021

@author: mvand
"""

import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


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



def buffer_validation(df, target_col, model, scaler): 
    
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

    n_total = n_correct + n_guess    
    print()
    print('-----'*10)        
    print('Total found: {}/{} ({:.2f}%)'.format( n_correct, n_gauges, (n_correct/n_gauges)*100) )
    print('Found in guess-range: {}/{} ({:.2f}%)'.format( n_guess, n_gauges, (n_guess/n_gauges)*100)  )
    print('In total: {}/{} ({:.2f}%) guessed'.format( n_total, 
                                                       n_gauges,
                                                       (n_total/n_gauges)*100))
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
    
    ## get gauge ids     
    idx = pd.Series(df.index.values).str.split('_', expand=True).values 
    ## add as separate columns 
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


def grid_viewer(df, y_col, y_hat_col, gauge_ids = None, buffer_size=2, plot_title=None):
    
    assert 'gauge_id' in df.columns, '[ERROR] gauge id column not found'
    
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
    return 










