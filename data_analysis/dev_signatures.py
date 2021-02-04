# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:43:32 2021

@author: mvand
"""

import numpy as np 
import pandas as pd 
from scipy import stats 
import hydrostats 
import matplotlib.pyplot as plt 
from pprint import pprint 


#### FEATURE TYPES 

sorted_features = {
    'stats':        ['normal', 'log', 'gev', 'gamma', 'poisson'],
    'correlation':  ['n-acorr', 'n-ccorr'],
    'fdc':          ['fdc-q', 'fdc-slope', 'lf-ratio'],
    'hydro':        ['bf-index', 'dld', 'rld', 'rbf']
    }


stat_options  = sorted_features['stats'] 
corr_options  = sorted_features['correlation']
fdc_options   = sorted_features['fdc']
hydro_options = sorted_features['hydro']

feature_keys = list(sorted_features.keys())
feature_options = []

for key in feature_keys:
    for val in sorted_features[key]:
        feature_options.append(val)


#### TIME WINDOW 
option_time_window = ['all', 'annual', 'seasonal', 'monthly', 'weekly']
time_format = {
    'annual':       ['Y'],
    'seasonal':     [month%12 // 3 + 1 for month in range(1, 13)],
    'monthly':      ['M'],
    'weekly':       ['W']}


##########################################
####       partial functions          #### 
##########################################


##### STAT FUNCTIONS #####
def calc_distr_normal(ts):
    return [np.mean(ts), np.std(ts)]

def calc_distr_log(ts):
    Y = np.log(ts) 
    Y_mu = np.mean(Y)
    Y_sigma = np.std(Y)
    
    X_mu = np.exp(Y_mu)
    X_sigma = X_mu * Y_sigma 
    return [Y_mu, Y_sigma, X_mu, X_sigma]

def calc_distr_gev(ts):
    a = np.pi / ((6**0.5)*np.std(ts))
    u = np.mean(ts) - (0.577/a)
    return [u,a]

def calc_distr_gamma(ts):
    mu = np.mean(ts)
    sigma = np.std(ts)

    k = (mu/sigma)**2 
    theta = sigma / (mu/sigma)
    
    # alpha = shape_k 
    # beta = 1 / scale_theta 
    # return [alpha, beta]
    return [k, theta]

def calc_distr_poisson(ts):
    return None 


###### CORRELATION ######

def calc_auto_correlation(ts, lag=0):
    if lag > 0:
        ts0 = ts[0:-lag]
        ts1 = ts[lag:]
    if lag ==0:
        ts0 = ts
        ts1 = ts        
    # if lag < 0: 
    return stats.pearsonr(ts0,ts1)[0]

def calc_cross_correlation(ts0, ts1, lag=0):
    if lag > 0:
        ts0 = ts0[0:-lag]
        ts1 = ts1[lag:] 
    if lag == 0:
        ts0 = ts0 
        ts1 = ts1 
    # if lag < 0:
    return stats.pearsonr(ts0, ts1)[0]

########## FDC ##########
def fit_FDC(ts, fdc_q, calc_slope = True, calc_LFratio = True):
    return 

#########################
def calc_limbs(ts, calc_RL = True, calc_DL = True):
    return 



func_dict = {
    'normal':   {
        'func': calc_distr_normal,
        'cols': ['Nm-{}', 'Ns-{}']
        },
    'log':      {
        'func': calc_distr_log,
        'cols': ['Lmy-{}', 'Lsy-{}', 'Lmx-{}', 'Lsx-{}']        
        },
    'gev':      {
         'func': calc_distr_gev,
        'cols': ['Gu-{}', 'Ga-{}']       
        },
    'gamma':    {
        'func': calc_distr_gamma,
        'cols': ['Gk-{}', 'Gt-{}']        
        },
    'poisson':  {
         'func': calc_distr_poisson,
        'cols': ['Pl-{}']       
        },
    'n-acorr':  {
         'func': calc_auto_correlation,
        'cols': ['alag-{}-{}']       
        },
    'n-ccorr':  {
         'func': calc_cross_correlation,
        'cols': ['clag-{}-{}']    
        }
    }


##############################################
####             OVERVIEW                 ####
##############################################


def reshape_data(df_obs, df_sim, locations, var='dis24'):
    
    ## create empty datarame to add all simulation and observations
    ## in columns 
    collect_df = pd.DataFrame() 
    
    for loc in locations:
        
        ## get location specific observation dataframe 
        obs_loc = df_obs[df_obs['loc_id'] == loc]
        
        ## extract observation timeseries and dates
        ## add to unique column 
        o_id = 'gauge_{}'.format(loc)
        collect_df['date'] =            obs_loc['date'].values 
        collect_df['obs_datetime'] =    obs_loc.index.values 
        collect_df[o_id] =              obs_loc['value'].values 
        
        ## get subset of simulations of pixels
        ## that potentially match with the gauge 
        ## and id of iterations 
        subset = df_sim[ df_sim['match_gauge']==loc]
        n_iterations = subset['iter_id'].unique() 
        
        ## loop through iterations 
        for n_iter in n_iterations:
            sim_loc = subset[subset['iter_id']==n_iter]
            
            ## extract simulation timeseries 
            ## and add as unique column to collect_df 
            s_id = 'iter_{}_{}'.format(loc, n_iter)
            collect_df[s_id] =           sim_loc[var].values  
            collect_df['sim_datetime'] = sim_loc.index.values 
    
    collect_df.set_index('date', inplace=True)
    return collect_df 

def calc_features(df_obs, df_sim, locations, features = feature_options, time_window = ['all'], n_lag=[0,1], n_cross=[0],
                  fdc_q = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99], mean_threshold = 0., var='dis24'):
        
    assert any(feature in feature_options for feature in features), '[ERROR] not all features can be calculated' 
    assert any(tw in option_time_window for tw in time_window), '[ERROR] not all time windows can be calculated' 
    
    ## empty dataframe for signatures output 
    out_df = pd.DataFrame()
    
    ## reshape data to calculate signatures 
    collect_df = reshape_data(df_obs, df_sim, locations, var=var)
    idx = [col for col in collect_df.columns.values if not 'date' in col]
    idx_gauges = [col for col in idx if 'gauge' in col]
    
    out_df['ID'] = idx 
    out_df.set_index('ID', inplace=True)  
    
    
    ## organize features 
    stat_features =  [feat for feat in features if feat in stat_options]
    corr_features =  [feat for feat in features if feat in corr_options]
    fdc_features =   [feat for feat in features if feat in fdc_options]
    hydro_features = [feat for feat in features if feat in hydro_options]
    # print( len(features), ( len(stat_features)+ len(corr_features) + len(fdc_features) + len(hydro_features) ))

    for tw in time_window:
        
        ### slice or resample all columns 
        if 'all' in tw:
            calc_df = collect_df  
        
        ### calculate STATISTIC features 
        for feature in stat_features:
            
            ## get expected column names 
            return_cols = func_dict[feature]['cols']
            
            ## calculate statistics 
            cdf = calc_df[idx].apply(func_dict[feature]['func'])
            
            ## add to out_df 
            for i in range(len(return_cols)):
                out_df[ return_cols[i].format(tw) ] = cdf.loc[i,:].values 
        
        ### calculate CORRELATION features 
        for feature in corr_features:
            
            ## n-lag autocorrelation 
            if 'n-acorr' in feature:
                
                ## get expected column names 
                return_cols = func_dict[feature]['cols']
                
                ## loop through given lag times 
                for i in range(len(n_lag)):
                    
                    ## check if lag time smaller than total timeseries time 
                    if n_lag[i] < len(calc_df):
                        
                        ## calculate n-lag autocorrelation 
                        cdf = calc_df[idx].apply(func_dict[feature]['func'], lag = n_lag[i])
                        
                        ## add to output df 
                        for j in range(len(return_cols)):
                            out_df[ return_cols[j].format(n_lag[i], tw) ] = cdf.values

            if 'n-ccorr' in feature:
                ## prepare cross correlation calculation 
                return_cols = func_dict[feature]['cols'][0]
                                
                for k in range(len(n_cross)):
                    ## create empty array 
                    results = []
                    
                    ## loop through individual gauges 
                    for i in range(len(idx_gauges)):
                            
                        ## extract gauge column - to calculate
                        ## cross-correlation wtih 
                        gauge_col = calc_df[idx_gauges[i]].values 
                        
                        ## find matching gauges 
                        gauge = idx_gauges[i].split('_')[-1]
                        gauge_matches = [col for col in idx if 'iter_{}'.format(gauge) in col] 
                        
                        ## check if lag time smaller than total timeseries time 
                        if n_cross[k] < len(calc_df):
                            
                            ## calculate lagged autocorrelation 
                            gauge_lag_autocorr = func_dict[feature]['func'](gauge_col, gauge_col, lag=n_cross[k])
                            results.append(gauge_lag_autocorr)
                            
                            ## loop through potential matches 
                            for j in range(len(gauge_matches)):
                                
                                ## get potential mtch 
                                match_col = calc_df[gauge_matches[j]]
                                
                                ## calculate lagged cross correlation 
                                lag_cross_corr = func_dict[feature]['func'](gauge_col, match_col, lag=n_cross[k])
                                results.append(lag_cross_corr)
                    
                    ## collect results in output dataframe 
                    out_df[ return_cols.format(n_cross[k], tw)] = results 
        
        for feature in fdc_features:
            print(feature)

    print(out_df)
    return out_df 




















