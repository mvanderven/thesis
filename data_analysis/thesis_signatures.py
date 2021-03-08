# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:43:32 2021

@author: mvand
"""

import numpy as np 
import pandas as pd 
from scipy import stats, optimize 
# import hydrostats 
import matplotlib.pyplot as plt 
from pprint import pprint 
from datetime import datetime 

import thesis_utils as utils 

#### FEATURE TYPES 

sorted_features = {
    'stats':        ['normal', 'log', 'gev', 'gamma', 'poisson'],
    'correlation':  ['n-acorr', 'n-ccorr'],
    'fdc':          ['fdc-q', 'fdc-slope', 'lf-ratio'],
    'hydro':        ['bf-index', 'dld', 'rld', 'rbf', 'src']
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

def calc_gof(model, stat):
    
    ###################### SAVE BUT NOT IMPLEMENTED
    ## chi-squared 
    ## transform both timeseries to frequencies 
    # model_hist, model_bins = np.histogram(model, bins=5) 
    # stat_hist, stat_bins = np.histogram(stat, bins = model_bins)
    
    ## calculate chi-squared 
    # chi_stat, chi_p = stats.chisquare(model_hist, stat_hist)
    # print(chi_stat, chi_p)
    ######################
    
    
    ## KS-test     
    # D, p = stats.kstest(model, stat)
    D, p = stats.ks_2samp(model, stat)
    
    ## return result of K-test 
    ## if D greater than critical p-value --> rejected 
    ## D > p 
    ## if D < p --> accepted 
    
    ## return: 
    ## 1 if goodness of fit accepted, 0 if not accepted ?? 
    return int(D<p)


def calc_distr_normal(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    mu = np.mean(ts)
    sigma = np.std(ts) 
        
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived mu and sigma (rvs)
    try:
        gof = calc_gof(ts, stats.norm.rvs(loc=mu, scale=sigma, size=len(ts)))
    except:
        print(mu, sigma, len(ts)) 
        print(ts)
        gof = 0
    return [mu, sigma, gof]

def calc_distr_log(ts, eps=1e-6):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    Y = np.log(ts+eps) 
    Y_mu = np.mean(Y)
    Y_sigma = np.std(Y)
    # gof_Y = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma, scale=Y_mu, size=len(ts)))  
    
    X_mu = np.exp(Y_mu)
    X_sigma = X_mu * Y_sigma 
    
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived mu and sigma (rvs)
    gof_X = calc_gof(ts, stats.lognorm.rvs(s=X_sigma, scale=X_mu, size=len(ts)))
    
    return [Y_mu, Y_sigma, X_mu, X_sigma, gof_X]

def calc_distr_gev(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    a = np.pi / ((6**0.5)*np.std(ts))
    u = np.mean(ts) - (0.577/a)
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived a and u 
    gof = calc_gof(ts, stats.genextreme.rvs(c=0, loc = u, scale = a, size=len(ts)))
    return [u,a, gof]

def calc_distr_gamma(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    mu = np.mean(ts)
    sigma = np.std(ts)

    k = (mu/sigma)**2 
    theta = sigma / (mu/sigma)
    
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived k and theta (rvs)
    gof = calc_gof(ts, stats.gamma.rvs(a = k, scale = theta, size=len(ts)))

    # alpha = shape_k 
    # beta = 1 / scale_theta 
    # return [alpha, beta]
    return [k, theta, gof]

def calc_distr_poisson(ts):
    return None 


###### CORRELATION ######

def calc_auto_correlation(ts, lag=0):
        
    # if lag > 0:
    #     ts0 = ts[0:-lag]
    #     ts1 = ts[lag:]
    # if lag ==0:
    #     ts0 = ts
    #     ts1 = ts        
    # # return stats.pearsonr(ts0,ts1)[0]
    return ts.corr(ts.shift(lag))

def calc_cross_correlation(ts0, ts1, lag=0):   
        
    # if lag > 0:
    #     ts0 = ts0[0:-lag]
    #     ts1 = ts1[lag:] 
    # if lag == 0:
    #     ts0 = ts0 
    #     ts1 = ts1 
    # # if lag < 0:
    # return stats.pearsonr(ts0, ts1)[0]
    return ts0.corr(ts1.shift(lag))

########## FDC ##########

def calc_FDC(ts):
    
    ## drop missing values 
    ts_drop = ts.dropna() 
    
    ## sort values from small to large 
    # ts_sorted = ts.sort_values()
    ts_sorted = ts_drop.sort_values()
    
    ## calculate ranks of data 
    ranks = stats.rankdata(ts_sorted, method='dense')

    ## reverse rank order
    ranks = ranks[::-1]
    
    ## calculate probability 
    prob = ((ranks / (len(ts)+1))) * 100
    return prob, ts_sorted 

def calc_FDC_q(ts, fdc_q):
    return_q = []
    
    ## calc FDC curve 
    fdc_prob, fdc_val = calc_FDC(ts)
    
    ## find corresponding quantiles 
    for q in fdc_q:        
        ix_nearest = ( np.abs(fdc_prob - q) ).argmin() 
        return_q.append(fdc_val[ix_nearest])   
    return return_q 

def calc_FDC_slope(ts, eps = 10e-6):
    
    #### from: https://github.com/naddor/camels/blob/master/hydro/hydro_signatures.R
    ####
    #### FDC slope from Sawicz et al. (2011)
    #### log(33)-log(66)/(0.66-0.33)    
    Q33, Q66 = calc_FDC_q(ts, [33, 66])  
    return (np.log10(Q33+eps) - np.log10(Q66+eps)) / (0.66-0.33)

def calc_LF_ratio(ts, eps = 10e-6):    
    Q90, Q50 = calc_FDC_q(ts, [90, 50])
    return (Q90+eps) / (Q50+eps)


########### HYDRO ##############

def calc_limb_slope(ts):
    
    ### calc if rising or declining limb between
    ### two points 
    slope = ts.diff() 
        
    ### calcualate number of peaks
    ### where d_ts[i] > 0 and d_ts[i+1] < 0 
    N_peaks = 0 
    
    ### IMPROVE 
    for i in range(len(slope)-1):
        if slope[i] > 0 and slope[i+1]<0:
            N_peaks += 1     
            
    return slope, N_peaks


def calc_RLD(ts):
    
    slope, N_peaks = calc_limb_slope(ts)
    
    delta_T = slope.index.to_series().diff()
    
    ### calculate total duration of positive limbs
    ### in given period
    T_rising_limbs = delta_T.where(slope>0).sum()

    return N_peaks/T_rising_limbs.days

def calc_DLD(ts):
    
    slope, N_peaks = calc_limb_slope(ts)
    
    delta_T = slope.index.to_series().diff()
    
    ### calculate total duration of negative limbs 
    ### in given period 
    T_declining_limbs = delta_T.where(slope<0).sum()
    
    return N_peaks/T_declining_limbs.days

def calc_i_bf(ts):
    ## from: https://github.com/naddor/camels/blob/master/hydro/hydro_signatures.R
    
    
    ## Calculate baseflow according to:
    ## from: https://raw.githubusercontent.com/TonyLadson/BaseflowSeparation_LyneHollick/master/BFI.R
    
    alpha = 0.925  
    
    Q = ts.values
    
    Q_quick = np.zeros(len(ts)) 
    Q_quick[0] = ts[0]
    
    for i in range(len(Q)-1):
        Q_quick[i+1] = alpha * Q_quick[i] + 0.5*(1+alpha)*(Q[i+1]-Q[i])
            
    Q_base = np.where(Q_quick > 0, Q - Q_quick, Q)
    return sum(Q_base)/ sum(Q)

def calc_RBF(ts):
    
    ## Kuentz et al (2017) - Understanding hydrologic variability across
    ## Europe through catchment classification 
    ## Richard-Baker Flashiness: 
    ## "sum of absolute values of day-to-day changes in mean daily flow
    ## divided by the sum of all daily flows"
    
    ## resample to daily values 
    daily_ts = ts.resample('D').mean() 
    
    ## calculate sum of absolute values of day-to-day changes 
    sum_abs_diff = daily_ts.diff().abs().sum() 
    
    ## calculate sum of daily flows 
    sum_flows = daily_ts.sum() 

    ## return Richard-Baker Flashiness
    return sum_abs_diff/sum_flows

def calc_recession_curve(ts):
    
    ## based on:
    ## Stoelzle, Stahl & Weiler (2013) Are streamflow recession characteristics really characteristic?
    ## Westerberg & McMillan (2015) Uncertainty in hydrological signatures
    
    ### calc differences 
    slope = ts.diff() 
    
    ### assume power-law relationship
    ### dS/dt = -Q
    ### -dQ/dt = a*Q^b
    ### log(-dQ/dt) = log(a)+b*log(Q)
    
    ### plot -dQ/dt vs Q 
    ### fit a straight line through it
    ### a = intercept 
    ### b = slope 
        
    ### get recession values 
    recession = slope.where(slope<0, 0)

    ### split timeseries in all recession times 
    lists = np.split(recession, np.where(recession>=0)[0]+1)
    
    collect_dQ = []
    collect_Q = [] 
    
    for _list in lists:
        
        ## only append recession periods 
        if len(_list) > 1:
            ixs = _list.index 
            Q_list = ts[ixs]
            
            ## save all values in recession period 
            for i in range(len(_list)):
                if (_list[i] < -1e-6) and (Q_list[i] > 1e-6):
                    collect_dQ.append( -_list[i] )
                    collect_Q.append( Q_list[i] )

    ### now all points collected of Q and dQ/dt (<0) 
    ## fit a straight line though these points in log-log plot 
    ## log(dQ/dt) = log(a) + b log(Q)
    ## T0 = intercept 
    ## b = slope 
    
    log_dQ = np.log10(collect_dQ) 
    log_Q = np.log10(collect_Q) 
    
    slope, intercept, log_r, log_p, log_s = stats.linregress(log_Q, log_dQ)     
    # slope, intercept, r_value, p_value, std_err = stats.linregress(collect_Q, collect_dQ)    
    
    b = slope 
    T0 = intercept 
      
    return [b, T0]

##### OVERVIEW 
func_dict = {
    'normal':   {
        'func': calc_distr_normal,
        'cols': ['Nm-{}', 'Ns-{}', 'N-gof-{}']
        },
    'log':      {
        'func': calc_distr_log,
        'cols': ['Lmy-{}', 'Lsy-{}', 'Lmx-{}', 'Lsx-{}', 'L-gof-{}']        
        },
    'gev':      {
         'func': calc_distr_gev,
        'cols': ['Gu-{}', 'Ga-{}', 'Gev-gof-{}']       
        },
    'gamma':    {
        'func': calc_distr_gamma,
        'cols': ['Gk-{}', 'Gt-{}', 'G-gof-{}']        
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
        },
    'fdc-q':  {
         'func': calc_FDC_q,
        'cols': ['fdcQ-{}-{}']       
        },
    'fdc-slope':  {
         'func': calc_FDC_slope,
        'cols': ['fdcS-{}']  
        },
    'lf-ratio':  {
         'func': calc_LF_ratio,
        'cols': ['lf-{}']  
        },
    'bf-index':  {
         'func': calc_i_bf,
        'cols': ['bfi-{}']  
        },
    'rld':  {
         'func': calc_RLD,
        'cols': ['rld-{}']  
        },
    'dld':  {
         'func': calc_DLD,
        'cols': ['dld-{}']  
        },
    'rbf':  {
         'func': calc_RBF,
        'cols': ['rbf-{}']  
        },
    'src':  {
         'func': calc_recession_curve,
        'cols': ['s_rc-{}', 'T0_rc-{}']  
        }
    }


##############################################
####             OVERVIEW                 ####
##############################################


def reshape_data(df_obs, df_sim, locations, var='dis24', T0 = '1991-01-01', T1 = '2020-12-31'):
    
    ## create empty dataframe to add all simulation and observations
    ## in columns 
    collect_df = pd.DataFrame() 
    
    ## create empty dataframe + lists for data collection 
    loc_df = pd.DataFrame()
    row_name = [] 
    row_id   = []
    list_lat = []
    list_lon = []
    list_X   = [] 
    list_Y   = []
    
    ## calculate time-delta from T0 and T1
    T0 = datetime.strptime(T0, '%Y-%m-%d') #.strftime('%Y-%m-%d')
    T1 = datetime.strptime(T1, '%Y-%m-%d') #.strftime('%Y-%m-%d') 
    delta_days = (T1-T0).days 
        
    ## set date column from T0 - T1
    dti = pd.date_range(T0, periods = delta_days+1, freq = 'D')
    
    collect_df['date'] = dti 
    collect_df.set_index('date', inplace=True)
    
    ## fill with observation and simulated values 
    for loc in locations:

        ## get location specific observation dataframe 
        obs_loc = df_obs[df_obs['loc_id'] == loc]
        
        if len(obs_loc) > 0:
                    
            ## extract observation timeseries and dates
            ## add to unique column 
            
            ## column name 
            o_id = 'gauge_{}'.format(loc)
            
            ## extract date indices to place values in
            ## appropriate places 
            obs_loc_ix = pd.to_datetime(obs_loc.index)
            
            ## gauge observations 
            collect_df.loc[obs_loc_ix, o_id] = obs_loc['value'].values 
    
            ## get subset of simulations of pixels
            ## that potentially match with the gauge 
            ## and id of iterations 
            subset = df_sim[ df_sim['match_gauge'] == loc]
            n_iterations = subset['iter_id'].unique() 
            
            ## collect data for loc_df 
            row_name.append(o_id) 
            row_id.append(loc) 
            list_lat.append( obs_loc['lat'].unique()[0] ) 
            list_lon.append( obs_loc['lon'].unique()[0] ) 
            list_X.append( obs_loc['x'].unique()[0] ) 
            list_Y.append( obs_loc['y'].unique()[0] ) 
                             
            ## loop through iterations 
            for n_iter in n_iterations:
                
                sim_loc = subset[subset['iter_id']==n_iter]
                
                sim_loc_ix = pd.to_datetime(sim_loc['date']) #, format = '%Y-%m-%d') 
                
                ## extract simulation timeseries 
                ## and add as unique column to collect_df 
                s_id = 'iter_{}_{}'.format(loc, n_iter)
                collect_df.loc[sim_loc_ix, s_id] = sim_loc[var].values  
                
                ## collect data for loc_df 
                row_name.append(s_id) 
                row_id.append(loc) 
                list_lat.append( sim_loc['lat'].unique()[0] ) 
                list_lon.append( sim_loc['lon'].unique()[0] ) 
                list_X.append( sim_loc['x'].unique()[0] ) 
                list_Y.append( sim_loc['y'].unique()[0] ) 
    
    
    loc_df = pd.DataFrame({'name_id': row_name, 'gauge_id': row_id, 
                           'lat': list_lat, 'lon': list_lon,
                           'x': list_X, 'y': list_Y})
    
    return collect_df, loc_df 


def calc_features(collect_df, locations, features = feature_options, time_window = ['all'], n_lag=[0,1], n_cross=[0],
                  fdc_q = [1, 5, 10, 50, 90, 95, 99], mean_threshold = 0., var='dis24', T_start = '1991-01-01', T_end = '2020-12-31'):
        
    assert any(feature in feature_options for feature in features), '[ERROR] not all features can be calculated' 
    assert any(tw in option_time_window for tw in time_window), '[ERROR] not all time windows can be calculated' 
    
    ## empty dataframe for signatures output 
    out_df = pd.DataFrame()
        
    idx = [col for col in collect_df.columns.values if not 'date' in col]
    idx_gauges = [col for col in idx if 'gauge' in col]
        
    ### ADD INITIAL ROWS with IDs & locations 
    out_df['ID'] = idx 
    out_df.set_index('ID', inplace=True) 
    
    for ix in idx_gauges: 
        
        ## get gauge id 
        gauge_id = ix.split('_')[-1]
        
        ## get subset of locations with matching id 
        sub_locations = locations[ locations['gauge_id'] == gauge_id] 
        
        for sub_loc in sub_locations['name_id']:
            
            for coord_type in ['x', 'y', 'lon', 'lat']:
                out_df.loc[sub_loc, coord_type] =  sub_locations[sub_locations['name_id'] == sub_loc][coord_type].values[0]
        
    ### FINISH PREP - start FEATURES                         
    ## organize features 
    stat_features =  [feat for feat in features if feat in stat_options]
    corr_features =  [feat for feat in features if feat in corr_options]
    fdc_features =   [feat for feat in features if feat in fdc_options]
    hydro_features = [feat for feat in features if feat in hydro_options]
    
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
                out_df[ return_cols[i].format(tw) ] = cdf.loc[i,:]#.values 
        
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
                            out_df[ return_cols[j].format(n_lag[i], tw) ] = cdf#.values

            if 'n-ccorr' in feature:
                ## prepare cross correlation calculation 
                return_cols = func_dict[feature]['cols'][0]
                                
                for k in range(len(n_cross)):
                                
                    return_col_name = return_cols.format(n_cross[k], tw)
                    
                    ## loop through individual gauges 
                    for i in range(len(idx_gauges)):
                            
                        ## extract gauge column - to calculate
                        ## cross-correlation with  
                        gauge_col = idx_gauges[i]
                        
                        ## find matching gauges 
                        gauge_id = idx_gauges[i].split('_')[-1]
                        gauge_matches = [col for col in idx if 'iter_{}'.format(gauge_id) in col] 
                        
                        ## create a list with all 
                        calc_gauges = [gauge_col] + gauge_matches 
                                                
                        ## check if lag time smaller than total timeseries time 
                        if n_cross[k] < len(calc_df):
                            
                            for gauge_i in calc_gauges: 
                                ## compute (lagged) cross correlation 
                                ## with gauge and loop over total list (includes (lagged) auto-correlation for specific gauge)
                                cross_corr = func_dict[feature]['func'](calc_df[gauge_col], calc_df[gauge_i], lag=n_cross[k] ) 
                                
                                ## add result to dataframe
                                out_df.loc[gauge_i, return_col_name] = cross_corr 
                            
        for feature in fdc_features:
            
            if 'fdc-q' in feature:
                                    
                cdf = calc_df[idx].apply(func_dict[feature]['func'], fdc_q = fdc_q) 
                
                for i in range(len(fdc_q)):
                    col_name = func_dict[feature]['cols'][0].format(fdc_q[i], tw)
                    out_df[col_name] = cdf.loc[i,:]
                
            else:
                return_cols = func_dict[feature]['cols']
                cdf = calc_df[idx].apply(func_dict[feature]['func']) 
                
                out_df[ return_cols[0].format(tw) ] = cdf
            
        for feature in hydro_features:
            return_cols = func_dict[feature]['cols']
            cdf = calc_df[idx].apply(func_dict[feature]['func'])
            
            if len(return_cols) > 1:
                
                for i in range(len(return_cols)):
                    col_name = return_cols[i].format(tw)    
                    out_df[col_name] = cdf.loc[i,:]
            else:
                col_name = return_cols[0].format(tw)
                out_df[ col_name ] = cdf
            
    return out_df 




















