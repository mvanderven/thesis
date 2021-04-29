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
    'all':          [':'],
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
    ## derived mu and sigma
    try:
        gof = calc_gof(ts, stats.norm.rvs(loc=mu, scale=sigma, size=len(ts)))
    except:
        gof = 0
        print(gof)
    return [mu, sigma, gof]

def calc_distr_log(ts, eps=1e-6):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    ## transform 
    Y = np.log(ts+eps) 
    Y_mu = np.mean(Y)
    Y_sigma = np.std(Y)
        
    ## calculate goodness-of-fit 
    ## create an artificial dataset
    ## based on derived distribution
    ## parameters 
    gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma, scale=np.exp(Y_mu), size=len(ts))) 
    
    ## alternative transformation - similar results  
    # Y_mu_alt = np.log( np.mean(ts)**2 / (( np.mean(ts)**2 + np.std(ts)**2 )**0.5 )  )
    # Y_sigma_alt = (np.log( 1 + ((np.std(ts)/np.mean(ts))**2) ))**0.5
    # gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma_alt, scale=np.exp(Y_mu_alt), size=len(ts))) 

    return [Y_mu, Y_sigma, gof]

def calc_distr_gev(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    ## calculate gev parameters 
    ## gumbel, so k = 0
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
    
    ## calculate gamma parameters 
    mu = np.mean(ts)
    sigma = np.std(ts)

    k = (mu/sigma)**2 
    theta = sigma / (mu/sigma)
    
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived k and theta (rvs)
    gof = calc_gof(ts, stats.gamma.rvs(a = k, scale = theta, size=len(ts)))
    
    return [k, theta, gof]

def calc_distr_poisson(ts):
    return None 


###### CORRELATION ######

def calc_auto_correlation(ts, lag=0):
    return ts.corr(ts.shift(lag))

def calc_cross_correlation(ts0, ts1, lag=0):   
    return ts0.corr(ts1.shift(lag))

########## FDC ##########

def calc_FDC(ts):
    
    ## FROM
    ## Searcy (1959) Flow-Duration Curves
    ##      Total period method: all discharges placed
    ##      according of magnitude
    
    ## drop missing values 
    ts_drop = ts.dropna() 
    
    ## sort values from small to large 
    ts_sorted = ts_drop.sort_values()
    
    ## calculate ranks of data 
    ## rankdata methods:
    ## (1)method=ordinal: unique rank value 
    ranks = stats.rankdata(ts_sorted, method='ordinal') 
    ## (2) method=dense: duplicate Q values get same rank value
    # ranks = stats.rankdata(ts_sorted, method='dense') 
    
    ## reverse rank order
    ranks = ranks[::-1] 
    
    ## calculate probability 
    prob = ((ranks / (len(ts_sorted)+1) )) * 100 
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
    #### FDC slope calculation procedure from:
    ####    Sawicz et al. (2011) Catchment classifcation: empirical analysis
    ####    of hydrological similarity based on catchment function in the
    ####    eastern USA
    ####
    #### FDC_slope = log(Q_33)-log(Q_66)/(0.66-0.33)    
    
    ## calculate FDC_Q33 and Q_66
    Q33, Q66 = calc_FDC_q(ts, [33, 66])  
    return (np.log10(Q33+eps) - np.log10(Q66+eps)) / (0.66-0.33)

def calc_LF_ratio(ts, eps = 10e-6):    
    
    #### Low Flow Ratio calculation from:
    ####    Nijzink et al. (2018) Constraining conceptual hydrological
    ####    models with multiple information sources
    ####
    #### LF = FDC_Q90 / FDC_Q50
    Q90, Q50 = calc_FDC_q(ts, [90, 50]) 
    return (Q90+eps) / (Q50+eps)


########### HYDRO ##############

def calc_limb_slope(ts):
    
    #### Morin et al. (2002) Objective, observations-based automatic
    #### estimation of catchment response timescale: 
    ####    Indicate if slope of hydrograph increases (+1) or decreases (-1) 
    ####    "A peak is a series of +1 points followed by -1 points, possibly
    ####    with zeros in between"
    
    #### Shamir et al. (2005) The role of hydrograph indices
    #### in parameter estimation of rainfall-runoff models 
    ####    All time steps showing a positive or negative change from 
    ####    previous time step, regardless of magnitude of change, were 
    ####    included in calculation of cumulative duration of rising or
    ####    declining steps. 
    ####
    ####    A peak is defined as a time step that has a higher value from
    ####    previous and latter time steps
    
    ## calculate differences between time steps 
    ts_diff = ts.diff()  
    
    ## indicate if increase or decrease between time steps 
    ts_slope = np.where( ts_diff > 0, 1, -1)
    
    ## detect peaks
    ## where a +1 point is followed by a -1 point 
    mask_peaks = (ts_slope[:-1] > 0) & (ts_slope[1:] < 0)
    return ts_diff, mask_peaks.sum()


def calc_RLD(ts):
    
    #### From:
    ####    Morin et al. (2002) Objective, observations-based automatic
    ####    estimation of catchment response timescale 
    
    #### Peak Density = 
    ####            Total peak numbers / total rising limb duration 
    ####    "Peak is a series of +1 (positive slope hydrograph) points
    ####    followed by -1 (negative slope hydrograph) points, possibly
    ####    with zeros in between"
    ####
    #### Also known as: Rising Limb Density
    
    ## get dQ and number peaks 
    slope, N_peaks = calc_limb_slope(ts)
    
    ## calculate timestep 
    delta_T = slope.index.to_series().diff()

    ### calculate total duration of positive limbs
    ### in given period 
    mask_positive_limbs = slope > 0     
    T_rising_limbs = delta_T.loc[ mask_positive_limbs ].sum()
    
    if T_rising_limbs.days > 0:
        return N_peaks / T_rising_limbs.days 
    else:
        return np.nan 

def calc_DLD(ts):
    
    #### Derived from "Peak Density" by Morin et al. (2002).
    #### Expanded by:
    ####    Shamir et al. (2005) The role of hydrograph indices
    ####    in parameter estimation of rainfall-runoff models
    
    #### Declining limb density = 
    ####              total peak numbers / total declining limb duration
                               
    
    slope, N_peaks = calc_limb_slope(ts)
    
    delta_T = slope.index.to_series().diff()
    
    ### calculate total duration of negative limbs 
    ### in given period 
    mask_declining_limbs = slope < 0 
    T_declining_limbs = delta_T.loc[mask_declining_limbs].sum() 
    
    if T_declining_limbs.days > 0:
        return N_peaks / T_declining_limbs.days 
    else:
        return np.nan 
    
    # T_declining_limbs = delta_T.where(slope<0).sum()
    
    # return N_peaks/T_declining_limbs.days

def calc_i_bf(ts):
    
    #### Base Flow Index following:
    ####    Sawicz et al. (2011) Catchment classification: empirical
    ####    analysis of hydrologic similarity based on catchment
    ####    function in the eastern USA
    ####
    #### Defines Base Flow Index as ratio of long-term baseflow to
    #### total streamflow. Calculated with following steps:
    #### (1) use one-parameter single-pass digital filter method 
    ####     to separate baseflow 
    ####     Q_Dt = c * Q_Dt-1 + (1+c)/2 (Q_t - Q_t-1), c = 0.925 
    #### (2) Baseflow equals: Q_Bt = Q_t - Q_Dt 
    #### (3) The baseflow index is then: I_BF = Sum (Q_B / Q)     
    
    #### coding examples from:
    #### https://github.com/naddor/camels/blob/master/hydro/hydro_signatures.R     
    #### with source code:
    ####    https://raw.githubusercontent.com/TonyLadson/BaseflowSeparation_LyneHollick/master/BFI.R 
        
    #### handle missing values 
    if ts.isnull().any():
        ts = ts.fillna(method='ffill').dropna() 
        
        ## if length of timeseries (still) 0, return nan 
        if len(ts) == 0:
            return np.nan 
    
    #### following formatting by Sawicz (2011)
    Q_t  = ts.values 
    Q_D = np.zeros(len(Q_t)) 
    
    ## set initial value equal to surface flow 
    Q_D[0] = Q_t[0] # * factor?
        
    #### Value of c determined by Eckhardt (2007) A comparison of baseflow 
    #### indices, which were cal-culated with seven different baseflow 
    #### separation methods
    c = 0.925 

    #### (1) Separate direct flow from baseflow 
    for i in range(len(Q_t)-1):
        Q_D[i+1] = (c * Q_D[i]) + ( ( (1+c)*0.5 ) * (Q_t[i+1] - Q_t[i]) )
    
    #### (2) Subtract direct flow from total flow to extract baseflow 
    Q_B = Q_t - Q_D  
    
    #### (3) Calculate baseflow index 
    return sum(Q_B)/ sum(Q_t)

def calc_RBF(ts):
    
    #### Richard-Baker Flashiness from:
    ####    Kuentz et al. (2017) Understanding hydrologic variability
    ####    across Europe through catchment classfication 
    ####
    #### Based on:
    ####    Baker et al. (2004) A new flashiness index: characeteristics
    ####    and applications to midswestern rivers and streams 1 
    ####
    #### Application derived from:
    ####    Holko et al. (2011) Flashiness of mountain streams in 
    ####    Slovakia and Austria 
    ####
    #### RBF defined as: Sum of absolute values of day-to-day changes in
    #### average daily flow divided by total discharge during time interval.
    #### Index calculated for seasonal or annual periods, can be averaged
    #### accross years 
        
    #### SPLIT TIMESERIES 
    #### ?? filling missing values?? 
    ts = ts.dropna()
    
    if len(ts) == 0:
        ## if length is 0, RBF cannot be calculated 
        ## return NaN 
        return np.nan 
    
    ## get delta_T to estimate if dates are continuous
    ## or with jumps 
    dT = ts.index.to_series().diff() 

    ## check for jumps in timeseries  
    if dT.max() > pd.Timedelta('1d'): #pd.Timedelta(value=1, unit='days'): 
        
        ## TEST FOR MULTIPLE MOMENTS 
        ## determine chunk size 
        t_start = ts.index.values[0] 
        delta_T = dT.max() 
        t_end = t_start + delta_T
        
        ts_chunk = ts.loc[(ts.index >= t_start) & (ts.index < t_end)]  
        n_chunks = int( len(ts) / len(ts_chunk) )
        
        ## get periods based on start date and frequency
        dti = pd.date_range(start=t_start, freq='12MS', periods = n_chunks)

    ## else split based on years 
    else:
        iter_list = ts.index.year.unique()
        dti = pd.date_range(start = ts.index[0], freq='YS', periods = len(iter_list))
    
    ## go over all periods 
    ## calculate RBF
    collect_RBF = []
    for i in range(len(dti)):

        if i < (len(dti)-1):  
            mask = (ts.index >= dti[i]) & (ts.index < dti[i+1])
            
        if i == (len(dti)-1):
            mask = (ts.index >= dti[i])
        
        ## get period of interest
        _ts = ts.loc[mask]
        
        if len(_ts) > 0:
            ## sum absolute day-to-day differences 
            _sum_abs_diff = _ts.diff().abs().sum() 
            
            ## sum total flow 
            _sum_flows = _ts.sum() 

            if (_sum_abs_diff > 0) & (_sum_flows > 0):
                collect_RBF.append((_sum_abs_diff/_sum_flows))
    
    ## return average Richard-Baker Flashiness
    if len(collect_RBF) > 0:
        return_RBF = np.nanmean(np.array(collect_RBF))
    else:
        return_RBF = np.nan 
    return return_RBF

def calc_recession_curve(ts):
    
    #### Recession curve characteristics based on:
    ####    Stoelzle et al. (2013) Are streamflow recession characteristics 
    ####    really characteristic?
    #### and:
    ####    Vogel & Kroll (1992) Regional geohydrologic-geomorphic 
    ####    relationships for the estimation of low-flow statistics
    ####    
    #### Method described by Vogel & Kroll:
    ####    "A recession period begins when a 3-day moving average begins
    ####    to decrease and ends when it begins to increase. Only recession 
    ####    of 10 or more consecutive daysare accepted as candidate 
    ####    hydrograph recessions."
    ####
    ####    "If the total length of a candidate recession segment is I,  
    ####    then the initial portion is predominatnly surface or storm 
    ####    runoff, so first lambda*I days were removed from the 
    ####    candidate recession segment."
    ####    First guess for lambda = 0.3 
    ####
    ####    "To avoid spurious observations, only accept pairs of stream-
    ####    flow where Q_t > 0.7*Q_t-1"
    ####
    ####    "Fit log-log relation:
    ####    ln(-dQ/dt) = ln(a) + b*len(Q) + eps "
    ####    Linear reservoir assumption: b = 1 
    ####    Use ordinary least squares estimators for a and b 
    ####    using all accepted dQ and Q pairs 
    
    ## Vogel & Kroll chose lambda = 0.3, looking at summer months only 
    ## Here, (multi-) annual periods, seasons and months are used
    ## In their study, lambda was varied between 0 and 0.8, to choose
    ## a value of lambda corresponding to an average value of b = 1 
    
    ## lower value of lambda decreases b 
    ## higher value of lambda increases b 
    # init_lambda = 0.3  
    init_lambda = 0.05
        
    ## calculate 3-day moving average 
    ts_mv = ts.rolling(window=3, center=True).mean() 
    
    ## calculate differences between two time steps 
    ts_mv_diff = ts_mv.diff() 

    ## mask recession periods 
    recession_mask = ts_mv_diff <= 0. 
    
    if recession_mask.sum() == 0:
        ## no recession periods detected 
        return [np.nan, np.nan]
        
    ## collect in dataframe 
    _df = pd.DataFrame()
    _df['Q'] = ts_mv[recession_mask]     
    _df['dQ'] = ts_mv_diff[recession_mask]    
    _df['dT'] = _df.index.to_series().diff() 
        
    ## identify periods 
    _df['periods'] =  _df['dT'].dt.days.ne(1).cumsum() 
    
    ## identify period length 
    for period_id in _df['periods'].unique():
        p_df = _df[ _df['periods'] == period_id]         
        _df.loc[ p_df.index, 'len'] = len(p_df)
    
    ## drop all periods with a length below 10 days 
    _df_dropped = _df[ _df['len'] >= 10].copy()
    
    ## check change in differences 
    ## Q_t > 0.7 Q_t-1
    Q_t1 = _df_dropped['Q'] 
    Q_t0 = 0.7 * _df_dropped['Q'].shift(-1)
    
    ## add results to df 
    _df_dropped['check_change'] = Q_t1 > Q_t0 
    
    ## collect final results 
    ## after checking change magnitudes 
    _df_dropped['analyse'] = 0 
    for period_id in _df_dropped['periods'].unique():
        p_df = _df_dropped[ _df_dropped['periods']==period_id] 
        
        ## compute number of False occurencies 
        n_false = len(p_df) - sum(p_df['check_change']) 
        last_false = False 
        
        if n_false == 1:
            ## get index of false value 
            idx_false = p_df[ p_df['check_change'] == False ].index         
                    
            ## if false idx and last idx match: continue 
            if idx_false == p_df.tail(1).index:
                last_false = True 
        
        if (n_false == 0) | (last_false) :
            ## if regression period is accepted, discrard the first 
            ## lambda * len(p_df) values from the period
            n_skip = round( init_lambda * len(p_df) )
            ix_to_keep = p_df.iloc[n_skip:].index  
            _df_dropped.loc[ix_to_keep, 'analyse'] = 1
    
    ## keep marked Q/dQ couples 
    _df_analyse = _df_dropped[ _df_dropped['analyse'] == 1][['Q', 'dQ']].copy()
    
    if not len(_df_analyse) > 0:
        return [np.nan, np.nan]
    
    ## plot Q and delta_Q on a natural logarithmic scale 
    _df_analyse['log_Q'] = np.log( _df_analyse['Q'] )
    _df_analyse['log_dQ'] = np.log( abs(_df_analyse['dQ'] + 10e-6) )
    
    ## use ordinary least squares regression to find a and b in:
    ## ln(-dQ) = ln(a) + b*len(Q) 
    slope, intercept, r_value, p_value, std_err = stats.linregress( _df_analyse['log_Q'], _df_analyse['log_dQ'] )     
    
    ## interpret results 
    b = slope 
    a = intercept 
    # a = np.exp(intercept)    
    return [b, a]

##### OVERVIEW 
func_dict = {
    'normal':   {
        'func': calc_distr_normal,
        'cols': ['Nm-{}', 'Ns-{}', 'N-gof-{}']
        },
    'log':      {
        'func': calc_distr_log,
        'cols': ['Lm-{}', 'Ls-{}', 'L-gof-{}']        
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
        'cols': ['b_rc-{}', 'a_rc-{}']  
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
    loc_df.set_index('name_id', inplace=True)
    
    return collect_df, loc_df 


def calc_features(collect_df, locations, features = feature_options, time_window = ['all'], n_lag=[0,1], n_cross=[0],
                  fdc_q = [1, 5, 10, 50, 90, 95, 99], mean_threshold = 0., T_start = '1991-01-01', T_end = '2020-12-31'):
     
    print('[START] feature calculation')
    
    assert any(feature in feature_options for feature in features), '[ERROR] not all features can be calculated' 
    assert any(tw in option_time_window for tw in time_window), '[ERROR] not all time windows can be calculated' 
    
    print('[INFO] prep')
    ## empty dataframe for signatures output 
    out_df = pd.DataFrame()
        
    idx = [col for col in collect_df.columns.values if not 'date' in col]
    idx_gauges = [col for col in idx if 'gauge' in col]
        
    ### ADD INITIAL ROWS with IDs & locations 
    out_df['ID'] = idx 
    out_df.set_index('ID', inplace=True) 
    
    for ix in idx_gauges: 
        
        ## give info about the type 
        out_df.loc[ix, 'is_gauge'] = 1 
        
        ## get gauge id 
        gauge_id = ix.split('_')[-1]
        
        ## get subset of locations with matching id 
        sub_locations = locations[ locations['gauge_id'] == gauge_id] 
                
        for sub_loc in sub_locations.index:
            out_df.loc[sub_loc, 'match'] = gauge_id
            for coord_type in ['x', 'y', 'lon', 'lat']:
                # out_df.loc[sub_loc, coord_type] =  sub_locations[sub_locations['name_id'] == sub_loc][coord_type].values[0]
                out_df.loc[sub_loc, coord_type] =  sub_locations.loc[sub_loc][coord_type]
                
    
    ## if not gauge, set is_gauge to 0 
    out_df['is_gauge'] = out_df['is_gauge'].fillna(0)
    # out_df['is_gauge'].fillna(0, inplace=True)
    
    print('[INFO] start feature calculation')    
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
            print('[INFO] calc stats: {}'.format(feature))
            
            ## get expected column names 
            return_cols = func_dict[feature]['cols']
            
            ## calculate statistics 
            cdf = calc_df[idx].apply(func_dict[feature]['func'])
            
            ## add to out_df 
            for i in range(len(return_cols)):
                out_df[ return_cols[i].format(tw) ] = cdf.loc[i,:]#.values 
        
        ### calculate CORRELATION features 
        for feature in corr_features:
            print('[INFO] calc correlation: {}'.format(feature))
            
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
            print('[INFO] calc FDC: {}'.format(feature))
            
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
            print('[INFO] calc index: {}'.format(feature))
            
            return_cols = func_dict[feature]['cols']
            cdf = calc_df[idx].apply(func_dict[feature]['func'])
            
            if len(return_cols) > 1:
                
                for i in range(len(return_cols)):
                    col_name = return_cols[i].format(tw)    
                    out_df[col_name] = cdf.loc[i,:]
            else:
                col_name = return_cols[0].format(tw)
                out_df[ col_name ] = cdf
    
    print('[FINISH] calculating features ')
    return out_df 

def calc_signatures(df_observations, df_simulations, id_col = 'loc_id',
                    features = feature_options, time_window = option_time_window,
                    fdc_q = [1, 5, 10, 50, 90, 95, 99],
                    n_alag = [1,365], n_clag = [0,1]):
    
    df_out = pd.DataFrame()
    
    ## organize features 
    stat_features =  [feat for feat in features if feat in stat_options]
    corr_features =  [feat for feat in features if feat in corr_options]
    fdc_features =   [feat for feat in features if feat in fdc_options]
    hydro_features = [feat for feat in features if feat in hydro_options] 
    
    for i, gauge_id in enumerate( df_observations[id_col].unique() ):
    
        gauge_ts = df_observations[ df_observations[id_col] == gauge_id][['date', 'value']].copy()
        gauge_ts['time'] = gauge_ts['date'] 
        gauge_ts = gauge_ts.set_index('time')
        
        cell_cols = [col for col in df_simulations.columns if gauge_id in col]
        df_buffer = df_simulations[cell_cols].copy() 
        
        gauge_col = 'gauge_{}'.format(gauge_id)
        df_buffer[gauge_col] = gauge_ts['value'] 
        
        ## how to handle missing data?
        # gauge_null_mask = gauge_ts.isnull().any(axis=1) 
        
        ## get ID values 
        ts_idx = df_buffer.columns  
        
        ## set temporary df to collect gauge output 
        tmp_df = pd.DataFrame()
        tmp_df['ID'] = ts_idx  
        
        ## create time window columns 
        for tw in time_window:

            date_ix = df_buffer.index 
            
            if tw == 'all':
                df_buffer['slice'] = 0 
                                   
            if tw == 'annual':
                df_buffer['slice'] = date_ix.year 
            
            if tw == 'seasonal':
                time_windows = dict(zip(range(1,13), time_format[tw])) 
                df_buffer['slice'] = date_ix.month.map(time_windows)
                
            if tw == 'monthly':
                df_buffer['slice'] = date_ix.month 

            if tw == 'weekly':
                df_buffer['slice'] = date_ix.isocalendar().week
                
            
            ## go over time windows and calc
            for time_index in df_buffer['slice'].unique(): 
                                
                tw_buffer = df_buffer[ df_buffer['slice'] == time_index][ts_idx] 
                
                if tw == 'all':
                    result_name = 'all' 
                else:
                    result_name = '{}_{}'.format(tw, time_index)
                
                for feature in features:
                    
                    if feature in stat_features:
                        print('[CALC] ', feature) 
                                                
                        ## get expected column names 
                        return_cols = func_dict[feature]['cols'] 
            
                        ## calculate statistics 
                        cdf = tw_buffer.apply(func_dict[feature]['func']) 
                                                
                        ## add to results 
                        for i, col in enumerate(return_cols):
                            col_name = col.format( result_name )
                            tmp_df[ col_name ] = cdf.loc[i,: ].values 
                            
                            
                    if feature in corr_features: 
                        print('[CALC] ', feature) 
                        
                        if 'n-acorr' in feature:
                            
                            ## get expected column names 
                            return_cols = func_dict[feature]['cols'] 
                            
                            ## loop throug given lag times 
                            for i, lag in enumerate(n_alag):

                                ## check if lag time smaller than total timeseries time 
                                if lag < len(tw_buffer):
                                    cdf = tw_buffer.apply( func_dict[feature]['func'], lag = lag ) 
                                    
                                    for j, col in enumerate(return_cols):
                                        col_name = col.format(lag, result_name) 
                                        tmp_df[ col_name ] = cdf.values 
                        
                        if 'n-ccorr' in feature:
                            return_cols = func_dict[feature]['cols'][0] 
                            
                            for i, lag in enumerate(n_clag):
                                
                                col_name = return_cols.format( lag, result_name )
                                                                
                                if lag < len(tw_buffer):
                                    
                                    for gauge_id in ts_idx:
                                        cross_corr = func_dict[feature]['func']( tw_buffer[gauge_col], tw_buffer[gauge_id], lag=lag  ) 
                                        tmp_df.loc[ tmp_df['ID'] == gauge_id, col_name ] = cross_corr 

                    if feature in fdc_features:
                        print('[CALC] ', feature) 
                        
                        if 'fdc-q' in feature: 
                            
                            try:
                                cdf = tw_buffer.apply(func_dict[feature]['func'], fdc_q = fdc_q) 
                            except:
                                cdf = None
                            
                            for i, q in enumerate(fdc_q ):
                                col_name = func_dict[feature]['cols'][0].format(q, result_name) 
                                if cdf is not None:
                                    tmp_df[col_name] = cdf.loc[i,:].values
                        
                        else:
                            return_col = func_dict[feature]['cols'][0].format(result_name) 

                            try:
                                cdf = tw_buffer.apply(func_dict[feature]['func'])
                                tmp_df[return_col] = cdf.values 
                            except:
                                cdf = None 
                                                           
                                             
                        
                    if feature in hydro_features:
                        print('[CALC] ', feature)  
                        
                        return_cols = func_dict[feature]['cols'] 
                        cdf = tw_buffer.apply(func_dict[feature]['func'])
                        
                        # try:
                        #     cdf = tw_buffer.apply(func_dict[feature]['func'])
                        # except:
                        #     cdf = None
                        
                        if cdf is not None:
                            
                            if len(return_cols) > 1:
                                for i, col in enumerate(return_cols):
                                    col_name = col.format(result_name) 
                                    tmp_df[col_name] = cdf.loc[i,:].values
                            
                            else:
                                col_name = return_cols[0].format(result_name) 
                                tmp_df[col_name] = cdf.values 
                                            
        tmp_df = tmp_df.set_index('ID')    
        df_out = df_out.append(tmp_df)
                
    return df_out

































