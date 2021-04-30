# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:02:05 2021

@author: mvand
"""

import pandas as pd 
import numpy as np 
from pathlib import Path
from scipy import stats
import dask.dataframe as dd 

##########################################
####       SUPPORT FUNCTIONS          #### 
##########################################

def read_gauge_data(fn_list, transform = False, src_proj = None, dst_proj = None, resample24hr=False):
    
    '''
    Function that reads different types of gauge data - 
    and returns a dataframe with same set/order of columns:
    [loc_id, quantity, unit, date, time, value, epsg, x, y]
    
    fn         path to (csv) datafile 
    dtype      datasource - file will be opened accordingly 
    transform  transform coordinate system 
    '''
    
    out_df = None
    meta = {}

    output_cols = ['date', 'time', 'value', 'quantity', 
                   'epsg', 'nan_val', 'loc_id', 'y', 
                   'x', 'upArea', 'unit']
    
    out_df = pd.DataFrame(columns=output_cols)
        
    for fn in fn_list:   
        
        assert Path(fn).exists(), '[ERROR] file not found'
        
        ## get gauge id nr from filename             
        gauge_id_nr = fn.name.split('_')[0]
        
        ## open file 
        ##  other options for encoding:
        ## encoding = 'ascii'  # encoding = 'mbcs' # encoding = 'ansi'
        df = pd.read_csv(fn, skiprows=36, delimiter=';', encoding = 'ansi') 
    
        ## extract data 
        discharge = df[' Value'].values 
        dt_dates = pd.to_datetime(df['YYYY-MM-DD'], yearfirst=True, 
                                  format='%Y-%m-%d')

        ## what time
        dt_time =  pd.Series(['00:00:00']*len(dt_dates))
        # dt_time =  pd.Series(['12:00:00']*len(dt_dates))

        ## create output dataframe 
        temp_df = pd.DataFrame( {'date':dt_dates, 'time':dt_time, 
                                'value':discharge})

        ## get metadata 
        byte_df = open(fn)
        lines = byte_df.readlines()[:36]
        
        temp_df['loc_id'] = gauge_id_nr 
        meta['loc_id'] = gauge_id_nr 
        
        for line in lines:
            vals = line.split(' ')

            if 'Station:' in vals:
                meta['loc_name'] = vals[-1].replace('\n', '').lower()
                temp_df['loc_id_name'] = meta['loc_name']

            if 'missing' in vals:
                meta['nan'] = float(vals[-1])
                temp_df['nan_val'] = float(vals[-1])

            if 'Latitude' in vals:
                meta['lat'] = float(vals[-1])
                # temp_df['y'] = meta['lat']
                temp_df['lat'] = meta['lat']

            if 'Longitude' in vals:
                meta['lon'] = float(vals[-1])
                # temp_df['x'] = meta['lon']
                temp_df['lon'] = meta['lon']

            if 'Unit' in vals:
                meta['unit'] = vals[-1].replace('\n', '')
                temp_df['unit'] = meta['unit']

            if 'area' in vals:
                meta['upArea(km2)'] = float(vals[-1])
                temp_df['upArea'] = meta['upArea(km2)']

            if 'Content:' in vals:
                meta['description'] = ' '.join( vals[-3:] ).replace('\n', '').lower()

            temp_df['quantity'] = 'Q'
            temp_df['epsg'] = 4326

            # meta['start_date'] = str(dt_dates.iloc[0])
            # meta['end_date'] = str(dt_dates.iloc[-1])

        ## set nan values 
        temp_df.loc[ temp_df['value'] == meta['nan'], 'value' ] = np.nan        

        ## close meta file 
        byte_df.close()
        lines = None 

        ## aggregate date and time - set as index 
        ## set format same as glofas/efas date format 
        temp_df.index = pd.to_datetime( temp_df['date'],
                                            format='%Y-%m-%dT%H:%M:%S.%f' 
                                           ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        out_df = out_df.append(temp_df)
    return out_df, meta

def rows_to_cols(df, id_col, index_col, target_col, resample_24hr = False):
    
    out_df = pd.DataFrame()
    
    unique_ids = df[id_col].unique() 
    
    for target_id in unique_ids:
        
        _df = df[ df[id_col] == target_id ] 
        col_names = _df.index.unique() 
        
        for col_name in col_names:
            col_df = _df[ _df.index == col_name ] 
            col_df = col_df.set_index(index_col) 
            
            out_df[col_name] = col_df[target_col]
    
    if resample_24hr:
        out_df.index = pd.to_datetime( out_df.index  )
        out_df = out_df.resample('D').mean()
        
    out_df.index = pd.to_datetime(out_df.index)    
    return out_df


##########################################
####       FEATURE FUNCTIONS          #### 
##########################################

#### FEATURE TYPES 
sorted_features = {
    # 'stats':        ['normal', 'log', 'gev', 'gamma', 'poisson'],
    'stats':        ['normal', 'log', 'gev', 'gamma'],
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
    
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]
    
    mu = np.mean(ts)
    sigma = np.std(ts) 
        
    ## calculate goodness of fit 
    ## create an artificial dataset based on
    ## derived mu and sigma
    try:
        gof = calc_gof(ts, stats.norm.rvs(loc=mu, scale=sigma, size=len(ts)))
    except:
        gof = 0
    return [mu, sigma, gof]

def calc_distr_log(ts, eps=1e-6):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]
    
    ## transform 
    Y = np.log(ts+eps) 
    Y_mu = np.mean(Y)
    Y_sigma = np.std(Y)
        
    ## calculate goodness-of-fit 
    ## create an artificial dataset
    ## based on derived distribution
    ## parameters 
    try:
        gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma, scale=np.exp(Y_mu), size=len(ts))) 
    except:
        gof = 0

    ## alternative transformation - similar results  
    # Y_mu_alt = np.log( np.mean(ts)**2 / (( np.mean(ts)**2 + np.std(ts)**2 )**0.5 )  )
    # Y_sigma_alt = (np.log( 1 + ((np.std(ts)/np.mean(ts))**2) ))**0.5
    # gof = calc_gof(ts, stats.lognorm.rvs(s=Y_sigma_alt, scale=np.exp(Y_mu_alt), size=len(ts))) 

    return [Y_mu, Y_sigma, gof]

def calc_distr_gev(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan]    
    
    ## calculate gev parameters 
    ## gumbel, so k = 0
    try:
        a = np.pi / ((6**0.5)*np.std(ts))
        u = np.mean(ts) - (0.577/a)
    
        ## calculate goodness of fit 
        ## create an artificial dataset based on
        ## derived a and u 
        gof = calc_gof(ts, stats.genextreme.rvs(c=0, loc = u, scale = a, size=len(ts))) 
    except:
        return [np.nan, np.nan, np.nan]
    return [u,a, gof]

def calc_distr_gamma(ts):
    
    ## drop remaining missing values 
    ts = ts.dropna() 
    if len(ts) == 0:
        return [np.nan, np.nan, np.nan] 
    
    try:
        ## calculate gamma parameters 
        mu = np.mean(ts)
        sigma = np.std(ts)
    
        k = (mu/sigma)**2 
        theta = sigma / (mu/sigma)
        
        ## calculate goodness of fit 
        ## create an artificial dataset based on
        ## derived k and theta (rvs)
        gof = calc_gof(ts, stats.gamma.rvs(a = k, scale = theta, size=len(ts)))
    except:
        return [np.nan, np.nan, np.nan] 
    
    return [k, theta, gof]

def calc_distr_poisson(ts):
    return [np.nan, np.nan]


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

    if len(ts_drop) == 0:
        return int(0), int(0)
    
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
    return prob, ts_sorted.values  


def calc_FDC_q(ts, fdc_q):
    return_q = []
    
    ## calc FDC curve 
    fdc_prob, fdc_val = calc_FDC(ts)
    
    if ( type(fdc_prob) == int) or (type(fdc_val) == int):
        return [np.nan]*len(fdc_q)
    
    
    ## find corresponding quantiles 
    for q in fdc_q:        
        ix_nearest = ( np.abs(fdc_prob - q) ).argmin() 
        return_q.append(fdc_val[ix_nearest])  
        
    # print( len(ts), len(return_q), type(return_q))
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
        if len(ts) == ts.isnull().sum():
            return np.nan 
        else:
            ts = ts.fillna(method='ffill').dropna() 
            
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
    if sum(Q_B) > 0 and sum(Q_t) > 0:
        return sum(Q_B) / sum(Q_t)
    else:
        return np.nan 


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


##########################################
####            overview              #### 
##########################################
 
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



def calc_signatures(df_observations, df_simulations, id_col = 'loc_id',
                    features = feature_options, time_window = option_time_window,
                    fdc_q = [1, 2, 5, 10, 50, 90, 95, 99],
                    n_alag = [1], n_clag = [0,1]):
        
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
                            
                            cdf = tw_buffer.apply(func_dict[feature]['func'], fdc_q = fdc_q) 
                            
                            for i, q in enumerate(fdc_q ):
                                col_name = func_dict[feature]['cols'][0].format(q, result_name) 
                                
                                try:                                
                                    tmp_df[col_name] = cdf.loc[i,:].values
                                except:
                                    ## bug in some fdc-q 
                                    ## cdf returns tw_buffer (with date index) 
                                    ## bug seems fixed now -->
                                    ##      as both fdc_q and length of
                                    ##      tw_buffer were 7, tw_buffer
                                    ##      returned as cdf instead of
                                    ##      cdf with apply function 
                                    ##      ??? 
                                    print('failed: ', col_name)
                                    col_name = None

                        else:
                            return_col = func_dict[feature]['cols'][0].format(result_name) 
                            
                            cdf = tw_buffer.apply(func_dict[feature]['func'])
                            tmp_df[return_col] = cdf.values 
                            
                                                           
                                             
                        
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



def pa_calc_signatures(gauge_id, fn_simulations, gauge_dir): 
        
    ## GET SIMULATION DATA
    ## load simulation data 
    df_model = dd.read_csv(fn_simulations) 

    ## select buffer corresponding with gauge & load into memory 
    df_model = df_model[ df_model['gauge'] == gauge_id]
    df_model = df_model.set_index('ID') 
    
    ## load data into memory 
    df_model = df_model.compute() 
    
    ## LOAD GAUGE DATA 
    gauge_fn = gauge_dir / '{}_Q_Day.Cmd.txt'.format(gauge_id)
    
    if gauge_fn.exists():
        df_gauge, meta = read_gauge_data([gauge_fn]) 
        
        # df_gauge = df_gauge[ (df_gauge['date'] >= '1991') &  (df_gauge['date'] < '2020')].copy()
        ## TEST 
        df_gauge = df_gauge[ (df_gauge['date'] >= '1991') &  (df_gauge['date'] < '1994')].copy()
        
        df_simulations = rows_to_cols(df_model, 'gauge', 'time', 'dis24')

        out_df = calc_signatures(df_gauge, df_simulations, time_window=['all'])
        return out_df
    
    else:
        return None 









