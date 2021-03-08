# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:41:43 2021

@author: mvand
"""

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import cartopy as cp 
import cartopy.feature as cfeature
import numpy as np

import thesis_utils as utils 

import seaborn as sns 

marker_styles = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h',
                 'H', 'X', 'D', 'd', 'P', 4, 5, 6, 7, 8, 9, 10, 11]


def dashboard(ds, df, gauge_data, locations, thresh_min = 500, coords=['x', 'y'], var = 'dis24', model = 'efas', cell_size=5000):
    
    ### GAUGE DATA 
    datas = gauge_data.date.unique()  
    max_obs = gauge_data.value.max()
    projs = gauge_data.epsg.unique() 
    gauge_locs = []
    
    for loc in locations:
        x_loc = gauge_data[ gauge_data['loc_id'] == loc]['lon'].unique()[0]
        y_loc = gauge_data[ gauge_data['loc_id'] == loc]['lat'].unique()[0]
                
        gauge_locs.append( [x_loc, y_loc])
    
    gauge_locs = np.array(gauge_locs)
    
    if 'efas' in model:
        ## check projection - 3035 
        if not 3035 in projs:
            gauge_locs_og = np.array(gauge_locs)            
            gauge_locs = utils.reproject_coordinates(gauge_locs_og, int(projs[0]), 3035)
    
    ### MODEL DATA - display 
    base_layer = np.log( ds[var].mean(dim='time') )
    X = ds[coords[0]].values 
    Y = ds[coords[1]].values 
    ds = None 
    
    ### BUFFER ITERATIONS  
    n_iters = df['iter_id'].unique()
    xy_iters = [] 
    lonlat_iters = [] 
    
    x_iters = [] 
    y_iters = []
    lat_iters = []
    lon_iters = []

    ### add buffer iterations 
    for loc in locations:
        df_iter = df[ df['match_gauge'] == loc]
        var_iter = len(df_iter[coords[0]].unique())
        
        for it in n_iters:
            subset = df_iter[ df_iter['iter_id'] == it]
            
            if 'efas' in model:
                x_iter = subset['x'].unique()[0]
                y_iter = subset['y'].unique()[0]
                
                xy_iters.append([x_iter, y_iter])
            
            lon_iter = subset['lon'].unique()[0]
            lat_iter = subset['lat'].unique()[0]
            lonlat_iters.append([lon_iter, lat_iter])
    
    xy_iters = np.array(xy_iters)
    lonlat_iters = np.array(lonlat_iters)
    
    ## determine window 
    if 'efas' in model:
        x_min_zoom, x_max_zoom = np.min(xy_iters[:,0]), np.max(xy_iters[:,0])
        y_min_zoom, y_max_zoom = np.min(xy_iters[:,1]), np.max(xy_iters[:,1])
        zoom_factor = 0.9

    x_min, x_max = np.min(lonlat_iters[:,0]), np.max(lonlat_iters[:,0])
    y_min, y_max = np.min(lonlat_iters[:,1]), np.max(lonlat_iters[:,1])
    scale_factor = 0.5
        
    
    ### PLOT 
    fig1 = plt.figure(figsize=(10,4))
    gs = fig1.add_gridspec(1,5)    
    
    
    ### OVERVIEW PLOT 
    ax1 = fig1.add_subplot(gs[0, :2], projection = cp.crs.PlateCarree())
    ax1.set_title('Case overview')
    
    ## imagery
    rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')
    # ax1.stock_img() 
    ax1.add_feature(cp.feature.OCEAN)
    ax1.add_feature(cp.feature.LAND, edgecolor='black')
    ax1.add_feature(cp.feature.BORDERS)
    ax1.add_feature(rivers_50m, facecolor='None', edgecolor='b')
    
    
    if 'efas' in model:
        ax1.scatter(gauge_locs_og[:,0], gauge_locs_og[:,1], color='r')
    else:
        ax1.scatter(gauge_locs[:,0], gauge_locs[:,1], color='r')
    
    ### plot in degrees - set zoomed out window 
    wd = 25.
    ax1.set_xlim( max(-180, x_min - wd), min(180, x_max + wd) )
    ax1.set_ylim( max(-90, y_min - wd),  min(90, y_max + wd) )
    
    ### add degree locators 
    ax1.gridlines(draw_labels=True)
    
    #### ZOOM PLOT 
    ax2 = fig1.add_subplot(gs[0,2:], projection = cp.crs.Mercator())
    ax2.set_title('Model grid zoom')
    
    ## imagery --> use xarray functionality, faster than matplotlib pcolormesh(?)
    base_layer.plot(ax=ax2, transform=cp.crs.Mercator(), vmin=0, vmax= np.log(max_obs),
                    cmap = 'terrain', edgecolor='white')
    # ax2.pcolormesh(X, Y, base_layer, shading='nearest', edgecolor='white',
    #                vmin=0, vmax=0.9*max_obs, cmap = 'cubehelix',
    #                linewidth = 0.5)
    
    
    ## plot iterators 
    colors = plt.cm.autumn(np.linspace(0,1, len(locations))) 
    marker_style_ix = np.random.randint(low=0, high=len(marker_styles)-1, size=(len(locations)) )
    
    offset_x = (0.5*cell_size) / len(locations)
    offset_spacing = np.linspace( -offset_x, offset_x, len(locations))
    
    for i in range(len(locations)):
        loc = locations[i]
        
        df_iter = df[df['match_gauge']==loc]
        
        
        for n_it in n_iters:
            subset = df_iter[ df_iter['iter_id'] == n_it]
            
            if 'efas' in model:
                x_iter = subset['x'].unique()[0]
                y_iter = subset['y'].unique()[0]
            
            else:
                x_iter = subset['lon'].unique()[0]
                y_iter = subset['lat'].unique()[0]
                
            
            ax2.scatter(x_iter + offset_spacing[i], y_iter, color=colors[i], s = 50, alpha=0.7,
                        marker = marker_styles[marker_style_ix[i]], edgecolor='w',
                        linewidths = 0.8)
        
        ax2.scatter(gauge_locs[i,0], gauge_locs[i,1], 
                    color=colors[i], marker='X', s = 100, edgecolor='k',
                    label = loc.capitalize())
        
    ## set window 
    if 'efas' in model:
        ax2.set_xlim(x_min_zoom-10000, x_max_zoom+10000)
        ax2.set_ylim(y_min_zoom-10000, y_max_zoom+10000)
    
    else:
        ax2.set_xlim(x_min-0.2, x_max+0.2)
        ax2.set_ylim(y_min-0.2, y_max+0.2)
    
    fig1.subplots_adjust(right=0.85) 
    ax2.legend(bbox_to_anchor=(1.25,1.), loc='upper left', borderaxespad=0., title='Gauges', numpoints=1)
    
    ##### TIMESERIES PLOT 
    fig2 = plt.figure(figsize=(12,8))      
    gs = fig2.add_gridspec(int(len(locations)), 5 )
    
    for i in range(len(locations)):        
        ax = fig2.add_subplot(gs[i, :4])
        
        loc = locations[i]
    
        ts_obs = gauge_data[gauge_data['loc_id'] == loc]['value'].values
        obs_mean = np.mean(ts_obs)
        ax.plot(datas, ts_obs, color=colors[i], linestyle='-.', label= 'Gauge {}'.format(loc.capitalize()))
        
        sub_df = df[ df['match_gauge'] == loc]
        
        for n_it in n_iters:
            ts_sim = sub_df[ sub_df['iter_id'] == n_it][var].values 
            data_sim = sub_df.index.unique() 
            sim_mean = np.mean(ts_sim)

            if (sim_mean/obs_mean) > 0.1 and (sim_mean/obs_mean) <= 2:
                ax.plot(data_sim, ts_sim, color = colors[i], linestyle=':', linewidth = 0.8, label = '{}'.format(n_it))
            
            
        ax.legend(bbox_to_anchor=(1.02,1.), loc='upper left')
        ax.set_xlim(datas[0], datas[-1])
        ax.grid()
        ax.set_ylabel('Average 24hr discharge [m3/s]')
        ax.set_xlabel('Date')
        
    plt.tight_layout()
    return 



def display_cross_correlation(df, cols, 
                              vmin = -1., vmax = 1., center = 0.,
                              cmap = sns.diverging_palette(220, 20, n=100) ):
    
    calc_corr = df[cols].corr() 
    
    fig, ax = plt.subplots(figsize=(14,9))
    
    ax = sns.heatmap( 
        calc_corr,
        vmin = vmin, vmax = vmax, center = center,
        cmap = cmap,
        square = True,
        annot = True,
        annot_kws={"size":4},
        fmt = '.1g',
        cbar=False
        )
    
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 60,
        horizontalalignment='right')
            
    return 



