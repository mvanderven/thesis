# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:04:24 2021

@author: mvand
"""

#%% Import modules 
import pandas as pd 
from pathlib import Path
import xarray as xr 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

#%% Set data paths 

gauge_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\code\cartesius")
fn_gauges = gauge_dir / 'V1_grdc_efas_selection-cartesius.csv'

efas_dir = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data\EFAS_6h_test") 
efas_file = [file for file in efas_dir.glob('*.nc')][0]


#%% Load data 

df_gauges = pd.read_csv(fn_gauges, index_col=0) 
ds = xr.open_dataset(efas_file)

#%% Define function 


## perform nearest cell searching function to find matching cell to given gauge coordinates
def find_nearest_cell(ds, x_coords, y_coords):
    
    return_coords = []
    assert len(x_coords) == len(y_coords), '[ERROR] x and y coord lists do not match'
    
    for i, x in enumerate(x_coords):
        ## find closest cell center form (gauge_X, gauge_Y)
        cell_search = ds.sel( {'x': x,
                               'y': y_coords[i]},
                               method = 'nearest') 
        
        return_coords.append([cell_search.x.values, cell_search.y.values])
        
    return np.array(return_coords) 


def analyse_cell_distances(df):
    
    df_out = df.copy()
    
    x_coords = df['proj_X'].values 
    y_coords = df['proj_Y'].values     
    
    xy_snapped = find_nearest_cell(ds, x_coords, y_coords)
    
    df_out['x_snap'] = xy_snapped[:,0]
    df_out['y_snap'] = xy_snapped[:,1]    
    
    df_out['d_X'] = df_out['Lisflood_X'] - df_out['x_snap'] 
    df_out['d_Y'] = df_out['Lisflood_Y'] - df_out['y_snap'] 
    
    df_out['d_X_cell'] = df_out['d_X'] / 5000.
    df_out['d_Y_cell'] = df_out['d_Y'] / 5000.
    
    return df_out 

#%% Run 

df_analysis = analyse_cell_distances(df_gauges)

#%% Show description 
print('----- Statistics -----')
print( df_analysis[['d_X_cell', 'd_Y_cell']].describe() ) 

#%% Plot shifts in bins 

bin_min = -260
bin_max = 35
bin_size = 1 

plt.figure(figsize=(8,6))
plt.suptitle('Cell shifts (binsize={})'.format(bin_size))
plt.subplot(121) 
plt.title('X-direction')
plt.yscale('log')
plt.xlim(-260, 35)
df_analysis['d_X_cell'].hist(bins = list(range(bin_min, bin_max, bin_size))) 

plt.subplot(122)
plt.title('Y-direction')
df_analysis['d_Y_cell'].hist(bins = list(range(bin_min, bin_max, bin_size))) 
plt.yscale('log')
plt.xlim(-260, 35)
plt.show()

#%% Iteratively plot n occurences in increasingly larger 'buffer' 

min_buffer_size = 0 
max_buffer_size = 20
delta_b = 5 
vary_buffer_size = list(range(min_buffer_size, max_buffer_size+1, delta_b)) 

size_found = []

show_heatmap = True  

for buffer_size in vary_buffer_size:
    
    ## perform search in df_analysis 
    x_filter = (df_analysis['d_X_cell'] >= -buffer_size) & (df_analysis['d_X_cell'] <= buffer_size)
    y_filter = (df_analysis['d_Y_cell'] >= -buffer_size) & (df_analysis['d_Y_cell'] <= buffer_size)
    
    df_buffer = df_analysis[ x_filter & y_filter ] 
    n_found = len(df_buffer)
    size_found.append(n_found)
    
    if show_heatmap: 
        ## get empty frame resembling buffer 
        n_vals = int( (2*buffer_size) + 1 )
        ar_buffer = np.zeros(( n_vals, n_vals  )) 
        
        
        for i in range(len(df_buffer)):
            
            rel_x = df_buffer.iloc[i]['d_X_cell']
            rel_y = df_buffer.iloc[i]['d_Y_cell']
            
            ar_buffer[ int(buffer_size - rel_y), int(rel_x+buffer_size) ] += 1 

        xy_ticks = np.arange(0, n_vals)
        x_labels = [ str( v - buffer_size) for v in xy_ticks]
        y_labels = [ str(buffer_size - v) for v in xy_ticks]
        
        plt.figure()
        plt.title('Buffersize = {} \n Percentage in buffer: {:.2f}% ({}/{})'.format( buffer_size, 
                                                                            (n_found/len(df_analysis))*100,
                                                                            n_found, len(df_analysis)))
        
        ## show percentages
        sns.heatmap( (ar_buffer/len(df_analysis)) , annot=True, cbar = False, 
                    linewidth=0.5, annot_kws = {"fontsize":8},
                    fmt = '.1%')
        
        ## show numbers 
        # sns.heatmap( ar_buffer , annot=True, cbar = False, 
        #             linewidth=0.5, annot_kws = {"fontsize":8})
        
        plt.xticks(xy_ticks + 0.5, x_labels)
        plt.yticks(xy_ticks + 0.5, y_labels)
        plt.show()

#%% Analyse buffer size and results 
    
plt.figure() 
plt.title('Buffer sensitivity') 
plt.plot(vary_buffer_size, size_found, marker='x')
plt.axhline(len(df_analysis), color = 'k', linestyle='--', linewidth=0.8)
plt.text(x = 0.1, y = len(df_analysis)+10, s = 'Max', color='k') 

for i, n in enumerate(size_found):
    plt.text( vary_buffer_size[i]+0.05, n-25, '{:.2f}%'.format(  (n/len(df_analysis))*100),
              fontsize = 7)

plt.ylim(0,650)
plt.xlim(-0.5, max(vary_buffer_size)+1.5)
plt.xticks( list(range(min_buffer_size, max_buffer_size+2,1))) 
plt.xlabel('buffer size') 
plt.ylabel('n found')
plt.grid()
plt.show() 


    







