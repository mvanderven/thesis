# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:00:54 2021

@author: mvand
"""

import numpy as np 
import pandas as pd 
import xarray as xr 
import cfgrib
from pathlib import Path 
from pprint import pprint
import pyproj
import datetime
import matplotlib.pyplot as plt 
import cartopy 


## import own functions  --> figure it out??
## 1) with anaconda prompt first navigate to data analysis directory 

# import sys 
# mod_dir = Path(__file__).parent
# sys.path.append( mod_dir )
# sys.path.append('.')

import dev_utils

#%%

## set data paths 
model_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")
gauge_data = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\gauge_data") 

## model data paths 
model_data_dict = dev_utils.get_file_paths(model_data, 'nc', long_name=True)
keys = list(model_data_dict.keys())

keys_efas = [key for key in keys if 'EFAS' in key]
keys_glofas = [key for key in keys if 'GLOFAS' in key]

## gauge data - GRDC
gauge_data_dict = dev_utils.get_file_paths(gauge_data, 'txt', long_name=True)
gauge_keys = list( gauge_data_dict.keys() )

#%%

## load gauge data 
gauge_file_names = gauge_data_dict[gauge_keys[0]] 
gauge_data_grdc, meta_grdc = dev_utils.read_gauge_data( gauge_file_names, dtype='grdc')
 
gauge_locs = gauge_data_grdc['loc_id'].unique()
lat_coords = gauge_data_grdc['Y'].unique()
lon_coords = gauge_data_grdc['X'].unique() 

coords_4326 = np.array([lon_coords, lat_coords]).transpose()
coords_3035 = dev_utils.reproject_coordinates(coords_4326, 4326, 3035)
print('Gauge data loaded')

#%% load EFAS data 

efas_dir = model_data_dict[keys_efas[2]]
ds_efas = dev_utils.open_efas(efas_dir)
print('EFAS data loaded')

#%% load glofas data 
glofas_dir = model_data_dict[keys_glofas[1]]
ds_glofas = dev_utils.open_glofas(glofas_dir)
print('GLOFAS data loaded')

#%% 










