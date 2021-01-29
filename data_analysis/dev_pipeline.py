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













