# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:51:53 2020

@author: mvand
"""

import cdsapi
from pathlib import Path
from calendar import monthrange, month_name 

c = cdsapi.Client()






def download_EFAS_cds(year, month, out_dir, do_download=True):
    #### monthly efas downloads    
    
    year_options = list(range(1991, 2021, 1))
    month_options = list(range(1,13,1))
    
    assert int(year) in year_options, '[ERROR] {} not in EFAS period from 1991-2020'
    assert int(month) in month_options, '[ERROR] {} month does not exist [1-12]'
    
    ### based on year & month, list day 
    n_days = monthrange(int(year), int(month))[1]    
    days = [ str(v) for v in list(range(1, n_days+1, 1))] 

    ### transform month number into name of month 
    month_string = month_name[int(month)].lower() 
    
    ### define download request
    request = {
        'system_version': 'version_4_0',
        'variable': 'river_discharge_in_the_last_6_hours',
        'model_levels': 'surface_level',
        'hyear': str(year),
        'hmonth': month_string,
        'hday': days,
        'time': ['00:00','06:00','12:00','18:00'],
        'format': 'netcdf'
        }    
    
    ### setup output name 
    if month < 10:
        fn = Path(out_dir) / 'EFAS_{}_0{}_6h.nc'.format(year, month)
    else:
        fn = Path(out_dir) / 'EFAS_{}_{}_6h.nc'.format(year, month)
        
    if do_download:
        c.retrieve(
            'efas-historical',
            request,
            fn
            )
    
    return fn 

# total_years = list(range(1991, 2020+1, 1))
total_years = list(range(1991, 1992, 1))
total_months = list(range(1, 12+1, 1))

download_dir_base = Path(r"C:\Users\mvand\Documents\Master EE\Year 4\Thesis\data\model_data")

for year in total_years:
    
    download_dir = download_dir_base / 'EFAS_{}_6h'.format(year)

    if not download_dir.exists():
        download_dir.mkdir() 
    
    for month in total_months:
        
        print(year, month)
        # download_EFAS_cds(year, month, out_dir = download_dir)
        

















