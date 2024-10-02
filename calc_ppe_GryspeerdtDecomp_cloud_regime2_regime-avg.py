#!/usr/bin/env python
# coding: utf-8

# # import libaries

# In[1]:


import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import sys
import os
import json
import pandas as pd 
import seaborn as sns 
import pickle

import matplotlib.patches as mpatches
import textwrap
import cartopy.crs as ccrs
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

import utils_v1v2 as v1v2
import utils_PPE as misc 
from matplotlib.gridspec import GridSpec


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 6.5
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 7 

#workdir = '/qfs/people/qiny108/ppe_scripts/'
workdir = '/compyfs/qiny108/home_dir_bigdata/ppe_scripts/'

#regime_method_tag = 'new'
#regime_method_tag = 'OceanOnly' # ocean only
regime_method_tag = 'OceanOnly30SN' # ocean only

if regime_method_tag == 'new':
    middir = workdir+'middata/regime_partition_m2/'
elif regime_method_tag in ['OceanOnly','OceanOnly30SN']:
    middir = workdir+'middata/regime_partition_'+regime_method_tag+'/'
else:
    print('Check your regime_method_tag!')
    exit()

figdir = workdir+'figure/'
datadir_fbk = '/qfs/people/qiny108/diag_feedback_E3SM/data_ppe_2010to2012/'

datadir = '/qfs/people/qiny108/ppe_scripts/decomp_ERFaci_Gryspeerdt/data/' 

lat_spc = np.arange(-90,90,2.5)
lon_spc = np.arange(0,360,2.5) 
print(f'lat_spc.shape={lat_spc.shape}, lon_spc.shape={lon_spc.shape}') 

from datetime import date
today = date.today()
dd = today.strftime("%Y-%m-%d")
print(dd)

import importlib
importlib.reload(v1v2) 
importlib.reload(misc) 


# # Define functions

# In[ ]:





# # DEFINE cases

# In[2]:


from utils_PPE import ctlc, noc, prc, mixc, turbc, deepc, aerc

regions = misc.define_regions() 
print('regions=',regions)
print()
catags = misc.define_catags() 
print(f'catags = {catags}') 

casesA = [
    ['BASE', ctlc],
    ['nomincdnc', noc],
    ['nomincdnc_2',noc],
    ['nomincdnc_3',noc],
    ['prc_exp1', prc],
    ['prc_exp1_2',prc], 
    ['prc_exp1_3',prc],
    # ['prc_exp', prc], 
    ['prc_exp_2',prc],
    ['prc_exp_3',prc],
    ['prc_coef1', prc],
    ['prc_coef1_2',prc], 
    ['prc_coef1_3',prc],
    ['accre_enhan', prc],
    ['accre_enhan_2',prc],  
    ['accre_enhan_3',prc], 
    ['berg', mixc],
    ['berg_2',mixc],
    ['berg_3',mixc],
    ['c1', turbc],
    ['c1_2',turbc],
    ['c1_3',turbc],
    ['c1_4',turbc], 
    ['gamma_coef', turbc],
    ['gamma_coef_2',turbc], 
    ['gamma_coef_3',turbc],
    ['gamma_coef_4',turbc], 
    ['gamma_coefb',turbc],
    ['gamma_coefb_2',turbc],
    ['gamma_coefb_3',turbc], 
    ['c8', turbc],
    ['c8_2',turbc],
    ['c8_3',turbc],
    ['c8_4',turbc], 
    ['wsub',aerc],
    ['wsub_2',aerc],
    ['wsub_3',aerc], 
    ['wsub_4',aerc], 
    ['clubb_tk1', mixc],
    ['clubb_tk1_2',mixc],
    ['clubb_tk1_3',mixc], 
    ['ice_deep', deepc],
    ['ice_deep_2',deepc], 
    ['ice_deep_3',deepc],
    ['ice_sed_ai', deepc],
    ['ice_sed_ai_2',deepc],
    ['ice_sed_ai_3',deepc],
    ['so4_sz', deepc],
    ['so4_sz_2',deepc], 
    ['so4_sz_3',deepc],
    ['dp1', deepc],
    ['dp1_2',deepc], 
    ['dp1_3',deepc],
    # ['nomincdnc.prc_exp1_2',prc], 
    # ['nomincdnc.prc_exp1_3',prc], 
    # ['nomincdnc.prc_exp_2',prc],
    # ['nomincdnc.prc_exp_3',prc],
    # ['nomincdnc.prc_coef1',prc],
    # ['nomincdnc.prc_coef1_3',prc], 
    # ['nomincdnc.berg',mixc],
    # ['nomincdnc.clubb_tk1_3',mixc],
    # ['nomincdnc.ice_deep_2',deepc],
    # ['nomincdnc.dp1_3',deepc],
    # ['nomincdnc.ice_sed_ai_3',deepc],
    # ['nomincdnc.so4_sz_3',deepc],
    ['BASE_01', ctlc],
    ['BASE_02', ctlc],
    ['BASE_03', ctlc],
    ]

cases = [case[0] for case in casesA]
colorsh = [case[1] for case in casesA]

print(len(cases),cases)


# # DEFINE parameter group

# In[4]:


Examine_minCDNC_PPEs = False

# ========================================== 
if Examine_minCDNC_PPEs: 
    exp_list = [
                # 'mincdnc',
                # 'prc_exp1','prc_exp','prc_coef1',
                # 'accre_enhan',
                # 'c1', 'gamma_coef','c8','wsub',
                # 'ice_deep','ice_sed_ai','so4_sz','dp1',
                # 'berg','clubb_tk1',
                'nomincdnc.prc_exp1', 'nomincdnc.prc_exp', 'nomincdnc.prc_coef1', 
                'nomincdnc.ice_deep','nomincdnc.ice_sed_ai','nomincdnc.so4_sz','nomincdnc.dp1',
                'nomincdnc.berg','nomincdnc.clubb_tk1',
                ]
else:
    exp_list = [
#             'mincdnc',
#             'prc_exp1','prc_exp','prc_coef1','accre_enhan',
            'c1', 'gamma_coef','c8','gamma_coefb', 
#             'wsub',
#             'ice_deep','ice_sed_ai','so4_sz','dp1',
#             'berg','clubb_tk1',
            # 'nomincdnc.prc_exp1', 'nomincdnc.prc_exp', 'nomincdnc.prc_coef1', 
            # 'nomincdnc.ice_deep','nomincdnc.ice_sed_ai','nomincdnc.so4_sz','nomincdnc.dp1',
            # 'nomincdnc.berg','nomincdnc.clubb_tk1',
            ]

# ========================================== 
dicc,dicr = misc.get_param_group_dic(exp_list) 

print(len(dicc.keys()),dicc.keys())
cases_t = [] 
for key in dicc.keys():
    for ii in range(len(dicc[key])): 
        tmp = dicc[key][ii][0]
        if tmp not in cases_t: 
            cases_t.append(tmp)

for tt in cases:
    if tt not in cases_t: 
        print(tt, 'is in cases, but not in cases_t.') 

print('total PPE count=',len(cases_t), 'len(cases)=',len(cases)) 

# # Cloud regime decomposition using Qin22

# ## Get ensemble mean of omega

# In[6]:


fomega = 'omega700_mm.nc'

if os.path.isfile(fomega): 
    ff = xr.open_dataset(fomega)
    omega700_pi_mm  = ff['omega700_pi_mm']
    omega700_ab_mm  = ff['omega700_ab_mm']
    omega700_avg_mm = ff['omega700_avg_mm']
    ff.close()

    print('Read available data: ',omega700_pi_mm.shape)

else: 
    Vars = ['OMEGA700']
    dicr = {}

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Temporary use cases with avaiable omega700
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for case in cases: 
        print(f'case = {case}') 
        
        dicr[case] = {}
        for var in Vars:
            # ============= read data ==================================
            fname = 'global_'+var+'_'+case+'.nc'

            f = xr.open_dataset(datadir_fbk+fname)
            data1 = f[var+'_pi_clim']
            data2 = f[var+'_ab_clim']
            f.close()

            # ensure the lat and lon names 
            latnew, lonnew = list(data1.coords.keys())[0], list(data1.coords.keys())[1]
            data1 = data1.rename({lonnew: 'lon',latnew: 'lat'})

            latnew, lonnew = list(data2.coords.keys())[0], list(data2.coords.keys())[1]
            data2 = data2.rename({lonnew: 'lon',latnew: 'lat'})

            if var == 'OMEGA700':
                scale = 864.
            else:
                scale = 1.0 
            dicr[case][var+'_pi'] = data1*scale
            dicr[case][var+'_ab'] = data2*scale  
            dicr[case][var+'_avg'] = (data1*scale+data2*scale)/2. 

    # =========== get ensemble mean
    omega700_pi_mm = xr.concat([dicr[case]['OMEGA700_pi'] for case in dicr.keys()],dim="case_dim").mean(axis=0)
    omega700_ab_mm = xr.concat([dicr[case]['OMEGA700_ab'] for case in dicr.keys()],dim="case_dim").mean(axis=0)
    omega700_avg_mm = xr.concat([dicr[case]['OMEGA700_avg'] for case in dicr.keys()],dim="case_dim").mean(axis=0)

    print(omega700_pi_mm.shape)

    # save to NC file 
    # ------------------------------------------------
    dicout = {}
    dicout['omega700_pi_mm'] = omega700_pi_mm
    dicout['omega700_ab_mm'] = omega700_ab_mm
    dicout['omega700_avg_mm'] = omega700_avg_mm

    v1v2.save_big_dataset(dicout,'omega700_mm.nc')
    

# ## DEFINE regimes and data source

# In[8]:


what_state = 'avgCTLP4K' # 'CTL','P4K','avgCTLP4K'

regimes = ['TropMarineLow','TropAscent','TropLand','NH-MidLat-Lnd','NH-MidLat-Ocn','SH-MidLat','HiLat','Global']
colorsh = v1v2.get_color('tab10',len(regimes))
regimes_abbr = ['TML','TA','TLD','NML','NMO','SML','HIL','GL']   


if regime_method_tag in ['OceanOnly','OceanOnly30SN']:
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','HiLat-Ocn', 'Global-Ocn']
    regimes_abbr = ['TML', 'TA', 'NMO', 'SMO', 'HLO', 'GLO']

    
outfile1 = 'dics_Gryspeerdt_Decomp_regime-avg_'+regime_method_tag+'.pkl'
print(f'outfile1={outfile1}')
print()

if os.path.isfile(middir+outfile1):
    print(f'{outfile1} already exists.')
    #continue

# ## Read Gryspperdt decomposition variables and do partitioning
datasource = 'Gryspeerdt_decomp'

dics_out = {} 
dics_out[datasource] = {}

dics1 = {}
for case in cases_t+['BASE']: 
    
    # ============= read data ==================================
    fname = datadir+'forcing_'+case+'_inative_retdata_RegAfFilter_ntime-2721_0425.nc'

    if not os.path.isfile(fname):
        print('Not found data for', case)
        continue 

    f1 = xr.open_dataset(fname) 

    dics1[case] = {}
    for svar in list(f1.data_vars): 
        data = f1[svar]

        # ensure the lat and lon names
        latnew, lonnew = list(data.coords.keys())[0], list(data.coords.keys())[1]
        data = data.rename({lonnew: 'lon',latnew: 'lat'})
        data = data.interp(lat=lat_spc, lon=lon_spc) 

        print(case, svar)
        
        # get OMEGA700
        omega700_pi = omega700_pi_mm.interp(lat=lat_spc,lon=lon_spc) 
        omega700_ab = omega700_ab_mm.interp(lat=lat_spc,lon=lon_spc)
        omega700_avg = omega700_avg_mm.interp(lat=lat_spc,lon=lon_spc) 
        
        regsum_pi = 0
        regsum_ab = 0 
        regsum_avg = 0 
        dics1[case][svar] = {}
        
        for ireg,reg in enumerate(regimes):
            ## do regime partitioning 
            if regime_method_tag == 'new':
                data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_2(reg,omega700_pi,omega700_ab,omega700_avg,data)
            elif regime_method_tag == 'OceanOnly':
                data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_3(reg,omega700_pi,omega700_ab,omega700_avg,data)
            elif regime_method_tag == 'OceanOnly30SN':
                data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_4(reg,omega700_pi,omega700_ab,omega700_avg,data)
            else:
                data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning(reg,omega700_pi,omega700_ab,omega700_avg,data)
           
            # ================ fractional area ============================         
            # check fractional area to ensure the sum equals to 1. 
            regsum_pi += v1v2.area_averager(data2m_pi).values
            regsum_ab += v1v2.area_averager(data2m_ab).values
            regsum_avg += v1v2.area_averager(data2m_avg).values
            
            if reg == 'Global': 
                print(reg, 
                    'pi_avg=',v1v2.area_averager(data2m_pi).values, 
                    'ab_avg=',v1v2.area_averager(data2m_ab).values, 
                    'avg_avg=',v1v2.area_averager(data2m_avg).values,
                    'cum pi_avg=',regsum_pi, 'cum ab_avg=',regsum_ab, 'cum avg_avg=',regsum_avg)
            
            dics1[case][svar][reg+'_pi'] = data2_pi
            dics1[case][svar][reg+'_ab'] = data2_ab
            dics1[case][svar][reg+'_avg'] = data2_avg
            dics1[case][svar][reg+'_cmb'] = xr.where((~np.isnan(data2_pi))&(~np.isnan(data2_ab)),data2_pi,0.0)
            dics1[case][svar][reg+'_frc_pi'] = data2m_pi
            dics1[case][svar][reg+'_frc_ab'] = data2m_ab
            dics1[case][svar][reg+'_frc_avg'] = data2m_avg
            
        print()
        
        if case != cases[0]:
            continue 

       
        # ========== plot regime map =================================
        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        ax.set_title(case) 
        
        ii = 0 
        patches = [] # for legend
        checksum = 0
        transitionsum = 0 
        for reg in regimes[:-1]: # ignore Global
            reg1 = reg+'_frc_pi'
            reg2 = reg+'_frc_ab'
            reg3 = reg+'_frc_avg'
            
            datap1_avg = v1v2.area_averager(dics1[case][svar][reg1]).values*100.
            datap2_avg = v1v2.area_averager(dics1[case][svar][reg2]).values*100.
            datap3_avg = v1v2.area_averager(dics1[case][svar][reg3]).values*100.
            # mask zero by nan 
            datap1 = xr.where(dics1[case][svar][reg1]==0,np.nan,1.0) 
            datap2 = xr.where(dics1[case][svar][reg2]==0,np.nan,1.0)
            datap3 = xr.where(dics1[case][svar][reg3]==0,np.nan,1.0)
            
            if what_state != 'avgCTLP4K':
                datap = xr.where((~np.isnan(datap1))&(~np.isnan(datap2)),1.0,0.0)
            else:
                datap = dics1[case][svar][reg3]
                
            datap_avg = v1v2.area_averager(datap).values*100. 
            
            checksum += datap_avg
            # print(reg, 'avg=',datap_avg, 'cum=',checksum)
            
            lon2d,lat2d = np.meshgrid(datap1.lon,datap1.lat)
            ax.scatter(lon2d,lat2d,datap*10.0,color=colorsh[ii],transform=ccrs.PlateCarree(),
                    )
            ax.coastlines()
            ax.set_global()
            
            patch = mpatches.Patch(color=colorsh[ii], label=reg.split('_')[0]+' ['+str(datap_avg.round(1))+'%]')
            patches.append(patch)

            if what_state != 'avgCTLP4K':
                # transition regime...
                datapp = xr.where((~np.isnan(datap1))&(datap==0),1.0,0.0)
                ax.scatter(lon2d,lat2d,datapp*1.0,color='grey',transform=ccrs.PlateCarree(),
                        )
                # print(v1v2.area_averager(datapp).values)
                transitionsum += v1v2.area_averager(datapp).values 

            ii += 1
        
        if what_state != 'avgCTLP4K':
            print(transitionsum)
            transitionsum = transitionsum *100 
            patch = mpatches.Patch(color='grey', label='transition regime'+' ['+str(transitionsum.round(1))+'%]')
            patches.append(patch)
            
        plt.legend(handles=patches,bbox_to_anchor=(1.04,0), loc='lower left')
        
        # if case == 'v2.OutTend':
        #     print('Figure Saved. Congrats!')
        #     fig.savefig(figdir+'LatLon_FixedRegimeNoEIS_regime_map_'+case+'_'+dd+'.png',dpi=300,bbox_inches='tight')


dics_out[datasource] = dics1 

print(len(dics_out.keys()), dics_out.keys(),) 


# In[9]:


with open(middir+outfile1, 'wb') as f:
    pickle.dump(dics_out, f) 
print('Done regime decomposition for Gryspeerdt decomposition')

