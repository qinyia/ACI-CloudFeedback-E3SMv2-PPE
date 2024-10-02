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

# --- For stratification ---- 
Do_stratification = False
metric_names = [
'LTS',
#'LWP_LS',
#'RELHUM_700to850',
#'WP2_700to850',
#'SKW_ZT_700to850',
#'LHFLX',
#'FREQZM',
] # Must be 2-D variables 
nbins = 5  
# --- 

#regime_method_tag = 'new2'
regime_method_tag = 'OceanOnly' # ocean only
#regime_method_tag = 'LndOcn'
regime_method_tag = 'OceanOnly30SN' # ocean only

if regime_method_tag == 'new2': 
    if Do_stratification: 
        middir = workdir+'middata/regime_partition_m4/'
    else:
        middir = workdir+'middata/regime_partition_m3/'
elif regime_method_tag == 'OceanOnly':
    middir = workdir+'middata/regime_partition_'+regime_method_tag+'/'
elif regime_method_tag == 'LndOcn':
    middir = workdir+'middata/regime_partition_'+regime_method_tag+'/'
elif regime_method_tag == 'OceanOnly30SN':
    middir = workdir+'middata/regime_partition_'+regime_method_tag+'/'
else:
    print('Check your regime_method_tag!')
    exit()

if not os.path.exists(middir):
    os.makedirs(middir)
    print("Directory", middir, "created successfully.")

figdir = workdir+'figure/'
datadir_aer = '/qfs/people/qiny108/colla/diag_ERFaer/data/'
datadir_fbk = '/qfs/people/qiny108/diag_feedback_E3SM/data_ppe_2010to2012/'
datadir_fbk_tmp = '/qfs/people/qiny108/diag_feedback_E3SM/data_ppe_2010to2012/' 

from datetime import date
today = date.today()
dd = today.strftime("%Y-%m-%d")
print(dd)

import importlib
importlib.reload(v1v2) 
importlib.reload(misc) 


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
    ['nomincdnc.prc_exp1_2',prc], 
    ['nomincdnc.prc_exp1_3',prc], 
    ['nomincdnc.prc_exp_2',prc],
    ['nomincdnc.prc_exp_3',prc],
    ['nomincdnc.prc_coef1',prc],
    ['nomincdnc.prc_coef1_3',prc], 
    ['nomincdnc.berg',mixc],
    ['nomincdnc.clubb_tk1_3',mixc],
    ['nomincdnc.ice_deep_2',deepc],
    ['nomincdnc.dp1_3',deepc],
    ['nomincdnc.ice_sed_ai_3',deepc],
    ['nomincdnc.so4_sz_3',deepc],
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
             'mincdnc',
             'prc_exp1','prc_exp','prc_coef1','accre_enhan',
            'c1', 'gamma_coef','c8','gamma_coefb', 
             'wsub',
             'ice_deep','ice_sed_ai','so4_sz','dp1',
             'berg','clubb_tk1',
             'nomincdnc.prc_exp1', 'nomincdnc.prc_exp', 'nomincdnc.prc_coef1', 
             'nomincdnc.ice_deep','nomincdnc.ice_sed_ai','nomincdnc.so4_sz','nomincdnc.dp1',
             'nomincdnc.berg','nomincdnc.clubb_tk1',
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

if regime_method_tag == 'OceanOnly':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','HiLat-Ocn', 'Global-Ocn']

if regime_method_tag == 'LndOcn':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','SH-HiLat-Ocn', 'NH-HiLat-Ocn', 
               'TropLand','SH-MidLat-Lnd','NH-MidLat-Lnd','SH-HiLat-Lnd','NH-HiLat-Lnd',
               'Global-Ocn','Global-Lnd','Global',
               ]
    
if regime_method_tag == 'OceanOnly30SN':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','SH-HiLat-Ocn', 'NH-HiLat-Ocn', 'Global-Ocn']

if Do_stratification:
    comps = ['Stratification']
else:
    comps = [
            'Net', 
            'SW', 
            'LW', 
#            'Net_HI','Net_LO',
#            'SW_HI','SW_LO',
#            'LW_HI','LW_LO',
            ]

for comp in comps: 

    # Define fbk and aer sources 
    if comp == 'Stratification':
        tag = '_'+comp
        fbk_datasources, aer_datasources = ['RK','CRK','CRK_tau'], ['Ghan','CRK','CRK_tau'] 
    elif comp == 'Net':
        tag = ''
        fbk_datasources, aer_datasources = ['RK','CRK','CRK_amt','CRK_tau','CRK_alt','APRP'], ['Ghan','CRK','CRK_amt','CRK_tau','CRK_alt','APRP'] 
    elif comp in ['SW','LW']:
        tag = '_'+comp
        fbk_datasources, aer_datasources = ['RK_'+comp,'CRK_'+comp,'CRK_'+comp+'_amt','CRK_'+comp+'_tau','CRK_'+comp+'_alt','APRP_'+comp], ['Ghan_'+comp,'CRK_'+comp,'CRK_'+comp+'_amt','CRK_'+comp+'_tau','CRK_'+comp+'_alt','APRP_'+comp] 
        if comp == 'SW':
            fbk_datasources.extend(['APRP_'+comp+'_amt','APRP_'+comp+'_scat','APRP_'+comp+'_abs','APRP_'+comp+'_tau',])
            aer_datasources.extend(['APRP_'+comp+'_amt','APRP_'+comp+'_scat','APRP_'+comp+'_abs','APRP_'+comp+'_tau',])

    elif comp in ['Net_LO','Net_HI']:
        comp1 = comp.split('_')[1]
        tag = '_'+comp
        fbk_datasources, aer_datasources = ['CRK_'+comp1,'CRK_'+comp1+'_amt','CRK_'+comp1+'_tau','CRK_'+comp1+'_alt'], ['CRK_'+comp1,'CRK_'+comp1+'_amt','CRK_'+comp1+'_tau','CRK_'+comp1+'_alt'] 
    elif comp in ['SW_LO','LW_LO','SW_HI','LW_HI']:
        tag = '_'+comp
        fbk_datasources, aer_datasources = ['CRK_'+comp,'CRK_'+comp+'_amt','CRK_'+comp+'_tau','CRK_'+comp+'_alt'], ['CRK_'+comp,'CRK_'+comp+'_amt','CRK_'+comp+'_tau','CRK_'+comp+'_alt'] 


    for datasource in ['fbk','aci']:

        if datasource == 'fbk':
            outfile = 'dics_fbk_4yr_regime-avg'+tag+'_'+regime_method_tag+'.pkl'
            xxx_datasources = fbk_datasources 
            datadir_xxx = datadir_fbk_tmp

        elif datasource == 'aci':
            outfile = 'dics_aci_regime-avg'+tag+'_'+regime_method_tag+'.pkl'
            xxx_datasources = aer_datasources 
            datadir_xxx = datadir_aer

        print(f'datasource={datasource}')
        print(f'outfile={outfile}')
        print(f'xxx_datasources = {xxx_datasources}')
        print(f'datadir_xxx = {datadir_xxx}')
        print()

        if (os.path.isfile(outfile)) and (not Do_stratification): 
            continue 

        # ## Read data and do partitioning
        dics_out = {} 

        for xxx_datasource in xxx_datasources: 

            if datasource == 'fbk':
                xxx_fname_append, Vars_xxx = misc.get_fbk_method_info(xxx_datasource)
            elif datasource == 'aci':
                xxx_fname_append, Vars_xxx = misc.get_aer_method_info(xxx_datasource)
        
            dics1 = {}
            for case in cases_t+['BASE']: 
    
                # ==========================================================
                if regime_method_tag == 'new2':
                    # ------ Read land fraction
                    fname = datadir_xxx + 'global_LANDFRAC_BASE.nc'
                    with xr.open_dataset(fname) as ff:
                        landfrac = ff['LANDFRAC_pi_clim'] 
                    
                    # ------ Read CLDLIQ as the reference to mask out high mountain regions
                    fname = datadir_xxx + 'global_CLDLIQ_BASE.nc'
                    with xr.open_dataset(fname) as ff:
                        cldliq = ff['CLDLIQ_pi_clim'].sel(plev=70000).squeeze()
                
                # ============= read data ==================================
                if not os.path.isfile(datadir_xxx+xxx_fname_append+case+'.nc'):
                    print('Not found data for', case)
                    continue 
        
                f1 = xr.open_dataset(datadir_xxx+xxx_fname_append+case+'.nc') 

                dics1[case] = {}
                for svar in Vars_xxx: 
                    if xxx_datasource == 'APRP_SW_tau' and svar == 'cld_tau':
                        data1 = f1['cld_scat']
                        data2 = f1['cld_abs']
                        data = xr.DataArray(data1+data2, coords=data1.coords)
                    else:
                        data = f1[svar]

                    # ensure the lat and lon names
                    latnew, lonnew = list(data.coords.keys())[0], list(data.coords.keys())[1]
                    data = data.rename({lonnew: 'lon',latnew: 'lat'})
        
                    print(case, svar)
                    
                    # get OMEGA700
                    omega700_pi = omega700_pi_mm 
                    omega700_ab = omega700_ab_mm
                    omega700_avg = omega700_avg_mm
    
                    
                    regsum_pi = 0
                    regsum_ab = 0 
                    regsum_avg = 0 

                    dics1[case][svar] = {}
                    
                    for ireg,reg in enumerate(regimes):
                        ## do regime partitioning 
                        if regime_method_tag in ['new']:
                            data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_2(reg,omega700_pi,omega700_ab,omega700_avg,data)
                        elif regime_method_tag == 'new2':
                            data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_3(reg,omega700_pi,omega700_ab,omega700_avg,data,landfrac=landfrac,high_mountain_mask=cldliq)
                        elif regime_method_tag in ['OceanOnly','LndOcn']:
                            data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_3(reg,omega700_pi,omega700_ab,omega700_avg,data)
                        elif regime_method_tag in ['OceanOnly30SN']:
                            data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_4(reg,omega700_pi,omega700_ab,omega700_avg,data)
                        else:
                            data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning(reg,omega700_pi,omega700_ab,omega700_avg,data)

                        # ====================== To do stratification =================
                        if Do_stratification: 
                            for metric_name in metric_names:
                                # Define outfile name 
                                outfile_stratification = middir+'Stratification_'+datasource+'_'+xxx_datasource+'_by_'+metric_name+'_'+reg+'_'+case+'_regime-avg_'+regime_method_tag+'.nc'
                                if os.path.isfile(outfile_stratification):
                                    print(outfile_stratification+' is ready. continue.')
                                    print()
                                    continue 
    
                                # ------ read metric variable to do stratification 
                                fname = datadir_fbk + 'global_'+metric_name+'_'+case+'.nc'
                                with xr.open_dataset(fname) as ff:
                                    metric0 = ff[metric_name+'_pi_clim'] 
                
                                _,_,datap2,_,_,_,_ = v1v2.regime_partitioning_3(reg, omega700_pi, omega700_ab, omega700_avg, metric0, 
                                                                            landfrac=landfrac, 
                                                                            high_mountain_mask=cldliq, 
                                                                            ) 
                                # ONLY Use the BASE field to get bin_edges - It would be better to put this as an offline file to read in...
                                fname = datadir_fbk + 'global_'+metric_name+'_BASE.nc'
                                with xr.open_dataset(fname) as ff:
                                    metric0 = ff[metric_name+'_pi_clim'] 
                
                                _,_,datap20,_,_,_,_ = v1v2.regime_partitioning_3(reg, omega700_pi, omega700_ab, omega700_avg, metric0, 
                                                                            landfrac=landfrac, 
                                                                            high_mountain_mask=cldliq, 
                                                                            ) 

                                bin_edges = v1v2.get_bin_edges(nbins,datap20)
                                print('bin_edges = ', bin_edges)

                                # Data stratification             
                                datap0_binned, datap0_binned_frac, bin_edges = v1v2.stratify_array(datap2, data2_avg, nbins, bin_edges=bin_edges) 
                
                                # Plot stratified data map 
                                figout_name = figdir+'Stratification_'+xxx_datasource+'_by_'+metric_name+'_'+reg+'_'+case+'.png'
                                v1v2.plot_stratified_map(nbins, bin_edges, datap0_binned, datap0_binned_frac, figout_name)
    
                                # --- Save stratified data into a new array 
                                attrs=dict(description="Binned "+xxx_datasource+" based on "+metric_name+" over "+reg)
    
                                if 'plev' in list(datap0_binned.dims): 
                                    ds = xr.Dataset(
                                        data_vars=dict(
                                                data_binned_gavg = (["bin","plev"], v1v2.area_averager(datap0_binned).data),
                                                data_binned_frac_gavg = (["bin","plev"], v1v2.area_averager(datap0_binned_frac).data),
                                                data_binned=(["bin","plev","lat","lon",], datap0_binned.data),
                                                data_binned_frac=(["bin","plev","lat","lon",], datap0_binned_frac.data),
                                                bin_edges=(["bin_edge"], bin_edges),
                                            ),
                                            coords=dict(
                                                lon=("lon", datap0_binned.lon.data),
                                                lat=("lat", datap0_binned.lat.data),
                                                plev=("plev", datap0_binned.plev.data),
                                                bin=("bin", datap0_binned.bin.data),
                                                bin_edge=("bin_edge",range(nbins+1)),
                                            ),
                                            attrs=attrs,
                                        )
                                else:
                                    ds = xr.Dataset(
                                        data_vars=dict(
                                                data_binned_gavg = (["bin"], v1v2.area_averager(datap0_binned).data),
                                                data_binned_frac_gavg = (["bin"], v1v2.area_averager(datap0_binned_frac).data),
                                                data_binned=(["bin","lat","lon",], datap0_binned.data),
                                                data_binned_frac=(["bin","lat","lon",], datap0_binned_frac.data),
                                                bin_edges=(["bin_edge"], bin_edges),
                                            ),
                                            coords=dict(
                                                lon=("lon", datap0_binned.lon.data),
                                                lat=("lat", datap0_binned.lat.data),
                                                bin=("bin", datap0_binned.bin.data),
                                                bin_edge=("bin_edge",range(nbins+1)),
                                            ),
                                            attrs=attrs,
                                        
                                        )
    
                                ds.to_netcdf(outfile_stratification)

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
                        

            dics_out[xxx_datasource] = dics1 
        
        print(len(dics_out.keys()), dics_out.keys(),) 
        
        
        # In[9]:
        
        with open(middir+outfile, 'wb') as f:
            pickle.dump(dics_out, f) 
        print('Done '+datasource+' regime decomposition for '+comp)


