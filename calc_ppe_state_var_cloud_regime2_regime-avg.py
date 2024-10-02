#!/usr/bin/env python
# coding: utf-8

# # import libaries

# In[1]:


import xarray as xr
import numpy as np
import sys
import os
import json
import pandas as pd 
import pickle

import utils_v1v2 as v1v2
import utils_PPE as misc 

from datetime import date
today = date.today()
dd = today.strftime("%Y-%m-%d")
print(dd)

import importlib
importlib.reload(v1v2) 
importlib.reload(misc) 

#workdir = '/qfs/people/qiny108/ppe_scripts/'
workdir = '/compyfs/qiny108/home_dir_bigdata/ppe_scripts/'

debug = False
do_wp2_budget = False 
do_norm_cloud = False

# --- For stratification ---- 
Do_stratification = False
metric_names = [
'LTS',
'LWP_LS',
'RELHUM_700to850',
'WP2_700to850',
'SKW_ZT_700to850',
'LHFLX',
'FREQZM',
] # Must be 2-D variables 
nbins = 5  
# --- 

#regime_method_tag = # (1): ''; (2): 'new'
#regime_method_tag = 'new2'
#regime_method_tag = 'OceanOnly' # ocean only
#regime_method_tag = 'LndOcn'
regime_method_tag = 'OceanOnly30SN' # ocean only

if regime_method_tag == 'new2': 
    if Do_stratification:
        middir = workdir+'middata/regime_partition_m4/'
    else:
        middir = workdir+'middata/regime_partition_m3/'
elif regime_method_tag in ['OceanOnly','LndOcn','OceanOnly30SN']:
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

lat_spc = np.arange(-90,90,2.5)
lon_spc = np.arange(0,360,2.5) 
print(f'lat_spc.shape={lat_spc.shape}, lon_spc.shape={lon_spc.shape}') 


# # DEFINE cases

from utils_PPE import ctlc, noc, prc, mixc, turbc, deepc 

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
    ['wsub',turbc],
    ['wsub_2',turbc],
    ['wsub_3',turbc], 
    ['wsub_4',turbc], 
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

# --- used for CLUBB budgets
if do_wp2_budget or debug:
    casesA = [
        ['BASE', ctlc],
        ['c8_3', turbc],
        ['c1', turbc],
        ]

cases = [case[0] for case in casesA]
colorsh = [case[1] for case in casesA]

print(len(cases),cases)


# # DEFINE parameter group

# In[12]:


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
            # 'mincdnc',
#            'prc_exp1','prc_exp','prc_coef1',
#            'accre_enhan',
            'c1', 'gamma_coef','c8','gamma_coefb', 
#            'wsub',
#            'ice_deep','ice_sed_ai','so4_sz','dp1',
#            'berg','clubb_tk1',
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

# In[4]:


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
    # Temporary use cases with available omega700
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

# In[5]:


what_state = 'avgCTLP4K' # 'CTL','P4K','avgCTLP4K'

regimes = ['TropMarineLow','TropAscent','TropLand','NH-MidLat-Lnd','NH-MidLat-Ocn','SH-MidLat','HiLat','Global'] # new

if regime_method_tag == 'OceanOnly':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','HiLat-Ocn', 'Global-Ocn']

if regime_method_tag == 'LndOcn':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','SH-HiLat-Ocn', 'NH-HiLat-Ocn', 
               'TropLand','SH-MidLat-Lnd','NH-MidLat-Lnd','SH-HiLat-Lnd','NH-HiLat-Lnd',
               'Global-Ocn','Global-Lnd','Global',
               ]
    
if regime_method_tag == 'OceanOnly30SN':
    regimes = ['TropMarineLow','TropAscent','NH-MidLat-Ocn','SH-MidLat-Ocn','HiLat-Ocn', 'Global-Ocn']


varlist = [
    'CLOUD', 
    'CLDLIQ', 'CLDICE',
    'CLDTOT','CLDLOW',
    'CDNUMC', 
    'EIS', 'LTS', 
    'LHFLX', 'SHFLX',
    'TS', 'TREFHT',
    'QRL', 'QRS',
    'DPDLFLIQ', 
    'ZMDLIQ', 'RCMTEND_CLUBB', 'MPDLIQ',
    'TGCLDCWP','TGCLDLWP','TGCLDIWP',
    'LWP_LS','LWP_CONV',
    'IWP_LS',
    'WP2_CLUBB', 
    'WP3_CLUBB', 
    'SKW_ZM', 'SKW_ZT',
    'FREQZM', 'CMFMCDZM', 
    'OMEGA', 'T', 'Q', 
    'WPTHVP_CLUBB','WPRTP_CLUBB', 'WPTHLP_CLUBB','WPRCP_CLUBB',
    'CDNUMC_INLIQCLD','LWP_INLIQCLD','LIQCLD',
    'CDNUMC','TAUTMODIS','TAUTLOGMODIS','TAUWMODIS','TAUWLOGMODIS','TAUIMODIS','TAUILOGMODIS',
    'CLDLIQ', 
    'ENTRAT','ENTRATP','ENTRATN',
    'ENTEFF','ENTEFFP','ENTEFFN',
    'ZMDT','TTEND_CLUBB','MPDT','QRL','QRS',
    'ZMDQ','RVMTEND_CLUBB','MPDQ',
    'DPDLFICE','ZMDICE','RIMTEND_CLUBB','MPDICE','PTECLDICE',
]

if debug:
    varlist = ['CLDLIQ']

if do_norm_cloud:
    #norm_method = 'TOPBOT'
    norm_method = 'TOP' 
    varlist = [
            'CLDLIQ',
            'CLOUD',
            'WP2_CLUBB',
            'WP3_CLUBB',
            'SKW_ZM', 'SKW_ZT',
            'WPRCP_CLUBB', 'WPRTP_CLUBB', 'WPTHLP_CLUBB', 'WPTHVP_CLUBB',
            'T', 'Q', 'RELHUM', 
            'QRL',
            'DPDLFLIQ', 'ZMDLIQ', 'RCMTEND_CLUBB', 'MPDLIQ',
            'CMFMCDZM',
    ]

    varlist = [xx+'_norm'+norm_method for xx in varlist]
    print('varlist = ', varlist)


if do_wp2_budget:
    varlist = [
             'WP2_CLUBB', 'wp2_bt', 'wp2_ma', 'wp2_ta', 'wp2_ac', 'wp2_bp', 'wp2_pr1', 'wp2_pr2', 'wp2_pr3', 'wp2_dp1', 'wp2_dp2', 'wp2_cl', 'wp2_pd', 'wp2_sf', 'wp2_sdmp', 'wp2_splat', # wp2 budget terms
#             'wprtp_bt', 'wprtp_ma', 'wprtp_ta', 'wprtp_tp', 'wprtp_ac', 'wprtp_bp', 'wprtp_pr1', 'wprtp_pr2', 'wprtp_pr3', 'wprtp_dp1', 'wprtp_mfl', 'wprtp_cl', 'wprtp_sicl', 'wprtp_pd', 'wprtp_forcing', # wprtp budget terms 
#             'wpthlp_bt', 'wpthlp_ma', 'wpthlp_ta', 'wpthlp_tp', 'wpthlp_ac', 'wpthlp_bp', 'wpthlp_pr1', 'wpthlp_pr2', 'wpthlp_pr3', 'wpthlp_dp1', 'wpthlp_mfl', 'wpthlp_cl', 'wpthlp_sicl', 'wpthlp_forcing', # wpthlp budget terms 
#             'wpthlp', 'wprtp', 'wprcp', 'wpthvp',
             ]

if Do_stratification:
    varlist = [
    'TGCLDLWP',
#    'LWP_LS',
#    'CLOUD',
#    'CLDLIQ',
#    'RELHUM',
#    'WP2_CLUBB', 'SKW_ZT',
    ]

# ## Read CFBK CTL state data and do partitioning

# In[13]:

for exph in ['fbk','aci']:
#for exph in ['aci']:
#for exph in ['fbk']:
    if (do_wp2_budget or do_norm_cloud) and exph == 'aci':
        continue 

    if exph == 'fbk':
        datadir_xxx = datadir_fbk_tmp
    elif exph == 'aci':
        datadir_xxx = datadir_aer

    for svar in varlist: 
        dics1 = {}

        casesh = cases_t + ['BASE']
        if do_wp2_budget or debug:
            casesh = cases

        for case in casesh: 
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

            # ==========================================================
            outfile = middir+'dics_'+exph+'_state_'+svar+'_'+case+'_regime-avg_'+regime_method_tag+'.pkl'

            if os.path.isfile(outfile) and (not Do_stratification):
                continue 

            # ============= read data ==================================
            svarh = svar
            if not os.path.isfile(datadir_xxx+'global_'+svarh+'_'+case+'.nc'):
                print('No file for svar =', svarh)
                print()
                continue

            f1 = xr.open_dataset(datadir_xxx+'global_'+svarh+'_'+case+'.nc')  

            dics1[case] = {}
            if Do_stratification:
                varlist_here = zip([svarh+'_pi_clim',svarh+'_ano_clim'],[svar+'_pi_clim',svar+'_ano_clim'])  # ignore xxx_ab_clim
            else:
                varlist_here = zip([svarh+'_pi_clim',svarh+'_ab_clim',svarh+'_ano_clim'],[svar+'_pi_clim',svar+'_ab_clim',svar+'_ano_clim'])

            for svar1,svar_out in varlist_here: 

                data = f1[svar1] 
                data = data.interp(lat=lat_spc, lon=lon_spc) 
    
                # get OMEGA700
                omega700_pi = omega700_pi_mm.interp(lat=lat_spc,lon=lon_spc) 
                omega700_ab = omega700_ab_mm.interp(lat=lat_spc,lon=lon_spc)
                omega700_avg = omega700_avg_mm.interp(lat=lat_spc,lon=lon_spc) 
    
                regsum_pi = 0
                regsum_ab = 0 
                regsum_avg = 0 
    
                dics1[case][svar_out] = {}
    
                for ireg,reg in enumerate(regimes):
                    ## do regime partitioning 
                    if regime_method_tag in ['new']:
                        data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_2(reg,omega700_pi,omega700_ab,omega700_avg,data)
                    elif regime_method_tag == 'new2':
                        data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_3(reg,omega700_pi,omega700_ab,omega700_avg,data,landfrac=landfrac,high_mountain_mask=cldliq)
                    elif regime_method_tag == 'OceanOnly':
                        data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_3(reg,omega700_pi,omega700_ab,omega700_avg,data)
                    elif regime_method_tag == 'OceanOnly30SN':
                        data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning_4(reg,omega700_pi,omega700_ab,omega700_avg,data)
                    else:
                        data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,_ = v1v2.regime_partitioning(reg,omega700_pi,omega700_ab,omega700_avg,data)
                    
                    # ====================== To do stratification =================
                    if Do_stratification: 
                        for metric_name in metric_names:
                            outfile_stratification = middir+'Stratification_'+exph+'_state_'+svar_out+'_by_'+metric_name+'_'+reg+'_'+case+'_regime-avg_'+regime_method_tag+'.nc'
    
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
                            if '_pi' in svar_out: 
                                figout_name = figdir+'Stratification_'+svar_out+'_by_'+metric_name+'_'+reg+'_'+case+'.png'
                                v1v2.plot_stratified_map(nbins, bin_edges, datap0_binned, datap0_binned_frac, figout_name)
    
                            # --- Save stratified data into a new array 
                            attrs=dict(description="Binned "+svar_out+" based on "+metric_name+" over "+reg)
    
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
                                        attrs = attrs, 
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
                                        attrs = attrs,
                                    )
    
                            ds.to_netcdf(outfile_stratification)
 
                    # ================ fractional area ============================         
                    # check fractional area to ensure the sum equals to 1. 
                    regsum_pi += v1v2.area_averager(data2m_pi).values
                    regsum_ab += v1v2.area_averager(data2m_ab).values
                    regsum_avg += v1v2.area_averager(data2m_avg).values
                    
                    if reg == 'Global': 
                        if len(data2m_pi.shape) == 3: 

                            print(case, svar1, svar_out,
                                reg, 
#                                'pi_avg=',v1v2.area_averager(data2m_pi[0,:]).values, 
#                                'ab_avg=',v1v2.area_averager(data2m_ab[0,:]).values, 
                                '|| avg_avg=',v1v2.area_averager(data2m_avg.sel(plev=50000)).values,
#                                'cum pi_avg=',regsum_pi[0], 
#                                'cum ab_avg=',regsum_ab[0], 
                                '|| cum avg_avg=',regsum_avg[18]) 
                        else: 
                            print(case, svar1, svar_out,
                                reg, 
#                                'pi_avg=',v1v2.area_averager(data2m_pi).values, 
#                                'ab_avg=',v1v2.area_averager(data2m_ab).values, 
                                '|| avg_avg=',v1v2.area_averager(data2m_avg).values,
#                                'cum pi_avg=',regsum_pi, 
#                                'cum ab_avg=',regsum_ab, 
                                '|| cum avg_avg=',regsum_avg)
                        print() 

                    dics1[case][svar_out][reg+'_pi'] = v1v2.area_averager(data2_pi)
                    dics1[case][svar_out][reg+'_ab'] = v1v2.area_averager(data2_ab)
                    dics1[case][svar_out][reg+'_avg'] = v1v2.area_averager(data2_avg)
    
    
            with open(outfile, 'wb') as f:
                pickle.dump(dics1[case], f) 
            print(exph+': Done-3 for '+case+' '+svar+'.') 
            print('====================================') 
            print()
    
