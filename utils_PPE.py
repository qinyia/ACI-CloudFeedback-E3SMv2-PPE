

import numpy as np 
import sys
import json
from scipy.stats import linregress
import seaborn as sns 
import matplotlib.patches as mpatches
import math
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt

# =================================================================================================
# Define color of each catagories 
ctlc = 'black'
noc = 'grey' 
prc = 'tab:blue'
mixc = 'tab:orange'
turbc = 'tab:red'
deepc = 'tab:purple'
aerc = 'tab:green'

# =================================================================================================
def get_param_group_dic(exp_list):
    '''
    Description: define the parameter group for PPEs. 

    Example:

    exp_list = [
        # 'mincdnc',
        'prc_exp1','prc_exp','prc_coef1',
        # 'accre_enhan',
        # 'c1', 'gamma_coef','c8','wsub','gamma_coefb',
        'ice_deep','ice_sed_ai','so4_sz','dp1',
        'berg','clubb_tk1',
        # 'nomincdnc.prc_exp1', 'nomincdnc.prc_exp', 'nomincdnc.prc_coef1', 
        # 'nomincdnc.ice_deep','nomincdnc.ice_sed_ai','nomincdnc.so4_sz','nomincdnc.dp1',
        # 'nomincdnc.berg','nomincdnc.clubb_tk1',
        ]

    dicc, dicr = get_param_group_dic(exp_list) 

    '''
    dicc = {}
    dicr = {} 
    for exp in exp_list: 
        if exp == 'nomincdnc.prc_exp1':
            casesA = [
                ['nomincdnc', ctlc, -1.40],
                ['nomincdnc.prc_exp1_2',prc, -2.0],   
                ['nomincdnc.prc_exp1_3',prc, -1.10], 
            ]
            rangeA = [-2.0,-1.10] 

        if exp == 'nomincdnc.prc_exp':
            casesA = [
                ['nomincdnc', ctlc, 3.19],
                ['nomincdnc.prc_exp_2',prc, 3.00],
                ['nomincdnc.prc_exp_3',prc, 3.40], 
            ]
            rangeA = [3.0,3.40] 

        if exp == 'nomincdnc.prc_coef1':
            casesA = [
                ['nomincdnc', ctlc, 30500],
                ['nomincdnc.prc_coef1', prc, 1350],
                ['nomincdnc.prc_coef1_3',prc, 3050],
            ]
            rangeA = [1000,30500]

        if exp == 'nomincdnc.berg':
            casesA = [
                ['nomincdnc', ctlc, 0.7],
                ['nomincdnc.berg', mixc, 0.1],
            ]
            rangeA = [0.1,0.7]
            
        if exp == 'nomincdnc.ice_deep':
            casesA = [
                ['nomincdnc', ctlc, 14],
                ['nomincdnc.ice_deep_2', deepc, 24],
            ]
            rangeA = [12,24]

        if exp == 'nomincdnc.clubb_tk1': 
            casesA = [
                ['nomincdnc', ctlc, 268],
                ['nomincdnc.clubb_tk1_3', mixc, 243.15], 
            ]
            rangeA = [243.15,268]

        if exp == 'nomincdnc.dp1':
            casesA = [
                ['nomincdnc', ctlc, 0.018],
                ['nomincdnc.dp1_3', deepc, 0.083], 
            ]
            rangeA = [0.018,0.083]

        if exp == 'nomincdnc.ice_sed_ai': 
            casesA = [
                ['nomincdnc', ctlc, 500],
                ['nomincdnc.ice_sed_ai_3',deepc, 1350], 
            ]
            rangeA = [400,1350]

        if exp == 'nomincdnc.so4_sz': 
            casesA = [
                ['nomincdnc', ctlc, 0.080],
                ['nomincdnc.so4_sz_3',deepc, 0.0273], 
            ]
            rangeA = [0.0273,0.080]

        if exp == 'nomincdnc.accre_enhan':
            casesA = [
                ['nomincdnc', ctlc, 1.75],
                ['nomincdnc.accre_enhan_2', prc, 0.80],
            ]
            rangeA = [0.80, 2.0]

        if exp == 'nomincdnc.c1':
            casesA = [
                ['nomincdnc', ctlc, 2.4],
                ['nomincdnc.c1_2', turbc, 4.0],
            ]
            rangeA = [1.0, 4.0]

        if exp == 'nomincdnc.gamma_coef':
            casesA = [
                ['nomincdnc', ctlc, 0.12],
                ['nomincdnc.gamma_coef', turbc, 0.32],
            ]
            rangeA = [0.12,0.42]

        if exp == 'nomincdnc.c8':
            casesA = [
                ['nomincdnc', ctlc, 5.2],
                ['nomincdnc.c8_3', turbc, 2.0],
            ]
            rangeA = [2.0,7.5]


        if exp == 'nomincdnc.wsub':
            casesA = [
                ['nomincdnc', ctlc, 0.1],
                ['nomincdnc.wsub_3', aerc, 0.9],
            ]
            rangeA = [0.1,0.9]

        # //////////////////////////////////////////////////
        if exp == 'mincdnc': 
            # mincdnc only 
            casesA = [
                ['BASE', ctlc, 10],
                ['nomincdnc', noc, 0],
                ['nomincdnc_2',noc, 5],
                ['nomincdnc_3',noc,15], 
            ]
            rangeA = [0,15] 

        if exp == 'prc_exp1': 
            # prc_exp1 only 
            casesA = [
                ['BASE', ctlc, -1.40],
                ['prc_exp1', prc, -1.79],
                ['prc_exp1_2',prc, -2.0],   
                ['prc_exp1_3',prc, -1.10], 
            ]
            rangeA = [-2.0,-1.10] 

        if exp == 'prc_exp': 
            # prc_exp only
            casesA = [
                ['BASE', ctlc, 3.19],
                # ['prc_exp', prc, 2.47], 
                ['prc_exp_2',prc, 3.00],
                ['prc_exp_3',prc, 3.40], 
            ]
            rangeA = [3.0,3.40] 

        if exp == 'prc_coef1': 
            # prc_coef1 only 
            casesA = [
                ['BASE', ctlc, 30500],
                ['prc_coef1', prc, 1350],
                ['prc_coef1_2',prc, 1000], 
                ['prc_coef1_3',prc, 3050],
            ]
            rangeA = [1000,30500]

        if exp == 'c1': 
            # c1 only
            casesA = [
                ['BASE', ctlc, 2.4],
                ['c1', turbc, 1.335],
                ['c1_2',turbc, 4.0],
                ['c1_3',turbc, 1.0],
                ['c1_4',turbc, 5.0],
            ] 
            rangeA = [1.0,5.0]

        if exp == 'gamma_coef': 
            # gamma_coef only
            casesA = [
                ['BASE', ctlc, 0.12],
                ['gamma_coef', turbc, 0.32],
                ['gamma_coef_2',turbc, 0.42], 
                ['gamma_coef_3',turbc,0.22],
                ['gamma_coef_4',turbc,0.52],
            ]
            rangeA = [0.12,0.52]

        if exp == 'gamma_coefb':
            casesA = [
                ['BASE', ctlc, 0.28],
                ['gamma_coefb', turbc, 0.6], 
                ['gamma_coefb_2', turbc, 0.83],
                ['gamma_coefb_3', turbc, 0.10],
            ]
            rangeA = [0.28, 0.83],

        if exp == 'c8': 
            # c8 only 
            casesA = [
                ['BASE', ctlc, 5.2],
                ['c8', turbc, 2.6],
                ['c8_2',turbc, 7.5],
                ['c8_3',turbc, 2.0],
                ['c8_4',turbc,8.0],
            ]
            rangeA = [2.0,8.0]

        if exp == 'accre_enhan': 
            # accre_enhan only 
            casesA = [
                ['BASE', ctlc, 1.75],
                ['accre_enhan', prc, 1.0],
                ['accre_enhan_2',prc, 0.8], 
                ['accre_enhan_3',prc,2.0],  
            ]
            rangeA = [0.8,2.0]

        if exp == 'wsub':
            # wsub only
            casesA = [
                ['BASE', ctlc, 0.1],
                ['wsub', aerc, 0.4],
                ['wsub_2', aerc, 0.7], 
                ['wsub_3', aerc, 0.9], 
                ['wsub_4', aerc, 1.0],
            ]
            rangeA = [0.1,1.0]

        if exp == 'ice_deep': 
            # ice_deep only 
            casesA = [
                ['BASE', ctlc, 14],
                ['ice_deep', deepc, 12],
                ['ice_deep_2',deepc, 24], 
                ['ice_deep_3',deepc,18], 
                # ['ice_deep_4',deepc,7],
                # ['ice_deep_5',deepc,35],
            ]
            rangeA = [12,24]

        if exp == 'ice_sed_ai': 
            # ice_sed_ai only 
            casesA = [
                ['BASE', ctlc, 500],
                 ['ice_sed_ai', deepc, 400],
                 ['ice_sed_ai_2',deepc, 1205],
                ['ice_sed_ai_3',deepc, 1350], 
                # ['ice_sed_ai_4',deepc, 250], 
            ]
            rangeA = [400,1350]

        if exp == 'so4_sz': 
            # so4_sz only
            casesA = [
                ['BASE', ctlc, 0.080],
                 ['so4_sz', deepc, 0.050],
                 ['so4_sz_2',deepc, 0.0335],
                ['so4_sz_3',deepc, 0.0273], 
                # ['so4_sz_4',deepc, 0.0187], 
            ]
            rangeA = [0.0273,0.080]
        
        if exp == 'dp1': 
            # dp1 only 
            casesA = [
            ['BASE', ctlc, 0.018],
            ['dp1', deepc, 0.0225],
            ['dp1_2',deepc, 0.063],   
            ['dp1_3', deepc, 0.083], 
            # ['dp1_4',deepc,0.009], 
            ]
            rangeA = [0.018,0.083]

        if exp == 'berg':
            # berg only
            casesA = [
                ['BASE',ctlc, 0.7],
                ['berg', mixc, 0.1],
                ['berg_2', mixc, 0.3],
                ['berg_3', mixc, 0.5], 
            ]
            rangeA = [0.1,0.7]
        
        if exp == 'clubb_tk1':
            # clubb_tk1
            casesA = [
                ['BASE', ctlc, 268],
                ['clubb_tk1', mixc, 253.15], 
                ['clubb_tk1_2', mixc, 263.15], 
                ['clubb_tk1_3', mixc, 243.15], 
            ]
            rangeA = [243.15,268]

        dicc[exp] = casesA
        dicr[exp] = rangeA 
    
    return dicc, dicr 

# =================================================================================================
def get_param_group_dic4compare_nominCDNC(exp_list):
    '''
    Description: define the parameter group for PPEs compared between minCDNC and nominCDNC groups. 

    Example:

    exp_list = [
        # 'mincdnc',
        'prc_exp1','prc_exp','prc_coef1',
        # 'accre_enhan',
        # 'c1', 'gamma_coef','c8','wsub',
        'ice_deep','ice_sed_ai','so4_sz','dp1',
        'berg','clubb_tk1',
        # 'nomincdnc.prc_exp1', 'nomincdnc.prc_exp', 'nomincdnc.prc_coef1', 
        # 'nomincdnc.ice_deep','nomincdnc.ice_sed_ai','nomincdnc.so4_sz','nomincdnc.dp1',
        # 'nomincdnc.berg','nomincdnc.clubb_tk1',
        ]

    dicc, dicr = get_param_group_dic(exp_list) 

    '''
    dicc = {}
    dicr = {} 
    for exp in exp_list: 
        if exp == 'nomincdnc.prc_exp1':
            casesA = [
                ['nomincdnc', ctlc, -1.40],
                ['nomincdnc.prc_exp1_2',prc, -2.0],   
                ['nomincdnc.prc_exp1_3',prc, -1.10], 
            ]
            rangeA = [-2.0,-1.10] 

        if exp == 'nomincdnc.prc_exp':
            casesA = [
                ['nomincdnc', ctlc, 3.19],
                ['nomincdnc.prc_exp_2',prc, 3.00],
                ['nomincdnc.prc_exp_3',prc, 3.40], 
            ]
            rangeA = [3.0,3.40] 

        if exp == 'nomincdnc.prc_coef1':
            casesA = [
                ['nomincdnc', ctlc, 30500],
                ['nomincdnc.prc_coef1', prc, 1350],
                ['nomincdnc.prc_coef1_3',prc, 3050],
            ]
            rangeA = [1000,30500]

        if exp == 'nomincdnc.berg':
            casesA = [
                ['nomincdnc', ctlc, 0.7],
                ['nomincdnc.berg', mixc, 0.1],
            ]
            rangeA = [0.1,0.7]
            
        if exp == 'nomincdnc.ice_deep':
            casesA = [
                ['nomincdnc', ctlc, 14],
                ['nomincdnc.ice_deep_2', deepc, 24],
            ]
            rangeA = [12,24]

        if exp == 'nomincdnc.clubb_tk1': 
            casesA = [
                ['nomincdnc', ctlc, 268],
                ['nomincdnc.clubb_tk1_3', mixc, 243.15], 
            ]
            rangeA = [243.15,268]

        if exp == 'nomincdnc.dp1':
            casesA = [
                ['nomincdnc', ctlc, 0.018],
                ['nomincdnc.dp1_3', deepc, 0.083], 
            ]
            rangeA = [0.018,0.083]

        if exp == 'nomincdnc.ice_sed_ai': 
            casesA = [
                ['nomincdnc', ctlc, 500],
                ['nomincdnc.ice_sed_ai_3',deepc, 1350], 
            ]
            rangeA = [400,1350]

        if exp == 'nomincdnc.so4_sz': 
            casesA = [
                ['nomincdnc', ctlc, 0.080],
                ['nomincdnc.so4_sz_3',deepc, 0.0273], 
            ]
            rangeA = [0.0273,0.080]

        if exp == 'nomincdnc.accre_enhan':
            casesA = [
                ['nomincdnc', ctlc, 1.75],
                ['nomincdnc.accre_enhan_2', prc, 0.80],
            ]
            rangeA = [0.80, 2.0]

        if exp == 'nomincdnc.c1':
            casesA = [
                ['nomincdnc', ctlc, 2.4],
                ['nomincdnc.c1_2', turbc, 4.0],
            ]
            rangeA = [1.0, 4.0]

        if exp == 'nomincdnc.gamma_coef':
            casesA = [
                ['nomincdnc', ctlc, 0.12],
                ['nomincdnc.gamma_coef', turbc, 0.32],
            ]
            rangeA = [0.12,0.42]

        if exp == 'nomincdnc.c8':
            casesA = [
                ['nomincdnc', ctlc, 5.2],
                ['nomincdnc.c8_3', turbc, 2.0],
            ]
            rangeA = [2.0,7.5]


        if exp == 'nomincdnc.wsub':
            casesA = [
                ['nomincdnc', ctlc, 0.1],
                ['nomincdnc.wsub_3', aerc, 0.9],
            ]
            rangeA = [0.1,0.9]


        # //////////////////////////////////////////////////
        if exp == 'mincdnc': 
            # mincdnc only 
            casesA = [
                ['BASE', ctlc, 10],
                ['nomincdnc', noc, 0],
                ['nomincdnc_2',noc, 5],
                ['nomincdnc_3',noc,15], 
            ]
            rangeA = [0,15] 

        if exp == 'prc_exp1': 
            # prc_exp1 only 
            casesA = [
                ['BASE', ctlc, -1.40],
                # ['prc_exp1', prc, -1.79],
                ['prc_exp1_2',prc, -2.0],   
                ['prc_exp1_3',prc, -1.10], 
            ]
            rangeA = [-2.0,-1.10] 

        if exp == 'prc_exp': 
            # prc_exp only
            casesA = [
                ['BASE', ctlc, 3.19],
                # ['prc_exp', prc, 2.47], 
                ['prc_exp_2',prc, 3.00],
                ['prc_exp_3',prc, 3.40], 
            ]
            rangeA = [3.0,3.40] 

        if exp == 'prc_coef1': 
            # prc_coef1 only 
            casesA = [
                ['BASE', ctlc, 30500],
                ['prc_coef1', prc, 1350],
                # ['prc_coef1_2',prc, 1000], 
                ['prc_coef1_3',prc, 3050],
            ]
            rangeA = [1000,30500]

        if exp == 'c1': 
            # c1 only
            casesA = [
                ['BASE', ctlc, 2.4],
                #['c1', turbc, 1.335],
                ['c1_2',turbc, 4.0],
                #['c1_3',prc, 1.0],
            ] 
            rangeA = [1.0,4.0]

        if exp == 'gamma_coef': 
            # gamma_coef only
            casesA = [
                ['BASE', ctlc, 0.12],
                ['gamma_coef', turbc, 0.32],
                #['gamma_coef_2',turbc, 0.42], 
                #['gamma_coef_3',turbc,0.22]
            ]
            rangeA = [0.12,0.42]

        if exp == 'c8': 
            # c8 only 
            casesA = [
                ['BASE', ctlc, 5.2],
                #['c8', turbc, 2.6],
                #['c8_2',turbc, 7.5],
                ['c8_3',turbc, 2.0],
            ]
            rangeA = [2.0,7.5]

        if exp == 'accre_enhan': 
            # accre_enhan only 
            casesA = [
                ['BASE', ctlc, 1.75],
                #['accre_enhan', prc, 1.0],
                ['accre_enhan_2',prc, 0.8], 
                #['accre_enhan_3',prc,2.0],  
            ]
            rangeA = [0.8,2.0]

        if exp == 'wsub':
            # wsub only
            casesA = [
                ['BASE', ctlc, 0.1],
                #['wsub', aerc, 0.4],
                #['wsub_2', aerc, 0.7], 
                ['wsub_3', aerc, 0.9], 
            ]
            rangeA = [0.1,0.9]

        if exp == 'ice_deep': 
            # ice_deep only 
            casesA = [
                ['BASE', ctlc, 14],
                # ['ice_deep', deepc, 12],
                ['ice_deep_2',deepc, 24], 
                # ['ice_deep_3',deepc,18], 
                # ['ice_deep_4',deepc,7],
                # ['ice_deep_5',deepc,35],
            ]
            rangeA = [12,24]

        if exp == 'ice_sed_ai': 
            # ice_sed_ai only 
            casesA = [
                ['BASE', ctlc, 500],
                # ['ice_sed_ai', deepc, 400],
                # ['ice_sed_ai_2',deepc, 1205],
                ['ice_sed_ai_3',deepc, 1350], 
                # ['ice_sed_ai_4',deepc, 250], 
            ]
            rangeA = [400,1350]

        if exp == 'so4_sz': 
            # so4_sz only
            casesA = [
                ['BASE', ctlc, 0.080],
                # ['so4_sz', deepc, 0.050],
                # ['so4_sz_2',deepc, 0.0335],
                ['so4_sz_3',deepc, 0.0273], 
                # ['so4_sz_4',deepc, 0.0187], 
            ]
            rangeA = [0.0273,0.080]
        
        if exp == 'dp1': 
            # dp1 only 
            casesA = [
            ['BASE', ctlc, 0.018],
            # ['dp1', deepc, 0.0225],
            # ['dp1_2',deepc, 0.063],   
            ['dp1_3', deepc, 0.083], 
            # ['dp1_4',deepc,0.009], 
            ]
            rangeA = [0.018,0.083]

        if exp == 'berg':
            # berg only
            casesA = [
                ['BASE',ctlc, 0.7],
                ['berg', mixc, 0.1],
                # ['berg_2', mixc, 0.3],
                # ['berg_3', mixc, 0.5], 
            ]
            rangeA = [0.1,0.7]
        
        if exp == 'clubb_tk1':
            # clubb_tk1
            casesA = [
                ['BASE', ctlc, 268],
                # ['clubb_tk1', mixc, 253.15], 
                # ['clubb_tk1_2', mixc, 263.15], 
                ['clubb_tk1_3', mixc, 243.15], 
            ]
            rangeA = [243.15,268]

        dicc[exp] = casesA
        dicr[exp] = rangeA 
    
    return dicc, dicr 

# =================================================================================================
def define_catags():
    catags = [
        #['BASE',ctlc],
        #['nomincdnc', noc],
        ['warm rain', prc],
        ['mixed phase',mixc],
        ['turbulence',turbc],
        ['deep convection',deepc],
        ['All', ctlc],
        # ['ice',icec],
    ]
    return catags 

# =================================================================================================
def define_regions():
    regions = {
    'Globe':[-90,90,0,360],
    '30-60S':[-60,-30,0,360],
    '30-60N':[30,60,0,360],
    '30S-30N':[-30,30,0,360],
    '60-90S':[-90,-60,0,360],
    '60-90N':[60,90,0,360],
    }

    return regions

# ================================================================================================
def get_regime_label(regime):
    if regime == 'TropMarineLow':
        regime_out = 'TML'
    if regime == 'TropAscent':
        regime_out = 'TA'
    if regime == 'TropLand':
        regime_out = 'TLD'

    if regime == 'TropLand-Africa':
        regime_out = 'TLD-Af'
    if regime == 'TropLand-SouthAmerica':
        regime_out = 'TLD-SAm'
    if regime == 'TropLand-SouthAsia':
        regime_out = 'TLD-SAs'
    if regime == 'TropLand-10':
        regime_out = 'TLD-10Sto10N'
    if regime == 'TropLand-10to30':
        regime_out = 'TLD-10to30'

    if regime == 'NH-MidLat-Lnd':
        regime_out = 'NML'
    if regime == 'NH-MidLat-Ocn':
        regime_out = 'NMO'
    if regime == 'SH-MidLat':
        regime_out = 'SML'
    if regime == 'HiLat':
        regime_out = 'HIL'
    if regime == 'Global':
        regime_out = 'GLB'

    if regime == 'SH-MidLat-Ocn':
        regime_out = 'SMO'
    if regime == 'Global-Ocn':
        regime_out = 'GLO'
    if regime == 'Global-Lnd':
        regime_out = 'GLL'

    if regime == 'NH-HiLat-Lnd':
        regime_out = 'NHL'
    if regime == 'NH-HiLat-Ocn':
        regime_out = 'NHO'

    if regime == 'SH-HiLat-Lnd':
        regime_out = 'SHL'
    if regime == 'SH-HiLat-Ocn':
        regime_out = 'SHO'


    return regime_out 

# ================================================================================================
def get_method_label(method):
    if method in ['RK', 'CRK']:
        method_out = 'Total'
    if method == 'CRK_amt':
        method_out = 'Cloud amount'
    if method == 'CRK_alt':
        method_out = 'Cloud altitude'
    if method == 'CRK_tau':
        method_out = 'Cloud optical depth'
    if method in ['RK_SW','CRK_SW']:
        method_out = 'SW Total'
    if method in ['RK_LW','CRK_LW']:
        method_out = 'LW Total'
    if method == 'CRK_SW_amt':
        method_out = 'SW Cloud amount'
    if method == 'CRK_SW_alt':
        method_out = 'SW Cloud altitude'
    if method == 'CRK_SW_tau':
        method_out = 'SW Cloud optical depth'
    if method == 'APRP_SW_scat':
        method_out = 'SW Cloud scattering'
    if method == 'APRP_SW_amt':
        method_out = 'SW Cloud amount'
    if method == 'APRP_SW_tau':
        method_out = 'SW Cloud optical depth'
    if method == 'APRP':
        method_out = 'Total$_{aprp}$'
    return method_out

# =================================================================================================
def calc_regress_line(XX, YY):
    slope, intercept, r_value, p_value, std_err = linregress(XX, YY) 

    XX2 = np.arange(np.min(XX),np.max(XX),(np.max(XX)-np.min(XX))/100) 
    regression_line = slope * np.array(XX2) + intercept
    regression_line2 = slope * np.array(XX) + intercept
   
    return r_value, p_value, regression_line2

# =================================================================================================
def add_regress_line(XX,YY,ax,xpos=0.03,ypos=0.15,add_line=False,color='black',show_text=True,show_pvalue=False,show_equ=False,transform_format='axes',prefix_str='',auto_move_text=False):
    '''
    Add regression line over the scatter plot
    Also add correlation coefficient, p-value 
    '''

    slope, intercept, r_value, p_value, std_err = linregress(XX, YY) 

    XX2 = np.arange(np.min(XX),np.max(XX),(np.max(XX)-np.min(XX))/100) 
    regression_line = slope * np.array(XX2) + intercept
    regression_line2 = slope * np.array(XX) + intercept
    r_squared = r_value ** 2

    std_y = np.std(YY)
    avg_y = np.mean(YY)   
    cof_of_var_y = std_y/avg_y
    std_x = np.std(XX)
    avg_x = np.mean(XX) 
    cof_of_var_x = std_x/avg_x
    null_hypo = np.sqrt((1-r_value)/2.) 

    if add_line:
        #lw = 2.0 
        lw = 1.0
        if p_value < 0.05:
            ls = '-'
            ax.plot(XX2, regression_line, color=color, ls=ls,lw=lw) 
        else:
#            ls = '--'   
            ls = '-'
            ax.plot(XX2, regression_line, color=color, ls=ls,lw=lw) 

#        sns.regplot(x=XX, y=YY, scatter=False,ci=95,ax=ax,line_kws=dict(color=color,lw=lw)) 

    # Add equation, variance, R-squared, correlation coefficient, and p-value to the plot
    # equation = f"y = {slope:.2f}x + {intercept:.2f}"
    # variance_text = f"Variance: {np.var(Y - regression_line):.2f}"
    # r_squared_text = f"R-squared: {r_squared:.2f}"
    if show_pvalue:
        if len(prefix_str) == 0: 
            correlation_text = f"R= {r_value:.2f}, p= {p_value:.3f}"
        else:
            if p_value < 0.05:
                correlation_text = f"{prefix_str}: R= {r_value:.2f}$^*$, p= {p_value:.3f}"
            else:
                correlation_text = f"{prefix_str}: R= {r_value:.2f}, p= {p_value:.3f}"
    else:
        if len(prefix_str) == 0: 
            if p_value < 0.05:
#                correlation_text = f"R= {r_value:.2f}$^*$, R$^2$= {r_squared:.2f}"
                correlation_text = f"R= {r_value:.2f}$^*$"
            else:
#                correlation_text = f"R= {r_value:.2f}, R$^2$= {r_squared:.2f}"
                correlation_text = f"R= {r_value:.2f}"
        else:
            if p_value < 0.05: 
#                correlation_text = f"{prefix_str}: R= {r_value:.2f}$^*$, R$^2$= {r_squared:.2f}"
                correlation_text = f"{prefix_str}: R= {r_value:.2f}$^*$"
            else:
#                correlation_text = f"{prefix_str}: R= {r_value:.2f}, R$^2$= {r_squared:.2f}"
                correlation_text = f"{prefix_str}: R= {r_value:.2f}"

    if show_text: 
        p_value_text = f"p= {p_value:.2f}"
        slope_text = f"b= {slope:.2f}" 
        stats_text = f"R={r_value:.2f},Vx/Vy={cof_of_var_x/cof_of_var_y:.2f},Null_hypo={null_hypo:.2f}"   
    
        if transform_format == 'axes':
            transform = ax.transAxes
        else:
            transform = ax.transData

        if auto_move_text:
#            if r_value > 0 and p_value < 0.05: 
            if r_value > 0: 
                ax.text(xpos+0.65, ypos, correlation_text, ha='center', va='top', c=color, transform=transform, fontsize=6)
            else:
                ax.text(xpos, ypos, correlation_text, ha='left', va='top', c=color, transform=transform, fontsize=6)
        else:
            ax.text(xpos, ypos, correlation_text, ha='left', va='top', c=color, transform=transform, fontsize=6)

        # ax.text(xpos, ypos-0.08, p_value_text, ha='left', va='top', c=color, transform=ax.transAxes)
        # ax.text(xpos, ypos-0.08*2, slope_text, ha='left', va='top', c=color, transform=ax.transAxes) 
        # ax.text(0.1, ypos-0.08*2, stats_text, ha='left', va='top', c=color, transform=ax.transAxes)

    if show_equ: # show regression equation
        equation = rf"$y = {slope:.2f}x + {intercept:.2f}$"

        if transform_format == 'axes':
            transform = ax.transAxes
        else:
            transform = ax.transData

        ax.text(0.01,0.95, equation, ha='left', va='top', c=color, transform=transform, fontsize=5)

    return r_value, p_value, slope

# =================================================================================================
def add_regress_line_estimator(XX,YY,ax,xpos=0.65,ypos=0.24,estimator_name='OLS',add_line=False,color='black',show_text=True,show_pvalue=False,transform_format='axes',prefix_str=''):
    '''
    Add regression line over the scatter plot based on selected sklearn estimators
    Also add correlation coefficient, p-value 
    '''

    from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
    )
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    np.random.seed(42)

    if estimator_name == 'OLS':
        estimator = LinearRegression()
    elif estimator_name == 'Theil-Sen':
        estimator = TheilSenRegressor(random_state=42)
    elif estimator_name == 'RANSAC':
        estimator = RANSACRegressor(random_state=42)
    elif estimator_name == 'HuberRegressor':
        estimator = HuberRegressor()

    XX2 = np.array(XX)[:, np.newaxis]
    model = make_pipeline(PolynomialFeatures(1), estimator)
    model.fit(XX2, YY)
    slope = estimator.coef_[0]
    YYpred = model.predict(XX2)
    mse = mean_squared_error(YYpred, YY)
    r_squared = r2_score(YY, YYpred)

    x_plot = np.linspace(np.array(XX).min(), np.array(XX).max())
    y_plot = model.predict(x_plot[:,np.newaxis])

    if add_line:
        lw = 1.0
        if r_squared > 0.5:
            ls = '-'
            ax.plot(x_plot, y_plot, color=color, ls=ls,lw=lw) 
        else:
            ls = '--'   
            ax.plot(x_plot, y_plot, color=color, ls=ls,lw=lw) 

    # Add equation, variance, R-squared, correlation coefficient, and p-value to the plot
    # equation = f"y = {slope:.2f}x + {intercept:.2f}"
    # variance_text = f"Variance: {np.var(Y - regression_line):.2f}"
    # r_squared_text = f"R-squared: {r_squared:.2f}"
    if len(prefix_str) == 0: 
        correlation_text = f"R$^2$= {r_squared:.2f}"
    else:
        correlation_text = f"{prefix_str}: R$^2$= {r_squared:.2f}"


    if show_text: 
        if transform_format == 'axes':
            transform = ax.transAxes
        else:
            transform = ax.transData
        ax.text(xpos, ypos, correlation_text, ha='left', va='top', c=color, transform=transform, fontsize=5)

    return r_squared, slope

# =================================================================================================
def add_case_legend(fig,nrow, ncol, figtype, ii):  
    '''
    Add the legend for each case 
    '''
    if ii == nrow*ncol-1:
        if figtype == '1x1': 
            fig.legend(bbox_to_anchor=(0.1,-0.3,0.8,0.1),loc='lower center',ncol=5)
        else: 
            fig.legend(bbox_to_anchor=(0.1,-0.1,0.8,0.1),ncol=5)
        
# =================================================================================================
def add_phys_legend(fig,nrow,ncol,ii,catags): 
    '''
    Add the legend for each physical category
    '''

    # Add fake legends
    lgs = []
    for catagname,catagcolor in catags:
        lg_spc = mpatches.Patch(color=catagcolor, label=catagname)
        lgs.append(lg_spc)

    if ii == ncol-1:
        fig.legend(handles=lgs,
                bbox_to_anchor=(1.02,0.2), loc='lower left',
                )

# =================================================================================================
def update_x_limits(existing_x_limits, x_range):
    '''
    Update the axis range based on a prescribed limit and range.
    '''
    
    spacing = x_range/20. 
    A, B = existing_x_limits
    C = spacing
    D = x_range
    num_points = D/C
    
    new_A = (B+A)/2. - D/2.
    new_B = (B+A)/2. + D/2. 

    # print('A=',A,'B=',B,'B-A=',B-A, 'C=',C,'D=',D, 'new_A=',new_A, 'new_B=',new_B) 

    if new_A <= A and new_B >= B: 
        return new_A, new_B
        # print('Good new limits')
    else:
        print('Inappropriate new limits. Check.') 
        print('xrange=',x_range) 
        print('Default limits=', A, B, B-A) 
        print('New limits=', new_A, new_B, new_B-new_A)  
        sys.exit() 

# =================================================================================================
def clustering(ax, XX, YY, x_ctl, y_ctl,scaling=0.15,angleIn=15,cc='grey',ls='--',lw=0.85):   
    '''
    Add reference lines to cluster all points into different groups for later analysis.
    '''
    
    xrange = max(XX) - min(XX)
    yrange = max(YY) - min(YY)

    # define the unsignificant ranges 
    xminn,xmaxx = [x_ctl-scaling*xrange, x_ctl+scaling*xrange]
    yminn,ymaxx = [y_ctl-scaling*yrange, y_ctl+scaling*yrange] 
    
    # Add center box that denotes the insigificant ranges 
    ax.plot([xminn, xminn], [yminn,ymaxx],ls=ls,c=cc,lw=lw)  
    ax.plot([xminn, xmaxx], [yminn,yminn],ls=ls,c=cc,lw=lw) 
    ax.plot([xminn, xmaxx], [ymaxx,ymaxx],ls=ls,c=cc,lw=lw) 
    ax.plot([xmaxx, xmaxx], [yminn,ymaxx],ls=ls,c=cc,lw=lw)  

    # specific degrees
    # distance = max([xrange,yrange])*5.0  # Distance from point1
    distance = 1.0 
    print('distance=',distance,'xrange=',xrange,'yrange=',yrange, 'x_ctl-min(XX)/xrange=',(x_ctl-min(XX))/xrange,
          '(y_ctl-min(YY))/yrange=',(y_ctl-min(YY))/yrange) 

    for angle in [0+angleIn, 90-angleIn, 90+angleIn, 180-angleIn, 180+angleIn, 270-angleIn, 270+angleIn, 360-angleIn]:  
        angle_rad = math.radians(angle)
        # Get the other point to plot the angle line. Note that the normalization is 
        # needed because x and y variables might not in the same unit.
        x_new = (x_ctl-min(XX))/xrange + distance * math.cos(angle_rad)
        y_new = (y_ctl-min(YY))/yrange + distance * math.sin(angle_rad)

        ax.plot([x_ctl,x_new*xrange+min(XX)],[y_ctl,y_new*yrange+min(YY)],ls=ls,c=cc,lw=lw)   

# =================================================================================================
def save_image(filename,figs):
    '''
    Save pdf figure into one PDF file 
    ''' 

    from matplotlib.backends.backend_pdf import PdfPages

    # PdfPages is a wrapper around pdf 
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(filename)
      
    # # get_fignums Return list of existing
    # # figure numbers
    # fig_nums = plt.get_fignums()  
    # figs = [plt.figure(n) for n in fig_nums]
      
    # iterating over the numbers in list
    for fig in figs: 
        
        # and saving the files
        fig.savefig(p, format='pdf') 
          
    # close the object
    p.close()  

# =================================================================================================
def area_averager(data_plot_xr):
    '''
    calculate weighted area mean
    input data is xarray DataArray
    '''
    weights = np.cos(np.deg2rad(data_plot_xr.lat))
    weights.name = "weights"
    # available in xarray version 0.15 and later
    data_weighted = data_plot_xr.weighted(weights)

    weighted_mean = data_weighted.mean(("lat", "lon"))

    return weighted_mean

# =================================================================================================
def get_uncertainty(for_var,datadir_fbk_tmp, datadir_aer, fbk_method,fbk_var,aer_method,aer_var):  
    '''
    for_var can be 'fbk' or 'aci'
    '''
    # =====================================================
    # Get the uncertainty estimate from BASE_01 to BASE_03
    # =====================================================

    case = 'BASE' 
    sample_values = [] 
    for realization in [0,1,2,3]:
        if realization == 0:  
            caser = case
        else: 
            caser = case+'_'+'{:02d}'.format(realization) 
        if for_var == 'fbk': 
            if fbk_method == 'CRK': 
                with xr.open_dataset(datadir_fbk_tmp+'global_cloud_feedback_'+caser+'.nc') as f1:  
                    data = f1[fbk_var]
                    # ensure the lat and lon names
                    latnew, lonnew = list(data.coords.keys())[0], list(data.coords.keys())[1]
                    data = data.rename({lonnew: 'lon',latnew: 'lat'})                   
                    datap = area_averager(data).values
            else:
                print('Warning: Check your fbk_method and setup according data reading.')  
                raise RuntimeError("unable to handle error") 
        
        elif for_var == 'aci': 
            with open(datadir_aer+aer_method+'_'+caser+'_gm.json') as f1:
                datap = json.load(f1)[caser][aer_var]                 
        sample_values.append(datap) 

    sample_mean = np.mean(sample_values) 
    sample_stderr_66 = 0.95*np.std(sample_values,ddof=1)/np.sqrt(4) 
    sample_stderr_90 = 1.64*np.std(sample_values,ddof=1)/np.sqrt(4) 
    print('sample_stderr=',sample_stderr_66, sample_stderr_90) 

    return sample_mean, sample_stderr_90 

# =================================================================================================
def get_uncertainty_regime(for_var,dics_in,svar_in,regime_in): 
    '''
    for_var can be 'fbk' or 'aci'
    dics_in: all regime values; dics_in[case][var][regime]
    svar_in:
    regime_in: 
    '''
    # =====================================================
    # Get the uncertainty estimate from BASE_01 to BASE_03
    # =====================================================
    case = 'BASE' 
    sample_values = [] 
    for realization in [0,1,2,3]:
        if realization == 0:  
            caser = case
        else: 
            caser = case+'_'+'{:02d}'.format(realization) 
        if for_var == 'fbk': 
            datap = area_averager(dics_in[caser][svar_in][regime_in]).values 
        elif for_var == 'aci': 
            datap = area_averager(dics_in[caser][svar_in][regime_in]).values 
            
        sample_values.append(datap) 
    # print('sample_values=', len(sample_values)) 
    # print() 
    sample_mean = np.mean(sample_values) 
    sample_stderr_66 = 0.95*np.std(sample_values,ddof=1)/np.sqrt(4) 
    sample_stderr_90 = 1.64*np.std(sample_values,ddof=1)/np.sqrt(4) 
    # print('sample_mean=',sample_mean, 'sample_stderr=',sample_stderr_66, sample_stderr_90) 
    
    return sample_mean, sample_stderr_90 

# =================================================================================================
def get_fbk_method_info(fbk_datasource): 
    if fbk_datasource == 'RK': 
        Vars_fbk = ['netCRE_ano_grd_adj'] 
    elif fbk_datasource == 'RK_SW': 
        Vars_fbk = ['SWCRE_ano_grd_adj'] 
    elif fbk_datasource == 'RK_LW': 
        Vars_fbk = ['LWCRE_ano_grd_adj'] 

    elif fbk_datasource == 'CRK':
        Vars_fbk = ['ALL_NETcld_tot']
    elif fbk_datasource == 'CRK_amt':
        Vars_fbk = ['ALL_NETcld_amt']
    elif fbk_datasource == 'CRK_alt':
        Vars_fbk = ['ALL_NETcld_alt']
    elif fbk_datasource == 'CRK_tau':
        Vars_fbk = ['ALL_NETcld_tau']

    elif fbk_datasource == 'CRK_LO':
        Vars_fbk = ['LO680_NETcld_tot']
    elif fbk_datasource == 'CRK_LO_amt':
        Vars_fbk = ['LO680_NETcld_amt']
    elif fbk_datasource == 'CRK_LO_alt':
        Vars_fbk = ['LO680_NETcld_alt']
    elif fbk_datasource == 'CRK_LO_tau':
        Vars_fbk = ['LO680_NETcld_tau']
       
    elif fbk_datasource == 'CRK_HI':
        Vars_fbk = ['HI680_NETcld_tot']
    elif fbk_datasource == 'CRK_HI_amt':
        Vars_fbk = ['HI680_NETcld_amt']
    elif fbk_datasource == 'CRK_HI_alt':
        Vars_fbk = ['HI680_NETcld_alt']
    elif fbk_datasource == 'CRK_HI_tau':
        Vars_fbk = ['HI680_NETcld_tau']

    elif fbk_datasource == 'CRK_SW':
        Vars_fbk = ['ALL_SWcld_tot']
    elif fbk_datasource == 'CRK_SW_amt':
        Vars_fbk = ['ALL_SWcld_amt']
    elif fbk_datasource == 'CRK_SW_alt':
        Vars_fbk = ['ALL_SWcld_alt']
    elif fbk_datasource == 'CRK_SW_tau':
        Vars_fbk = ['ALL_SWcld_tau']

    elif fbk_datasource == 'CRK_SW_LO':
        Vars_fbk = ['LO680_SWcld_tot']
    elif fbk_datasource == 'CRK_SW_LO_amt':
        Vars_fbk = ['LO680_SWcld_amt']
    elif fbk_datasource == 'CRK_SW_LO_alt':
        Vars_fbk = ['LO680_SWcld_alt']
    elif fbk_datasource == 'CRK_SW_LO_tau':
        Vars_fbk = ['LO680_SWcld_tau']
       
    elif fbk_datasource == 'CRK_SW_HI':
        Vars_fbk = ['HI680_SWcld_tot']
    elif fbk_datasource == 'CRK_SW_HI_amt':
        Vars_fbk = ['HI680_SWcld_amt']
    elif fbk_datasource == 'CRK_SW_HI_alt':
        Vars_fbk = ['HI680_SWcld_alt']
    elif fbk_datasource == 'CRK_SW_HI_tau':
        Vars_fbk = ['HI680_SWcld_tau']

    elif fbk_datasource == 'CRK_LW':
        Vars_fbk = ['ALL_LWcld_tot']
    elif fbk_datasource == 'CRK_LW_amt':
        Vars_fbk = ['ALL_LWcld_amt']
    elif fbk_datasource == 'CRK_LW_alt':
        Vars_fbk = ['ALL_LWcld_alt']
    elif fbk_datasource == 'CRK_LW_tau':
        Vars_fbk = ['ALL_LWcld_tau']

    elif fbk_datasource == 'CRK_LW_LO':
        Vars_fbk = ['LO680_LWcld_tot']
    elif fbk_datasource == 'CRK_LW_LO_amt':
        Vars_fbk = ['LO680_LWcld_amt']
    elif fbk_datasource == 'CRK_LW_LO_alt':
        Vars_fbk = ['LO680_LWcld_alt']
    elif fbk_datasource == 'CRK_LW_LO_tau':
        Vars_fbk = ['LO680_LWcld_tau']
       
    elif fbk_datasource == 'CRK_LW_HI':
        Vars_fbk = ['HI680_LWcld_tot']
    elif fbk_datasource == 'CRK_LW_HI_amt':
        Vars_fbk = ['HI680_LWcld_amt']
    elif fbk_datasource == 'CRK_LW_HI_alt':
        Vars_fbk = ['HI680_LWcld_alt']
    elif fbk_datasource == 'CRK_LW_HI_tau':
        Vars_fbk = ['HI680_LWcld_tau']

    elif fbk_datasource == 'APRP': 
        Vars_fbk = ['cldnet'] 
    elif fbk_datasource == 'APRP_SW': 
        Vars_fbk = ['cld'] 
    elif fbk_datasource == 'APRP_LW': 
        Vars_fbk = ['cldlw'] 
    elif fbk_datasource == 'APRP_SW_amt':
        Vars_fbk = ['cld_amt']
    elif fbk_datasource == 'APRP_SW_scat':
        Vars_fbk = ['cld_scat']
    elif fbk_datasource == 'APRP_SW_abs':
        Vars_fbk = ['cld_abs']
    elif fbk_datasource == 'APRP_SW_tau': # sum of cld_scat and cld_abs
        Vars_fbk = ['cld_tau']
    else:
        print('Error: cannot find Vars_fbk for '+fbk_datasource+'. Please check.')

    if fbk_datasource in ['RK','RK_SW','RK_LW']:
         fbk_fname_append = 'lat-lon-gfdbk-CMIP6-'
    elif 'CRK' in fbk_datasource: 
         fbk_fname_append = 'global_cloud_feedback_obscu_'
    elif 'APRP' in fbk_datasource:
        fbk_fname_append = 'APRP_' 
    else:
        print('Error: cannot find fbk_fname_append for '+fbk_datasource+'. Please check.')

    return fbk_fname_append, Vars_fbk

# =================================================================================================
def get_aer_method_info(aer_datasource): 

    if aer_datasource == 'Ghan': 
        Vars_aer = ['ACI'] 
    elif aer_datasource == 'Ghan_SW': 
        Vars_aer = ['ACI_SW'] 
    elif aer_datasource == 'Ghan_LW': 
        Vars_aer = ['ACI_LW'] 

    elif aer_datasource == 'CRK':
        Vars_aer = ['ALL_NETcld_tot']
    elif aer_datasource == 'CRK_amt':
        Vars_aer = ['ALL_NETcld_amt']
    elif aer_datasource == 'CRK_alt':
        Vars_aer = ['ALL_NETcld_alt']
    elif aer_datasource == 'CRK_tau':
        Vars_aer = ['ALL_NETcld_tau']

    elif aer_datasource == 'CRK_LO':
        Vars_aer = ['LO680_NETcld_tot']
    elif aer_datasource == 'CRK_LO_amt':
        Vars_aer = ['LO680_NETcld_amt']
    elif aer_datasource == 'CRK_LO_alt':
        Vars_aer = ['LO680_NETcld_alt']
    elif aer_datasource == 'CRK_LO_tau':
        Vars_aer = ['LO680_NETcld_tau']
       
    elif aer_datasource == 'CRK_HI':
        Vars_aer = ['HI680_NETcld_tot']
    elif aer_datasource == 'CRK_HI_amt':
        Vars_aer = ['HI680_NETcld_amt']
    elif aer_datasource == 'CRK_HI_alt':
        Vars_aer = ['HI680_NETcld_alt']
    elif aer_datasource == 'CRK_HI_tau':
        Vars_aer = ['HI680_NETcld_tau']


    elif aer_datasource == 'CRK_SW':
        Vars_aer = ['ALL_SWcld_tot']
    elif aer_datasource == 'CRK_SW_amt':
        Vars_aer = ['ALL_SWcld_amt']
    elif aer_datasource == 'CRK_SW_alt':
        Vars_aer = ['ALL_SWcld_alt']
    elif aer_datasource == 'CRK_SW_tau':
        Vars_aer = ['ALL_SWcld_tau']

    elif aer_datasource == 'CRK_SW_HI':
        Vars_aer = ['HI680_SWcld_tot']
    elif aer_datasource == 'CRK_SW_HI_amt':
        Vars_aer = ['HI680_SWcld_amt']
    elif aer_datasource == 'CRK_SW_HI_alt':
        Vars_aer = ['HI680_SWcld_alt']
    elif aer_datasource == 'CRK_SW_HI_tau':
        Vars_aer = ['HI680_SWcld_tau']

    elif aer_datasource == 'CRK_SW_LO':
        Vars_aer = ['LO680_SWcld_tot']
    elif aer_datasource == 'CRK_SW_LO_amt':
        Vars_aer = ['LO680_SWcld_amt']
    elif aer_datasource == 'CRK_SW_LO_alt':
        Vars_aer = ['LO680_SWcld_alt']
    elif aer_datasource == 'CRK_SW_LO_tau':
        Vars_aer = ['LO680_SWcld_tau']

    elif aer_datasource == 'CRK_LW':
        Vars_aer = ['ALL_LWcld_tot']
    elif aer_datasource == 'CRK_LW_amt':
        Vars_aer = ['ALL_LWcld_amt']
    elif aer_datasource == 'CRK_LW_alt':
        Vars_aer = ['ALL_LWcld_alt']
    elif aer_datasource == 'CRK_LW_tau':
        Vars_aer = ['ALL_LWcld_tau']

    elif aer_datasource == 'CRK_LW_HI':
        Vars_aer = ['HI680_LWcld_tot']
    elif aer_datasource == 'CRK_LW_HI_amt':
        Vars_aer = ['HI680_LWcld_amt']
    elif aer_datasource == 'CRK_LW_HI_alt':
        Vars_aer = ['HI680_LWcld_alt']
    elif aer_datasource == 'CRK_LW_HI_tau':
        Vars_aer = ['HI680_LWcld_tau']

    elif aer_datasource == 'CRK_LW_LO':
        Vars_aer = ['LO680_LWcld_tot']
    elif aer_datasource == 'CRK_LW_LO_amt':
        Vars_aer = ['LO680_LWcld_amt']
    elif aer_datasource == 'CRK_LW_LO_alt':
        Vars_aer = ['LO680_LWcld_alt']
    elif aer_datasource == 'CRK_LW_LO_tau':
        Vars_aer = ['LO680_LWcld_tau']

    elif aer_datasource == 'APRP': 
        Vars_aer = ['cldnet'] 
    elif aer_datasource == 'APRP_SW': 
        Vars_aer = ['cld'] 
    elif aer_datasource == 'APRP_LW': 
        Vars_aer = ['cldlw'] 
    elif aer_datasource == 'APRP_SW_amt':
        Vars_aer = ['cld_amt']
    elif aer_datasource == 'APRP_SW_scat':
        Vars_aer = ['cld_scat']
    elif aer_datasource == 'APRP_SW_abs':
        Vars_aer = ['cld_abs']
    elif aer_datasource == 'APRP_SW_tau': # sum of cld_scat and cld_abs
        Vars_aer = ['cld_tau']
    else:
        print('Error: cannot find Vars_aer for '+aer_datasource+'. Please check.')

    if 'Ghan' in aer_datasource:
        aer_fname_append = 'Ghan_'
    elif  'CRK' in aer_datasource:
#        aer_fname_append = 'CRK_obscu_'
        aer_fname_append = 'CRK_' # 09/10/24: reset to old one due to missing raw data for warm-rain analysis
    elif 'APRP' in aer_datasource:
        aer_fname_append = 'APRP_'
    else:
        print('Error: cannot find aer_fname_append for '+aer_datasource+'. Please check.')

    return aer_fname_append, Vars_aer


# =================================================================================================
def get_fbk_method_info_cordecomp(fbk_datasource,decomp_type): 
    '''
    decomp_type could be ['SWLW', 'SWcomp', 'total']
    '''

    if fbk_datasource == 'RK': 
        fbk_fname_append = 'lat-lon-gfdbk-CMIP6-'

        if decomp_type == 'SWLW':
            Vars_fbk = ['netCRE_ano_grd_adj','SWCRE_ano_grd_adj','LWCRE_ano_grd_adj']
            Vars_fbk_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            print('RK method does not support decompositions based on SW components.')
            return None, None, None
        elif decomp_type == 'total':
            Vars_fbk = ['netCRE_ano_grd_adj','SWCRE_ano_grd_adj','LWCRE_ano_grd_adj']
            Vars_fbk_out = ['Total','SW','LW']

    elif fbk_datasource == 'CRK':
        fbk_fname_append = 'global_cloud_feedback_'

        if decomp_type == 'SWLW':
            Vars_fbk = ['ALL_NETcld_tot','ALL_SWcld_tot','ALL_LWcld_tot']
            Vars_fbk_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            Vars_fbk = ['ALL_SWcld_tot','ALL_SWcld_amt', 'ALL_SWcld_alt', 'ALL_SWcld_tau', 'ALL_SWcld_err']
            Vars_fbk_out = ['SW','SW cloud amount','SW cloud altitude','SW cloud optical depth','SW residual']
        elif decomp_type == 'total':
            Vars_fbk = ['ALL_NETcld_tot','ALL_SWcld_amt','ALL_SWcld_alt','ALL_SWcld_tau','ALL_SWcld_err','ALL_LWcld_amt','ALL_LWcld_alt','ALL_LWcld_tau','ALL_LWcld_err']
            Vars_fbk_out = ['Total','SW cloud amount','SW cloud altitude','SW cloud optical depth','SW residual','LW cloud amount','LW cloud altitude','LW cloud optical depth','LW residual']

    elif fbk_datasource == 'APRP':
        fbk_fname_append = 'APRP_'

        if decomp_type == 'SWLW':
            Vars_fbk = ['cldnet', 'cld', 'cldlw']
            Vars_fbk_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            Vars_fbk = ['cld','cld_amt','cld_scat','cld_abs']
            Vars_fbk_out = ['SW','Cloud amount','Scattering','Absorption']
        elif decomp_type == 'total':
            Vars_fbk = ['cldnet','cld_amt','cld_scat','cld_abs','cldlw']
            Vars_fbk_out = ['Total','Cloud amount','Scattering','Absorption','Cloud LW']

    return fbk_fname_append, Vars_fbk, Vars_fbk_out


# =================================================================================================
def get_aer_method_info_cordecomp(aer_datasource,decomp_type): 
    '''
    decomp_type could be ['SWLW', 'SWcomp', 'total']
    '''

    if aer_datasource == 'Ghan': 
        aer_fname_append = 'Ghan_'

        if decomp_type == 'SWLW':
            Vars_aer = ['ACI','ACI_SW','ACI_LW']
            Vars_aer_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            print('Ghan method does not support decompositions based on SW components.')
            return None, None, None

        elif decomp_type == 'total':
            #Vars_aer = ['AER_TOT','ACI_SW','ARI_SW','SURF_SW','ACI_LW','ARI_LW','SURF_LW',]
            Vars_aer = ['ACI','ACI_SW','ACI_LW']
            Vars_aer_out = ['Total','SW','LW']

    elif aer_datasource == 'CRK':
        aer_fname_append = 'CRK_'

        if decomp_type == 'SWLW':
            Vars_aer = ['ALL_NETcld_tot','ALL_SWcld_tot','ALL_LWcld_tot']
            Vars_aer_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            Vars_aer = ['ALL_SWcld_tot','ALL_SWcld_amt', 'ALL_SWcld_alt', 'ALL_SWcld_tau', 'ALL_SWcld_err']
            Vars_aer_out = ['Total','Cloud amount','Cloud altitude','Cloud optical depth','Residual']
        elif decomp_type == 'total':
            Vars_aer = ['ALL_NETcld_tot','ALL_SWcld_amt','ALL_SWcld_alt','ALL_SWcld_tau','ALL_SWcld_err','ALL_LWcld_amt','ALL_LWcld_alt','ALL_LWcld_tau','ALL_LWcld_err']
            Vars_aer_out = ['Total','SW cloud amount','SW cloud altitude','SW cloud optical depth','SW residual','LW cloud amount','LW cloud altitude','LW cloud optical depth','LW residual']

    elif aer_datasource == 'APRP':
        aer_fname_append = 'APRP_'

        if decomp_type == 'SWLW':
            Vars_aer = ['cldnet', 'cld', 'cldlw']
            Vars_aer_out = ['Total','SW','LW']
        elif decomp_type == 'SWcomp':
            Vars_aer = ['cld','cld_amt','cld_scat','cld_abs']
            Vars_aer_out = ['Total','Cloud amount','Scattering','Absorption']
        elif decomp_type == 'total':
            Vars_aer = ['cldnet','cld_amt','cld_scat','cld_abs','cldlw']
            Vars_aer_out = ['Total','Cloud amount','Scattering','Absorption','Cloud LW']

       
    return aer_fname_append, Vars_aer, Vars_aer_out

# ================================================================================================
def pearsonr_ci(x, y, ci=95, n_boots=10000):
    import random
    random.seed(42)

    x = np.asarray(x)
    y = np.asarray(y)
    
   # (n_boots, n_observations) paired arrays
    rand_ixs = np.random.randint(0, x.shape[0], size=(n_boots, x.shape[0]))
    x_boots = x[rand_ixs]
    y_boots = y[rand_ixs]
    
    # differences from mean
    x_mdiffs = x_boots - x_boots.mean(axis=1)[:, None]
    y_mdiffs = y_boots - y_boots.mean(axis=1)[:, None]
    
    # sums of squares
    x_ss = np.einsum('ij, ij -> i', x_mdiffs, x_mdiffs)
    y_ss = np.einsum('ij, ij -> i', y_mdiffs, y_mdiffs)
    
    # pearson correlations
    r_boots = np.einsum('ij, ij -> i', x_mdiffs, y_mdiffs) / np.sqrt(x_ss * y_ss)
    
    # upper and lower bounds for confidence interval
    ci_low = np.percentile(r_boots, (100 - ci) / 2)
    ci_high = np.percentile(r_boots, (ci + 100) / 2)
    return ci_low, ci_high

# ================================================================================================
def corr_x_ycomp(x,y,ycomps,debugging=False):

    '''
    Description: decompose the correlation and quantify the relative contribution of each component of y.

    x: ndarray
    y: ndarray
    ycomps: list, including each y component 
    debugging: True or False
    '''

    # correlation between x and y
    corr_x_y,_ = stats.pearsonr(x,y)
    # Get confidence interval 
    ci_low,ci_high = pearsonr_ci(x,y, ci=90)
    if debugging: 
        print(f'corr_x_y = {corr_x_y}, ci_low={ci_low}, ci_high={ci_high}')

    # standard deviation of x 
    stdd_x = np.std(x) 
    if debugging:
        print(f'stdd_x = {stdd_x}')

    # standard deviation of y 
    stdd_y = np.std(y)
    if debugging:
        print(f'stdd_y = {stdd_y}')

    corr_x_y_constructed = 0.0 
    corr_x_y_contribution = []
    corr_x_y_weights = []
    corr_x_ycomps = []
    for ycomp in ycomps:
        # correlation between x and each y component 
        corr_x_ycomp,_ = stats.pearsonr(x,ycomp)
        if debugging:
            print(f'corr_x_ycomp = {corr_x_ycomp}')

        # standard deviation of each y component 
        stdd_ycomp = np.std(ycomp)
        if debugging:
            print(f'stdd_ycomp = {stdd_ycomp}')

        corr_x_y_constructed += stdd_ycomp/stdd_y * corr_x_ycomp 
        corr_x_y_contribution.append(stdd_ycomp/stdd_y * corr_x_ycomp)
        corr_x_y_weights.append(stdd_ycomp/stdd_y)
        corr_x_ycomps.append(corr_x_ycomp)

    if debugging:
        print(f'corr_x_y =  {corr_x_y}')
        print(f'corr_x_y_constructed = {corr_x_y_constructed}') 

    if corr_x_y - corr_x_y_constructed > 2e-3:
        print('Oops. Your data do not pass the correlation decomposition test. Please check:', 'corr_x_y=',corr_x_y, 'corr_x_y_constructed=',corr_x_y_constructed)
        sys.exit()
    
    return corr_x_y, ci_low, ci_high, np.array(corr_x_y_contribution), np.array(corr_x_y_weights), np.array(corr_x_ycomps)

# ================================================================================================
def corr_x_ycomp_map(x,y,ycomps_map,debugging=False):

    '''
    Description: decompose the correlation and quantify the relative contribution of each map component of y.

    x: ndarray
    y: ndarray
    ycomps_map: list, including each spatial map of y component
    debugging: True or False
    '''

    # correlation between x and y
    corr_x_y,_ = stats.pearsonr(x,y)
    if debugging: 
        print(f'corr_x_y = {corr_x_y}')

    # standard deviation of x 
    stdd_x = np.std(x) 
    if debugging:
        print(f'stdd_x = {stdd_x}')

    # standard deviation of y 
    stdd_y = np.std(y)
    if debugging:
        print(f'stdd_y = {stdd_y}')

    corr_x_y_constructed = 0.0 
    corr_x_y_contribution = []
    corr_x_y_weights = []
    corr_x_ycomps = []
    for ycomp_tmp in ycomps_map:
        ycomp = ycomp_tmp
        # correlation between x and each y component 
        x_nd = np.transpose(np.tile(x,(ycomp.shape[1],ycomp.shape[2],1)),(2,0,1))
        corr_x_ycomp,_ = pearsonr_nd(x_nd,ycomp)
        if debugging:
            print(f'corr_x_ycomp.shape = {corr_x_ycomp.shape}')

        # standard deviation of each y component 
        stdd_ycomp = np.nanstd(ycomp,axis=0)
        if debugging:
            print(f'stdd_ycomp.shape = {stdd_ycomp.shape}')

        corr_x_y_constructed += stdd_ycomp/stdd_y * corr_x_ycomp 
        corr_x_y_contribution.append(stdd_ycomp/stdd_y * corr_x_ycomp)
        corr_x_y_weights.append(stdd_ycomp/stdd_y)
        corr_x_ycomps.append(corr_x_ycomp)

    corr_x_y_constructed = area_averager(xr.DataArray(corr_x_y_constructed, coords={'lat':ycomp.lat,'lon':ycomp.lon})).values

    if debugging:
        print(f'corr_x_y =  {corr_x_y}')
        print(f'corr_x_y_constructed = {corr_x_y_constructed}') 

    if corr_x_y - corr_x_y_constructed > 2e-3:
        print('Oops. Your data do not pass the correlation decomposition test. Please check:', 'corr_x_y=',corr_x_y, 'corr_x_y_constructed=',corr_x_y_constructed, 'the difference =', corr_x_constructed - corr_x_y)
        sys.exit()
    else:
        print('You pass the construction test. Congratulations!')
    
    return corr_x_y, np.array(corr_x_y_contribution), np.array(corr_x_y_weights), np.array(corr_x_ycomps)


# ================================================================================================
# ## Func get_color
# -----------------------------------------------------------------------
def get_color(colormap,color_nums):
    '''
    get color based on user-defined colormap and number or colors needed.
    '''
    palette = plt.get_cmap(colormap)
    colors = []
    for ic in range(color_nums):
        color = palette(ic)
        colors.append(color)
    return colors

# ================================================================================================

def get_marker_size(fsa): 
    '''
    Get the marker size in the scatter plot based on the third array (fsa)
    '''
    import matplotlib.colors as mcolors

    offset = 0.01
    norm = mcolors.Normalize(vmin=min(fsa)-offset, vmax=max(fsa)) 
    Z_normalized = norm(fsa) 
    min_marker_size = 10
    ss = Z_normalized*(50 - min_marker_size)+ min_marker_size
    return ss 

# ================================================================================================
def wgt_p_LevLatLon(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top
    '''
    
    # fixed parameters
    dt = 1800 # time step unit: s
    cpair = 1.00464e3 #J/(kg K)
    rho_w = 1000 # kg/m3
    gravit = 9.8 # m/s2

    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1].values - levs[1:].values
    # print(dp)

    data_mid = (data[:-1,:].values+data[1:,:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = np.moveaxis(data_mid,0,-1)
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)

    ## covert to xarray
    data_integral = xr.DataArray(data_wgt, coords=data[0,:].coords,dims=data[0,:].dims)

    return data_integral

# ================================================================================================

def get_var_unit(var):
    var_label = var
    scale = 1.0 

    if var == 'RK':
        var_unit = 'W/m$^2$/K'
        var_label = 'CF$_{RK}$'
    if var == 'CRK_tau':
        var_unit = 'W/m$^2$/K'
        var_label = 'CRK$_{tau}$'
    if var == 'CRK_amt':
        var_unit = 'W/m$^2$/K'
        var_label = 'CRK$_{amt}$'
    if var == 'CRK':
        var_unit = 'W/m$^2$/K'
        var_label = 'CRK'
    if var == 'CRK_SW':
        var_unit = 'W/m$^2$/K'
        var_label = 'CRK$_{SW}$'
    if var == 'CRK_LW':
        var_unit = 'W/m$^2$/K'
        var_label = 'CRK$_{LW}$'
    if var == 'CRK_SW_tau':
        var_unit = 'W/m$^2$/K'
        var_label = 'SW CRK$_{tau}$'
    if var == 'CRK_LW_tau':
        var_unit = 'W/m$^2$/K'
        var_label = 'LW CRK$_{tau}$'
    if var == 'Gryspeerdt_decomp':
        var_unit = 'W/m$^2$'
    if var in ['LWP_LS','TGCLDLWP','LWP_LSlow','LWP_LShigh','TGCLDCWP','TGCLDIWP','LWP_CONV','IWP_LS','IWP_CONV']:
        var_unit = 'g/m$^2$'
        scale = 1e3
        if var == 'LWP_LS':
            var_label = 'LWP$_{LS}$'
        if var == 'TGCLDLWP':
            var_label = 'LWP'
        if var == 'TGCLDIWP':
            var_label = 'IWP'
    if var == 'WP3_CLUBB':
        var_unit = 'x10$^{-2}$ m$^3$/s$^3$' 
        var_label = r'$\overline{{w}^{\prime}}^{3}$'
        scale = 1e2
    if var in ['WP2_CLUBB','WP2_700to850','WP2_850to925']:
        var_unit = 'x10$^{-2}$ m$^2$/s$^2$' 
        var_label = r'$\overline{{w}^{\prime}}^{2}$'
#        var_label = r'$\langle\overline{{w}^{\prime}}^{2}\rangle_{700-950}$' # default for 700-950 average
        scale = 1e2
#        if var == 'WP2_850to925':
#            var_label = r'$\left.\overline{{w}^{\prime}}^{2}\right|_{850-925}$'
        if var == 'WP2_700to850':
            var_label = r'$\left.\overline{{w}^{\prime}}^{2}\right|_{700..850}$'
    if var in ['SKW_ZT', 'SKW_ZT_700to850', 'SKW_ZT_850to925']:
        var_label = r'${Skw}$'
        var_unit = '-' 
        if var == 'SKW_ZT_700to850':
            var_label = r'$\left.{Skw}\right|_{700..850}$'
        if var == 'SKW_ZT_850to925':
            var_label = r'$\left.{Skw}\right|_{850..925}$'
    if var == 'DCPPBL':
        var_unit = 'm'
    if var == 'FRQPBL':
        var_unit = '-'
    if var in ['ENTRAT','ENTRATP','ENTRATN']:
        var_unit = 'm/day'
        scale = 24*60*60 
        if var == 'ENTRAT':
            var_label = 'w$_e$'
    if var in ['ENTEFF','ENTEFFP','ENTEFFN']:
        scale = 1e4
        var_unit = 'x10$^{-4}$'
        if var == 'ENTEFF':
            var_label = 'A'
    if var == 'CF':
        var_unit = 'W/m$^2$/K'
    if var == 'ERFaci':
        var_unit = 'W/m$^2$'
    if var in ['EIS','LTS']:
        var_unit = 'K'
    if var == 'TMQ':
        var_unit = 'mm'
    if var == 'FREQZM':
        var_unit = '-'
    if var == 'DP_WCLDBASE':
        var_unit = 'm/s'
    if var == 'DP_MFUP_MAX':
        var_unit = 'kg/m$^2$'
    if var == 'CMFMCDZM':
        scale = 86400
        var_unit = 'kg/m$^2$/day'
        var_label = 'ZM mass flux'
    if var == 'CDNUMC':
        var_unit = 'x10$^4$ #/cm$^2$'
        scale = 1e-4*1e-4
        var_label = 'CDNC'
    if var == 'T':
        var_unit = 'K'
    if var in ['CLDLIQ','CLDICE']:
        var_unit = 'g/kg'
        scale = 1e3 * 1e3
        var_unit = 'x10$^{-3}$ g/kg'
    if var in ['Q']:
        var_unit = 'g/kg'
        scale = 1e3 
    if var in ['LHFLX','SHFLX','buoy_flux']:
        var_unit = 'W/m$^2$'
    if var  in ['Bowen_ratio']:
        var_unit = '-'
    if var in ['TS','TREFHT','tas']:
        var_unit = 'K' 
    if var in ['CLDTOT','CLDLOW','CLDMED','CLDTOT4POP']:
        scale = 100
        var_unit = '%'
    if var in ['RELHUM','RELHUM_850to925','RELHUM_700to850']:
        var_label = 'RH'
        var_unit = '%'
        if var == 'RELHUM_850to925':
            var_label = r'$\left.{RH}\right|_{850..925}$'
        if var == 'RELHUM_700to850':
            var_label = r'$\left.{RH}\right|_{700..850}$'
    if var == 'CLOUD':
        var_unit = '%'
        scale = 100

    if var in ['DPDLFLIQ', 'ZMDLIQ', 'MPDLIQ','RCMTEND_CLUBB']:
        scale = 1e3*86400.
        var_unit = 'g/kg/day'
        if var == 'DPDLFLIQ':
            var_label = 'ZMdetrain'
        if var == 'ZMDLIQ':
            var_label = 'ZM'
        if var == 'MPDLIQ':
            var_label = 'MP'
        if var == 'RCMTEND_CLUBB':
            var_label = 'CLUBB'
   
    if var in ['ZMDQ', 'RVMTEND_CLUBB', 'MPDQ']:
        scale = 1e3*86400
        var_unit = 'g/kg/day'
        if var == 'ZMDQ':
            var_label = 'ZM'
        if var == 'RVMTEND_CLUBB':
            var_label = 'CLUBB'
        if var == 'MPDQ':
            var_label = 'MP'
    
    if var in ['ZMDT', 'TTEND_CLUBB', 'MPDT','QRL', 'QRS', 'EVAPTZM', 'ZMMTT', 'DYNT']:
        var_unit = 'K/day'
        cpair = 1.00464e3 #J/(kg K) 
        if var == 'MPDT': # W/kg --> K/s
            scale = 1.0/cpair*86400.
        else:
            scale = 86400.
        if var == 'ZMDT':
            var_label = 'ZM'
        if var == 'TTEND_CLUBB':
            var_label = 'CLUBB'
        if var == 'MPDT':
            var_label = 'MP'
        if var == 'QRL':
            var_label = 'LW'
        if var == 'QRS':
            var_label = 'SW'
        if var == 'EVAPTZM':
            var_label = 'ZM evap'
        if var == 'ZMMTT':
            var_label = 'ZM momentum'
            

    if var in ['OMEGA']:
        var_unit = 'hPa/day'
        scale = 864.
        var_label = '$\omega$'

    if var in ['PRECT','PRECC','PRECL']:
        var_unit = 'mm/day'
        scale = 8.64e7

    if var in ['PE', 'PE_LS', 'PE_CONV','PErgavg']:
        var_unit = 'x10$^{-3}$ s$^{-1}$'
        scale = 1e3 * 1e3

    if var in ['POP', 'POP_LS', 'POP_CONV']:
        var_unit = '-'

    if var == 'WPTHVP_CLUBB':
        var_label = r"$\overline{w'\theta_{v}'}$"
        var_unit = 'W/m$^2$' 
    if var == 'WPTHLP_CLUBB':
        var_label = r"$\overline{w'\theta_{l}'}$"
        var_unit = 'W/m$^2$' 
    if var == 'WPRCP_CLUBB':
        var_label = r"$\overline{w'r_{c}'}$"
        var_unit = 'W/m$^2$' 
    if var == 'WPRTP_CLUBB':
        var_label = r"$\overline{w'r_{t}'}$"
        var_unit = 'W/m$^2$' 
    if var == 'RTP2_CLUBB':
        var_label = r"$\overline{r_{t}'^2}$"
        var_unit = r"$\mathrm{g^{2}\,kg^{-2}}$"
    if var == 'RTPTHLP_CLUBB':
        var_label = r"$\overline{r_{t}'\theta_{l}}'$"
        var_unit = r"$\mathrm{K\,g\,kg^{-1}\,}$"
    if var == 'THLP2_CLUBB':
        var_label = r"$\overline{\theta_{l}'^2}$"
        var_unit = 'K$^2$'

    if var == 'TAUTMODIS':
        scale = 1e-2
        var_label = r'$\tau$'
        var_unit = '-'
    if var == 'TAUTLOGMODIS':
        var_label = r'$log(\tau)$'
        var_unit = '-' 
    if var == 'TAUWMODIS':
        scale = 1e-2
        var_label = r'$\tau_{liq}$'
        var_unit = '-'
    if var == 'TAUWLOGMODIS':
        var_label = r'$log(\tau_{liq})$'
        var_unit = '-' 
    if var == 'TAUIMODIS':
        scale = 1e-2
        var_label = r'$\tau_{ice}$'
        var_unit = '-'
    if var == 'TAUILOGMODIS':
        var_label = r'$log(\tau_{ice})$'
        var_unit = '-' 

    if var == 'CDNUMC_INLIQCLD':
        var_label = 'in-cld CDNC'
        var_unit = 'x10$^4$ #/cm$^2$'
        scale = 1e-4*1e-4
    if var == 'LWP_INLIQCLD':
        var_label = 'in-cld LWP'
        scale = 1e3 
        var_unit = 'g/m$^2$'
    if var == 'LIQCLD':
        var_label = 'Liq cloud fraction'
        scale = 100
        var_unit = '%'

    return var_unit, scale, var_label

# ================================================================================================

def append_band_to_CF(fbk_datasource, ylabel):  
    import re

    '''

    Example:
    append_band_to_CF('CRK_LW_HI_tau','CF')

    '''

    # Define your pre-defined string
    pre_defined_string = ylabel

    # Define the string containing 'SW' or 'LW'
    string_with_sw_or_lw = fbk_datasource

    # Define a regular expression pattern to match 'SW' or 'LW'
    pattern = re.compile(r'(?:^|[_\s])(SW|LW)(?:[_\s]|$)')

    # Find all matches of 'SW' or 'LW' in the string
    matches = re.findall(pattern, string_with_sw_or_lw)

    # If matches are found, append them to the pre-defined string
    if matches:
        for match in matches:
            pre_defined_string =  match +' '+ pre_defined_string


    # Define a regular expression pattern to match 'HI' or 'LO'
    pattern = re.compile(r'(?:^|[_\s])(HI|LO)(?:[_\s]|$)')

    # Find all matches of 'SW' or 'LW' in the string
    matches = re.findall(pattern, string_with_sw_or_lw)

    # If matches are found, append them to the pre-defined string
    if matches:
        for match in matches:
            pre_defined_string =  match +' '+ pre_defined_string 
            

    # Print the updated pre-defined string
    print("Updated string:", pre_defined_string)

    return pre_defined_string 


# ================================================================================================
# 04/26/24 Stolen from https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/atmosphere.html
def alt2pres(altitude):
    '''
    Determine site pressure from altitude.

    Parameters
    ----------
    altitude : numeric
        Altitude above sea level. [m]

    Returns
    -------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    press = 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

    return press
# ================================================================================================

def pearsonr_nd(x, y):
    """
    Parameters
    ----------
    x : (N,,,) array_like
        Input
    y : (N,,,) array_like
        Input
    #<qinyi 2021-02-12 #------------------
    Description: revised based on 1D pearsonr function. Please keep the correlation axis at the most left.
    #>qinyi 2021-02-12 #------------------
    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)
    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[0]
    mx = np.mean(x,axis=0)
    my = np.mean(y,axis=0)
    
    if len(x.shape) == 1:
        newshape = n
    if len(x.shape) == 2:
        newshape = np.append(n,1)
    elif len(x.shape) == 3:
        newshape = np.append(n,[1,1])
        
    mx_new = np.tile(mx,newshape)
    my_new = np.tile(my,newshape)
    
    xm, ym = x-mx_new, y-my_new    
    r_num = np.add.reduce(xm * ym,axis=0)
    r_den = np.sqrt(ss(xm,axis=0) * ss(ym,axis=0))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = np.where(r>1.0, 1.0, r)
    r = np.where(r<-1.0, -1.0, r)

    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df + t_squared))    
    prob = np.where(r==1.0,0.0,prob)

    return r, prob

# ================================================================================================

