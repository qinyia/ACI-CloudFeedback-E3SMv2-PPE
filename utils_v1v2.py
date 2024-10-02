
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import sys
from global_land_mask import globe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd

#import matplotlib as mpl
#mpl.rcParams['font.size'] = 14
#mpl.rcParams['xtick.labelsize'] = 15
#mpl.rcParams['ytick.labelsize'] = 15
#mpl.rcParams['axes.labelsize'] = 15
#mpl.rcParams["legend.handlelength"] = 1.0
#mpl.rcParams["legend.frameon"] = True

# fixed parameters 
dt = 1800 # time step unit: s
cpair = 1.00464e3 #J/(kg K)
rho_w = 1000 # kg/m3
gravit = 9.8 # m/s2

# # define functions

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


# ## Func area_averager
# -----------------------------------------------------------------------
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


# ## Func wgt_p_tp
# -----------------------------------------------------------------------
def wgt_p_tp(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
    '''
    if levs[-1] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[1:] - levs[:-1]

    data_mid = (data[:-1].values+data[1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    # covert to xarray 
    # data_integral = xr.DataArray(data_wgt, coords=data[:,0].coords,dims=data[:,0].dims)

    data_integral = data_wgt
    return data_integral


# ## Func wgt_p_1d
# -----------------------------------------------------------------------
def wgt_p_1d(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1] - levs[1:]

    data_mid = (data[:-1].values+data[1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[0].coords,dims=data[0].dims)

    return data_integral


# ## Func wgt_p_TimeLev
# -----------------------------------------------------------------------
def wgt_p_TimeLev(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1] - levs[1:]

    data_mid = (data[:,:-1].values+data[:,1:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = data_mid
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[:,0].coords,dims=data[:,0].dims)

    return data_integral


# ## Func wgt_p_TimeLevLatLon
# -----------------------------------------------------------------------
def wgt_p_TimeLevLatLon(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1].values - levs[1:].values

    data_mid = (data[:,:-1,:].values+data[:,1:,:].values)/2.
    # print('data_mid',data_mid.shape)

    ## move the level to the last axis
    data_mid_trans = np.moveaxis(data_mid,1,-1)
    # print('data_mid_trans',data_mid_trans.shape)

    data_wgt = np.nansum(data_mid_trans*dp,axis=-1)/gravit # kg/m2
    # print('data_wgt',data_wgt.shape)
    
    ## covert to xarray 
    data_integral = xr.DataArray(data_wgt, coords=data[:,0,:].coords,dims=data[:,0,:].dims)

    return data_integral


# ## Func wgt_p_LevLatLon
# -----------------------------------------------------------------------
def wgt_p_LevLatLon(data,levs):
    '''
    vertical integral weighted by pressure thickness
    inputs: data(time,level)
            levs -- must be bottom to top 
    '''
    if levs[0] < 1300: # in hPa --> Pa
        levs = levs*100.

    dp = levs[:-1].values - levs[1:].values

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


# ## Func regime-partitioning
# -----------------------------------------------------------------------
def regime_partitioning(reg,omega700_pi_in,omega700_ab_in,omega700_avg_in,data):  
    '''
    Statc regime partitioning 
    ''' 

    # use the same lon as omega700 for data and EIS
    data['lon'] = omega700_pi_in.lon 
        
    fillvalue = np.nan
    # ============== get land mask ========================
    
    lons = data.coords['lon'].data
    lats = data.coords['lat'].data
    lons_here = np.where(lons>180,lons-360,lons)
    lon_grid,lat_grid = np.meshgrid(lons_here,lats)
    globe_land_mask = globe.is_land(lat_grid,lon_grid)

    if len(data.shape) == 4: # (time,lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],data.shape[1],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
    elif len(data.shape) == 3: # (lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],1,1)), coords=data.coords)
    elif len(data.shape) == 2: 
        tmp = globe_land_mask
        omega700_pi = omega700_pi_in
        omega700_ab = omega700_ab_in
        omega700_avg = omega700_avg_in
        
    globe_land_mask = xr.DataArray(tmp,coords=data.coords)
    # print('globe_land_mask.shape=',globe_land_mask.shape,'tmp.shape=',tmp.shape)
    
    avg_flag = xr.zeros_like(data)
    
    if reg == 'TropMarineLow':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi>0),data1,fillvalue)
        data2_ab = xr.where((omega700_ab>0),data1,fillvalue)
        data2_avg = xr.where((omega700_avg>0),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 1)
    if reg == 'TropAscent':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi<0),data1,fillvalue)                
        data2_ab = xr.where((omega700_ab<0),data1,fillvalue)       
        data2_avg = xr.where((omega700_avg<0),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 2)  
    if reg == 'MidLat':
        data1 = data.where((((data.lat>=-60)&(data.lat<-30))|((data.lat<60)&(data.lat>=30))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat':
        data1 = data.where(((data.lat<60)&(data.lat>=30)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat-Lnd':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat-Ocn':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 

    if reg == 'SH-MidLat':
        data1 = data.where(((data.lat>=-60)&(data.lat<-30)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 

    if reg == 'HiLat':
        data1 = data.where((((data.lat<-60))|((data.lat>=60))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'TropLand':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-15':
        data1 = data.where((data.lat>=-15)&(data.lat<15)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-15-30':
        data1 = data.where((((data.lat>=-30)&(data.lat<-15))|((data.lat<30)&(data.lat>=15)))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global':
        data1 = data
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = 1.0 

    # ================ fractional area ============================
    data2m_pi = xr.where(np.isnan(data2_pi),0.0,1.0)
    data2m_ab = xr.where(np.isnan(data2_ab),0.0,1.0)
    data2m_avg = xr.where(np.isnan(data2_avg),0.0,1.0)
    
    return data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,avg_flag


# ## Func regime-partitioning
# -----------------------------------------------------------------------
def regime_partitioning_2(reg,omega700_pi_in,omega700_ab_in,omega700_avg_in,data):  
    '''
    Statc regime partitioning:
    change omega700 threshold from 0 to 10 hPa/day to have more restricted tropical marine low cloud regime.
    also, expand the marine low cloud regime range from 30S-30N to 45S-45N. 

    04/30/24: add another three TropLand regimes based on longitudes:
    TropLandAfrica: 0~60E, -30S~30S
    TropLandSouthAmerica: 120W~30W, 10N~30S
    TropLandSouthAsia: 60E~130E, 10S~30N

    05/05/24: decompose TropLand into TropLand-10 and TropLand-10to30
    ''' 

    # use the same lon as omega700 for data and EIS
    data['lon'] = omega700_pi_in.lon 
        
    fillvalue = np.nan
    # ============== get land mask ========================
    
    lons = data.coords['lon'].data
    lats = data.coords['lat'].data
    lons_here = np.where(lons>180,lons-360,lons)
    lon_grid,lat_grid = np.meshgrid(lons_here,lats)
    globe_land_mask = globe.is_land(lat_grid,lon_grid)

    if len(data.shape) == 4: # (time,lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],data.shape[1],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
    elif len(data.shape) == 3: # (lev,lat,lon)
        tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],1,1)), coords=data.coords)
    elif len(data.shape) == 2: 
        tmp = globe_land_mask
        omega700_pi = omega700_pi_in
        omega700_ab = omega700_ab_in
        omega700_avg = omega700_avg_in
        
    globe_land_mask = xr.DataArray(tmp,coords=data.coords)
    # print('globe_land_mask.shape=',globe_land_mask.shape,'tmp.shape=',tmp.shape)
    
    avg_flag = xr.zeros_like(data)
    
    w_threshold = 10. # 10 hPa

    if reg == 'TropMarineLow':
        data1 = data.where((data.lat>=-45)&(data.lat<45)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi>=w_threshold),data1,fillvalue)
        data2_ab = xr.where((omega700_ab>=w_threshold),data1,fillvalue)
        data2_avg = xr.where((omega700_avg>=w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 1)
    if reg == 'TropAscent':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi<w_threshold),data1,fillvalue)                
        data2_ab = xr.where((omega700_ab<w_threshold),data1,fillvalue)       
        data2_avg = xr.where((omega700_avg<w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 2)  
    if reg == 'NH-MidLat-Lnd':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat-Ocn':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_pi<w_threshold)),fillvalue)
        data2_ab = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_ab<w_threshold)),fillvalue)
        data2_avg = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_avg<w_threshold)),fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat':
        data1 = data.where(((data.lat>=-60)&(data.lat<-45)),fillvalue) # select big region and mask land 
        data2_pi = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_pi<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        data2_ab = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_ab<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        data2_avg = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_avg<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 

    if reg == 'HiLat':
        data1 = data.where((((data.lat<-60))|((data.lat>=60))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'TropLand':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10':
        data1 = data.where((data.lat>=-10)&(data.lat<10)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10to30':
        data1 = data.where((((data.lat>=-30)&(data.lat<-10))|((data.lat<30)&(data.lat>=10)))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-Africa':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(data.lon>0)&(data.lon<60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAmerica':
        data1 = data.where((data.lat>=-30)&(data.lat<10)&(data.lon>240)&(data.lon<330)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAsia':
        data1 = data.where((data.lat>=10)&(data.lat<30)&(data.lon>60)&(data.lon<120)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 


    if reg == 'Global':
        data1 = data
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = 1.0 

    # ================ fractional area ============================
    data2m_pi = xr.where(np.isnan(data2_pi),0.0,1.0)
    data2m_ab = xr.where(np.isnan(data2_ab),0.0,1.0)
    data2m_avg = xr.where(np.isnan(data2_avg),0.0,1.0)
    
    return data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,avg_flag

# -----------------------------------------------------------------------
def regime_partitioning_3(reg,omega700_pi_in,omega700_ab_in,omega700_avg_in,data,landfrac=None,high_mountain_mask=None):  
    '''
    Statc regime partitioning:
    change omega700 threshold from 0 to 10 hPa/day to have more restricted tropical marine low cloud regime.
    also, expand the marine low cloud regime range from 30S-30N to 45S-45N. 

    04/30/24: add another three TropLand regimes based on longitudes:
    TropLandAfrica: 0~60E, -30S~30S
    TropLandSouthAmerica: 120W~30W, 10N~30S
    TropLandSouthAsia: 60E~130E, 10S~30N

    05/05/24: decompose TropLand into TropLand-10 and TropLand-10to30

    05/07/24: use landfrac to do the mask rather than global_land_mask

    05/09/24: introduce high mountain mask for 2-D variables

    05/16/24: add SHMidLat-Ocn, HiLat-Ocn and Global-Ocn

    07/16/24: add Global-Lnd, SH-MidLat-Lnd, SH-HiLat-Ocn
    ''' 

    fillvalue = np.nan

    latspc = np.arange(-90,92.5,2.5)
    lonspc = np.arange(0,362.5,2.5)

    interp_method = 'nearest'
    kwargs = {"fill_value": "extrapolate"}
    #kwargs = {"fill_value": np.nan}
    #print(kwargs)

    # ============== Regrid data onto the same grid ========================
    data = data.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)

    omega700_pi_in = omega700_pi_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)
    omega700_ab_in = omega700_ab_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)
    omega700_avg_in = omega700_avg_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)

    # ============== get land mask ========================
    if landfrac is not None: 
        landfrac = landfrac.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)
        globe_land_mask = xr.where(landfrac>0,True,False) # (lat,lon)
        if len(data.shape) == 3: # (lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        elif len(data.shape) == 2: # (lat,lon)
            tmp = globe_land_mask
        globe_land_mask = xr.DataArray(tmp,coords=data.coords)
    else:
        lons_here = np.where(lonspc>180,lonspc-360,lonspc)
        lon_grid,lat_grid = np.meshgrid(lons_here,latspc)
        globe_land_mask = globe.is_land(lat_grid,lon_grid)
    
        if len(data.shape) == 4: # (time,lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],data.shape[1],1,1,))
        elif len(data.shape) == 3: # (lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        elif len(data.shape) == 2: 
            tmp = globe_land_mask
            
        globe_land_mask = xr.DataArray(tmp,coords=data.coords)
        

    # ============== Mask high mountain regions ===========
    if (high_mountain_mask is not None) and (len(data.shape) == 2): 
        high_mountain_mask = high_mountain_mask.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)
        data = xr.where(np.isnan(high_mountain_mask), np.nan, data)
        
       
    # ============== Map omega to the same dimension of data ===========
    if len(data.shape) == 4: # (time,lev,lat,lon)
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
    elif len(data.shape) == 3: # (lev,lat,lon)
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],1,1)), coords=data.coords)
    elif len(data.shape) == 2: 
        omega700_pi = omega700_pi_in
        omega700_ab = omega700_ab_in
        omega700_avg = omega700_avg_in
    

    # =============
    avg_flag = xr.zeros_like(data)
    
    w_threshold = 10. # 10 hPa

    if reg == 'TropMarineLow':
        data1 = data.where((data.lat>=-45)&(data.lat<45)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi>=w_threshold),data1,fillvalue)
        data2_ab = xr.where((omega700_ab>=w_threshold),data1,fillvalue)
        data2_avg = xr.where((omega700_avg>=w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 1)
    if reg == 'TropAscent':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi<w_threshold),data1,fillvalue)                
        data2_ab = xr.where((omega700_ab<w_threshold),data1,fillvalue)       
        data2_avg = xr.where((omega700_avg<w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 2)  
    if reg == 'NH-MidLat-Lnd':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat-Ocn':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_pi<w_threshold)),fillvalue)
        data2_ab = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_ab<w_threshold)),fillvalue)
        data2_avg = data.where(((data.lat<60)&(data.lat>=45)&(globe_land_mask==False)) | ((data.lat<45)&(data.lat>=30)&(globe_land_mask==False)&(omega700_avg<w_threshold)),fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat':
        data1 = data.where(((data.lat>=-60)&(data.lat<-45)),fillvalue) # select big region and mask land 
        data2_pi = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_pi<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        data2_ab = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_ab<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        data2_avg = data.where(((data.lat>=-60)&(data.lat<-45)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_avg<w_threshold)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(globe_land_mask==True)),fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat-Ocn':
        data1 = data.where(((data.lat>=-60)&(data.lat<-45)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data.where(((data.lat>=-60)&(data.lat<-45)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_pi<w_threshold)&(globe_land_mask==False)) ,fillvalue)
        data2_ab = data.where(((data.lat>=-60)&(data.lat<-45)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_ab<w_threshold)&(globe_land_mask==False)) ,fillvalue)
        data2_avg = data.where(((data.lat>=-60)&(data.lat<-45)&(globe_land_mask==False)) | ((data.lat>=-45)&(data.lat<-30)&(omega700_avg<w_threshold)&(globe_land_mask==False)),fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat-Lnd':
        data1 = data.where(((data.lat>=-60)&(data.lat<-30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'HiLat':
        data1 = data.where((((data.lat<-60))|((data.lat>=60))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'HiLat-Ocn':
        data1 = data.where(((data.lat<-60)|(data.lat>=60))&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'HiLat-Lnd':
        data1 = data.where(((data.lat<-60)|(data.lat>=60))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'NH-HiLat-Ocn':
        data1 = data.where((data.lat>=60)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'NH-HiLat-Lnd':
        data1 = data.where((data.lat>=60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'SH-HiLat-Ocn':
        data1 = data.where((data.lat<-60)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'SH-HiLat-Lnd':
        data1 = data.where((data.lat<-60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 

    if reg == 'TropLand':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10':
        data1 = data.where((data.lat>=-10)&(data.lat<10)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10to30':
        data1 = data.where((((data.lat>=-30)&(data.lat<-10))|((data.lat<30)&(data.lat>=10)))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-Africa':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(data.lon>0)&(data.lon<60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAmerica':
        data1 = data.where((data.lat>=-30)&(data.lat<10)&(data.lon>240)&(data.lon<330)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAsia':
        data1 = data.where((data.lat>=10)&(data.lat<30)&(data.lon>60)&(data.lon<120)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global-Ocn':
        data1 = data.where(globe_land_mask==False,fillvalue)
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global-Lnd':
        data1 = data.where(globe_land_mask==True,fillvalue)
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global':
        data1 = data
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = 1.0 

    # ================ fractional area ============================
    data2m_pi = xr.where(np.isnan(data2_pi),0.0,1.0)
    data2m_ab = xr.where(np.isnan(data2_ab),0.0,1.0)
    data2m_avg = xr.where(np.isnan(data2_avg),0.0,1.0)
    
    return data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,avg_flag


# -----------------------------------------------------------------------
def regime_partitioning_4(reg,omega700_pi_in,omega700_ab_in,omega700_avg_in,data,landfrac=None,high_mountain_mask=None):  
    '''
    Statc regime partitioning:
    change omega700 threshold from 0 to 10 hPa/day to have more restricted tropical marine low cloud regime.
    also, expand the marine low cloud regime range from 30S-30N to 45S-45N. 

    04/30/24: add another three TropLand regimes based on longitudes:
    TropLandAfrica: 0~60E, -30S~30S
    TropLandSouthAmerica: 120W~30W, 10N~30S
    TropLandSouthAsia: 60E~130E, 10S~30N

    05/05/24: decompose TropLand into TropLand-10 and TropLand-10to30

    05/07/24: use landfrac to do the mask rather than global_land_mask

    05/09/24: introduce high mountain mask for 2-D variables

    05/16/24: add SHMidLat-Ocn, HiLat-Ocn and Global-Ocn

    07/16/24: add Global-Lnd, SH-MidLat-Lnd, SH-HiLat-Ocn

    08/05/24: revert TropMarineLow back to 30S-30N 
              revert SH-MidLat and NH-MidLat  back to 30-60 N/S
    ''' 

    fillvalue = np.nan

    latspc = np.arange(-90,92.5,2.5)
    lonspc = np.arange(0,362.5,2.5)

    interp_method = 'nearest'
    kwargs = {"fill_value": "extrapolate"}
    #kwargs = {"fill_value": np.nan}
    #print(kwargs)

    # ============== Regrid data onto the same grid ========================
    data = data.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)

    omega700_pi_in = omega700_pi_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)
    omega700_ab_in = omega700_ab_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)
    omega700_avg_in = omega700_avg_in.interp(lat=latspc,lon=lonspc,method=interp_method,kwargs=kwargs)

    # ============== get land mask ========================
    if landfrac is not None: 
        landfrac = landfrac.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)
        globe_land_mask = xr.where(landfrac>0,True,False) # (lat,lon)
        if len(data.shape) == 3: # (lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        elif len(data.shape) == 2: # (lat,lon)
            tmp = globe_land_mask
        globe_land_mask = xr.DataArray(tmp,coords=data.coords)
    else:
        lons_here = np.where(lonspc>180,lonspc-360,lonspc)
        lon_grid,lat_grid = np.meshgrid(lons_here,latspc)
        globe_land_mask = globe.is_land(lat_grid,lon_grid)
    
        if len(data.shape) == 4: # (time,lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],data.shape[1],1,1,))
        elif len(data.shape) == 3: # (lev,lat,lon)
            tmp = np.tile(globe_land_mask,(data.shape[0],1,1,))
        elif len(data.shape) == 2: 
            tmp = globe_land_mask
            
        globe_land_mask = xr.DataArray(tmp,coords=data.coords)
        

    # ============== Mask high mountain regions ===========
    if (high_mountain_mask is not None) and (len(data.shape) == 2): 
        high_mountain_mask = high_mountain_mask.interp(lon=lonspc,lat=latspc,method=interp_method,kwargs=kwargs)
        data = xr.where(np.isnan(high_mountain_mask), np.nan, data)
        
       
    # ============== Map omega to the same dimension of data ===========
    if len(data.shape) == 4: # (time,lev,lat,lon)
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],data.shape[1],1,1)), coords=data.coords)
    elif len(data.shape) == 3: # (lev,lat,lon)
        omega700_pi = xr.DataArray(np.tile(omega700_pi_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_ab = xr.DataArray(np.tile(omega700_ab_in,(data.shape[0],1,1)), coords=data.coords)
        omega700_avg = xr.DataArray(np.tile(omega700_avg_in,(data.shape[0],1,1)), coords=data.coords)
    elif len(data.shape) == 2: 
        omega700_pi = omega700_pi_in
        omega700_ab = omega700_ab_in
        omega700_avg = omega700_avg_in
    

    # =============
    avg_flag = xr.zeros_like(data)
    
    w_threshold = 0. # 10 hPa

    if reg == 'TropMarineLow':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi>=w_threshold),data1,fillvalue)
        data2_ab = xr.where((omega700_ab>=w_threshold),data1,fillvalue)
        data2_avg = xr.where((omega700_avg>=w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 1)
    if reg == 'TropAscent':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = xr.where((omega700_pi<w_threshold),data1,fillvalue)                
        data2_ab = xr.where((omega700_ab<w_threshold),data1,fillvalue)       
        data2_avg = xr.where((omega700_avg<w_threshold),data1,fillvalue)
        avg_flag = xr.where(np.isnan(data1), avg_flag, 2)  
    if reg == 'NH-MidLat-Lnd':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'NH-MidLat-Ocn':
        data1 = data.where(((data.lat<60)&(data.lat>=30)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat':
        data1 = data.where(((data.lat>=-60)&(data.lat<-30)),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat-Ocn':
        data1 = data.where(((data.lat>=-60)&(data.lat<-30)&(globe_land_mask==False)),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'SH-MidLat-Lnd':
        data1 = data.where(((data.lat>=-60)&(data.lat<-30)&(globe_land_mask==True)),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 3) 
    if reg == 'HiLat':
        data1 = data.where((((data.lat<-60))|((data.lat>=60))),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'HiLat-Ocn':
        data1 = data.where(((data.lat<-60)|(data.lat>=60))&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'HiLat-Lnd':
        data1 = data.where(((data.lat<-60)|(data.lat>=60))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'NH-HiLat-Ocn':
        data1 = data.where((data.lat>=60)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'NH-HiLat-Lnd':
        data1 = data.where((data.lat>=60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'SH-HiLat-Ocn':
        data1 = data.where((data.lat<-60)&(globe_land_mask==False),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 
    if reg == 'SH-HiLat-Lnd':
        data1 = data.where((data.lat<-60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1 
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 4) 

    if reg == 'TropLand':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10':
        data1 = data.where((data.lat>=-10)&(data.lat<10)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-10to30':
        data1 = data.where((((data.lat>=-30)&(data.lat<-10))|((data.lat<30)&(data.lat>=10)))&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-Africa':
        data1 = data.where((data.lat>=-30)&(data.lat<30)&(data.lon>0)&(data.lon<60)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAmerica':
        data1 = data.where((data.lat>=-30)&(data.lat<10)&(data.lon>240)&(data.lon<330)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'TropLand-SouthAsia':
        data1 = data.where((data.lat>=10)&(data.lat<30)&(data.lon>60)&(data.lon<120)&(globe_land_mask==True),fillvalue) # select big region and mask land 
        data2_pi = data1
        data2_ab = data1 
        data2_avg = data1 
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global-Ocn':
        data1 = data.where(globe_land_mask==False,fillvalue)
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global-Lnd':
        data1 = data.where(globe_land_mask==True,fillvalue)
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = xr.where(np.isnan(data1), avg_flag, 5) 

    if reg == 'Global':
        data1 = data
        data2_pi = data1 
        data2_ab = data1
        data2_avg = data1
        avg_flag = 1.0 

    # ================ fractional area ============================
    data2m_pi = xr.where(np.isnan(data2_pi),0.0,1.0)
    data2m_ab = xr.where(np.isnan(data2_ab),0.0,1.0)
    data2m_avg = xr.where(np.isnan(data2_avg),0.0,1.0)
    
    return data2_pi,data2_ab,data2_avg,data2m_pi,data2m_ab,data2m_avg,avg_flag



# ============================================================================= 
def stratify_array(metric,datap0, nbins, bin_edges=None):  
    '''
    Stratify data based on binned metric.

    Inputs:
    metric: Array based on which datap0 will be stratified. (lat,lon)
    datap0: Array to be stratified. (lat,lon)
    nbins: number of bins for stratification. 
    bin_edges: pre-defined bin_edges. optional.

    Returns:
    stratified array (bin,lat,lon)

    '''
    # Define bin edges based on percentiles 
    if bin_edges is None: 
        bin_edges = np.nanquantile(metric.values.flatten(), np.linspace(0, 1, nbins+1))
        print('Using local bin_edges = ', bin_edges) 
    else:
        print('Using pre-defined bin_edges = ', bin_edges)

    # Define output
    if len(datap0.shape) == 2: # (lat,lon) 
        metric_3d = metric
        tmp = xr.DataArray(data=np.zeros((nbins,datap0.shape[0],datap0.shape[1])), 
                           coords={'bin': range(nbins), 'lat': datap0.lat, 'lon': datap0.lon}) 
        tmp[:] = np.nan 
        tmp_frac = xr.zeros_like(tmp) # Array to save the fractional area from each bin 

    elif len(datap0.shape) == 3: # (lev,lat,lon) 
         # Expand the metric into 3-D as the datap0 
         metric_3d = np.tile(metric,(datap0.shape[0],1,1)) 
         tmp = xr.DataArray(data=np.zeros((nbins,datap0.shape[0],datap0.shape[1],datap0.shape[2])),
                            coords={'bin': range(nbins), 'plev':datap0.plev, 'lat': datap0.lat, 'lon': datap0.lon})
         tmp[:] = np.nan
         tmp_frac = xr.zeros_like(tmp) 
        
    for ibin in range(nbins): 
        tmp[ibin,:] = xr.where((metric_3d>=bin_edges[ibin])&(metric_3d<bin_edges[ibin+1]), datap0, np.nan) 
        tmp_frac[ibin,:] = xr.where((metric_3d>=bin_edges[ibin])&(metric_3d<bin_edges[ibin+1]), 1.0, 0.0) 
            
    return tmp, tmp_frac, bin_edges

# ============================================================================= 
def get_bin_edges(nbins, data):
    '''
    Define bin edges. 

    Inputs:
    nbins: number of bins.
    data: input data.

    Return:
    bin_edges: bin edges. 
    '''
    bin_edges = np.zeros(nbins+1)
    if nbins == 5: 
        bin_edges[0] = -1e3
        bin_edges[-1] = 1e3
        bin_edges[1] = np.nanquantile(data.values.flatten(),0.2)
        bin_edges[2] = np.nanquantile(data.values.flatten(),0.4)
        bin_edges[3] = np.nanquantile(data.values.flatten(),0.6)
        bin_edges[4] = np.nanquantile(data.values.flatten(),0.8)
    else:
        print('You need to explicitly define your bin edges for nbins = ', nbins)
        exit()

    return bin_edges 

# ============================================================================= 
def plot_stratified_map(nbins, bin_edges, datap0_binned, datap0_binned_frac, figname): 
    '''
    Plot the stratified map based on stratified data

    Inputs:
    nbins: number of bins 
    bin_edges: bin edges for colorbar tick labeling 
    datap0_binned: stratified data array
    datap0_binned_frac: fractional area of each bin
    figname: output figure name

    Returns:
    None 
    '''
    from matplotlib.gridspec import GridSpec
    
    # --- Select one level for 3-D variables 
    if 'plev' in list(datap0_binned.dims): 
        datap = datap0_binned.sel(plev=50000) 
        datap_frac = datap0_binned_frac.sel(plev=50000) 
    else: 
        datap = datap0_binned
        datap_frac = datap0_binned_frac 

    figt = plt.figure(figsize=(6.5,6.5,),dpi=200) 
    nrowt, ncolt = 4,2
    
    gs = GridSpec(nrowt,ncolt) 
    
    # ----- Plot bin map 
    axt = figt.add_subplot(gs[0,0:2],projection=ccrs.PlateCarree(160))

    # Define customed colormap                 
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    cmap = mcolors.ListedColormap(tab10_colors[:nbins]) 

    sum = 0 # sum of fractional area which should be equal to the fractional area of this regime  
    jj = 0
    for jj in range(nbins): 
        lon2d,lat2d = np.meshgrid(datap.lon,datap.lat) # get meshed lat and lon

        # datap = xr.where(~np.isnan(datap0_binned[jj]),jj,np.nan) # set grid boxes where values are meaningful to bin index for plotting  
        datapp = xr.where(datap_frac[jj]==1,jj,np.nan) # set grid boxes where values are meaningful to bin index for plotting  

        axt.scatter(lon2d,lat2d,c=datapp, s=1, transform=ccrs.PlateCarree(),cmap=cmap,vmin=0,vmax=nbins)   
        axt.coastlines()

        avg = area_averager(datap_frac[jj]).values
        print(jj, avg) 

        sum += avg 
    print('sum = ', sum) 

    frac_avg = area_averager(datap_frac).values 
    axt.set_title('Fractional area of each bin = '+str(frac_avg.round(3))) 

    # Create a ScalarMappable object
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=nbins)) 
    sm.set_array([])
    cbar = plt.colorbar(sm,
                        orientation='horizontal'
                        ) 
    cbar.set_ticks(range(nbins+1))
    cbar.set_ticklabels([f'{bin_edges[i].round(2)}' for i in range(nbins+1)]) 
    cbar.set_label('Bins') 

    # -------- Plot data for each bin 
    for jj in range(nbins):
        irow = jj//ncolt + 1
        icol = jj%ncolt 
        # print(irow,icol) 
        axt = figt.add_subplot(gs[irow,icol],projection=ccrs.PlateCarree(160))

        datapp = datap[jj,:] 
        im = axt.contourf(datapp.lon, datapp.lat, datapp, transform=ccrs.PlateCarree())  
        axt.coastlines()

        axt.set_xlim([-180, 180]) 

        figt.colorbar(im,
                    #   fraction=0.05,
                      orientation='horizontal')  
        
        axt.set_title(str(bin_edges[jj].round(2))+' ~ '+str(bin_edges[jj+1].round(2))) 

    figt.tight_layout() 
    figt.savefig(figname,bbox_inches='tight',dpi=200)
    
    return None 

# ============================================================================= 

# ============================================================================= 

# -----------------------------------------------------------------------
# ## Func StretechOutNormalize
# -----------------------------------------------------------------------
# define a colormap
# cmap0 = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Blues(np.arange(255))])

class StretchOutNormalize(plt.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# ## Func save_big_dataset
# -----------------------------------------------------------------------
def save_big_dataset(dic_mod,outfile):
    '''
    create a big dataset based on all variables in a dictionary and save to netcdf file.
    '''
    datalist = []
    for svar in dic_mod.keys():
        data = xr.DataArray(dic_mod[svar],name=svar)
        datalist.append(data)

    data_big = xr.merge(datalist,compat='override')

    #data_big.to_netcdf(outfile,encoding={'time':{'dtype': 'i4'},'bin':{'dtype':'i4'}})
    data_big.to_netcdf(outfile)
