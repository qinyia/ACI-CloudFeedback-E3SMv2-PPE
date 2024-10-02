'''Builds on the 2020 forcing decomposition paper, creating a mimimal version
without all the extra diagnostics used for testing'''

'''
11/16/23 added by Yi Qin: use offline regression to calculate dalbedo/dLWP with filtered samples
                          based on kmean discretization method to avoid the albedo saturation effect.
'''

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import traceback
import glob
#import misc
#import misc.stats
#import misc.geo
from calc_solar_insolation import calculate_insolation # YQIN
import datetime
import cartopy
import cartopy.crs as ccrs
from shapely.geometry import MultiLineString
from copy import copy
import pickle
import progressbar
import os

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
import time as timer

# Careful!!!
import warnings
warnings.filterwarnings("ignore")

datadir = '/compyfs/qiny108/decomp_ERFaci_Gryspeerdt_postdata/'
outdir = '/qfs/people/qiny108/ppe_scripts/decomp_ERFaci_Gryspeerdt/data/'
#outdir = '/qfs/people/qiny108/ppe_scripts/decomp_ERFaci_Gryspeerdt/'


# total number of time stamp
#ntime = 29201
#ntime = 100
ntime = 2721

# only for test
lat1 = 45
lon1 = 100 


#tag = 'logLWPtrue_ntime-'+str(ntime)
#tag = 'RegAfFilter_ntime-'+str(ntime)
#tag = 'RegAfFilter_ntime-'+str(ntime)+'_0412'
tag = 'RegAfFilter_ntime-'+str(ntime)+'_0425'


# //////////////////////////////////////////////////////////////////////////
def get_offline_albcld_lwp_sens(lwp,albcld,min_samples=20):
    '''
    Calculate d(albedo)/d(lwp) after excluding albedo saturation feature.

	Usage:
		output = get_offline_albcld_lwp_sens(datax[:1000,:], datay[:1000,:])
		print(output.shape)
    '''
#    data_nan_indices = np.isnan(lwp)
#    target_nan_indices = np.isnan(albcld)
#    nan_indices_combined = data_nan_indices | target_nan_indices

    data = lwp#[~nan_indices_combined]
    target = albcld#[~nan_indices_combined]

	# How many bins do you separate the raw data? 
    n_bin = 2 

    # Reshape the data and target arrays
    time, lat, lon = data.shape
    data_reshaped = data.reshape((time, lat * lon))
    target_reshaped = target.reshape((time, lat * lon))

    regression_model = LinearRegression()

    coefs = np.zeros_like(data_reshaped[0,:]) 
    coefs_filtered = np.zeros_like(data_reshaped[0,:]) 

    time0 = timer.time() 
    for i in range(lat * lon):
        # =========== Use all samples =======================================
        mask = ~np.isnan(data_reshaped[:,i]) & ~np.isnan(target_reshaped[:,i])
        this_X = data_reshaped[:,i][mask].reshape(-1,1)
        this_Y = target_reshaped[:,i][mask]

        if len(this_X) > min_samples: 
            regression_model.fit(this_X, this_Y)
            coef = regression_model.coef_[0]   
            coefs[i] = coef 

        # =========== Use samples after binning =======================================    
        if len(this_X) > min_samples: 
            enc = KBinsDiscretizer(n_bins=n_bin, encode="onehot",strategy='kmeans')
            X_binned = enc.fit_transform(this_X)
            X_limit = enc.bin_edges_[0][-2]
            mask = (~np.isnan(data_reshaped[:,i]))&(~np.isnan(target_reshaped[:,i]))&(data_reshaped[:,i]<X_limit)
            this_Xnew = data_reshaped[:,i][mask].reshape(-1,1)        
            this_Ynew= target_reshaped[:,i][mask] 

            if len(this_Xnew) > min_samples:
                regression_model.fit(this_Xnew, this_Ynew)
                coef2 = regression_model.coef_[0]
                coefs_filtered[i] = coef2 

#         # ========== Plotting check ====================== 
#         if i%10000 == 0 and len(this_X) > min_samples: 
#             fig = plt.figure(figsize=(9,9))
#             ax = fig.add_subplot(1,1,1)
#             ax.scatter(this_X, this_Y)

#             X_plot = np.linspace(this_X.min(), this_X.max())
#             regression_model.fit(this_X, this_Y)
#             Y_plot = regression_model.predict(X_plot[:, np.newaxis]) 
#             ax.plot(X_plot, Y_plot, label='All samples') 

#             X_plot_new = np.linspace(this_Xnew.min(), this_Xnew.max())
#             regression_model.fit(this_Xnew, this_Ynew)
#             Y_plot_new = regression_model.predict(X_plot_new[:, np.newaxis]) 
#             ax.plot(X_plot_new, Y_plot_new, label='Filtered samples') 

#             ax.legend()
#             print(f'i={i}, coef={coef}, coef2={coef2}') 

#             time1 = timer.time() 
#             # Print the execution time
#             execution_time = time1 - time0
#             print(f'i = {i}, Execution Time: {execution_time} seconds') 

    time2 = timer.time() 
    # Print the execution time
    execution_time = time2 - time0
    print(f'Total execution Time: {execution_time} seconds') 
    return coefs_filtered.reshape((lat,lon)) 

# //////////////////////////////////////////////////////////////////////////
def lat_weighted_av(data,latitudes):
    weights = np.cos(np.radians(latitudes))
    # expand into the same size as data
    weights_nd = np.transpose(np.tile(weights,(data.shape[1],1)),(1,0))
    mask = ~np.isnan(data)
    weighted_mean = np.ma.average(data[mask], weights=weights_nd[mask])
    return weighted_mean

def lwav(data,latitudes):
    return lat_weighted_av(data,latitudes)


#def custom_setup():
#    # Reduce the resolution of the cartopy map to make smaller size images
#    feature = cartopy.feature.NaturalEarthFeature(
#        'physical', 'coastline', '110m')
#    f = []
#    for x in list(feature.geometries()):
#        b = x.bounds
#        if misc.geo.haversine(b[1], b[0], b[3], b[2]) > 450:
#            f.append(MultiLineString(
#                [x.simplify(0.75, preserve_topology=False)]))
#
#    cartopy.feature._NATURAL_EARTH_GEOM_CACHE[
#        ('coastline', 'physical', 'custom')] = f
#
#
#custom_setup()


def get_rsdt_time(utc, lon=None, lat=None, atm_corr=False):
    if lon is None:
        lat = np.arange(-88.75, 90, 2.5)
        lon = np.arange(1.25, 360, 2.5)

    try:
        doy = utc.timetuple().tm_yday
    except:
        doy = (utc -
               netcdftime._netcdftime.Datetime360Day(
                   utc.year, 1, 1)).days + 1
    utc_hour = utc.hour + utc.minute/24
        
    rsdt = np.fromfunction(
        lambda y, x: astro.insolation(
            np.radians(lat)[y.astype('int')],
            doy,
            lon[x.astype('int')]/15 + utc_hour,
            atm_corr),
        (len(lat), len(lon)))

    rsdt[rsdt<134] = np.nan
    return rsdt


class ModelData():
    def __init__(self, model, toffset, doffset, full_analysis=True, ens='r1i1p1', pd=False):
        # This sets up the arrays for storing model output. To save on memory, timesteps
        # are processed individually, requiring these arrays to accumulate output
        self.model = model
        # Time offset for each 3 hour period - required for calcualting rsdt
        # Not all models have the timestep at the time they claim
        self.toffset = toffset
        self.doffset = doffset  # Day offset (some models have a different calendar)
        self.pd = pd
        # Only for CMIP5 models (some are missing required output)
        self.full_analysis = full_analysis
        self.variables = ['cdnc', 'tcc', 'lwp', 'icc', 'lcc', 'iwp',
                          'rsut', 'rsutcs', 'rsutnoa', 'rsutcsnoa',
                          'rlut', 'rlutcs', 'od550', 'rsdt']

        modelfolder = (datadir+model+'/latlon/')
#        modelfolder = (datadir+model+'/latlon_73x144/')

        self.files = {}
        for name in self.variables:
            if pd is True:
                pattern = (
                    f'{modelfolder}/' +
                    f'{name}_PD_{model}.nc')
                #print(pattern)
                filename = [f for f in glob.glob(pattern)]
            else:
                pattern = (
                    f'{modelfolder}/' +
                    f'{name}_PI_{model}.nc')
                filename = [f for f in glob.glob(pattern)]
            if (len(filename) > 1) or (len(filename) == 0):
                print(pattern, filename)
                raise(IOError)
            self.files[name] = Dataset(filename[0], 'r')

        # A lot of this time calculation is required to make sure the PD and PI data match
        # and for calculating the incoming solar radiation
        Vtime = self.files[self.variables[0]].variables['time']
        vtimes = Vtime[:]
        vtimes[vtimes > 80000] = 6000
        dates = num2date(vtimes, units=Vtime.units, calendar=Vtime.calendar)
        self.doys = np.array(list(
            map(lambda x: int(x.strftime("%j")), dates)))
        self.doys[vtimes > 80000] = -1
        self.hours = np.array(list(
            map(lambda x: float(x.strftime("%H")), dates)))
        self.lon = self.files[self.variables[0]].variables['lon'][:]
        self.lat = self.files[self.variables[0]].variables['lat'][:]
        
        self.shape = self.files[self.variables[0]].variables[self.variables[0]].shape[1:]
        # Accumulated variables
        # - final char
        #  - l - low/liquid
        #  - h - high/ice
        #  - i - high/ice but not thin (IWP>thresh). This is required 
        self.outputvars = [
            # Incoming solar - can be calculated if required (1)
            'rsdt',
            # Various albedos are needed for the different components (8)
            'alb', 'albnoa',
            'albcs', 'albcsnoa',
            'albcld', 'albcldnoa',
            'albcldcs', 'albcldcsnoa',
            # These are for the LW decomposition (4)
            'rlut', 'rlutcs', 'rlutcld', 'rlutcldcs',
            # Required for forcing calculation (2)
            'TCC', 'LWP',
            # Needed for threshold calculation ('i' variables) (1)
            'IWP',
            # # Probably no longer needed - cfl, no ice gridboxes (1)
            'cfl_noi',
            # For the LWP regression against albcldl (6)
            'LWPl', 'lsLWPl', 'lsalbcldlnoa',
            'lsLWPlsq', 'lsalbcldlnoasq', 
            'lsLWPlalbcldlnoa', 
            # For the LWP regression against albcldl but using log LWP - YQIN
            'lslnLWPl', 
            'lslnLWPlsq',
            'lslnLWPlalbcldlnoa',
            # Required for liquid-ice decomposition (3)
            'cfl', 'cfh', 'cfi',
            # Required for liquid-ice decomposition (6)
            'albcldl', 'albcldlnoa',
            'albcldh', 'albcldhnoa',
            'albcldi', 'albcldinoa',
            # cloud-clear sky albedo - used for cloud fraction compoennt (6)
            'albcldcsl', 'albcldcslnoa',
            'albcldcsh', 'albcldcshnoa',
            'albcldcsi', 'albcldcsinoa',
            # Liquid-ice decomposition for longwave (6)
            'rlutcldl', 'rlutcldh', 'rlutcldi',
            'rlutcldcsl', 'rlutcldcsh', 'rlutcldcsi',
        ]

        self.output = {}
        for opname in self.outputvars:
            self.output[opname] = np.zeros(self.shape)
            self.output[opname+'_num'] = np.zeros(self.shape)

            if opname in ['lsLWPl', 'lsalbcldlnoa','lslnLWPl']:
                self.output[opname+'_3d'] = np.zeros((ntime, self.shape[0], self.shape[1]))

        self.file_index = 0        
        self.rsdt = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()
        
    def __del__(self):
        for name in self.files.keys():
            self.files[name].close()

    def read_data(self, index):
        # Read in the required variables - not all are required for
        # the forcing calculation though, Nd and AOD are only used for
        # subsequent analysis
        outputdata = {}
        for name in self.variables:
            outputdata[name] = self.files[name].variables[name][index]
            outputdata[name][np.where((outputdata[name] < -1) |
                                      (outputdata[name] > 1e15))] = np.nan
        if ('CND' in self.model) or ('anthsca' in self.model) or ('HadGEM' in self.model):
            outputdata['rsutnoa'] = outputdata['rsut']
            outputdata['rsutcsnoa'] = outputdata['rsutcs']

        if ('HadGEM' not in self.model) and ('UKESM' not in self.model):
            outputdata['cdnc'] /= 1000000
            
        # SPRINTARS appears to not have CF weighted cloud properties
        if ('SPRINTARS' not in self.model) and ('AACR' not in self.model):
            outputdata['cdnc'] /= outputdata['tcc']
            outputdata['lwp'] /= outputdata['tcc']
            outputdata['iwp'] /= outputdata['tcc']
            outputdata['lwp'][outputdata['tcc'] < 0.05] = 0
            outputdata['lwp'][outputdata['lwp'] <= 0] = 0

            # Added by YQIN
            outputdata['cdnc'][outputdata['tcc']==0] = np.nan
            outputdata['lwp'][outputdata['tcc']==0] = np.nan
            outputdata['iwp'][outputdata['tcc']==0] = np.nan 

        datatime = num2date(
            self.files['rsut'].variables['time'][index],
            self.files['rsut'].variables['time'].units)
#        outputdata['rsdt'] = get_rsdt_time(
#            datatime+datetime.timedelta(
#                hours=self.toffset(index)+24*self.doffset),
#            self.lon,
#            self.lat)

        mask = (outputdata['rsut'] == 0)
        for name in outputdata.keys():
            outputdata[name][mask] = np.nan

        # Mask cases when rsdt is small - suggested by Ed G. - YQIN 04/12/24
        outputdata['rsdt'][outputdata['rsdt']<134] = np.nan

        outputdata['rsdtnan'] = copy(outputdata['rsdt'])
        #outputdata['rsdt'] = misc.stats.zero_nans(outputdata['rsdt']) # YQIN: replace it by the following command
        outputdata['rsdt'] = np.nan_to_num(outputdata['rsdt'],nan=0.0)

        # Set MaskedArray to ndarray  - YQIN
        for name in outputdata.keys():
            outputdata[name] = outputdata[name].filled(np.nan) 

        return outputdata


    def process_index(self, index, icclim, cflim, iwplim):
        # This function processes the data for each timestep, storing the results in
        # output arrays, if running the decomposition online, I would suggest this is the
        # part that could be implemented in the model itself (reducing required output)
        data = self.read_data(index)

        # Determine where data is liquid, ice or thick ice (iwp lim)
        liqmask = data['icc'] < icclim
        icemask = data['lcc'] < icclim
        iwpmask = (data['iwp'] > iwplim)
        clrmask = data['tcc'] < cflim
        
        procdata = {}
        # Variables copied
        procdata['cfl'] = data['lcc']
        procdata['cfh'] = data['icc']
        procdata['TCC'] = data['tcc']
        # Thick ice cloud fraction
        procdata['cfi'] = np.where(iwpmask, procdata['cfh'], 0)
        # Liquid cloud fraction is cases with no ice. Required for comparison with satellite values
        procdata['cfl_noi'] = np.where(liqmask, procdata['cfl'], np.nan)
        
        # Set values in clear-sky regions to zero
        procdata['LWP'] = data['lwp']
        procdata['LWP'][clrmask==1] = 0
        procdata['IWP'] = data['iwp']
        procdata['IWP'][clrmask==1] = 0

        # Shortwave variables
        procdata['rsdt'] = data['rsdt']
        procdata['alb'] = data['rsut']/data['rsdtnan']
        procdata['albnoa'] = data['rsutnoa']/data['rsdtnan']
        procdata['albcs'] = data['rsutcs']/data['rsdtnan']
        procdata['albcsnoa'] = data['rsutcsnoa']/data['rsdtnan']
        
        # Aim to catch errors in rsdt calculation (if calculated offline)
        flag = 100*((procdata['alb']>=1).sum() + (procdata['alb']<=0.05).sum())/np.isfinite(procdata['alb']).sum()
        if flag > 1:
            raise ValueError('Error in rsdt calculation')

        # Calculate cloud albedos
        procdata['albcld'] = ((procdata['alb']-procdata['albcs']*(1-procdata['TCC'])) /
                              (procdata['TCC']))
        procdata['albcld'][(procdata['albcld'] > 1) *
                           np.isfinite(procdata['albcld'])] = 1
        procdata['albcld'][clrmask==1] = np.nan

        procdata['albcldnoa'] = ((procdata['albnoa']-procdata['albcsnoa']*(1-procdata['TCC'])) /
                                 (procdata['TCC']))
        procdata['albcldnoa'][(procdata['albcldnoa'] > 1) *
                              np.isfinite(procdata['albcldnoa'])] = 1
        procdata['albcldnoa'][clrmask==1] = np.nan

        procdata['albcldcs'] = procdata['albcld']-procdata['albcs']
        procdata['albcldcsnoa'] = procdata['albcldnoa']-procdata['albcsnoa']

        # Longwave variables
        procdata['rlut'] = data['rlut']
        procdata['rlutcs'] = data['rlutcs']
        procdata['rlutcld'] = ((data['rlut']-data['rlutcs']*(1-data['tcc'])) /
                               (data['tcc']))
        procdata['rlutcld'][clrmask==1] = np.nan
        procdata['rlutcldcs'] = procdata['rlutcld'] - procdata['rlutcs']

        # LWP and CDNC for liquid cloud only
        procdata['LWPl'] = np.where(liqmask, procdata['LWP'], np.nan)

        # Removing the zero LWP points has a significant effect on the slope
        #  If LWP zeros are out, removing CDNC zeros does very little
        # We dont' care so much about the completely clear cases (they are rare in multi-year means)
#        lsliqmask = liqmask*(procdata['LWP']>0)
        lsliqmask = liqmask*(procdata['LWP']>1e-7)

        procdata['lsLWPl'] = np.where(lsliqmask, procdata['LWP'], np.nan)
        procdata['lsLWPlalbcldlnoa'] = np.where(lsliqmask, procdata['LWP']*procdata['albcldnoa'], np.nan)
        procdata['lsLWPlsq'] = np.where(lsliqmask, procdata['LWP']**2, np.nan)
        procdata['lsalbcldlnoasq'] = np.where(lsliqmask, procdata['albcldnoa']**2, np.nan)
        procdata['lsalbcldlnoa'] = np.where(lsliqmask, procdata['albcldnoa'], np.nan)

        procdata['lslnLWPl'] = np.where(lsliqmask, np.log(procdata['LWP']), np.nan)
        procdata['lslnLWPlsq'] = np.where(lsliqmask, np.log(procdata['LWP'])**2, np.nan)
        procdata['lslnLWPlalbcldlnoa'] = np.where(lsliqmask, np.log(procdata['LWP'])*procdata['albcldnoa'], np.nan)

        # Store cloud albedos for the cloud types
        procdata['albcldl'] = np.where(liqmask, procdata['albcld'], np.nan)
        procdata['albcldh'] = np.where(icemask, procdata['albcld'], np.nan)
        procdata['albcldi'] = np.where(iwpmask, procdata['albcld'], np.nan)
        procdata['albcldcsl'] = np.where(liqmask, procdata['albcldcs'], np.nan)
        procdata['albcldcsh'] = np.where(icemask, procdata['albcldcs'], np.nan)
        procdata['albcldcsi'] = np.where(iwpmask, procdata['albcldcs'], np.nan)
        procdata['albcldlnoa'] = np.where(liqmask, procdata['albcldnoa'], np.nan)
        procdata['albcldhnoa'] = np.where(icemask, procdata['albcldnoa'], np.nan)
        procdata['albcldinoa'] = np.where(iwpmask, procdata['albcldnoa'], np.nan)
        procdata['albcldcslnoa'] = np.where(liqmask, procdata['albcldcsnoa'], np.nan)
        procdata['albcldcshnoa'] = np.where(icemask, procdata['albcldcsnoa'], np.nan)
        procdata['albcldcsinoa'] = np.where(iwpmask, procdata['albcldcsnoa'], np.nan)

        procdata['rlutcldl'] = np.where(liqmask, procdata['rlutcld'], np.nan)
        procdata['rlutcldh'] = np.where(icemask, procdata['rlutcld'], np.nan)
        procdata['rlutcldi'] = np.where(iwpmask, procdata['rlutcld'], np.nan)
        procdata['rlutcldcsl'] = np.where(liqmask, procdata['rlutcldcs'], np.nan)
        procdata['rlutcldcsh'] = np.where(icemask, procdata['rlutcldcs'], np.nan)
        procdata['rlutcldcsi'] = np.where(iwpmask, procdata['rlutcldcs'], np.nan)

        return procdata

    def append_index(self, index, icclim, cflim, iwplim):
        data = self.process_index(index, icclim, cflim, iwplim)
        for name in data.keys():
            locs = np.isfinite(data[name]).data
            np.add.at(self.output[name], locs, data[name][locs])
            np.add.at(self.output[name+'_num'], locs, 1)

            if name in ['lsLWPl', 'lsalbcldlnoa','lslnLWPl']:
                self.output[name+'_3d'][index,:,:] = data[name]

    def get_output(self, name):
        return self.output[name]/self.output[name+'_num']

    def get_albcld_lwp_sens(self, log=False):
        num = self.output['lsLWPl_num']
        if not log:
            sxx = self.output['lsLWPlsq']-(self.output['lsLWPl']**2)/num
            syy = self.output['lsalbcldlnoasq']-(self.output['lsalbcldlnoa']**2)/num
            sxy = self.output['lsLWPlalbcldlnoa']-(self.output['lsLWPl']*self.output['lsalbcldlnoa'])/num
        else:
            sxx = self.output['lslnLWPlsq']-(self.output['lslnLWPl']**2)/num
            syy = self.output['lsalbcldlnoasq']-(self.output['lsalbcldlnoa']**2)/num
            sxy = self.output['lslnLWPlalbcldlnoa']-(self.output['lslnLWPl']*self.output['lsalbcldlnoa'])/num
        op = (sxy/sxx), (np.sqrt((1./(num-2))*((syy/sxx)-((sxy/sxx)**2))))
        return op[0], op[1], np.where(np.abs(op[0])>2*np.abs(op[1]), op[0], 0)


######################################################################
# Main program
######################################################################

if __name__=='__main__':
    icclim = 0.02
    cflim = 0.01
    iwplim = 0.0087 # 8.7gm-2 ~ tau=0.4 ~ MODIS cloud mask limit

    exps = ['BASE','c1','c1_2','c1_3','c1_4','gamma_coef','gamma_coef_2','gamma_coef_3','gamma_coef_4','c8','c8_2','c8_3','c8_4','gamma_coefb','gamma_coefb_2','gamma_coefb_3']
#    exps = ['Casey.exp']

    # model name, toffset, doffset
    models = [(exp, lambda x: 0, 0) for exp in exps]
#    models = [
#              ('Casey.exp', lambda x: 0, 0),
#              ]

    for model in models:
        if os.path.isfile(outdir+'Table_ERF_decomp_'+model[0]+'_'+tag+'.txt'):
            print(model[0]+' is done.')
            print()
            continue 

        print(model[0])
        modeldata = {'PD': ModelData(*model, pd=True),
                     'PI': ModelData(*model, pd=False)}

        bar = progressbar.ProgressBar(max_value=modeldata['PD'].files['rsut'].variables['rsut'].shape[0], term_width=80)
        i = 0
        missed = 0
        while True:
            try:
                if i%25 == 0:
                    bar.update(i)
                modeldata['PD'].append_index(i, icclim=icclim, cflim=cflim, iwplim=iwplim)
                modeldata['PI'].append_index(i, icclim=icclim, cflim=cflim, iwplim=iwplim)
#                if i > 365*8*5:
                if i > ntime:
                    break
                i += 1
            except IndexError:
                break
            except KeyboardInterrupt:
                break
            except OverflowError:
                i += 1
                missed += 1
                continue
            except ValueError:
                i += 1
                missed += 1
                continue
        print('\nMissed: ', missed)

        # YQIN: get lat and lon for global-average calculation
        latitudes = modeldata['PD'].files['rsut'].variables['lat'][:]
        longitudes = modeldata['PD'].files['rsut'].variables['lon'][:]

        # Given the processed/accumulated timestep data, calculate the forcing terms
        # This part could in theory be done online, but is really a post-processing
        # step, so I recommend doing it with output data.
        output = {}
        # First, get the PD-PI diffference in values
        for name in modeldata['PD'].outputvars:
            output[name] = modeldata['PD'].get_output(name)
            output['d'+name] = modeldata['PD'].get_output(name) - modeldata['PI'].get_output(name)

        output['num_liq'] = modeldata['PD'].output['lsLWPl_num']
        output['alb_lwp_sens'], output['alb_lwp_sens_err'], albsens = modeldata['PD'].get_albcld_lwp_sens(log=False)

        output['alb_lwp_sens_log'], _, albsens_log = modeldata['PD'].get_albcld_lwp_sens(log=True)

        ## ============= Save data for offline test ================================
        #for name in ['lsLWPl', 'lsalbcldlnoa','lslnLWPl']:
        #    np.save(name+'_3d',modeldata['PD'].output[name+'_3d'])

        # ============= Offline regression to get dalbedo/dLWP ==========================
        albsens_offline = get_offline_albcld_lwp_sens(modeldata['PD'].output['lsLWPl_3d'], modeldata['PD'].output['lsalbcldlnoa_3d'])
        print(f'albsens_offline.shape = {albsens_offline.shape}')
        print()

        retdata = {}
        # Shortwave forcing
        retdata['F_SW'] = -output['rsdt']*output['dalb'] # Delta_SW
        # Clearsky changes
        retdata['F_dalbcs'] = -output['rsdt']*output['dalbcs']*(1-output['TCC'])
        retdata['F_dalbsurf'] = -output['rsdt']*output['dalbcsnoa']*(1-output['TCC'])   # Delta_Surf
        # SW RFari (clear-sky - surface)
        retdata['F_RFari'] = retdata['F_dalbcs'] - retdata['F_dalbsurf'] # SW_ari_clr
        # Cloud albedo changes (total)
        retdata['F_dalbc'] = -output['rsdt']*output['dalbcld']*output['TCC']
        retdata['F_dalbcnoa'] = -output['rsdt']*output['dalbcldnoa']*output['TCC'] # Delta_SW_c
        # SW RFari (cloudy-skies)
        retdata['F_RFaric'] = retdata['F_dalbc'] - retdata['F_dalbcnoa'] # SW_aric_c

        #########################################################
        # Now let's start calculating the liquid/ice components #
        #########################################################

        # Here is the cloud albedo part #
        #################################
        retdata['F_dalbclnoa'] = -output['rsdt']*output['dalbcldlnoa']*output['cfl']  # Delta_SW_cl
        retdata['F_dalbchnoa'] = -output['rsdt']*output['dalbcldhnoa']*output['cfh']  # Delta_SW_ch - all ice cld fraction
        retdata['F_dalbcinoa'] = -output['rsdt']*output['dalbcldinoa']*output['cfi']  # Delta-SW_ci - thick ice cld fraction
        # And the relevant forcings, attributing thin high cloud changes to low level liquid
        retdata['thin_high_adj'] = (retdata['F_dalbchnoa']-retdata['F_dalbcinoa'])
        retdata['F_dalbcl'] = retdata['thin_high_adj'] + retdata['F_dalbclnoa']
        retdata['F_dalbci'] = retdata['F_dalbcinoa']

        # LWP forcing (constant Nd) - linear regression
        retdata['albsens'] = albsens
        retdata['albsens_log'] = albsens_log
        retdata['albsens_off'] = albsens_offline

        retdata['F_dalbc_lwp_off'] = -output['rsdt']*output['cfl']*albsens_offline*output['dlsLWPl']
        retdata['F_dalbc_lwp_log'] = -output['rsdt']*output['cfl']*albsens_log*output['dlslnLWPl']  # Delta_albl_L Eq (8)
        retdata['F_dalbc_lwp_off'][output['num_liq'] < 100] = 0
        retdata['F_dalbc_lwp_log'][output['num_liq'] < 100] = 0

        retdata['F_dalbc_lwp'] = -output['rsdt']*output['cfl']*albsens*output['dlsLWPl']  # Delta_albl_L Eq (8)
        retdata['F_dalbc_lwp'][output['num_liq'] < 100] = 0

        # Twomey effect (constant LWP)
        retdata['F_dalbc_nd_off'] = retdata['F_dalbclnoa']-retdata['F_dalbc_lwp_off'] # Delta_albl_Nd Eq (9)
        retdata['F_dalbc_nd_log'] = retdata['F_dalbclnoa']-retdata['F_dalbc_lwp_log'] # Delta_albl_Nd Eq (9)
        retdata['F_dalbc_nd'] = retdata['F_dalbclnoa']-retdata['F_dalbc_lwp'] # Delta_albl_Nd Eq (9)

        # Distribute the ice-masked cloud forcing between the LWP and Nd components
        extra_f = retdata['F_dalbc_nd']/(retdata['F_dalbc_nd']+retdata['F_dalbc_lwp'])
        extra_f = np.clip(extra_f, 0, 1)
        retdata['F_dalbc_nd'] += extra_f*retdata['thin_high_adj']
        retdata['F_dalbc_lwp'] += (1-extra_f)*retdata['thin_high_adj']

        extra_f = retdata['F_dalbc_nd_off']/(retdata['F_dalbc_nd_off']+retdata['F_dalbc_lwp_off'])
        extra_f = np.clip(extra_f, 0, 1)
        retdata['F_dalbc_nd_off'] += extra_f*retdata['thin_high_adj']
        retdata['F_dalbc_lwp_off'] += (1-extra_f)*retdata['thin_high_adj']

        extra_f = retdata['F_dalbc_nd_log']/(retdata['F_dalbc_nd_log']+retdata['F_dalbc_lwp_log'])
        extra_f = np.clip(extra_f, 0, 1)
        retdata['F_dalbc_nd_log'] += extra_f*retdata['thin_high_adj']
        retdata['F_dalbc_lwp_log'] += (1-extra_f)*retdata['thin_high_adj']

        # And the cloud fraction change components #
        ############################################
        retdata['F_dcfnoa'] = -output['rsdt']*output['dTCC']*output['albcldcsnoa']  # Delta_SW_cf
        retdata['F_dcflnoa'] = -output['rsdt']*output['dcfl']*output['albcldcslnoa']
        retdata['F_dcfhnoa'] = -output['rsdt']*output['dcfh']*output['albcldcshnoa']
        retdata['F_dcfinoa'] = -output['rsdt']*output['dcfi']*output['albcldcsinoa']

        retdata['F_SW_sum'] = retdata['F_dalbsurf'] + retdata['F_RFari'] + retdata['F_RFaric'] + retdata['F_dalbcnoa'] + retdata['F_dcfnoa']  # YQIN
        retdata['F_SW_res'] = retdata['F_SW'] - retdata['F_SW_sum']  # YQIN
        
        # The liquid cloud fraction change has to be corrected for changes in overlying ice cloud
        # Assumes the ice and liquid cloud fraction changes are uncorrelated
        output['dcfl_corr'] = output['dcfl'] + output['dcfh']*(output['cfl']/(1-output['dcfh']))
        # Forcing from liquid cloud changes (corrected)
        retdata['F_dcfl'] = -output['rsdt']*output['dcfl_corr']*output['albcldcslnoa']
        # And make the corresponging adjustment to the ice cloud fraction forcing
        retdata['F_dcfi'] = retdata['F_dcfhnoa']+(retdata['F_dcflnoa']-retdata['F_dcfl'])

        # Mask cases with little liquid cloud? Not sure if this is necessary though
        retdata['F_dalbcl'][output['cfl'] < 0.02] = 0
        retdata['F_dalbc_lwp'][output['cfl'] < 0.02] = 0
        retdata['F_dcfl'][output['cfl'] < 0.02] = 0

        # YQIN: Find strange value...
        idx = np.argwhere(retdata['F_dalbc_lwp_off']>1000)
        if len(idx) > 0:
            print('Check your F_dalbc_lwp_off data.')
            #exit()

        #####################
        # Longwave forcings #
        #####################
        retdata['F_LW'] = -output['drlut']
        retdata['F_dlwcs'] = -output['drlutcs']*(1-output['TCC'])  # LW_ari_cs
        retdata['F_dlwc'] = -output['drlutcld']*output['TCC'] # Delta_LW_c
        retdata['F_dlwcl'] = -output['drlutcldl']*output['cfl']
        retdata['F_dlwch'] = -output['drlutcldh']*output['cfh']
        retdata['F_dlwci'] = -output['drlutcldi']*output['cfi']

        retdata['F_dlwcf'] = -output['dTCC']*output['rlutcldcs'] # Delta_LW_cf
        retdata['F_dlwcfl'] = -output['dcfl']*output['rlutcldcsl']
        retdata['F_dlwcfl_corr'] = -output['dcfl_corr']*output['rlutcldcsl']
        retdata['F_dlwcfh'] = -output['dcfh']*output['rlutcldcsh']

        # Adjust the LW forcing for the thin overlying cirrus effect
        retdata['F_LWcfi'] = retdata['F_dlwcfh']+(retdata['F_dlwcfl']-retdata['F_dlwcfl_corr'])

        retdata['F_LW_sum'] = retdata['F_dlwcs'] + retdata['F_dlwc'] + retdata['F_dlwcf']
        retdata['F_LW_res'] = retdata['F_LW'] - retdata['F_LW_sum'] 


        with open(outdir+'raw_output_summary_'+model[0]+'_'+tag+'.txt',"w") as f:
            for name in retdata.keys():
                print('{: <15} {:.2f}'.format(name, lwav(retdata[name],latitudes)))
                print('{: <15} {:.2f}'.format(name, lwav(retdata[name],latitudes)),file=f)

        #####################
        # Save global mean values #
        #####################
        print()
        table = outdir+'Table_ERF_decomp_'+model[0]+'_'+tag+'.txt'
        with open(table, "w") as f:
            dataA = [ 
#            ['Model', model[0]],
            ['DeltaSW', retdata['F_SW']],
            ['DeltaSurf', retdata['F_dalbsurf']],
            ['SWari_cs', retdata['F_RFari']],
            ['SWari_cld', retdata['F_RFaric']], # I am not sure about this term... Please ask.
            ['DeltaSW_cl', retdata['F_dalbcl']],
            ['DeltaSW_ci', retdata['F_dalbci']],
            ['DeltaSW_cfl', retdata['F_dcflnoa']], # equal to 'fc' term below
            ['DeltaSW_cfi', retdata['F_dcfinoa']],
            ['SW_res', retdata['F_SW_res']],
            ['Nd', retdata['F_dalbc_nd']],
            ['Liquid', retdata['F_dalbc_lwp']],
            ['Nd_off', retdata['F_dalbc_nd_off']],
            ['Liquid_off', retdata['F_dalbc_lwp_off']],
            ['Nd_log', retdata['F_dalbc_nd_log']],
            ['Liquid_log', retdata['F_dalbc_lwp_log']],
            ['fc', retdata['F_dcflnoa']],
            ['fc(corr)', retdata['F_dcfl']],
            ['fci(corr)', retdata['F_dcfi']],
            ['DeltaLW', retdata['F_LW']],
            ['LWari_cs', retdata['F_dlwcs']],
            ['DeltaLW_cl', retdata['F_dlwcl']],
            ['DeltaLW_ci', retdata['F_dlwci']],
            ['DeltaLW_cfl', retdata['F_dlwcfl']],
            ['DeltaLW_cfi', retdata['F_LWcfi']],
            ['LW_res', retdata['F_LW_res']],
            ]

            for xx_name, xx in dataA:
                print('{: <20}{:.2f}'.format(xx_name,lwav(xx,latitudes)), file=f)
                print('{: <20}{:.2f}'.format(xx_name,lwav(xx,latitudes)))

        #Store the output
        modelfolder = (datadir+model[0]+'/latlon/')
#        modelfolder = (datadir+model[0]+'/latlon_73x144/')


        pattern = (
            f'{modelfolder}/' +
            f'rsut_PD_{model[0]}.nc')
        filename = [f for f in glob.glob(pattern)][0]
        print(filename)

        opfilename1 = f'forcing_{model[0]}_inative_output_'+tag+'.nc'
        ncdf1 = Dataset(outdir+opfilename1, 'w', 'NETCDF4')

        opfilename2 = f'forcing_{model[0]}_inative_retdata_'+tag+'.nc'
        ncdf2 = Dataset(outdir+opfilename2, 'w', 'NETCDF4')


        ncdf1.title = "Aerosol forcing decomposition from AeroCom IND3 data"
        ncdf1.setncattr('institution', "Space and Atmospheric Physics, Imperial College London.")
        ncdf1.history = "CMIP5"
        ncdf1.contact = "Edward Gryspeerdt (e.gryspeerdt@imperial.ac.uk)"
        ncdf1.Conventions = "CF-1.6 "

        with Dataset(filename) as ncdf_old:
            ncdf1.createDimension('lat', ncdf_old.dimensions['lat'].size)
            ncdf1.createDimension('lon', ncdf_old.dimensions['lon'].size)
            ncdf1.createDimension('time', 1)
            ncdf1.createDimension('bnds', 2)

            Vlat = ncdf1.createVariable('lat', 'f', ('lat',))
            Vlat.long_name = "latitude"
            Vlat.standard_name = "latitude"
            Vlat.units = "degrees_north"
            Vlat[:] = ncdf_old.variables['lat'][:]

            Vlon = ncdf1.createVariable('lon', 'f', ('lon',))
            Vlon.long_name = "longitude"
            Vlon.standard_name = "longitude"
            Vlon.units = "degrees_east"
            Vlon[:] = ncdf_old.variables['lon'][:]

            Vtime = ncdf1.createVariable('time', 'f8', ('time',))
            Vtime.long_name = "time"
            Vtime.units = "days since 1900-1-1 0:0:0"
            Vtime.calendar = "standard"
            Vtime[:] = np.array([0])

        for name in output.keys():
            Var = ncdf1.createVariable(name, 'f8', ('time', 'lat', 'lon'))
            Var[:] = output[name][None, :, :].astype('float')

        ncdf1.close()

        with Dataset(filename) as ncdf_old:
            ncdf2.createDimension('lat', ncdf_old.dimensions['lat'].size)
            ncdf2.createDimension('lon', ncdf_old.dimensions['lon'].size)
            ncdf2.createDimension('time', 1)
            ncdf2.createDimension('bnds', 2)

            Vlat = ncdf2.createVariable('lat', 'f', ('lat',))
            Vlat.long_name = "latitude"
            Vlat.standard_name = "latitude"
            Vlat.units = "degrees_north"
            Vlat[:] = ncdf_old.variables['lat'][:]

            Vlon = ncdf2.createVariable('lon', 'f', ('lon',))
            Vlon.long_name = "longitude"
            Vlon.standard_name = "longitude"
            Vlon.units = "degrees_east"
            Vlon[:] = ncdf_old.variables['lon'][:]

            Vtime = ncdf2.createVariable('time', 'f8', ('time',))
            Vtime.long_name = "time"
            Vtime.units = "days since 1900-1-1 0:0:0"
            Vtime.calendar = "standard"
            Vtime[:] = np.array([0])

        for name in retdata.keys():
            Var = ncdf2.createVariable(name, 'f8', ('time', 'lat', 'lon'))
            Var[:] = retdata[name][None, :, :].astype('float')

        ncdf2.close()


