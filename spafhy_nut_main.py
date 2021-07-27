# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:12:55 2020

@author: alauren
"""
#import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import datetime
import os
import time
import spafhy
from spafhy_nut_utils import get_spafhy_results, get_nutbal_results, get_Motti_results, daily_to_monthly,estimate_imm
from spafhy_io import preprocess_soildata, create_catchment, read_FMI_weather, get_clear_cuts
from spafhy_parameters_default import soil_properties, parameters, nutpara, immpara
from nutrient_balance import Grid_nutrient_balance
from nutrient_export import Export
import spafhy_stand as stand
import seaborn as sns
import matplotlib.pylab as plt
sns.set()

hydrology_calc = True
nut_balance_calc = True
export_calc = True

folder =r'C:/Users/alauren/Documents/SVE_catchments/'

def nsy(iN, iP, pgen, pcpy, pbu, ptop, pnut, psoil, gisdata, clear_cuts, forcing, outfold = None):
                       
    """run hydrology"""
    if hydrology_calc: 
        """ create SpatHy object """
                
        spa = spafhy.initialize(pgen, pcpy, pbu, ptop, psoil, gisdata, cpy_outputs=False, 
                         bu_outputs=False, top_outputs=False, flatten=True)
    
        """ create netCDF output file """
       
        dlat, dlon = np.shape(spa.GisData['cmask'])
        end_date = datetime.datetime.strptime(pgen['end_date'], '%Y-%m-%d')
        start_date = datetime.datetime.strptime(pgen['start_date'], '%Y-%m-%d')           
        Nsteps = (end_date-start_date).days
    
        ncf, ncf_file = spafhy.initialize_netCDF(ID=spa.id, fname=spa.ncf_file, lat0=spa.GisData['lat0'], 
                                                 lon0=spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=Nsteps)
        
     
        """ read forcing data and catchment runoff file """
        FORC = forcing.copy()
        FORC['Prec'] = FORC['Prec'] / spa.dt                            # mms-1
        FORC['U'] = 2.0                                                 # use constant wind speed ms-1
        
    
        for k in range(0, Nsteps):                                      # keep track on dates
            current_date = datetime.datetime.strptime(pgen['start_date'],'%Y-%m-%d').date() + datetime.timedelta(days=k)
            current_datetime = datetime.datetime.strptime(pgen['start_date'],'%Y-%m-%d') + datetime.timedelta(days=k)
    
            if k%30==0: print('step: ' + str(k), current_date.year, current_date.month, current_date.day)
    
            forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
            spa.run_timestep(forc, current_datetime, ncf=ncf)
    
            if current_date in clear_cuts.keys():
                print ('     +Clear cut ', current_date)            
                spa.set_clear_cut(clear_cuts[current_date])
                
        
        ncf.close()                                                             # close output file
        print (ncf_file, ' closed') 
    
    #*********************************************************************************************************            
    if nut_balance_calc:
        
        soildata = preprocess_soildata(pbu, psoil, gisdata['soilclass'], gisdata['cmask'], pgen['spatial_soil'])
                
        TAmr =stand.TAmr(forcing['T'])                                          # growing season air temperature
        sp_res = get_spafhy_results(pnut['spafhyfile'])                      # retrieve results from netCDF file
        
        
        Wm, lsm, Rm, Dm, Dretm, Infil,s_run,Tm, ddsm, Prec = daily_to_monthly(sp_res, forcing, pnut, balance=True)  # Change hydrology to monthly values
        lat= float(gisdata['loc']['lat'])                                    # coordinates in Spathy
        lon= float(gisdata['loc']['lon'])    
        motti = get_Motti_results(pnut['mottisims'], lat, lon)          # retrieve motti parameters
        Nsteps = np.shape(Rm)[0]                                         # number of months in the smulation

        print ('Nsteps', Nsteps)
        print ('***************************')
        """ Create nutrient balance object """
        
        nutSpafhy = Grid_nutrient_balance(forcing, pbu, pgen, pnut, gisdata, \
                                          soildata, motti, Wm, ddsm, lat, lon, iN=iN, iP=iP)
        
        for k in range(0, Nsteps):                             
            current_date = datetime.date(ddsm.index[k].year, ddsm.index[k].month,ddsm.index[k].day) # keep track on the date
            Ta = Tm[k]                                                                             # Mean monthly air temperature
            Pm = Prec[k]                                                                    # Precipitation monthly, mm
            Infi = Infil[k,:,:]
            local_s= lsm[k,:,:]                                                          # Mean monthly local saturation deficits in the catchment, m
            Wliq = Wm[k,:,:]                                                             # Mean monthly water content in root layer [m3 m-3]
            draindown = Dm[k,:,:]                                                        # Sum of monthly percolation of water from root zone to down m dt-1
            drainretflow = Dretm[k,:,:]                                                  # Sum of monthly returnflow, vertical upwards [mm dt-1], 
            Q_s = s_run[k,:,:]                                                           # Sum of monthly runoff, [mm dt-1]                     
            dt = (current_date - datetime.date(current_date.year, current_date.month, 1)).days  # length of time step in days (different lenght in different months)

            """ Run time step"""
            nutSpafhy.run_timestep(Ta, Pm, Infi, Q_s, TAmr, Wliq, local_s, draindown, drainretflow, dt)        
        
            print ('step: ' + str(k), current_date.year, current_date.month,current_date.day,) 
            print ('   -> Concentration mg L-1 N:', np.round(np.nanmean(nutSpafhy.nconc),3), 'P:', np.round(np.nanmean(nutSpafhy.pconc),3)) 
            
            print ('date: ', current_date) 
             
            if current_date in clear_cuts.keys():
                print ('     +Clear cut ', current_date)            
                nutSpafhy.set_clear_cut(clear_cuts[current_date])        

        nutSpafhy.close_ncf()
                
    #**************************************************************************************************************            
    if export_calc:
        
        soildata = preprocess_soildata(pbu, psoil, gisdata['soilclass'], gisdata['cmask'], pgen['spatial_soil'])            
        
        sp_res = get_spafhy_results(pnut['spafhyfile'])    
        _, lsm, Rm, _, _, _, _,_, ddsm, _ = daily_to_monthly(sp_res, forcing, pnut, balance=False)  # Change hydrology to monthly values: s_local and runoff 
        
        nsp_res = get_nutbal_results(f=pgen['output_folder']+ pnut['nutspafhyfile'])
    
        """ Initialize and run """                
        ex = Export(pgen, lsm, Rm, ddsm, nsp_res, gisdata,soildata)       
        nsp_res.close()
        dfOut = ex.run()
        print ('Computation done')
        
        df_annual_load = dfOut.resample('Y', convention ='end').sum()
        df_annual_conc = dfOut.resample('Y', convention ='end').mean()
     
        print ('******** Export load, kg ha-1 yr-1 **************') 
        print ( np.round(df_annual_load[['nexport[kg/ha]','pexport[kg/ha]' ]] ,3))  
        
        print ('******** Concentration, mg l-1 *****************')
        print (np.round(df_annual_conc[['nconc[mg/l]','pconc[mg/l]']], 3))
     
                    
        print ('')
        print ('COMPLETED: ', cat  +'  '+ scen)

scens = ['scen_no_log',
         'scen_gt_100m','scen_lt_35m']  
scen = scens[1] 
cat ='2'
outfile = r'C:/Users/alauren/Documents/sve_catchments/Nopt.csv'

print('************************************************')
print ('****** Catchment: ', cat ,' *******************')
print('************************************************')
    
pgen,pcpy,pbu,ptop=parameters(cat,scen)                         # general, canopy grid, bucket, topmodel
pnut = nutpara(pgen, cat, scen)
psoil = soil_properties()

gisdata = create_catchment(pgen, fpath=pgen['gis_folder'],
                               plotgrids=False, plotdistr=False)

clear_cuts = get_clear_cuts(pgen, gisdata['cmask'])

FORC = read_FMI_weather(pgen['catchment_id'],
                        pgen['start_date'],
                        pgen['end_date'],
                        sourcefile=pgen['forcing_file'])

iN, iP = estimate_imm(gisdata)

outfold =  r'C:/Users/alauren/Documents/sve_catchments/'+ str(int(cat)) + '/' + scen + '/'
a = nsy(iN, iP, pgen, pcpy, pbu, ptop, pnut, psoil, gisdata, clear_cuts, FORC, outfold=outfold)
print ('*******************************') 