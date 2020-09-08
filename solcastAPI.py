import json
import time
import urllib.request
import solcast
from opencage.geocoder import OpenCageGeocode
from PMUforecast import FileInf
from PMUforecast import SolcastDataForecast
from PMUforecast import SolcastHistorical
import os  #access files and so on
import matplotlib
import matplotlib.pyplot as plt
from PMUforecast import SolarF
import pandas as pd
import datetime
from time import sleep
import schedule
#%%


def saving_task():      
    address='17 goodyear, Irvine'
    if not address:
        address = input('where is your site address?  ')
    location_api_key='23e6edd3ccc7437b90c589fd7c9c6213'
    solcast_API_key='osmO54Z_7TKYMgJFi3vrQenczYLbErBk'

    SDF = SolcastDataForecast(location_api_key, solcast_API_key, address)
    act, pred = SDF.get_solcast_forecast()
    day = pred['local_time'][1].strftime("%Y-%m-%d")
    f_dst = 'data/solcast/Irvine/forecast-' + day + '.pkl'
    a_dst = 'data/solcast/Irvine/actual-' + day + '.pkl'

    pred.to_pickle(f_dst)
    act.to_pickle(a_dst)
    print('the files for %s has been saved!' % day)
    
#%%
def show_sth():
    print('I am still working!!!')
    print(datetime.datetime.now())
#%%
schedule.clear()
schedule.every().day.at('00:00').do(saving_task)
schedule.every().hour.do(show_sth)

#%%
while True:
    schedule.run_pending()
    # time.sleep(60*60)

#%%
address = '17 goodyear, Irvine'
if not address:
    address = input('where is your site address?  ')
location_api_key = '23e6edd3ccc7437b90c589fd7c9c6213'
solcast_API_key = 'osmO54Z_7TKYMgJFi3vrQenczYLbErBk'


SDF = SolcastDataForecast(location_api_key, solcast_API_key, address)
act, pred = SDF.get_solcast_forecast()
# #%%
# temp_data = SolacastData(location_api_key, solcast_API_key, address)
# # temp_data.get_address_lat_lng()
# temp_data.get_solcast_forecast()
day = pred['local_time'][1].strftime("%Y-%m-%d")
f_dst = 'data/solcast/Irvine/forecast-' + day + '.pkl'
a_dst = 'data/solcast/Irvine/actual-' + day + '.pkl'

pred.to_pickle(f_dst)
act.to_pickle(a_dst)
print('the files for %s has been saved!' % day)
#%%
