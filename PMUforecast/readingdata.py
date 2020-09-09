# =============================================================================
# Liberaries
# =============================================================================
import pandas as pd
import os  # access files and so on
import sys  # for handling exceptions
import re  # for checking letter in a string
import numpy as np
import random
import time
import xlrd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import solcast
from opencage.geocoder import OpenCageGeocode
import datetime
import math
import json
import urllib.request
from datetime import datetime, timezone


class UCRPMU:

    def __init__(self, dst):
        self.dst = dst
        self.each_day_horizon = 12*24  # each 5 minutes

    def get_clean_data(self):
        whole_data = pd.DataFrame(columns=["DateTime", "RealPower", "ReactivePower", "ApprentPower"])
        for i in os.listdir(self.dst):
            folder = os.listdir(self.dst + i)
            for file in folder:
                data = pd.read_csv(self.dst + i + '/' + file)
                data['DateTime'] = pd.to_datetime(data['DateTime'])
                data = data.set_index(data['DateTime'])
                start = data.iloc[0]['DateTime'].replace(minute=00)
                end = data.iloc[-1]['DateTime'].replace(minute=55)
                idx = pd.date_range(start, end, periods=self.each_day_horizon)
                data = data.reindex(idx, fill_value=0)
                data['DateTime'] = idx
                whole_data = whole_data.append(data, ignore_index=True)

        ts = []
        for i in whole_data['DateTime']:
            ts.append(int(i.timestamp()))
        whole_data['ts'] = ts
        return whole_data

    def get_train_test(self, train_chunk, window_horizon):
        data = self.get_clean_data()
        sample_numbers = int(data.shape[0] / self.each_day_horizon) - window_horizon

        x_start = 0
        x_end = self.each_day_horizon * window_horizon
        y_start = x_end
        y_end = y_start + self.each_day_horizon

        separated_data = {'x': [], 'y': []}

        for i in range(sample_numbers):
            separated_data['x'].append(data.iloc
                                       [x_start:x_end][['DateTime', 'RealPower']])
            separated_data['y'].append(data.iloc
                                       [y_start:y_end][['DateTime', 'RealPower']])

            x_start = x_start + self.each_day_horizon
            x_end = x_end + self.each_day_horizon
            y_start = x_end
            y_end = y_end + self.each_day_horizon

        random.seed(1)
        train_number = int(train_chunk * sample_numbers)
        data_indexes = np.arange(0, sample_numbers)
        np.random.shuffle(data_indexes)
        train_index = data_indexes[0:train_number]
        test_index = data_indexes[train_number:]
        train_data_x = {i: separated_data['x'][i]['RealPower'].values for i in train_index}
        train_data_y = {i: separated_data['y'][i]['RealPower'].values for i in train_index}
        test_data_x = {i: separated_data['x'][i]['RealPower'].values for i in test_index}
        test_data_y = {i: separated_data['y'][i]['RealPower'].values for i in test_index}

        return train_data_x, train_data_y, test_data_x, test_data_y


class SolcastFiveMinHistorical:
    def __init__(self, dst):
        # self.train_index = train_index
        # self.test_index = test_index
        self.dst = dst
        self.data = self.__time_add()

        # self.train,  self.test = self.__train_test_historical()

    def __time_add(self):
        historical = pd.read_csv(self.dst)

        def __utc_to_local(utc_dt):
            return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)

        ts = []
        tsold = []
        for i in historical['PeriodEnd']:
            date = datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
            t = datetime.timestamp(date)  #
            ts.append(int(t))

            date = __utc_to_local(datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ'))
            t = datetime.timestamp(date)  # Irvine to UTC difference
            more_shift = 7*3600
            tsold.append(int(t) - more_shift)
        historical['t'] = ts
        historical['told'] = tsold

        return historical

    # def __train_test_historical(self):
    #
    #     historical = self.__time_add()
    #
    #     train = {}
    #     count = 0
    #     for i, t in enumerate(self.train_index):
    #         start = t
    #         end = start + 24 * 3600
    #         part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
    #         if part.shape[0] == 24:
    #             train[t] = part
    #             count += 1
    #     test = {}
    #     count = 0
    #     for i, t in enumerate(self.test_index):
    #         start = t
    #         end = start + 24 * 3600
    #         part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
    #         if part.shape[0] == 24:
    #             test[t] = part
    #             count += 1
    #
    #     return train, test
    #


class SolcastHistorical:

    def __init__(self, dst, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index
        self.dst = dst
        self.train,  self.test = self.__train_test_historical()

    def __time_add(self):
        historical = pd.read_csv(self.dst)
        ts = []
        for i in historical['PeriodEnd']:
            date = datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
            t = datetime.timestamp(date) - 7 * 3600  # Irvine to UTC difference
            ts.append(int(t))
        historical['t'] = ts
        return historical

    def __train_test_historical(self):

        historical = self.__time_add()

        train = {}
        count = 0
        for i, t in enumerate(self.train_index):
            start = t
            end = start + 24 * 3600
            part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
            if part.shape[0] == 24:
                train[t] = part
                count += 1
        test = {}
        count = 0
        for i, t in enumerate(self.test_index):
            start = t
            end = start + 24 * 3600
            part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
            if part.shape[0] == 24:
                test[t] = part
                count += 1

        return train, test

class SolcastDataForecast:
    
    def __init__(self, location_api_key, solcast_api_key, address):
        self.location_api_key = location_api_key
        self.solcast_api_key = solcast_api_key
        self.address = address

    def __get_address_lat_lng(self):
        geocoder = OpenCageGeocode(self.location_api_key)
        self.whole_location_info = geocoder.geocode(self.address)[0]
        geo = self.whole_location_info['geometry']
        lat,  lng = geo['lat'], geo['lng']
        self.lat = lat
        self.lng = lng
# location_api_key='23e6edd3ccc7437b90c589fd7c9c6213'

    def get_solcast_forecast(self):
        # solcast_API_key='osmO54Z_7TKYMgJFi3vrQenczYLbErBk'
        self.__get_address_lat_lng()
        radiation_forecasts = solcast.RadiationForecasts(self.lat, self.lng, self.solcast_api_key)
        self.forecasts_data=pd.DataFrame(radiation_forecasts.forecasts)
        radiation_actuals = solcast.RadiationEstimatedActuals(self.lat, self.lng, self.solcast_api_key)
        self.actuals_data=pd.DataFrame(radiation_actuals.estimated_actuals)
        self.__local_time()
        return self.actuals_data, self.forecasts_data

    def __local_time(self):
        #timezonefinder and get append UNIX time as well
        temp_time=[]
        desired_tz=self.whole_location_info['annotations']['timezone']['name']
        def utc_to_local(utc_dt,desired_tz):
            return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=desired_tz)
        for t in self.actuals_data['period_end']:
            temp_time.append(utc_to_local(t,desired_tz))
        
        self.actuals_data['local_time']=temp_time
        
        temp_time=[]
        desired_tz=self.whole_location_info['annotations']['timezone']['name']
        for t in self.forecasts_data['period_end']:
            temp_time.append(utc_to_local(t,desired_tz))
        
        self.forecasts_data['local_time'] = temp_time

class OpenWeatherAPI:
    def __init__(self, location_api, address, openweather_api):
        self.location_api = location_api
        self.address = address
        self.openweather_api = openweather_api


    def __get_address_lat_lng(self):
        geocoder = OpenCageGeocode(self.location_api)
        self.whole_location_info = geocoder.geocode(self.address)[0]
        geo = self.whole_location_info['geometry']
        lat, lng = geo['lat'], geo['lng']
        self.lat = lat
        self.lng = lng
# location_api_key='23e6edd3ccc7437b90c589fd7c9c6213'

    def get_forecasted_data(self):
        self.__get_address_lat_lng()
        base_url = 'https://api.openweathermap.org/data/2.5/onecall?lat=' + str(self.lat) + \
                   '&lon=' + str(self.lng) +'&%20exclude=hourly&appid=' + self.openweather_api

        with urllib.request.urlopen(base_url) as base_url:
            # print(search_address)
            data = json.loads(base_url.read().decode())


        hourly = data['hourly']
        hourly = pd.DataFrame(hourly)
        return hourly

class FileInf():
    def __init__(self,directory):
        self.dir=directory
        self.files=os.listdir(self.dir)

    def load_data(self,file):
        """

        Parameters
        ----------
        file : string
            selected dataset file.

        Returns
        -------
        df : pd.Datafframe
            dataset for the selected file.
        """
        print(file)
        self.file=file
        filepath=os.path.join(self.dir, file)
        try:
            #with regard to website file format
            if file.split('_')[0]=='JRC':
                with open(filepath) as fd:
                    headers = [ next(fd) for i in range(10) ]
                    df = pd.read_csv(fd)
                  
                  #remove descriptions of the file located at the end of the file
                while re.search('[a-zA-Z]', df.iloc[-1][0]):
                    df.drop(df.tail(1).index,inplace=True)
            
            elif file.split('_')[0]=='NREL':
                with open(filepath) as fd:
                    headers = [ next(fd) for i in range(2) ]
                    df = pd.read_csv(fd)
                    self.data=df
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
        
        return df
    
    
    def train_dev_test(self,file,features,output_f,resolution,**kwarg):
        """
        

        Parameters
        ----------
        file : string
            selected data set .
        **kwarg : 0 - 1
            train, dedv and test percentage.

        Returns
        -------
        new_data : three array each of the are dataframe
            train, dedv and test sets.

        """
        train_chunk,dev_chunk,test_chunk=kwarg['train'],kwarg['dev'],kwarg['test']
        
        data=self.load_data(self.file)
        k=data.keys()
        
        # data[k] = StandardScaler().fit_transform(data[k])            
        scaler = MinMaxScaler() 
        scaled_values = scaler.fit_transform(data) 
        data.loc[:,:] = scaled_values
        #irrediaction features for today
        
        
        
        feature_irradiations=data[['Clearsky DHI','Clearsky DNI','Clearsky GHI',
                      'DHI','DNI','GHI']].iloc[0:(data.shape[0]-resolution)]

        #features are for tomorrow 
        feature_weather=data[[ 'Temperature','Cloud Type', 'Dew Point',
               'Fill Flag', 'Relative Humidity', 'Solar Zenith Angle',
               'Surface Albedo', 'Pressure', 'Precipitable Water', 'Wind Direction',
           'Wind Speed']].iloc[resolution:(data.shape[0])]
        
        same_index=np.array(range(0,data.shape[0]-resolution))
        
        feature_irradiations.index=same_index
        feature_weather.index=same_index
        
        whole_features=pd.concat([feature_irradiations,feature_weather],axis=1)
        
        #should be forecasted
        output_irradiations=data[['Clearsky DHI','Clearsky DNI','Clearsky GHI',
                      'DHI','DNI','GHI']].iloc[resolution:(data.shape[0])]
        old_columns=[]
        new_columns=[]
        for i in output_irradiations:
            old_columns.append(i)
            new_columns.append(i+' tmrw')
        for i,k in enumerate(old_columns):
            output_irradiations.rename(columns={k:new_columns[i]},inplace=True)
        
        output_irradiations.index=same_index
        
        new_data=pd.concat([whole_features,output_irradiations],axis=1)
        new_data=np.array_split(new_data,np.floor(data.shape[0]/resolution-1))#one day shift for prediciton 
        
        
        
        #shuffle data with seed
        random.seed(4)
        random.shuffle(new_data)
        
        size_data=len(new_data)
        trian_pointer=int(np.floor(train_chunk*size_data))
        dev_pointer=int(trian_pointer+np.floor(dev_chunk*size_data))
        
        train=new_data[0:trian_pointer]
        dev=new_data[trian_pointer:dev_pointer]
        test=new_data[dev_pointer:]
        
        x_train=[]
        y_train=[]
        for i in train:
            x_train.append(i.loc[:,features].values)
            y_train.append(i.loc[:,output_f].values)
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        
        
        x_dev=[]
        y_dev=[]
        for i in dev:
            x_dev.append(i.loc[:,features].values)
            y_dev.append(i.loc[:,output_f].values)
        x_dev=np.array(x_dev)
        y_dev=np.array(y_dev)
            
        x_test=[]
        y_test=[]
        for i in test:
            x_test.append(i.loc[:,features].values)
            y_test.append(i.loc[:,output_f].values)
        x_test=np.array(x_test)
        y_test=np.array(y_test)
        
        tdt_output=[x_train,y_train,x_dev,y_dev,x_test,y_test]
        
        
            

        return tdt_output
    
    
    
    
    
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# # # Detail for data      
# =============================================================================
# =============================================================================
# =============================================================================

#JRC
# =============================================================================
# =============================================================================
# # P: PV system power (W)
# Gb(i): Beam (direct) irradiance on the inclined plane (plane of the array) (W/m2)
# Gd(i): Diffuse irradiance on the inclined plane (plane of the array) (W/m2)
# Gr(i): Reflected irradiance on the inclined plane (plane of the array) (W/m2)
# H_sun: Sun height (degree)
# T2m: 2-m air temperature (degree Celsius)
# WS10m: 10-m total wind speed (m/s)
# Int: 1 means solar radiation values are reconstructed
# =============================================================================
# =============================================================================
