import importlib
import PMUforecast
import tensorflow as tf
PMUforecast = importlib.reload(PMUforecast)
from PMUforecast import FileInf
from PMUforecast import SolcastHistorical
from PMUforecast import SolcastDataForecast
from PMUforecast import SolcastFiveMinHistorical
from PMUforecast import UCRPMU
import os  #access files and so on
import matplotlib
import matplotlib.pyplot as plt
from PMUforecast import PMUF
import numpy as np
# import keras
import pandas as pd
import datetime
import time
from time import sleep
import schedule
#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#%%
####################################################
####################################################
####################################################
"""

UCR PMU forecasting real power

"""
####################################################
####################################################
####################################################
#%%
dst = 'data/pmu/'

PMU = UCRPMU(dst)
pmu_data = PMU.get_clean_data()

start = pmu_data.iloc[0]['ts']
end = pmu_data.iloc[-1]['ts']

historical = SolcastFiveMinHistorical('data/weather.csv')
data = historical.data

selected_historical = data.loc[(data['t']>=start) & (data['t']<=end)]
ss = data.loc[(data['told']>=start) & (data['told']<=end)]
#%%
# %matplotlib auto
a=10*288
b=41*288
plt.plot(selected_historical['Ghi'].values)
plt.plot(ss['Ghi'][a:b].values)
plt.plot(pmu_data['RealPower'].values)
plt.legend(['t','told','p'])
plt.show()
#%%
#making new data file with weather and pmusolar data together

index = np.intersect1d(ss['told'],pmu_data['ts'])

pmu_align = pmu_data.loc[pmu_data['ts'].isin(index)].reset_index()

weather_align = ss.loc[ss['told'].isin(index)].reset_index()
#%%
merged_data = pd.concat([pmu_align, weather_align], axis=1, sort=False)
#%%
merged_data.to_pickle('data/mereged_data.pkl')

#%%
# train based on last data
PMUforecast = importlib.reload(PMUforecast)

# window_horizon = 24*7 #we see a week before for forecasting

#align all the available data
dst = 'data/pmu/'
each_day_horizon = 12*24
train_chunk, window_horizon = 0.7, 7
feature_numbers = 1
PMU = UCRPMU(dst)
pmu_data = PMU.get_clean_data()
train_data_x, train_data_y, test_data_x, test_data_y = PMU.get_train_test(train_chunk, window_horizon)

pmumax = np.max(list(train_data_x.values()))
pmumin = np.min(list(train_data_x.values()))
x_train, y_train, x_test, y_test = (list(train_data_x.values()) - pmumin)/(pmumax - pmumin),\
                                   (list(train_data_y.values()) - pmumin)/(pmumax - pmumin),\
                                   (list(test_data_x.values()) - pmumin)/(pmumax - pmumin),\
                                   (list(test_data_y.values()) - pmumin)/(pmumax - pmumin)


#%%
#clean train

should_be_removed = [22, 48, 55, 58, 61, 71, 84]
x_train = np.delete(x_train,should_be_removed,0)
y_train = np.delete(y_train,should_be_removed,0)
#%%
new_x = []
for i in list(x_train):
    new_x.append(i.reshape(2016,1))

new_y = []
for i in list(y_train):
    new_y.append(i.reshape(288,1))
#%%
# x_train_shaped = np.reshape(x_train, newshape=(-1, 2016, 1))
x_train = x_train.reshape(x_train.shape[0], each_day_horizon* window_horizon, feature_numbers)
y_train = y_train.reshape(y_train.shape[0], each_day_horizon)


#%%

pmu_forecaster = PMUF(feature_numbers, each_day_horizon * window_horizon, each_day_horizon)
pmu_forecaster.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
# #train
#%%
# y_train=y_train.reshape(327,48,1)
pmu_forecaster.train(x_train, y_train, batch=1, epoch=20)
#evaluation on train set
# pmu_forecaster.solar_eval(x_train, y_train)
# #evaluation on dev set

# pmu_forecaster.solar_eval(x_train, y_train)
# pmu_forecaster.solar_eval(x_dev, y_dev)
# pmu_forecaster.solar_eval(x_test, y_test)

# pmu_forecaster.model.save('models/'+sc)


#%%
pred = pmu_forecaster.solar_predict(x_train)

# pred = tf.transpose(pred)
for i, k in enumerate(pred[0:20]):
    print(i)
    # plt.plot(x_train[i])
    plt.plot(y_train[i])
    plt.plot(pred[i])
    plt.legend(['real','pred'])
    plt.show()


#%%
for j,i in enumerate(x_train):
    plt.plot(i)
    plt.savefig('figs/xtr'+str(j)+'.png')
    plt.show()
    print(j,np.mean(i))


#%%
a = pmu_forecaster.model(x_train[0])

b = pmu_forecaster.model(x_train[0])
#%%



forecasted_features = ['Ghi', 'Ghi90', 'Ghi10', 'Ebh', 'Dni', 'Dni10', 'Dni90', 'Dhi',
       'air_temp', 'Zenith', 'Azimuth', 'cloud_opacity', 'period_end',
       'Period']
historical_features = ['PeriodEnd', 'PeriodStart', 'Period', 'AirTemp', 'AlbedoDaily',
       'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi',
       'PrecipitableWater', 'RelativeHumidity', 'SnowDepth', 'SurfacePressure',
       'WindDirection10m', 'WindSpeed10m', 'Zenith']

whole_training_features = ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'] #for today which will predict tomorrow
output_feature = ['PV_power']#for the next day power generation

%reload_ext solarforecast

etap_power=EtapData(0.8)
train_times = etap_power.train_data.keys()
test_times = etap_power.test_data.keys()
#historical data from solcast
dst='data/solcast_etap_historical.csv'
hist = SolcastHistorical(dst, train_times, test_times)

#%%

#%%
# select features for training
scenarios = {
    'whole': ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'],
    'radiations':['Ghi', 'Ebh', 'Dni', 'Dhi'],
    'normal':['Ghi', 'AirTemp', 'CloudOpacity'],
    'minimal':['Ghi', 'CloudOpacity'],
    'Ghi':['Ghi']
}

for sc in scenarios:
    print(sc)
    selected_features = scenarios[sc]

    feature_numbers=len(selected_features)
    resolution=24
    x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)
    solar_forecaster = SolarF(feature_numbers,resolution)

    solar_forecaster.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
# #train

    # y_train=y_train.reshape(327,48,1)
    solar_forecaster.train(x_train, y_train, batch=1, epoch=100)
    #evaluation on train set
    solar_forecaster.solar_eval(x_train, y_train)
    # #evaluation on dev set

    solar_forecaster.solar_eval(x_train, y_train)
    # solar_forecaster.solar_eval(x_dev, y_dev)
    solar_forecaster.solar_eval(x_test, y_test)

    solar_forecaster.model.save('models/'+sc)


#%%
x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)
solar_forecaster = SolarF(feature_numbers,resolution)

solar_forecaster.opt_ls_mtr(optimizer='adam',
                            loss='mse',
                            metric='mse')
# #train

# y_train=y_train.reshape(327,48,1)
solar_forecaster.train(x_train, y_train, batch=1, epoch=1)
#evaluation on train set
solar_forecaster.solar_eval(x_train, y_train)
# #evaluation on dev set

solar_forecaster.solar_eval(x_train, y_train)
# solar_forecaster.solar_eval(x_dev, y_dev)
solar_forecaster.solar_eval(x_test, y_test)

solar_forecaster.model.save('models/whole_features')

#%%
scenarios = {
    'whole': ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'],
    'radiations':['Ghi', 'Ebh', 'Dni', 'Dhi'],
    'normal':['Ghi', 'AirTemp', 'CloudOpacity'],
    'minimal':['Ghi', 'CloudOpacity'],
    'Ghi':['Ghi']
}

## loading model and compare their performance
mses={}
for sc in scenarios:
    if sc == 'normal':
        print(sc)
        selected_features = scenarios[sc]

        feature_numbers = len(selected_features)
        resolution = 24
        x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)

        loaded_model = keras.models.load_model('models/'+sc)
        print(loaded_model)
        predicted = loaded_model.predict(x_test)
        mse_error = loaded_model.evaluate(x_test, y_test)
        print(mse_error)
        mses[sc] = mse_error[0]
        # os.mkdir('models/figs/'+sc)
        # for i, k in enumerate(predicted):
        #     print(i)
        #     plt.plot(y_test[i])
        #     plt.plot(predicted[i])
        #     plt.legend(['real', 'pred'])
        #     plt.savefig('models/figs/' + sc + '/' + str(i) + '.png')
        #     plt.show()





#%%
#prediction
pred = solar_forecaster.solar_predict(x_test)
for i, k in enumerate(pred):
    # print(i[30])
    # plt.plot(x_train[i])
    plt.plot(y_test[i])
    plt.plot(pred[i])
    plt.legend(['real','pred'])
    plt.show()
# selected_data.head()

#%%
#saving keras model
solar_forecaster.model.save('models/whole_features')
#%%
loaded=keras.models.load_model('models/normal')



#%%
import sklearn
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)
# reg.score(x_train, y_train)
pred = reg.predict(x_train)

