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
from tensorflow.keras.models import load_model

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
data = pd.read_pickle('data/mereged_data.pkl')

features = ['Ghi','CloudOpacity','AirTemp']
output_criteria = ['RealPower']

selected_data = data[['Ghi','CloudOpacity', 'AirTemp','RealPower']]

#train test
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(selected_data)

day_num = int(selected_data.shape[0]/(12*24))
splited_data = np.split(scaled_data,day_num)

print(len(splited_data))
del_id = []
for i,c in enumerate(splited_data):
    if np.mean(c,axis=0)[-1] < 0.01:
        print(i,np.mean(c,axis=0))
        del_id.append(i)


indices = 0, 2
splited_data = [i for j, i in enumerate(splited_data) if j not in del_id]
# splited_data = np.delete(splited_data,del_id)

print(len(splited_data))

np.random.seed(0)
np.random.shuffle(splited_data)

train_chunk = 0.8
x_train = []
y_train = []
x_test = []
y_test = []

total_feature_size = len(features)

for i in range(0,int(day_num*train_chunk)):
    x_train.append(splited_data[i][:,0:total_feature_size])
    y_train.append(splited_data[i][:,total_feature_size])

for i in range(int(day_num*train_chunk),len(splited_data)):
    x_test.append(splited_data[i][:,0:total_feature_size])
    y_test.append(splited_data[i][:,total_feature_size])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


#%%
feature_numbers =len(features)
each_day_horizon = 288
pmu_forecaster = PMUF(feature_numbers, each_day_horizon)
pmu_forecaster.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
pred = pmu_forecaster.solar_predict(x_train)

# #train
#%%
# y_train=y_train.reshape(327,48,1)
pmu_forecaster.train(x_train, y_train, batch=5, epoch=500)
#evaluation on train set
# pmu_forecaster.solar_eval(x_train, y_train)
# #evaluation on dev set

# pmu_forecaster.solar_eval(x_train, y_train)
# pmu_forecaster.solar_eval(x_dev, y_dev)
# pmu_forecaster.solar_eval(x_test, y_test)
#%%
# pmu_forecaster.model.save_weights('models/w500_ghi_cloud')
#%%
feature_numbers =len(features)
each_day_horizon = 288
newpmu = PMUF(feature_numbers, each_day_horizon)
newpmu.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
newpmu.model.load_weights('models/w500')

#%%
pred = newpmu.solar_predict(x_train)
#%%
for i, k in enumerate(pred[0:20]):
    print(i)
    plt.plot(x_train[i][:,0], color = 'magenta')
    plt.plot(x_train[i][:,1], color = 'cyan')
    plt.plot(x_train[i][:,2],  color = 'orange')
    plt.plot(y_train[i], color = 'blue')
    plt.plot(pred[i], color = 'red')
    plt.legend(['Ghi','cloud', 'temp','real','pred'])
    # plt.savefig('figs/pred/train/' + str(i) + '.png')
    plt.show()
#%%
pred = newpmu.solar_predict(x_train)


mx = np.max(selected_data, axis=0)
mn = np.min(selected_data, axis=0)


main_pred = pred * (mx[-1] -mn[-1]) +mn[-1]
main_act = y_train * (mx[-1] -mn[-1]) +mn[-1]

from sklearn.metrics import mean_squared_error
from math import sqrt
total_errors = []
count = 0
start = 6*12
end = 20*12
for i in range(main_pred.shape[0]):
    mse = mean_squared_error(main_pred[i][start:end], main_act[i][start:end])
    rmse = sqrt(mse)
    nrmse = rmse/(mx[-1] - mn[-1])
    total_errors.append(nrmse)
    if nrmse < 0.3:
        count += 1
        print(nrmse, rmse, i)
print(count/main_pred.shape[0])

