import numpy as np
import tensorflow as tf
import pandas as pd

def weather_processor(path_to_weather_data):

        pd_array = pd.read_csv(path_to_weather_data)
        data = np.array(pd_array)
        data[:,0] = np.linspace(-1,1,data.shape[0])
        data = data.astype("float32")

        training_data = data[:int(0.7*data.shape[0])]
        val_data = data[int(0.7*data.shape[0]):int(0.8*data.shape[0])]
        test_data = data[int(0.8*data.shape[0]):]

        #scale

        training_data_scaled = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        val_data_scaled = (val_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        test_data_scaled =  (test_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)

        x_train, y_train = training_data_scaled[:,:1], training_data_scaled[:,-1:]
        x_val, y_val = val_data_scaled[:,:1], val_data_scaled[:,-1:]
        x_test, y_test = test_data_scaled[:,:1], test_data_scaled[:,-1:]

        return x_train, y_train, x_val, y_val, x_test, y_test
