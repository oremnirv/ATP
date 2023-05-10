import numpy as np
import tensorflow as tf
import pandas as pd

def weather_processor(path_to_weather_data):

        pd_array = pd.read_csv(path_to_weather_data)
        data = np.array(pd_array)
        data[:,0] = np.linspace(-1,1,data.shape[0])
        # we need to have it between -1 to 1 for each batch item not just overall!!!!!!!!
        data = data.astype("float32")

        training_data = data[:int(0.7*data.shape[0])]
        val_data = data[int(0.7*data.shape[0]):int(0.8*data.shape[0])]
        test_data = data[int(0.8*data.shape[0]):]

        #scale

        training_data_scaled = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        val_data_scaled = (val_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        test_data_scaled =  (test_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)

        x_train, y_train = training_data_scaled[:,:1], training_data_scaled[:,4:5]
        x_val, y_val = val_data_scaled[:,:1], val_data_scaled[:,4:5]
        x_test, y_test = test_data_scaled[:,:1], test_data_scaled[:,4:5]

        return x_train[:,:,np.newaxis], y_train[:,:,np.newaxis], x_val[:,:,np.newaxis], y_val[:,:,np.newaxis], x_test[:,:,np.newaxis], y_test[:,:,np.newaxis]


def dataset_processor(path_to_data):
        # works for exchange and ETTm2 dataset w/o extra features 

        pd_array = pd.read_csv(path_to_data)
        data = np.array(pd_array)
        data[:,0] = np.linspace(-1,1,data.shape[0])
        # we need to have it between -1 to 1 for each batch item not just overall!!!!!!!!

        data = data.astype("float32")

        training_data = data[:int(0.69*data.shape[0])]
        val_data = data[int(0.69*data.shape[0]):int(0.8*data.shape[0])]
        test_data = data[int(0.8*data.shape[0]):]

        #scale

        training_data_scaled = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        val_data_scaled = (val_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
        test_data_scaled =  (test_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)

        x_train, y_train = training_data_scaled[:,:1], training_data_scaled[:,-1:]
        x_val, y_val = val_data_scaled[:,:1], val_data_scaled[:,-1:]
        x_test, y_test = test_data_scaled[:,:1], test_data_scaled[:,-1:]

        return x_train[:,:,np.newaxis], y_train[:,:,np.newaxis], x_val[:,:,np.newaxis], y_val[:,:,np.newaxis], x_test[:,:,np.newaxis], y_test[:,:,np.newaxis]

def gp_data_processor(path_to_data_folder):

        x = np.load(path_to_data_folder + "x.npy")
        y = np.load(path_to_data_folder + "y.npy")

        x_train = x[:int(0.99*x.shape[0])]
        y_train = y[:int(0.99*y.shape[0])]
        x_val = x[int(0.99*x.shape[0]):]
        y_val = y[int(0.99*y.shape[0]):]

        x_test = np.load(path_to_data_folder + "x_test.npy")
        y_test = np.load(path_to_data_folder + "y_test.npy")

        context_n_test = np.load(path_to_data_folder + "context_n_test.npy")

        return x_train, y_train, x_val, y_val, x_test, y_test, context_n_test

