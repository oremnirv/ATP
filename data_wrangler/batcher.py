import numpy as np
import tensorflow as tf
import pandas as pd 

def batcher(t, y, idx_list, batch_s = 32, window = 288):
    '''
    cutting one long array to sequences of length 'window'.
    'batch_s' must be ≤ full array - window length

    input to forecast: (None, 1, 1) for t,y.
    input to NP tasks: (None, seq_len, 1) for t,y. window = 1.
    idx_list: list of indices, must be ≤ full array - window length.
    '''
    
    if len(idx_list) < 1:
        print("warning- you didn't loop over the correct range")
        
    print("length of idx list:", len(idx_list))
    batch_s = min(batch_s, y.shape[0]-window) 
    idx = np.random.choice(len(idx_list), batch_s, replace = False)
    selected_idx = [idx_list[i] for i in idx]

    y = np.array([np.array(y)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
    t = np.array([np.array(t)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
    for i in sorted(idx, reverse=True): del idx_list[i]
        
    t = t.squeeze()
    y = y.squeeze()
    
    if len(t.shape) == 2:
        t = t[:,:,np.newaxis]
        y = y[:,:,np.newaxis]
        
    return t,y, idx_list, selected_idx

def batcher_np(t,y,batch_s=32):

    idx = np.random.choice(y.shape[0], batch_s, replace = False)

    y = y[idx, :, :]
    t = t[idx, :, :]

    return t,y

def batcher_multi_ts(array, n_C, n_T, batch_size=32):
    """
    array: numpy array with shape (number of samples, number of time series + 1)
    n_C: number of context points
    n_T: number of target points
    batch_size: batch size
    returns: y with shape (batch size, (n_C + n_T) x number of time series, 1)
             x with shape (batch size, (n_C + n_T) x number of time series)
    """
    number_of_ts = array.shape[1] - 1 ## -1 because the first column is time
    t = []; y = []
    for i in range(batch_size):
        index = int(np.random.randint(0, array.shape[0] - n_C - n_T, 1))
        t_all =  array[index:index + n_C + n_T , 0]
        t_all = np.repeat(t_all, number_of_ts)[:, np.newaxis]
        y_temp_all =  array[index:index + n_C + n_T , 1]

        for i in range(2, number_of_ts + 1):
            y_temp_all = np.concatenate([y_temp_all, array[index:index + n_C + n_T, i]], axis=0)

        t.append(t_all)
        y.append(y_temp_all)
    return np.array(y), np.array(t)

def batcher_bc(array, seq_l, batch_size=32):
    """
    This is useful when we have multuple ts and the one we want to predict is sparser (shorter) than the others.
    ** Seq to predict is the column 1  in the array. **

    array: numpy array with shape (number of samples, number of time series + 1)
    seq_l: length of the sequence for each ts
    batch_size: batch size
    returns: y with shape (batch size, (seq_l) x number of time series, 1)
             x with shape (batch size, (seq_l) x number of time series)
    """
    number_of_ts = array.shape[1] - 1 ## -1 because the first column is time
    t = []; y = []
    for i in range(batch_size):
        index = int(np.random.randint(0, array.shape[0] - seq_l, 1))
        t_all =  array[index:index + seq_l , 0]
        t_all = np.repeat(t_all, number_of_ts)[:, np.newaxis]
        y_temp_all =  array[index:index +seq_l , 1]

        for i in range(2, number_of_ts + 1):
            y_temp_all = np.concatenate([y_temp_all, array[index:index + seq_l, i]], axis=0)

        t.append(t_all)
        y.append(y_temp_all)
    return np.array(y), np.array(t)

if __name__ == 'main':
    ## test batcher_multi_ts example with exchange rate data
    exchange = pd.read_csv('datasets/exchange.csv')
    array = np.array(exchange)
    array[:,0] = np.linspace(-1,1,array.shape[0])
    array = array.astype("float32")
    x, y = batcher_multi_ts(array[:5000, :], 20, 10) 


    ## test that batcher_bc gives all values from one ts and then all the vals from the next ts
    batch_size = 1
    seq_l = 8
    array = np.random.noraml(0,1, size=(300, 3))
    number_of_ts = array.shape[1] - 1 ## -1 because the first column is time
    t = []; y = []
    for i in range(batch_size):
        index = int(np.random.randint(0, array.shape[0] - seq_l, 1))
        print(index)
        t_all =  array[index:index + seq_l , 0]
        t_all = np.repeat(t_all, number_of_ts)[:, np.newaxis]
        y_temp_all =  array[index:index +seq_l , 1]

        for i in range(2, number_of_ts + 1):
            y_temp_all = np.concatenate([y_temp_all, array[index:index + seq_l, i]], axis=0)

        t.append(t_all)
        y.append(y_temp_all)
    
    array.iloc[index :index+seq_l , :]
    ## compare to  output y
