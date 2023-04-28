import numpy as np
import pandas as pd

### let's create data batches that fit the task
def batcher(t, y, batch_s = 32,window=100):
    
    batch_s = min(batch_s, bat)
    idx = np.random.choice(np.arange(0, len(y) - window), batch_s, replace=False)
    
    y = np.array([np.array(y)[i:i+window] for i in idx])
    t = np.array([np.array(t)[i:i+window] for i in idx])
    
    return t, y
  ######## need to edit to make it clearer how to use the batcher so it returns what is desired

def create_batch(input_list, batch_s=32):
    
    batch_list = []
    shape_label = input_list[0].shape[0]
    batch_idx_la = np.random.choice(list(range(shape_label)), batch_s)
    for i in input_list: 
        batch_item = (i[batch_idx_la,])
        batch_list.append(batch_item)

    return batch_list


### It is usually helpful to make gaps during predictions as opposed to providing the full sequence. Let's then run the batcher outputs through a sampler 
def batch_sampler(t, y, given_hr = 96, pred_hr = 192):
  
    m = np.random.choice(np.arange(3, given_hr), 1)
    n = np.random.choice(np.arange(3, pred_hr), 1)
    idx_cont = np.sort(np.random.choice(np.arange(given_hr), m))
    idx_tar = np.sort(np.random.choice(np.arange(given_hr, given_hr + pred_hr), n))

    t = np.concatenate([t[:, idx_cont], t[:, idx_tar]], axis=-1)
    y = np.concatenate([y[:, idx_cont], y[:, idx_tar]], axis=-1)
    return t, y, m, n

def make_features(t, y, context_points, batch_s=32):
    x = PE(t, d=28, TΔmin=0.1, Tmax=2)
    
    value_x = y[:, :, np.newaxis]
    context_points = int(context_points)

    mask = np.tri(y.shape[1], y.shape[1], 0) - np.eye(y.shape[1])
    mask[:context_points, :context_points] = 1 
    mask = np.repeat(mask[np.newaxis, :, :], batch_s, axis=0)

    diff_y, diff_x, d, x_n, y_n = DE(y, t, context_points, embed=True)
    y_prime = np.concatenate([y[:, :, np.newaxis], diff_y.reshape(batch_s, -1, 1), d.reshape(batch_s, -1, 1), y_n.reshape(batch_s, -1, 1)], axis=2)
    query_x = key_x = x_prime = np.concatenate([x, diff_x, x_n], axis=2)
    
    query_xy_label = np.ones((batch_s, y.shape[1], 1))
    key_xy_label = np.concatenate([np.ones((batch_s, context_points, 1)), np.zeros((batch_s, y.shape[1]-context_points, 1))], axis=1)


    key_xy = value_xy = np.concatenate([y_prime, key_xy_label, x_prime], axis=2)
    query_xy = np.concatenate([y_prime, query_xy_label, x_prime], axis=2)
    query_xy[:, context_points:, :3] = 0

    return query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n

def PE(t, d=28, TΔmin=0.1, Tmax=2):  # return.shape=(T,B,d)
    # t.shape=(T,B)   T=sequence_length, B=batch_size
    """A position-embedder, similar to the Attention paper, but tweaked to account for
    floating point positions, rather than integer.
    """
    R = Tmax / TΔmin * 100
    drange_even = TΔmin * R**(np.arange(0,d,2)/d)
    drange_odd = TΔmin * R**((np.arange(1,d,2) - 1)/d)
    x = np.concatenate([np.sin(t[:,:,None] / drange_even), np.cos(t[:,:,None] / drange_odd)], 2)
    return x


def DE(ŷ, x̂, c, embed=False):
    d=1
    if embed:
        d=28
    
    m, n = ŷ.shape[0], ŷ.shape[1]
    diff_y = np.zeros((m , n))
    diff_x = np.zeros((m, n, d))
    dd = np.zeros((m, n))
    y_n = np.zeros((m , n))
    x_n = np.zeros((m , n, d))
    
    for i in range(m):
        for j in range(c):
            current_x = (x̂[i, :c][j])
            current_y = ŷ[i, :c][j]
            x_temp = (x̂[i, :c])
            y_temp = ŷ[i , :c]
            ix = np.argsort(np.abs(current_x - x_temp))[1] 

            x_rep = current_x - x_temp[ix]
            y_rep = current_y - y_temp[ix]
            deriv = y_rep / (x_rep + 0.0001)
            
            diff_y[i, j] = y_rep
            diff_x[i, j, :] = x_rep
            x_n[i, j, :] = x_temp[ix]
            if embed:
                diff_x[i, j, :] = PE(np.array([current_x])[:, np.newaxis]) -  PE(np.array([x_temp[ix]])[:, np.newaxis])
                x_n[i, j, :] = PE(np.array([x_temp[ix]])[:, np.newaxis])
            
            dd[i, j] = deriv
            y_n[i, j] = y_temp[ix]
        
        for j in range(c, ŷ.shape[1]):
    
            x_temp = x̂[i, :j+1]
            y_temp = ŷ[i , :j+1]

            ix = np.argmin(np.abs(x_temp[-1] - x_temp[:-1]))
            x_rep = x_temp[-1] - x_temp[ix]
            y_rep = y_temp[-1] - y_temp[ix]

            deriv = y_rep / (x_rep + 0.0001)
            
            diff_y[i, j] = y_rep
            diff_x[i, j, :] = x_rep
            dd[i, j] = deriv
            x_n[i, j, :] = x_temp[ix]

            if embed:
                diff_x[i, j, :] = PE(np.array([x_temp[-1]])[:, np.newaxis]) -  PE(np.array([x_temp[ix]])[:, np.newaxis])
                x_n[i, j, :] = PE(np.array([x_temp[ix]])[:, np.newaxis])
            
            
            y_n[i, j] = y_temp[ix]

    return diff_y, diff_x, dd, x_n, y_n

## We will need the date information in a numeric version 
def date_to_numeric(col):
    datetime = pd.to_datetime(col)
    return datetime.dt.hour, datetime.dt.day, datetime.dt.month, datetime.dt.year