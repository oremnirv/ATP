import numpy as np

### let's create data batches that fit the task
def batcher(t, y, batch_s = 32, training=True, given_hr = 96, pred_hr = 192):
    win_l = given_hr + pred_hr
    tr_last_ix = int((y.shape[0]) *0.7)
    val_last_ix = int((y.shape[0]) *0.8)

    if training:
        y = y[:tr_last_ix]
        t = t[:tr_last_ix]
    else: 
        y = y[tr_last_ix:val_last_ix]
        t = t[tr_last_ix:val_last_ix]
    idx = np.random.choice(np.arange(win_l, len(y) - win_l), batch_s, replace=False)

    y = np.array([np.array(y)[i:i+win_l] for i in idx])
    t = np.array([np.array(t)[i:i+win_l] for i in idx])
    
    return t, y

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

    diff_y, diff_x, d, x_n, y_n = derivative_extr(y.squeeze(), t, context_points, embed=True)
    y_prime = np.concatenate([y[:, :, np.newaxis], diff_y.reshape(batch_s, -1, 1), d.reshape(batch_s, -1, 1), y_n.reshape(batch_s, -1, 1)], axis=2)
    query_x = key_x = x_prime = np.concatenate([x, diff_x, x_n], axis=2)

    key_xy = value_xy = np.concatenate((y_prime, x_prime), axis=2)
    query_xy = key_xy.copy()
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


def derivative_extr(ŷ, x̂, c, embed=False):
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

            x_rep = float(current_x) - float(x_temp[ix])
            y_rep = current_y - y_temp[ix]
            deriv = float(y_rep) / (float(x_rep) + 0.0001)
            
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
            x_rep = float(x_temp[-1]) - float(x_temp[ix])
            y_rep = float(y_temp[-1]) - float(y_temp[ix])

            deriv = float(y_rep) / (float(x_rep) + 0.0001)
            
            diff_y[i, j] = y_rep
            diff_x[i, j, :] = x_rep
            dd[i, j] = deriv
            
            x_n[i, j, :] = x_temp[ix]

            if embed:
                diff_x[i, j, :] = PE(np.array([x_temp[-1]])[:, np.newaxis]) -  PE(np.array([x_temp[ix]])[:, np.newaxis])
                x_n[i, j, :] = PE(np.array([x_temp[ix]])[:, np.newaxis])
            
            
            y_n[i, j] = y_temp[ix]

    return diff_y, diff_x, dd, x_n, y_n