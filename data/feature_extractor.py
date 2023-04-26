import numpy as np
import tensorflow as tf
    

class feature_wrapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
    
        x_emb,y,y_diff,x_diff,d,x_n,y_n,n_C,n_T = inputs ##think about clearer notation

        
        dim_y = y.shape[-1]
        dim_x = x_n.shape[-1]
        deriv_shape = d.shape[-1]

        ##### inputs for the MHA-X head ######


        value_x =  tf.identity(y) #check if identity is needed

        x_prime =  tf.concat([x_emb, x_diff, x_n], axis=2) ### check what is happening with embedding
        query_x = tf.identity(x_prime)
        key_x = tf.identity(x_prime)

        ##### inputs for the MHA-XY head ######

        y_prime = tf.concat([y,y_diff,d,y_n],axis=-1)
        batch_s = tf.shape(y_prime)[0]
        key_xy_label = tf.zeros((batch_s,n_C+n_T,1))
        value_xy = tf.concat([y_prime,key_xy_label,x_prime],axis=-1)
        key_xy = tf.identity(value_xy)

        query_xy_label = tf.concat([tf.zeros((batch_s,n_C,1)),tf.ones((batch_s,n_T,1))],axis=1)
        y_prime_1 = tf.concat([y[:,:n_C],tf.zeros((batch_s,n_T,dim_y))],axis=1)
        y_prime_2 = tf.concat([y_diff[:,:n_C],tf.zeros((batch_s,n_T,dim_y*dim_x))],axis=1)
        y_prime_3 = tf.concat([d[:,:n_C],tf.zeros((batch_s,n_T,deriv_shape))],axis=1)
        y_prime_masked = tf.concat([y_prime_1,y_prime_2,y_prime_3,y_n],axis=2)

        query_xy = tf.concat([y_prime_masked,query_xy_label,x_prime],axis=-1)

        return query_x, key_x, value_x, query_xy, key_xy, value_xy
    
    def permute(self,inputs):

        x,y,n_C,n_T,num_permutation_repeats = inputs

        if num_permutation_repeats <= 1:
            return x,y
        
        else: 
            batch_size = x.shape[0]
            dim_x = x.shape[-1]
            dim_y = y.shape[-1]

            permute_indices_a = tf.argsort(tf.random.uniform((num_permutation_repeats,n_T)),axis=-1)
            permute_indices = tf.repeat(permute_indices_a,axis=0,repeats=batch_size)
            repeated_x = tf.tile(x,multiples=[num_permutation_repeats,1,1])
            repeated_y = tf.tile(y,multiples=[num_permutation_repeats,1,1])

            selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(num_permutation_repeats*batch_size),n_T),(-1,1)),
                                tf.reshape(permute_indices,(-1, 1))],axis=1)

            x_target = tf.reshape(tf.gather_nd(repeated_x[:,n_C:],selection_indices),(num_permutation_repeats*batch_size,n_T,dim_x))
            y_target = tf.reshape(tf.gather_nd(repeated_y[:,n_C:],selection_indices),(num_permutation_repeats*batch_size,n_T,dim_y))

            x_permuted = tf.concat([repeated_x[:,:n_C],x_target],axis=1)
            y_permuted = tf.concat([repeated_y[:,:n_C],y_target],axis=1)
            return x_permuted,y_permuted
        
        
    def PE(self,inputs):  # return.shape=(T,B,d)
        # t.shape=(T,B)   T=sequence_length, B=batch_size
        """A position-embedder, similar to the Attention paper, but tweaked to account for
        floating point positions, rather than integer.

        """

        x, enc_dim, xΔmin, xmax = inputs

        R = xmax / xΔmin * 100
        drange_even = tf.cast(xΔmin * R**(tf.range(0,enc_dim,2)/enc_dim),"float32")
        drange_odd = tf.cast(xΔmin * R**((tf.range(1,enc_dim,2) - 1)/enc_dim),"float32")
        x = tf.concat([tf.math.sin(x / drange_even), tf.math.cos(x / drange_odd)], axis=2)
        return x            

class DE(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        y, x, n_C, n_T,training = inputs

        if x.shape[-1] == 1:
            y_diff,x_diff,derivative,x_n,y_n = self.derivative_function([ y, x, n_C, n_T])
        else: 
            y_diff,x_diff,derivative,x_n,y_n = self.derivative_function_2d([ y, x, n_C, n_T])

        derivative_new = tf.where(tf.math.is_nan(derivative), 10000.0, derivative)
        derivative_new_2 = tf.where(tf.abs(derivative_new) > 200., 0., derivative_new)

        derivative_scaled = self.batch_norm_layer(derivative_new_2,training=training)

        if tf.math.reduce_any(derivative_new != derivative_new_2):
            # you will need a label as some of the derivatives aren't "real"
            label_deriv = tf.cast(derivative_new_2 == derivative_new,"float32")
            derivative_scaled = tf.concat([derivative_scaled,label_deriv],axis=-1)

        return y_diff,x_diff,derivative_scaled,x_n,y_n

 ############ check what to do for 2d derivatives - should y diff just be for one point? for residual trick that would make most sense.
 #### but you need mutli-dimensional y for the derivative

 ###### and explain why we do this

    def derivative_function(self,inputs):
        
        y_values,x_values,context_n,target_m = inputs

        epsilon = 0.000002 

        batch_size,length = y_values.shape[0],context_n + target_m

        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]


        #context section

        current_x = tf.expand_dims(x_values[:,:context_n],axis=2)
        current_y = tf.expand_dims(y_values[:,:context_n],axis=2)

        x_temp = x_values[:,:context_n]
        x_temp = tf.repeat(tf.expand_dims(x_temp,axis=1),axis=1,repeats=context_n)
        
        y_temp = y_values[:,:context_n]
        y_temp = tf.repeat(tf.expand_dims(y_temp,axis=1),axis=1,repeats=context_n)
        

        ix = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp),axis=-1),axis=-1)[:,:,1]        
        selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n),1),(-1,1)),
                                       tf.reshape(ix,(-1,1))],axis=1)


        x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,context_n,dim_x)),selection_indices),
                               (batch_size,context_n,dim_x)) 
        
        
        y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,context_n,dim_y)),selection_indices),
                       (batch_size,context_n,dim_y))
        
        x_rep = current_x[:,:,0] - x_closest
        y_rep = current_y[:,:,0] - y_closest            

        deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep,axis=-1,keepdims=True))

        dydx_dummy = deriv
        diff_y_dummy = y_rep
        diff_x_dummy =x_rep
        closest_y_dummy = y_closest
        closest_x_dummy = x_closest

        #target selection

        current_x = tf.expand_dims(x_values[:,context_n:context_n+target_m],axis=2)
        current_y = tf.expand_dims(y_values[:,context_n:context_n+target_m],axis=2)

        x_temp = tf.repeat(tf.expand_dims(x_values[:,:target_m+context_n],axis=1),axis=1,repeats=target_m)
        y_temp = tf.repeat(tf.expand_dims(y_values[:,:target_m+context_n],axis=1),axis=1,repeats=target_m)


        x_mask = tf.linalg.band_part(tf.ones((target_m,context_n + target_m),tf.bool),-1,context_n)
        x_mask_inv = (x_mask == False)
        x_mask_float = tf.cast(x_mask_inv,"float32")*1000
        x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float,axis=0),axis=0,repeats=batch_size)

        ix = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp),
                                            axis=-1),dtype="float32") + x_mask_float_repeat,axis=-1)[:,:,1]

       
        selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m),1),(-1,1)),
                                   tf.reshape(ix,(-1,1))],axis=1)

        x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,target_m+context_n,dim_x)),selection_indices),
                               (batch_size,target_m,dim_x)) 
        
        y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,target_m+context_n,dim_y)),selection_indices),
                       (batch_size,target_m,dim_y))
        
        
        x_rep = current_x[:,:,0] - x_closest
        y_rep = current_y[:,:,0] - y_closest            

        deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep,axis=-1,keepdims=True))

        dydx_dummy = tf.concat([dydx_dummy,deriv],axis=1)
        diff_y_dummy = tf.concat([diff_y_dummy,y_rep],axis=1)
        diff_x_dummy = tf.concat([diff_x_dummy,x_rep],axis=1)
        closest_y_dummy = tf.concat([closest_y_dummy,y_closest],axis=1)
        closest_x_dummy = tf.concat([closest_x_dummy,x_closest],axis=1)

        return diff_y_dummy,diff_x_dummy,dydx_dummy,closest_x_dummy,closest_y_dummy


    def derivative_function_2d(self,inputs):

            epsilon = 0.0000
        
            def dydz(current_y,y_closest_1,y_closest_2,current_x,x_closest_1,x_closest_2):
                #"z" is the second dim of x input
                numerator = y_closest_2 - current_y[:,:,0] - ((x_closest_2[:,:,:1]-current_x[:,:,0,:1])*(y_closest_1-current_y[:,:,0] ))/(x_closest_1[:,:,:1]-current_x[:,:,0,:1] +epsilon)
                denom = x_closest_2[:,:,1:2] - current_x[:,:,0,1:2] - (x_closest_1[:,:,1:2]-current_x[:,:,0,1:2])*(x_closest_2[:,:,:1]-current_x[:,:,0,:1])/(x_closest_1[:,:,:1]-current_x[:,:,0,:1]+epsilon)
                dydz_pred = numerator/(denom+epsilon)
                return dydz_pred
            
            def dydx(dydz,current_y,y_closest_1,current_x,x_closest_1):
                dydx = (y_closest_1-current_y[:,:,0] - dydz*(x_closest_1[:,:,1:2]-current_x[:,:,0,1:2]))/(x_closest_1[:,:,:1]-current_x[:,:,0,:1]+epsilon)
                return dydx

            y_values,x_values,context_n,target_m = inputs

            batch_size,length = y_values.shape[0],context_n + target_m

            dim_x = x_values.shape[-1]
            dim_y = y_values.shape[-1]


            #context section

            current_x = tf.expand_dims(x_values[:,:context_n],axis=2)
            current_y = tf.expand_dims(y_values[:,:context_n],axis=2)

            x_temp = x_values[:,:context_n]
            x_temp = tf.repeat(tf.expand_dims(x_temp,axis=1),axis=1,repeats=context_n)

            y_temp = y_values[:,:context_n]
            y_temp = tf.repeat(tf.expand_dims(y_temp,axis=1),axis=1,repeats=context_n)

            ix_1 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp),axis=-1),axis=-1)[:,:,1]        
            selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n),1),(-1,1)),
                                                tf.reshape(ix_1,(-1,1))],axis=1)

            ix_2 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp),axis=-1),axis=-1)[:,:,2]        
            selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n),1),(-1,1)),
                                        tf.reshape(ix_2,(-1,1))],axis=1)


            x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,context_n,dim_x)),selection_indices_1),
                                (batch_size,context_n,dim_x)) +   tf.random.normal(shape=(batch_size,context_n,dim_x),stddev=0.01)

            x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,context_n,dim_x)),selection_indices_2),
                                (batch_size,context_n,dim_x)) +   tf.random.normal(shape=(batch_size,context_n,dim_x),stddev=0.01)



            y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,context_n,dim_y)),selection_indices_1),
                        (batch_size,context_n,dim_y))


            y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,context_n,dim_y)),selection_indices_2),
                        (batch_size,context_n,dim_y))


            x_rep_1 = current_x[:,:,0] - x_closest_1
            x_rep_2 = current_x[:,:,0] - x_closest_2

            y_rep_1 = current_y[:,:,0] - y_closest_1
            y_rep_2 = current_y[:,:,0] - y_closest_2

            dydx_2 = dydz(current_y,y_closest_1,y_closest_2,current_x,x_closest_1,x_closest_2)
            dydx_1 = dydx(dydx_2,current_y,y_closest_1,current_x,x_closest_1)

            deriv_dummy = tf.concat([dydx_1,dydx_2],axis=-1)

            diff_y_dummy = tf.concat([y_rep_1,y_rep_2],axis=-1)

            diff_x_dummy =tf.concat([x_rep_1,x_rep_2],axis=-1)

            closest_y_dummy = tf.concat([y_closest_1,y_closest_2],axis=-1)
            closest_x_dummy = tf.concat([x_closest_1,x_closest_2],axis=-1)

            #target selection

            current_x = tf.expand_dims(x_values[:,context_n:context_n+target_m],axis=2)
            current_y = tf.expand_dims(y_values[:,context_n:context_n+target_m],axis=2)

            x_temp = tf.repeat(tf.expand_dims(x_values[:,:target_m+context_n],axis=1),axis=1,repeats=target_m)
            y_temp = tf.repeat(tf.expand_dims(y_values[:,:target_m+context_n],axis=1),axis=1,repeats=target_m)


            x_mask = tf.linalg.band_part(tf.ones((target_m,context_n + target_m),tf.bool),-1,context_n)
            x_mask_inv = (x_mask == False)
            x_mask_float = tf.cast(x_mask_inv,"float32")*1000
            x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float,axis=0),axis=0,repeats=batch_size)
            
            
            ix_1 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp),
                                                axis=-1),dtype="float32") + x_mask_float_repeat,axis=-1)[:,:,1]
            selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m),1),(-1,1)),
                                                tf.reshape(ix_1,(-1,1))],axis=1)
            
            
            
            ix_2 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp),
                                                axis=-1),dtype="float32") + x_mask_float_repeat,axis=-1)[:,:,2]
            selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m),1),(-1,1)),
                                                tf.reshape(ix_2,(-1,1))],axis=1)
            
            
            
            x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,target_m+context_n,dim_x)),selection_indices_1),
                                (batch_size,target_m,dim_x)) +   tf.random.normal(shape=(batch_size,target_m,dim_x),stddev=0.01)

            x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp,(-1,target_m+context_n,dim_x)),selection_indices_2),
                                (batch_size,target_m,dim_x)) +   tf.random.normal(shape=(batch_size,target_m,dim_x),stddev=0.01)



            y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,target_m+context_n,dim_y)),selection_indices_1),
                        (batch_size,target_m,dim_y))


            y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp,(-1,target_m+context_n,dim_y)),selection_indices_2),
                        (batch_size,target_m,dim_y))
        

            x_rep_1 = current_x[:,:,0] - x_closest_1
            x_rep_2 = current_x[:,:,0] - x_closest_2

            y_rep_1 = current_y[:,:,0] - y_closest_1
            y_rep_2 = current_y[:,:,0] - y_closest_2

            dydx_2 = dydz(current_y,y_closest_1,y_closest_2,current_x,x_closest_1,x_closest_2)
            dydx_1 = dydx(dydx_2,current_y,y_closest_1,current_x,x_closest_1)
            
            deriv_dummy_2 = tf.concat([dydx_1,dydx_2],axis=-1)

            diff_y_dummy_2 = tf.concat([y_rep_1,y_rep_2],axis=-1)

            diff_x_dummy_2 =tf.concat([x_rep_1,x_rep_2],axis=-1)

            closest_y_dummy_2 = tf.concat([y_closest_1,y_closest_2],axis=-1)
            closest_x_dummy_2 = tf.concat([x_closest_1,x_closest_2],axis=-1)
            
            ########## concat all ############


            deriv_dummy_full = tf.concat([deriv_dummy,deriv_dummy_2],axis=1)
            diff_y_dummy_full = tf.concat([diff_y_dummy,diff_y_dummy_2],axis=1)
            diff_x_dummy_full = tf.concat([diff_x_dummy,diff_x_dummy_2],axis=1)
            closest_y_dummy_full = tf.concat([closest_y_dummy,closest_y_dummy_2],axis=1)
            closest_x_dummy_full = tf.concat([closest_x_dummy,closest_x_dummy_2],axis=1)

            return diff_y_dummy_full,diff_x_dummy_full,deriv_dummy_full,closest_x_dummy_full,closest_y_dummy_full




# def make_features(t, y, num_context_points, batch_s=32):
#     x = PE(t, d=28, TΔmin=0.1, Tmax=2)
    
#     value_x = y[:, :, np.newaxis]
#     context_points = int(context_points)

#     mask = np.tri(y.shape[1], y.shape[1], 0) - np.eye(y.shape[1])
#     mask[:context_points, :context_points] = 1 
#     mask = np.repeat(mask[np.newaxis, :, :], batch_s, axis=0)

#     diff_y, diff_x, d, x_n, y_n = DE(y, t, context_points, embed=True)
#     y_prime = np.concatenate([y[:, :, np.newaxis], diff_y.reshape(batch_s, -1, 1), d.reshape(batch_s, -1, 1), y_n.reshape(batch_s, -1, 1)], axis=2)
#     query_x = key_x = x_prime = np.concatenate([x, diff_x, x_n], axis=2)
    
#     query_xy_label = np.ones((batch_s, y.shape[1], 1))
#     key_xy_label = np.concatenate([np.ones((batch_s, context_points, 1)), np.zeros((batch_s, y.shape[1]-context_points, 1))], axis=1)


#     key_xy = value_xy = np.concatenate([y_prime, key_xy_label, x_prime], axis=2)
#     query_xy = np.concatenate([y_prime, query_xy_label, x_prime], axis=2)
#     query_xy[:, context_points:, :3] = 0

#     return query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n


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


# def DE(ŷ, x̂, c, embed=False):
#     d=1
#     if embed:
#         d=28
    
#     m, n = ŷ.shape[0], ŷ.shape[1]
#     diff_y = np.zeros((m , n))
#     diff_x = np.zeros((m, n, d))
#     dd = np.zeros((m, n))
#     y_n = np.zeros((m , n))
#     x_n = np.zeros((m , n, d))
    
#     for i in range(m):
#         for j in range(c):
#             current_x = (x̂[i, :c][j])
#             current_y = ŷ[i, :c][j]
#             x_temp = (x̂[i, :c])
#             y_temp = ŷ[i , :c]
#             ix = np.argsort(np.abs(current_x - x_temp))[1] 

#             x_rep = current_x - x_temp[ix]
#             y_rep = current_y - y_temp[ix]
#             deriv = y_rep / (x_rep + 0.0001)
            
#             diff_y[i, j] = y_rep
#             diff_x[i, j, :] = x_rep
#             x_n[i, j, :] = x_temp[ix]
#             if embed:
#                 diff_x[i, j, :] = PE(np.array([current_x])[:, np.newaxis]) -  PE(np.array([x_temp[ix]])[:, np.newaxis])
#                 x_n[i, j, :] = PE(np.array([x_temp[ix]])[:, np.newaxis])
            
#             dd[i, j] = deriv
#             y_n[i, j] = y_temp[ix]
        
#         for j in range(c, ŷ.shape[1]):
    
#             x_temp = x̂[i, :j+1]
#             y_temp = ŷ[i , :j+1]

#             ix = np.argmin(np.abs(x_temp[-1] - x_temp[:-1]))
#             x_rep = x_temp[-1] - x_temp[ix]
#             y_rep = y_temp[-1] - y_temp[ix]

#             deriv = y_rep / (x_rep + 0.0001)
            
#             diff_y[i, j] = y_rep
#             diff_x[i, j, :] = x_rep
#             dd[i, j] = deriv
#             x_n[i, j, :] = x_temp[ix]

#             if embed:
#                 diff_x[i, j, :] = PE(np.array([x_temp[-1]])[:, np.newaxis]) -  PE(np.array([x_temp[ix]])[:, np.newaxis])
#                 x_n[i, j, :] = PE(np.array([x_temp[ix]])[:, np.newaxis])
            
            
#             y_n[i, j] = y_temp[ix]

#     return diff_y, diff_x, dd, x_n, y_n

## We will need the date information in a numeric version 
def date_to_numeric(col):
    datetime = pd.to_datetime(col)
    return datetime.dt.hour, datetime.dt.day, datetime.dt.month, datetime.dt.year