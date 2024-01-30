import numpy as np
import tensorflow as tf
from data_wrangler import feature_extractor, dataset_preparer
from data_wrangler import batcher
import math as m
    

class feature_wrapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self,  inputs):
    
        x_emb,  y,  y_diff,  x_diff,  d,  x_n,  y_n,  n_C,  n_T = inputs # (batch_s, multiply * (n_C + n_T), enc_dim + multiply) (batch_s, multiply * (n_C + n_T), 1) (batch_s, multiply * (n_C + n_T), 1) (batch_s, multiply * (n_C + n_T), 1) (batch_s, multiply * (n_C + n_T), 2) (batch_s, multiply * (n_C + n_T), 1) (batch_s, multiply * (n_C + n_T), 1)
        ##### inputs for the MHA-X head ######
        value_x =  tf.identity(y) #check if identity is needed
        ### check what is happening with embedding
        x_prime =  tf.concat([x_emb,  x_diff,  x_n],  axis=2) # (batch_s, multiply * (n_C + n_T), enc_dim + multiply + 2)

        query_x = tf.identity(x_prime)
        key_x = tf.identity(x_prime)

        ##### inputs for the MHA-XY head ######
        y_prime = tf.concat([y,  y_diff,  d,  y_n], axis=-1)
        batch_s = (y_prime.shape)[0]
        key_xy_label = tf.zeros((batch_s,  tf.shape(y_prime)[1],  1))
        value_xy = tf.concat([y_prime,  key_xy_label,  x_prime], axis=-1)
        key_xy = tf.identity(value_xy)
        query_xy_label = tf.concat([tf.zeros((batch_s,  n_C ,  1)), tf.ones((batch_s,  n_T,  1))],  axis=1)
        y_prime_masked = tf.concat([self.mask_target_pt([y,  n_C,  n_T]),  self.mask_target_pt([y_diff,  n_C,  n_T]),  self.mask_target_pt([d,  n_C,  n_T]),  y_n],  axis=2)
        query_xy = tf.concat([y_prime_masked,  query_xy_label,  x_prime], axis=-1)
        return query_x,  key_x,  value_x,  query_xy,  key_xy,  value_xy

    def mask_target_pt(self,  inputs):
        y,  n_C,  n_T = inputs
        dim_y = y.shape[-1]
        batch_s = y.shape[0]

        mask_y = tf.concat([y[:,  :n_C],  tf.zeros((batch_s,  n_T,  dim_y))],  axis=1)
        return mask_y
    
    def masker(self, n_C, n_T):

        context_part = tf.concat([tf.ones((n_C, n_C), tf.bool), tf.zeros((n_C, n_T ), tf.bool)], axis=-1)
        diagonal_mask = tf.linalg.band_part(tf.ones(((n_C+n_T), (n_C+n_T)), tf.bool),-1,0)
        # print("diagonal_mask: ", diagonal_mask.shape)
        lower_diagonal_mask = tf.linalg.set_diag(diagonal_mask, tf.zeros(diagonal_mask.shape[0:-1],tf.bool)) 
        mask = tf.concat([context_part, lower_diagonal_mask[n_C:(n_C+n_T), :(n_C+n_T)]], axis=0) 
        return mask
    
    
    def permute(self,  inputs):

        x,  y,  n_C,  _,  num_permutation_repeats = inputs

        if (num_permutation_repeats < 1):
            return x,  y
        
        else: 
            # Shuffle traget only. tf.random.shuffle only works on the first dimension so we need tf.transpose.
            x_permuted = tf.concat([tf.concat([x[:,  :n_C, :], tf.transpose(tf.random.shuffle(tf.transpose(x[:, n_C:, :], perm=[1, 0, 2])), perm =[1, 0, 2])], axis=1) for j in range(num_permutation_repeats)], axis=0)            
            y_permuted = tf.concat([tf.concat([y[:,  :n_C, :], tf.transpose(tf.random.shuffle(tf.transpose(y[:, n_C:, :], perm=[1, 0, 2])), perm =[1, 0, 2])], axis=1) for j in range(num_permutation_repeats)], axis=0)

            return x_permuted,  y_permuted
        
    def sorted_rand_idx(self,  n, num_idxs):
        # print("n: ", n)
        # print("num_idxs: ", num_idxs)
        return  tf.sort((tf.reshape(tf.argsort(tf.random.normal([n])), [-1]))[:num_idxs])
        
    def gather_idx_from_tensors(self,  tensor_list,  indices_list):
        for i in range(len(tensor_list)):
            tensor_list[i] = tf.gather(tensor_list[i], indices_list[i], axis=1)
        return tensor_list

    def subsample(self, x, y, n_C, n_T, n_C_s, n_T_s):
        """
        Subsample the context and target points.
        x and y are tensors of shape (B, n_C+n_T, d)
        n_C_s and n_T_s are the number of context and target points to subsample

        Returns:
            x, y: tensors of shape (B, n_C_s+n_T_s, d)
        """

        indices_c = self.sorted_rand_idx(n_C, n_C_s)
        # print("indices c: ", indices_c)
        indices_t = self.sorted_rand_idx(n_T, n_T_s)

        tensors_x = self.gather_idx_from_tensors([x[:, :n_C, :], x[:, n_C:n_C+n_T, :]], [indices_c, indices_t])
        x = tf.concat(tensors_x, axis=1)
        tensors_y = self.gather_idx_from_tensors([y[:, :n_C, :], y[:, n_C:n_C+n_T, :]], [indices_c, indices_t])
        y = tf.concat(tensors_y, axis=1)
        new_shape_x = [tf.shape(x)[0], n_C_s + n_T_s, x.shape[-1]]
        x = tf.reshape(x, new_shape_x)
        # print('n_C_s_A', n_C_s)
        new_shape_y = [tf.shape(y)[0], n_C_s + n_T_s, y.shape[-1]]
        y = tf.reshape(y, new_shape_y)
        return x, y
        
        
    def PE(self,  inputs):  # return.shape=(T, B, d)
        """
        # t.shape=(T, B)   T=sequence_length,  B=batch_size
        A position-embedder,  similar to the Attention paper,  but tweaked to account for
        floating point positions,  rather than integer.
        """
        x,  enc_dim,  xΔmin,  xmax = inputs
        π  = tf.constant(m.pi)
        R = tf.cast(xmax / (2 * π), "float32")
        # drange_even = tf.cast(xΔmin * R**(tf.range(0, enc_dim, 2) / enc_dim), "float32")
        # drange_odd = tf.cast(xΔmin * R**((tf.range(1, enc_dim, 2) - 1) / enc_dim), "float32")

        drange_even = tf.cast(tf.cast(1, "float32")/R**(tf.range(0, enc_dim, 2, dtype=tf.float32) / enc_dim), "float32")
        drange_odd = tf.cast(tf.cast(1, "float32")/R**((tf.range(1, enc_dim, 2, dtype=tf.float32) - 1) / enc_dim), "float32")
        
        x = tf.concat([tf.math.sin(x / drange_even),  tf.math.cos(x / drange_odd)],  axis=2)
        return x            

class DE(tf.keras.layers.Layer):
    def __init__(self, n_C_s, n_T_s, img_seg=False):
        super().__init__()
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.img_seg = img_seg
        self.n_C_s = n_C_s
        self.n_T_s = n_T_s

    def call(self,  inputs):
        y_all, x_all, y_temp_context, y_temp_target, x_temp_context, x_temp_target,  context_n, target_m, i,  training = inputs

        if (x_temp_context.shape[-1] == 1):
            print("1d")
            y_diff,  x_diff,  d,  x_n,  y_n = self.derivative_function([y_all, x_all, y_temp_context, y_temp_target, x_temp_context, x_temp_target,  context_n,  target_m, i])
        else: 
            # print("2d")
            y_diff,  x_diff,  d,  x_n,  y_n = self.derivative_function_2d([y,  x,  self.n_C_s,  self.n_T_s])

        d_1 = tf.where(tf.math.is_nan(d),  10000.0,  d)
        d_2 = tf.where(tf.abs(d) > 200.,  0.,  d)
        d = self.batch_norm_layer(d_2,  training=training)

        d_label = tf.cast(tf.math.equal(d_2,  d_1),  "float32")
        d = tf.concat([d,  d_label],  axis=-1)

        return y_diff,  x_diff,  d,  x_n,  y_n


###### i think here what we do is calculate the derivative at the given y value and add that in as a feature. This is masked when making predictions
# so the derivative of other y values are what are seen
#  Based on taylor expansion, a better feature would be including the derivative of the closest x point, where only seen y values are used for the differencing. 
#this derivative wouldn't need masking.

 ############ check what to do for 2d derivatives - should y diff just be for one point? for residual trick that would make most sense.
 #### but you need mutli-dimensional y for the derivative

 ###### and explain why we do this
    @tf.function
    def derivative_function(self,  inputs):
        
        y_all, x_all, y_temp_context, y_temp_target, x_temp_context, x_temp_target,  _,  _, _ = inputs
        epsilon = 0.000002 

        batch_size = y_temp_context.shape[0]

        dim_x = x_temp_context.shape[-1]
        dim_y = y_temp_context.shape[-1]
        # print('y_temp_context.shape', y_temp_context.shape)
        # print('y_temp_target.shape', y_temp_target.shape)
        # print('x_temp_context.shape', x_temp_context.shape)
        # print('x_temp_target.shape', x_temp_target.shape)
        c_m = tf.shape(y_temp_context)[1]
        t_m = tf.shape(y_temp_target)[1]


        #context section

        current_x = tf.expand_dims(x_temp_context, axis=2) # (32, 20, 1, 1)
        current_y = tf.expand_dims(y_temp_context, axis=2) # (32, 20, 1, 1)
        
        x_temp = tf.repeat(tf.expand_dims(x_temp_context,  axis=1),  axis=1,  repeats=tf.shape(current_x)[1]) #x_temp.shape (32, 20, 20, 1)
        y_temp = tf.repeat(tf.expand_dims(y_temp_context,  axis=1),  axis=1,  repeats=tf.shape(current_y)[1])
        

        ix = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 1]
        ix = tf.cast(ix, tf.int32)
        selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*c_m), 1), (-1, 1)), 
                                    tf.reshape(ix, (-1, 1))], axis=1)


        x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, c_m, dim_x)), selection_indices), 
                            (batch_size, c_m, dim_x)) 
        # print("x_closest.shape", x_closest.shape)
        y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, c_m, dim_y)), selection_indices), 
                    (batch_size, c_m, dim_y))
        # print("y_closest.shape", y_closest.shape)
        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest
        # print("x_rep_DFG.shape", x_rep.shape)
        # print("y_rep_DFG.shape", y_rep.shape)            
        if not self.img_seg: # or not bc for heatwave

            deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep, axis=-1, keepdims=True))
            # print("deriv.shape", deriv.shape)

        else:
            deriv = tf.zeros_like(y_rep)

        dydx_dummy = deriv
        diff_y_dummy = y_rep
        diff_x_dummy =x_rep
        closest_y_dummy = y_closest
        closest_x_dummy = x_closest

        if t_m > 0:
            print('target_m > 0')
            current_x = tf.expand_dims(x_temp_target, axis=2) # (32, 20, 1, 1)
            current_y = tf.expand_dims(y_temp_target, axis=2) # (32, 20, 1, 1)
        
            x_temp = tf.repeat(tf.expand_dims(x_all[:, :c_m + t_m],  axis=1),  axis=1,  repeats=t_m) #x_temp.shape (32, 20, 20, 1)
            y_temp = tf.repeat(tf.expand_dims(y_all[:, :c_m + t_m],  axis=1),  axis=1,  repeats=t_m)
            
            x_temp = tf.reshape(x_temp, (batch_size, tf.shape(x_temp_target)[1],tf.shape(x_temp_target)[1] + tf.shape(x_temp_context)[1], 1))
            y_temp = tf.reshape(y_temp, (batch_size, tf.shape(y_temp_target)[1],tf.shape(y_temp_target)[1] + tf.shape(y_temp_context)[1], 1))

            print("x_temp.shape", x_temp.shape)
            print("y_temp.shape", y_temp.shape)

            # print("x_temp_target.shape", x_temp.shape)
            # print("y_temp_target.shape", y_temp.shape)

            t_m = tf.shape(current_x)[1]
            x_mask = tf.linalg.band_part(tf.ones((t_m, c_m + t_m), tf.bool),  tf.cast(-1, tf.int32), c_m)
            x_mask = tf.reshape(x_mask, (1, t_m, c_m+t_m))
            x_mask_inv = (x_mask == False)
            x_mask_float = tf.cast(x_mask_inv, "float32")*1000
            x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float, axis=0), axis=0, repeats=batch_size)
            x_mask_float_repeat = tf.reshape(x_mask_float_repeat, (batch_size, t_m, c_m+t_m))

            # print('diff shape: ', (current_x - x_temp).shape)
            # print("x_mask_float_repeat.shape", x_mask_float_repeat.shape)
            # print('current_x.shape', current_x.shape)
            # print('x_temp.shape', x_temp.shape)
            ix = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                                                axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 1]

            # print("ix.shape", ix.shape)
            ix = tf.cast(ix, tf.int32)
            selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*t_m), 1), (-1, 1)), 
                                    tf.reshape(ix, (-1, 1))], axis=1)

            x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, t_m+c_m, dim_x)), selection_indices), 
                                (batch_size, t_m, dim_x)) 
            
            y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, t_m+c_m, dim_y)), selection_indices), 
                        (batch_size, t_m, dim_y))
            # print("x_closest.shape", x_closest.shape)
            # print("y_closest.shape", y_closest.shape)
            
            
            x_rep = current_x[:, :, 0] - x_closest
            y_rep = current_y[:, :, 0] - y_closest  
            # print("x_rep.shape", x_rep.shape)
            # print("y_rep.shape", y_rep.shape)          

            if not self.img_seg: # or not bc for heatwave

                deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep, axis=-1, keepdims=True))

            else:
                deriv = tf.zeros_like(y_rep)

            dydx_dummy = tf.concat([dydx_dummy, deriv], axis=1)
            diff_y_dummy = tf.concat([diff_y_dummy, y_rep], axis=1)
            diff_y_dummy =tf.reshape(diff_y_dummy, (batch_size, c_m+t_m, dim_y))
            diff_x_dummy = tf.concat([diff_x_dummy, x_rep], axis=1)
            diff_x_dummy =tf.reshape(diff_x_dummy, (batch_size, c_m+t_m, dim_x))
            closest_y_dummy = tf.concat([closest_y_dummy, y_closest], axis=1)
            closest_y_dummy =tf.reshape(closest_y_dummy, (batch_size, c_m+t_m, dim_y))
            closest_x_dummy = tf.concat([closest_x_dummy, x_closest], axis=1)
            closest_x_dummy =tf.reshape(closest_x_dummy, (batch_size, c_m+t_m, dim_x))
            # print("dydx_dummy.shapeTARGET", dydx_dummy.shape)
            # print("diff_y_dummy.shape", diff_y_dummy.shape)
            # print("diff_x_dummy.shape", diff_x_dummy.shape)
            # print("closest_y_dummy.shape", closest_y_dummy.shape)
            # print("closest_x_dummy.shape", closest_x_dummy.shape)

        return diff_y_dummy, diff_x_dummy, dydx_dummy, closest_x_dummy, closest_y_dummy


    def derivative_function_2d(self, inputs):

            epsilon = 0.0000
        
            def dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2):
                #"z" is the second dim of x input
                numerator = y_closest_2 - current_y[:, :, 0] - ((x_closest_2[:, :, :1]-current_x[:, :, 0, :1])*(y_closest_1-current_y[:, :, 0] ))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1] +epsilon)
                denom = x_closest_2[:, :, 1:2] - current_x[:, :, 0, 1:2] - (x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2])*(x_closest_2[:, :, :1]-current_x[:, :, 0, :1])/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
                dydz_pred = numerator/(denom+epsilon)
                return dydz_pred
            
            def dydx(dydz, current_y, y_closest_1, current_x, x_closest_1):
                dydx = (y_closest_1-current_y[:, :, 0] - dydz*(x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2]))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
                return dydx

            y_values, x_values, context_n, target_m = inputs
            batch_size, length = y_values.shape[0], context_n + target_m

            context_n = tf.cast(context_n, tf.int64)
            target_m = tf.cast(target_m, tf.int64)
            dim_x = x_values.shape[-1]
            dim_y = y_values.shape[-1]


            #context section
            # if (context_n > 0):
            current_x = tf.expand_dims(x_values[:, :context_n], axis=2)
            current_y = tf.expand_dims(y_values[:, :context_n], axis=2)

            x_temp = x_values[:, :context_n]
            x_temp = tf.repeat(tf.expand_dims(x_temp, axis=1), axis=1, repeats=context_n)

            y_temp = y_values[:, :context_n]
            y_temp = tf.repeat(tf.expand_dims(y_temp, axis=1), axis=1, repeats=context_n)


            ix_1 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 1]  
            ix_1 = tf.cast(ix_1, tf.int64)
            selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n), 1), (-1, 1)), 
                                                tf.reshape(ix_1, (-1, 1))], axis=1)

            ix_2 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 2]        
            ix_2 = tf.cast(ix_2, tf.int64)
            selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n), 1), (-1, 1)), 
                                        tf.reshape(ix_2, (-1, 1))], axis=1)


            x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_1), 
                                (batch_size, context_n, dim_x)) +   tf.random.normal(shape=(batch_size,  context_n,  dim_x), stddev=0.01)

            # print(x_closest_1.shape)
            x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_2), 
                                (batch_size, context_n, dim_x)) +   tf.random.normal(shape=(batch_size, context_n, dim_x), stddev=0.01)

            # print(x_closest_2.shape)

            y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_1), 
                        (batch_size, context_n, dim_y))

            # print(y_closest_1.shape)
            y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_2), 
                        (batch_size, context_n, dim_y))

            # print(y_closest_2.shape)
            x_rep_1 = current_x[:, :, 0] - x_closest_1
            x_rep_2 = current_x[:, :, 0] - x_closest_2

            y_rep_1 = current_y[:, :, 0] - y_closest_1
            y_rep_2 = current_y[:, :, 0] - y_closest_2
            # print("y_rep_1", y_rep_1.shape)
            # print("y_rep_2", y_rep_2.shape)
            if not self.img_seg:

                dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
                dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)

            else:
                # print("img_seg no derivs")
                dydx_2 = tf.zeros_like(x_rep_1)
                dydx_1 = tf.zeros_like(x_rep_1)
    
            deriv_dummy = tf.concat([dydx_1, dydx_2], axis=-1)

            diff_y_dummy = tf.concat([y_rep_1, y_rep_2], axis=-1)

            diff_x_dummy =tf.concat([x_rep_1, x_rep_2], axis=-1)

            closest_y_dummy = tf.concat([y_closest_1, y_closest_2], axis=-1)
            closest_x_dummy = tf.concat([x_closest_1, x_closest_2], axis=-1)

            deriv_dummy_full = deriv_dummy
            diff_y_dummy_full = diff_y_dummy
            diff_x_dummy_full = diff_x_dummy
            closest_y_dummy_full = closest_y_dummy
            closest_x_dummy_full = closest_x_dummy
            # print("closest_x_dummy_full", closest_x_dummy_full.shape)
            # print("deriv_dummy_full", deriv_dummy_full.shape)
            # print("diff_x_dummy_full", diff_x_dummy_full.shape)
            # print("diff_y_dummy_full", diff_y_dummy_full.shape)

            #target selection
            if (target_m > 0):
                # print("target_m > 0")
                current_x = tf.expand_dims(x_values[:, context_n:context_n+target_m], axis=2)
                current_y = tf.expand_dims(y_values[:, context_n:context_n+target_m], axis=2)

                x_temp = tf.repeat(tf.expand_dims(x_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)
                y_temp = tf.repeat(tf.expand_dims(y_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)


                x_mask = tf.linalg.band_part(tf.ones((target_m, context_n + target_m), tf.bool), tf.cast(-1, tf.int64), tf.cast(context_n, tf.int64))
                x_mask_inv = (x_mask == False)
                x_mask_float = tf.cast(x_mask_inv, "float32")*1000
                x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float, axis=0), axis=0, repeats=batch_size)
                
                
                ix_1 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                                                    axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 1]
                ix_1 = tf.cast(ix_1, tf.int64)
                selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m), 1), (-1, 1)), 
                                                    tf.reshape(ix_1, (-1, 1))], axis=1)
                
                
                
                ix_2 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                                                    axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 2]
                ix_2 = tf.cast(ix_2, tf.int64)
                selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m), 1), (-1, 1)), 
                                                    tf.reshape(ix_2, (-1, 1))], axis=1)
                
                
                
                x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices_1), 
                                    (batch_size, target_m, dim_x)) +   tf.random.normal(shape=(batch_size, target_m, dim_x), stddev=0.01)

                x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices_2), 
                                    (batch_size, target_m, dim_x)) +   tf.random.normal(shape=(batch_size, target_m, dim_x), stddev=0.01)



                y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices_1), 
                            (batch_size, target_m, dim_y))


                y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices_2), 
                            (batch_size, target_m, dim_y))
            

                x_rep_1 = current_x[:, :, 0] - x_closest_1
                x_rep_2 = current_x[:, :, 0] - x_closest_2

                y_rep_1 = current_y[:, :, 0] - y_closest_1
                y_rep_2 = current_y[:, :, 0] - y_closest_2
                if not self.img_seg:
                    dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
                    dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)
                
                else:
                    dydx_2 = tf.zeros_like(x_rep_1)
                    dydx_1 = tf.zeros_like(x_rep_1)
                deriv_dummy_2 = tf.concat([dydx_1, dydx_2], axis=-1)

                diff_y_dummy_2 = tf.concat([y_rep_1, y_rep_2], axis=-1)

                diff_x_dummy_2 =tf.concat([x_rep_1, x_rep_2], axis=-1)

                closest_y_dummy_2 = tf.concat([y_closest_1, y_closest_2], axis=-1)
                closest_x_dummy_2 = tf.concat([x_closest_1, x_closest_2], axis=-1)
                
                ########## concat all ############    
                deriv_dummy_full = tf.concat([deriv_dummy_full, deriv_dummy_2], axis=1)
                diff_y_dummy_full = tf.concat([diff_y_dummy_full, diff_y_dummy_2], axis=1)
                diff_x_dummy_full = tf.concat([diff_x_dummy_full, diff_x_dummy_2], axis=1)
                closest_y_dummy_full = tf.concat([closest_y_dummy_full, closest_y_dummy_2], axis=1)
                closest_x_dummy_full = tf.concat([closest_x_dummy_full, closest_x_dummy_2], axis=1)
                # print("closest_x_dummy_full.shape", closest_x_dummy_full.shape)
            return diff_y_dummy_full, diff_x_dummy_full, deriv_dummy_full, closest_x_dummy_full, closest_y_dummy_full




# ## We will need the date information in a numeric version 
# def date_to_numeric(col):
#     datetime = pd.to_datetime(col)
#     return datetime.dt.hour,  datetime.dt.day,  datetime.dt.month,  datetime.dt.year


if __name__ == 'main':
    ## test subsample function with real data
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    idx_list = list(np.arange(0, x_train.shape[0] - 25, 1))
    x, y, _, _ = batcher.batcher(x_train, y_train, idx_list=idx_list, window=70, batch_s=1)
    f = feature_extractor.feature_wrapper()
    x1, y1 = f.subsample(x[np.newaxis, :, np.newaxis], y[np.newaxis, :, np.newaxis], 20, 10, 5, 5)
