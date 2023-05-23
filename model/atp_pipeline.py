import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_wrangler.feature_extractor import  DE, feature_wrapper
from model.atp import ATP
from model.atp_no_leakage import ATP as ATP_no_leakage
from model.atp_no_leakage_new_block import ATP as ATP_new_block
from model.atp_no_leakage_xxx import ATP as ATP_no_leakage_xxx



class atp_pipeline(keras.models.Model):
    
    def __init__(self, num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                 bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, multiply=1, MHAX_leakage=True, subsample =True):
        super().__init__()
        # for testing set permutation_repeats=0
   
        self._permutation_repeats = permutation_repeats
        self.enc_dim = enc_dim
        self.xmin = xmin
        self.xmax = xmax
        self.multiply = multiply
        self._subsample = subsample
        self._feature_wrapper = feature_wrapper(multiply=self.multiply)
        if MHAX_leakage == True:
            self._atp = ATP(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == False:
            self._atp = ATP_no_leakage(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == "new_block":
            self._atp = ATP_new_block(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        elif MHAX_leakage == "xxx":
            self._atp = ATP_no_leakage_xxx(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
                        projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
        self._DE = DE()
    
    def concat_context_multi_ts(self, list_of_inputs, dim, n_C):
        return tf.concat([list_of_inputs[i][dim][:, :n_C, :] for i in range(len(list_of_inputs))], axis=1)

    def concat_target_multi_ts(self, list_of_inputs, dim, n_C, n_T, last_dim):
        x = tf.concat([list_of_inputs[i][dim][:, n_C:n_C+n_T, :][:, :, tf.newaxis, :] for i in range(len(list_of_inputs))], axis=2)
        x = tf.reshape(x, (x.shape[0], -1, last_dim))
        return x

    def inputs_for_multi_ts(self, x, y, n_C, n_T, n_C_s, n_T_s):
        batch_size = x.shape[0]
        inputs_for_processing = []
        eye = tf.eye(self.multiply)
        for i in range(self.multiply):

            # embed each ts separately and each dimension separately
            ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C + n_T), axis=0), (batch_size, -1, self.multiply)) # one hot encoding of the ts
            ts_start = i*(n_C + n_T)
            ts_end = (i+1)*(n_C + n_T)

            if self._subsample == True:
                x_temp, y_temp = self._feature_wrapper.subsample(x[:, ts_start:ts_end, :], y[:, ts_start:ts_end, :], n_C, n_T, n_C_s, n_T_s)
                n_C = n_C_s
                n_T = n_T_s
                ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C + n_T), axis=0), (batch_size, -1, self.multiply)) # overwrites the previous ts_label

            else:
                x_temp = x[:, ts_start:ts_end, :]
                y_temp = y[:, ts_start:ts_end, :]

            x_emb = [tf.concat([self._feature_wrapper.PE([x_temp[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]), ts_label], axis=-1) for dim_num in range(x_temp.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1) # (32, 30, 34)
            # take derivative of each ts separately
            y_diff, x_diff, d, x_n, y_n = self._DE([y_temp, x_temp, n_C, n_T, True]) #  (32, 30, 1),  (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1)

            inputs_for_processing.append([x_emb, y_temp[:, :, np.newaxis], y_diff, x_diff, d, x_n, y_n])  

        # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C, y1*, y2*, ..yk*, y1**, y2**, ..yk**)

        x_context = self.concat_context_multi_ts(inputs_for_processing, 0, n_C)
        x_target = self.concat_target_multi_ts(inputs_for_processing, 0, n_C, n_T, x_context.shape[-1])
        x_emb_c = tf.concat([x_context, x_target], axis=1)


        y_context = self.concat_context_multi_ts(inputs_for_processing, 1, n_C)[:, :, :, 0]
        y_target = self.concat_target_multi_ts(inputs_for_processing, 1, n_C, n_T, 1)
        zz3 = tf.concat([y_context, y_target], axis=1)

        y_context_diff = self.concat_context_multi_ts(inputs_for_processing, 2, n_C)
        y_target_diff = self.concat_target_multi_ts(inputs_for_processing, 2, n_C, n_T, y_context_diff.shape[-1])
        y_diff_c = tf.concat([y_context_diff, y_target_diff], axis=1)

        x_context_diff = self.concat_context_multi_ts(inputs_for_processing, 3, n_C)
        x_target_diff = self.concat_target_multi_ts(inputs_for_processing, 3, n_C, n_T, x_context_diff.shape[-1])
        x_diff_c = tf.concat([x_context_diff, x_target_diff], axis=1)

        d_context = self.concat_context_multi_ts(inputs_for_processing, 4, n_C)
        d_target = self.concat_target_multi_ts(inputs_for_processing, 4, n_C, n_T, d_context.shape[-1])
        d_c = tf.concat([d_context, d_target], axis=1)

        x_n_context = self.concat_context_multi_ts(inputs_for_processing, 5, n_C)
        x_n_target = self.concat_target_multi_ts(inputs_for_processing, 5, n_C, n_T, x_n_context.shape[-1])
        x_n_c = tf.concat([x_n_context, x_n_target], axis=1)

        y_n_context = self.concat_context_multi_ts(inputs_for_processing, 6, n_C)
        y_n_target = self.concat_target_multi_ts(inputs_for_processing, 6, n_C, n_T, y_n_context.shape[-1])

        y_n_c = tf.concat([y_n_context, y_n_target], axis=1)
        y_n = y_n_c
        inputs_for_processing = [x_emb_c, zz3, y_diff_c, x_diff_c, d_c, x_n_c, y_n_c, n_C , n_T]
        return inputs_for_processing, y_n, n_C, n_T





    def call(self, inputs):

        x, y, n_C, n_T, training, n_C_s, n_T_s = inputs #  (batch_size, n_C + n_T, 1), (batch_size, n_C + n_T, 1)
        
        x = x[:,:(n_C+n_T) * self.multiply,:]
        y = y[:,:(n_C+n_T) * self.multiply,:]

        if (self._subsample == True) and (self.multiply == 1):
            x, y = self._feature_wrapper.subsample(x, y, n_C, n_T, n_C_s, n_T_s)
            n_C = n_C_s
            n_T = n_T_s

        #x and y have shape batch size x length x dim
        # if training == True:    
            # x,y = self._feature_wrapper.permute([x, y, n_C * self.multiply, n_T * self.multiply, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?
        # print("y shape after permute:", y.shape)
        # permute returns y with shape batch size x NONE x dim ****** check this issue! ******
        ######## make mask #######
        

        if self.multiply == 1:
            x_emb = [self._feature_wrapper.PE([x[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for dim_num in range(x.shape[-1])] 
            x_emb = tf.concat(x_emb, axis=-1)  #(32, 30, 32) (batch_size, n_C + n_T, enc_dim) 

            ######## create derivative ########
            y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training]) #  (32, 30, 1), (32, 30, 1), (32, 30, 2), (32, 30, 1), (32, 30, 1) [n_C = 20 and n_T = 10]
            inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]

        else: 

            # # the end sequence will be (y_11,.., y_1n_C, y21, ..y_2n_C, y1*, y2*, ..yk*, y1**, y2**, ..yk**)
            inputs_for_processing, y_n, n_C, n_T = self.inputs_for_multi_ts(x, y, n_C, n_T, n_C_s, n_T_s)

        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing) #  (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim), (batch_size, multiply *(n_C_s + n_T_s), enc_dim + multiply + x_diff.dim + x_n.dim) , (batch_size, multiply *(n_C_s + n_T_s), 1), (batch_size, multiply *(n_C_s + n_T_s), enc.dim + multiply + 2 + label.dim + y.dim + y_diff.dim + d.dim + y_n.dim)
        value_x = tf.reshape(value_x, (value_x.shape[0], self.multiply * (n_C + n_T), value_x.shape[-1]))
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point
        mask = self._feature_wrapper.masker(n_C, n_T)
        μ, log_σ = self._atp([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest],training=training)

        return μ[:, n_C*self.multiply:], log_σ[:, n_C*self.multiply:]
      

def instantiate_atp(dataset,training=True):
    if dataset == "ETT":
        return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                    bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)
    elif dataset == "traffic":
        return atp_pipeline(num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
                    bound_std=False, num_layers=2, enc_dim=32, xmin=0.1, xmax=2)
    elif dataset == "exchange":
        # return atp_pipeline(num_heads=6, projection_shape_for_head=12, output_shape=32, rate=0.0, permutation_repeats=0,
        #          bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1)
        #the above is what i used for previous reults

        return atp_pipeline(num_heads=10, projection_shape_for_head=9, output_shape=32, rate=0.05, permutation_repeats=0,
                 bound_std=False, num_layers=6, enc_dim=32, xmin=0.1, xmax=1,MHAX_leakage="new_block")

    else:
        print('choose a valid dataset name')         
            
        

