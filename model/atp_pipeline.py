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
    



    def call(self, inputs):

        x, y, n_C, n_T, training, n_C_s, n_T_s = inputs
        batch_size = x.shape[0]
        
        x = x[:,:(n_C+n_T) * self.multiply,:]
        y = y[:,:(n_C+n_T) * self.multiply,:]

        if self._subsample == True:
            x, y = self._feature_wrapper.subsample(x, y, n_C, n_T, n_C_s, n_T_s)
            n_C = n_C_s
            n_T = n_T_s

        #x and y have shape batch size x length x dim
        # if training == True:    
            # x,y = self._feature_wrapper.permute([x, y, n_C * self.multiply, n_T * self.multiply, self._permutation_repeats]) ##### clean permute, and check permute target and/or context?
        # print("y shape after permute:", y.shape)
        # permute returns y with shape batch size x NONE x dim ****** check this issue! ******
        ######## make mask #######
        mask = self._feature_wrapper.masker(n_C, n_T)

        if self.multiply == 1:
            print(x.shape)
            x_emb = [self._feature_wrapper.PE([x[:, :, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for dim_num in range(x.shape[-1])] 
            print("x_emb shape:", x_emb[0].shape)
            x_emb = tf.concat(x_emb, axis=-1)
            # print("x_emb shape:", x_emb.shape)  (32, 30, 32) (batch_size, n_C + n_T, enc_dim)

            ######## create derivative ########
            y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training])
            # if n_C = 20 and n_T = 10:
                # print("y_diff shape:", y_diff.shape)   (32, 30, 1)
                # print("x_diff shape:", x_diff.shape) (32, 30, 1)
                # print("d shape:", d.shape) (32, 30, 2)
                # print("x_n shape:", x_n.shape) (32, 30, 1)
                # print("y_n shape:", y_n.shape) (32, 30, 1)

            inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]

            

        else: 

            batch_size = x.shape[0]
            inputs_for_processing = []
            eye = tf.eye(self.multiply)
            for i in range(self.multiply):
                # embed each ts separately and each dimension separately
                ts_label = tf.reshape(tf.repeat(eye[:, i][tf.newaxis, :], batch_size*(n_C + n_T), axis=0), (batch_size, -1, self.multiply))
                ts_start = i*(n_C + n_T)
                ts_end = (i+1)*(n_C + n_T)
                x_emb = [tf.concat([self._feature_wrapper.PE([x[:, ts_start:ts_end, dim_num][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]), ts_label], axis=-1) for dim_num in range(x.shape[-1])] 
                print("x_emb shape:", x_emb[0].shape)
                x_emb = tf.concat(x_emb, axis=-1)
                print("x_emb shape:", x_emb.shape)
                # take derivative of each ts separately
                y_diff, x_diff, d, x_n, y_n = self._DE([y[:, i*(n_C + n_T):(i+1)*(n_C + n_T)], x[:, i*(n_C + n_T):(i+1)*(n_C + n_T), 0][:, :, np.newaxis], n_C, n_T, True])
                print("y_diff shape:", y_diff.shape)
                print("x_diff shape:", x_diff.shape)
                print("d shape:", d.shape)
                print("x_n shape:", x_n.shape)
                print("y_n shape:", y_n.shape)

                inputs_for_processing.append([x_emb, y[:, i*(n_C + n_T):(i+1)*(n_C + n_T)][:, :, np.newaxis], y_diff, x_diff, d, x_n, y_n, n_C, n_T])  

            x_emb_a = tf.concat([inputs_for_processing[i][0][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_emb_b = tf.concat([inputs_for_processing[i][0][:, n_C:n_C+n_T, :][:, :, tf.newaxis, :] for i in range(len(inputs_for_processing))], axis=2)
            x_emb_b = tf.reshape(x_emb_b, (x_emb_b.shape[0], -1, x_emb_a.shape[-1]))
            x_emb_c = tf.concat([x_emb_a, x_emb_b], axis=1)

            zz1 = tf.concat([inputs_for_processing[i][1][:, :n_C, :, 0] for i in range(len(inputs_for_processing))], axis=1)
            zz2 = tf.concat([inputs_for_processing[i][1][:, n_C:n_C+n_T, :, 0] for i in range(len(inputs_for_processing))], axis=2)
            zz2 = tf.reshape(zz2, (zz2.shape[0], -1, 1))
            zz3 = tf.concat([zz1, zz2], axis=1)


            y_diff_a = tf.concat([inputs_for_processing[i][2][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            y_diff_b = tf.concat([inputs_for_processing[i][2][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            y_diff_b = tf.reshape(y_diff_b, (y_diff_b.shape[0], -1, y_diff_a.shape[-1]))
            y_diff_c = tf.concat([y_diff_a, y_diff_b], axis=1)

            x_diff_a = tf.concat([inputs_for_processing[i][3][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_diff_b = tf.concat([inputs_for_processing[i][3][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            x_diff_b = tf.reshape(x_diff_b, (x_diff_b.shape[0], -1, x_diff_a.shape[-1]))
            x_diff_c = tf.concat([x_diff_a, x_diff_b], axis=1)

            d_a = tf.concat([inputs_for_processing[i][4][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            d_b = tf.concat([inputs_for_processing[i][4][:, n_C:n_C+n_T, :][:, :, tf.newaxis, :] for i in range(len(inputs_for_processing))], axis=2)
            d_b = tf.reshape(d_b, (d_b.shape[0], -1, d_a.shape[-1]))
            d_c = tf.concat([d_a, d_b], axis=1)

            x_n_a = tf.concat([inputs_for_processing[i][5][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            x_n_b = tf.concat([inputs_for_processing[i][5][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            x_n_b = tf.reshape(x_n_b, (x_n_b.shape[0], -1, x_n_a.shape[-1]))
            x_n_c = tf.concat([x_n_a, x_n_b], axis=1)

            y_n_a = tf.concat([inputs_for_processing[i][6][:, :n_C, :] for i in range(len(inputs_for_processing))], axis=1)
            y_n_b = tf.concat([inputs_for_processing[i][6][:, n_C:n_C+n_T, :] for i in range(len(inputs_for_processing))], axis=2)
            y_n_b = tf.reshape(y_n_b, (y_n_b.shape[0], -1, y_n_a.shape[-1]))
            y_n_c = tf.concat([y_n_a, y_n_b], axis=1)
            y_n = y_n_c
            inputs_for_processing = [x_emb_c, zz3, y_diff_c, x_diff_c, d_c, x_n_c, y_n_c, n_C , n_T]

        query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing)
        value_x = tf.reshape(value_x, (value_x.shape[0], n_C + n_T, value_x.shape[-1]))
        y_n_closest = y_n[:, :, :y.shape[-1]] #### need to update this based on how we pick closest point

        # print("query_x", query_x.shape)
        # print("key_x", key_x.shape)
        # print("value_x", value_x.shape)
        # print("query_xy", query_xy.shape)
        # print("key_xy", key_xy.shape)
        # print("value_xy", value_xy.shape)
        # print("mask", mask.shape)
        # print("y_n_closest", y_n_closest.shape)
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
            
        

